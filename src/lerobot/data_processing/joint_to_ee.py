#!/usr/bin/env python3
"""
Convert existing joint-space LeRobot datasets to end-effector (EE) representation.

Handles full mobile-manipulator schemas (arms + head + base + lift) by:
  1. Identifying arm joint groups vs pass-through non-arm dims (by name).
  2. Running forward kinematics per arm with either:
     - ``classic`` — analytic SO-101 2-link model (same conventions as ``xlerobot_vr`` /
       ``lerobot-umi-retarget``); no URDF required.
     - ``urdf`` — placo ``RobotKinematics`` on an SO-101 arm URDF (full 6-DOF pose).

Non-arm components (``head_*``, ``x/y/theta.vel``, ``gantry.*``, etc.) are copied
unchanged from the source so base/lift/head trajectories stay aligned with EE arms.

Usage (xlerobot 18-DOF master → 20-DOF EE + head/base/lift pass-through)::

    lerobot-joint-to-ee \\
        --source-repo Odog16/master_home_v1 \\
        --target-repo-id Odog16/master_home_v1_ee \\
        --profile xlerobot_pass_through \\
        --solver urdf \\
        --joint-units degrees

Usage (bimanual arms only, 12 joint dims → 14 EE dims)::

    lerobot-joint-to-ee \\
        --source-repo Odog16/task_v1 \\
        --target-repo-id Odog16/task_v1_ee \\
        --profile bimanual12 \\
        --solver classic \\
        --ee-format euler

See ``DATA_CONVERSION.md`` §3.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from tqdm import tqdm

from lerobot.data_processing.co_training_utils import (
    _compute_resample_indices,
    _ensure_depth_chw_numpy,
    _ensure_image_hwc_numpy,
    _remap_vector_by_names,
)
from lerobot.data_processing.extract_subset import (
    XLEROBOT_ARM_NAMES,
    XLEROBOT_FULL_NAMES,
    _feature_names,
    build_subset_features,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import FrameTimestampError
from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.rotation import Rotation

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s", force=True)
logger = logging.getLogger(__name__)

# Canonical SO-101 arm joints (URDF / teleop naming, without ``.pos`` suffix).
SO101_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

LEFT_ARM_JOINTS = [f"left_arm_{j}" for j in SO101_JOINTS]
RIGHT_ARM_JOINTS = [f"right_arm_{j}" for j in SO101_JOINTS]

# xlerobot / OB15 non-arm dims copied verbatim when using pass-through profiles.
XLEROBOT_PASS_THROUGH = [
    "head_pan.pos",
    "head_tilt.pos",
    "x.vel",
    "y.vel",
    "theta.vel",
    "gantry.height_mm",
]

EE_DIM_ROTVEC = ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]
EE_DIM_EULER = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


class EeFormat(str, Enum):
    ROTVEC = "rotvec"
    EULER = "euler"


class JointUnits(str, Enum):
    DEGREES = "degrees"
    NORMALIZED = "normalized"


@dataclass(frozen=True)
class ArmGroup:
    """One manipulator arm: LeRobot motor names and optional URDF joint names."""

    name: str
    joint_names: tuple[str, ...]
    urdf_joint_names: tuple[str, ...] = field(default_factory=lambda: tuple(SO101_JOINTS))

    def ee_output_names(self, ee_format: EeFormat) -> list[str]:
        dims = EE_DIM_ROTVEC if ee_format == EeFormat.ROTVEC else EE_DIM_EULER
        prefix = f"{self.name}_" if self.name else ""
        if ee_format == EeFormat.ROTVEC:
            return [f"{prefix}ee.{d}" if d != "gripper_pos" else f"{prefix}ee.gripper_pos" for d in dims]
        return [f"{prefix}{d}" for d in dims]


@dataclass(frozen=True)
class ConversionProfile:
    """Describes which arms to FK-convert and which source dims to pass through."""

    arms: tuple[ArmGroup, ...]
    pass_through_names: tuple[str, ...]

    def output_state_names(self, ee_format: EeFormat) -> list[str]:
        names: list[str] = []
        for arm in self.arms:
            names.extend(arm.ee_output_names(ee_format))
        names.extend(self.pass_through_names)
        return names


PROFILES: dict[str, ConversionProfile] = {
    "so101": ConversionProfile(
        arms=(ArmGroup(name="", joint_names=tuple(SO101_JOINTS)),),
        pass_through_names=(),
    ),
    "bimanual12": ConversionProfile(
        arms=(
            ArmGroup(name="left", joint_names=tuple(LEFT_ARM_JOINTS)),
            ArmGroup(name="right", joint_names=tuple(RIGHT_ARM_JOINTS)),
        ),
        pass_through_names=(),
    ),
    "xlerobot_pass_through": ConversionProfile(
        arms=(
            ArmGroup(name="left", joint_names=tuple(LEFT_ARM_JOINTS)),
            ArmGroup(name="right", joint_names=tuple(RIGHT_ARM_JOINTS)),
        ),
        pass_through_names=tuple(XLEROBOT_PASS_THROUGH),
    ),
    "xlerobot_full": ConversionProfile(
        arms=(
            ArmGroup(name="left", joint_names=tuple(LEFT_ARM_JOINTS)),
            ArmGroup(name="right", joint_names=tuple(RIGHT_ARM_JOINTS)),
        ),
        pass_through_names=tuple(n for n in XLEROBOT_FULL_NAMES if n not in XLEROBOT_ARM_NAMES),
    ),
}


class ForwardKinematicsSolver(Protocol):
    def arm_joints_to_ee(self, joints_deg: dict[str, float], arm: ArmGroup) -> np.ndarray:
        """Map arm joint values (degrees / gripper native) to a 7-dim EE vector."""


def _suffix_pos(name: str) -> str:
    return name if name.endswith(".pos") or name.endswith(".vel") or name.endswith("_mm") else f"{name}.pos"


def _joint_values_from_vector(
    vec: np.ndarray,
    src_names: list[str],
    arm_joint_names: tuple[str, ...],
) -> dict[str, float]:
    keyed = {_suffix_pos(n): n for n in arm_joint_names}
    out: dict[str, float] = {}
    for jn in arm_joint_names:
        key = _suffix_pos(jn)
        if key not in src_names:
            raise KeyError(f"Joint {key!r} not found in source names (have {src_names})")
        out[jn] = float(vec[src_names.index(key)])
    return out


def _to_degrees(value: float, joint_name: str, units: JointUnits, norm_scale: float) -> float:
    if units == JointUnits.DEGREES:
        return value
    if joint_name == "gripper" or joint_name.endswith("_gripper"):
        return value
    return value * norm_scale


def _prepare_arm_joints_deg(
    joints: dict[str, float],
    units: JointUnits,
    norm_scale: float,
) -> dict[str, float]:
    return {k: _to_degrees(v, k, units, norm_scale) for k, v in joints.items()}


def _rotation_matrix_to_euler_xyz(rot: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw (rad) from a 3×3 rotation matrix (XYZ convention)."""
    sy = math.sqrt(float(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    if sy >= 1e-6:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


def _pack_ee_vector(
    pos: np.ndarray,
    orientation: np.ndarray,
    gripper: float,
    ee_format: EeFormat,
) -> np.ndarray:
    if ee_format == EeFormat.ROTVEC:
        return np.array(
            [pos[0], pos[1], pos[2], orientation[0], orientation[1], orientation[2], gripper],
            dtype=np.float32,
        )
    roll, pitch, yaw = orientation
    return np.array([pos[0], pos[1], pos[2], roll, pitch, yaw, gripper], dtype=np.float32)


def _normalize_gripper_value(gripper: float) -> float:
    """Return gripper in [0, 1] when possible; otherwise pass through (e.g. 0–100)."""
    if 0.0 <= gripper <= 1.0:
        return gripper
    if 0.0 <= gripper <= 100.0:
        return gripper / 100.0
    return gripper


@dataclass
class ClassicSO101Solver:
    """Analytic FK matching ``xlerobot_vr`` / ``umi_retarget`` decomposed conventions."""

    ee_format: EeFormat
    kinematics: SO101Kinematics = field(default_factory=SO101Kinematics)

    def arm_joints_to_ee(self, joints_deg: dict[str, float], arm: ArmGroup) -> np.ndarray:
        canonical = {short: joints_deg[full] for short, full in zip(SO101_JOINTS, arm.joint_names, strict=True)}
        pan_rad = math.radians(canonical["shoulder_pan"])
        lift = canonical["shoulder_lift"]
        elbow = canonical["elbow_flex"]
        wrist_flex = canonical["wrist_flex"]
        wrist_roll = canonical["wrist_roll"]
        gripper = _normalize_gripper_value(canonical["gripper"])

        x_local, y_local = self.kinematics.forward_kinematics(lift, elbow)
        fwd = x_local * math.cos(pan_rad) - y_local * math.sin(pan_rad)
        lat = x_local * math.sin(pan_rad) + y_local * math.cos(pan_rad)
        up = 0.0

        pitch = math.radians(wrist_flex + lift + elbow)
        roll = math.radians(wrist_roll)
        yaw = pan_rad

        if self.ee_format == EeFormat.EULER:
            return _pack_ee_vector(np.array([fwd, lat, up]), np.array([roll, pitch, yaw]), gripper, self.ee_format)

        # rotvec: build rotation from euler components (yaw * pitch * roll approx via matrix)
        rot = (
            Rotation.from_rotvec(np.array([0.0, 0.0, yaw]))
            .as_matrix()
            @ Rotation.from_rotvec(np.array([0.0, pitch, 0.0])).as_matrix()
            @ Rotation.from_rotvec(np.array([roll, 0.0, 0.0])).as_matrix()
        )
        rotvec = Rotation.from_matrix(rot).as_rotvec()
        return _pack_ee_vector(np.array([fwd, lat, up]), rotvec, gripper, self.ee_format)


@dataclass
class UrdfSolver:
    """Placo URDF forward kinematics (full 6-DOF EE pose)."""

    ee_format: EeFormat
    urdf_path: str
    target_frame_name: str = "gripper_frame_link"
    _kinematics: Any = field(default=None, init=False, repr=False)

    def _get_kinematics(self, urdf_joint_names: tuple[str, ...]):
        from lerobot.model.kinematics import RobotKinematics

        if self._kinematics is None:
            self._kinematics = RobotKinematics(
                urdf_path=self.urdf_path,
                target_frame_name=self.target_frame_name,
                joint_names=list(urdf_joint_names),
            )
        return self._kinematics

    def arm_joints_to_ee(self, joints_deg: dict[str, float], arm: ArmGroup) -> np.ndarray:
        kin = self._get_kinematics(arm.urdf_joint_names)
        q = np.array([joints_deg[jn] for jn in arm.joint_names], dtype=float)
        transform = kin.forward_kinematics(q)
        pos = transform[:3, 3]
        rotvec = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
        gripper = joints_deg[arm.joint_names[-1]]
        if self.ee_format == EeFormat.ROTVEC:
            return _pack_ee_vector(pos, rotvec, gripper, self.ee_format)
        roll, pitch, yaw = _rotation_matrix_to_euler_xyz(transform[:3, :3])
        return _pack_ee_vector(pos, np.array([roll, pitch, yaw]), gripper, self.ee_format)


def ensure_so101_urdf() -> str:
    """Fetch or return cached SO-101 URDF (meshes included) from the HF robot-urdfs bucket."""
    dest_dir = HF_LEROBOT_HOME / "robot-urdfs" / "so101"
    urdf_path = dest_dir / "so101_new_calib.urdf"
    marker = dest_dir / ".sync_complete"
    if not marker.exists():
        from huggingface_hub import sync_bucket

        logger.info("Downloading SO-101 URDF bundle to %s …", dest_dir)
        sync_bucket("hf://buckets/lerobot/robot-urdfs/so101", str(dest_dir), quiet=True)
        marker.touch()
    return str(urdf_path)


def build_solver(
    solver_name: Literal["classic", "urdf"],
    ee_format: EeFormat,
    urdf_path: str | None,
    target_frame_name: str,
) -> ForwardKinematicsSolver:
    if solver_name == "classic":
        return ClassicSO101Solver(ee_format=ee_format)
    path = urdf_path or ensure_so101_urdf()
    if not Path(path).is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    return UrdfSolver(ee_format=ee_format, urdf_path=path, target_frame_name=target_frame_name)


def convert_joint_vector(
    vec: np.ndarray,
    src_names: list[str],
    profile: ConversionProfile,
    solver: ForwardKinematicsSolver,
    ee_format: EeFormat,
    units: JointUnits,
    norm_scale: float,
) -> np.ndarray:
    """Convert one flat joint vector to EE (+ optional pass-through dims)."""
    out_names = profile.output_state_names(ee_format)
    parts: list[float] = []

    for arm in profile.arms:
        raw_joints = _joint_values_from_vector(vec, src_names, arm.joint_names)
        joints_deg = _prepare_arm_joints_deg(raw_joints, units, norm_scale)
        ee_vec = solver.arm_joints_to_ee(joints_deg, arm)
        parts.extend(float(x) for x in ee_vec)

    if profile.pass_through_names:
        passthrough = _remap_vector_by_names(vec, src_names, list(profile.pass_through_names))
        parts.extend(float(x) for x in passthrough)

    if len(parts) != len(out_names):
        raise RuntimeError(f"Internal error: expected {len(out_names)} outputs, got {len(parts)}")
    return np.array(parts, dtype=np.float32)


def build_ee_features(
    src_features: dict[str, Any],
    output_names: list[str],
    cameras: list[str] | None,
    drop_depth: bool,
    target_image_size: tuple[int, int] | None,
    effective_fps: int,
) -> tuple[dict[str, Any], list[str]]:
    return build_subset_features(
        src_features,
        output_names,
        cameras,
        drop_depth,
        target_image_size,
        effective_fps,
    )


def infer_profile_from_names(state_names: list[str]) -> ConversionProfile | None:
    """Auto-detect xlerobot / SO-101 layout from ``observation.state`` names."""
    name_set = set(state_names)
    if all(_suffix_pos(j) in name_set for j in LEFT_ARM_JOINTS) and all(
        _suffix_pos(j) in name_set for j in RIGHT_ARM_JOINTS
    ):
        passthrough = tuple(n for n in XLEROBOT_PASS_THROUGH if n in name_set)
        if passthrough:
            return PROFILES["xlerobot_pass_through"]
        return PROFILES["bimanual12"]
    if all(_suffix_pos(j) in name_set for j in SO101_JOINTS):
        return PROFILES["so101"]
    return None


def convert_joint_dataset_to_ee(
    source_repo: str,
    target_repo_id: str,
    profile: ConversionProfile,
    solver: ForwardKinematicsSolver,
    ee_format: EeFormat,
    joint_units: JointUnits = JointUnits.DEGREES,
    norm_scale: float = 1.8,
    output_root: Path | None = None,
    cameras: list[str] | None = None,
    drop_depth: bool = True,
    target_fps: int | None = None,
    target_image_size: tuple[int, int] | None = None,
    robot_type: str | None = None,
    push_to_hub: bool = False,
    force_rebuild: bool = False,
) -> Path:
    """Rewrite *source_repo* with arm joints replaced by EE poses."""
    src_ds = LeRobotDataset(source_repo)
    src_meta = src_ds.meta
    src_fps = src_meta.fps
    effective_fps = min(src_fps, target_fps) if target_fps else src_fps

    src_state_names = _feature_names(src_meta.features, "observation.state")
    src_action_names = _feature_names(src_meta.features, "action")

    out_state_names = profile.output_state_names(ee_format)
    for label, src_names, arm_joints in (
        ("state", src_state_names, [j for arm in profile.arms for j in arm.joint_names]),
        ("action", src_action_names, [j for arm in profile.arms for j in arm.joint_names]),
    ):
        missing = [_suffix_pos(j) for j in arm_joints if _suffix_pos(j) not in src_names]
        if missing:
            raise ValueError(
                f"Source {label} is missing arm joints required by profile: {missing}. "
                f"Source has: {src_names}"
            )
        for pt in profile.pass_through_names:
            if pt not in src_names:
                raise ValueError(
                    f"Pass-through dim {pt!r} not in source {label} names. Source has: {src_names}"
                )

    target_features, dropped_cams = build_ee_features(
        src_meta.features,
        out_state_names,
        cameras,
        drop_depth,
        target_image_size,
        effective_fps,
    )
    cam_keys = [k for k in target_features if k.startswith("observation.images.")]
    if not cam_keys:
        raise ValueError("No cameras left after --cameras / --drop-depth filtering.")

    if output_root is None:
        output_root = Path(f"~/.cache/huggingface/lerobot/{target_repo_id}").expanduser()
    if output_root.exists():
        if not force_rebuild:
            raise FileExistsError(f"{output_root} exists. Pass --force-rebuild to overwrite.")
        logger.warning("Force rebuild: removing %s", output_root)
        shutil.rmtree(output_root)

    logger.info(
        "Joint→EE %s → %s | solver=%s format=%s | dims %d→%d | fps %d→%d",
        source_repo,
        target_repo_id,
        type(solver).__name__,
        ee_format.value,
        len(src_state_names),
        len(out_state_names),
        src_fps,
        effective_fps,
    )

    out_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=effective_fps,
        robot_type=robot_type or src_meta.robot_type or "xlerobot",
        features=target_features,
        root=output_root,
        use_videos=True,
        image_writer_threads=4,
    )

    target_keys = set(target_features)
    num_episodes = src_meta.total_episodes
    for ep_idx in tqdm(range(num_episodes), desc="Convert episodes", unit="ep", dynamic_ncols=True):
        ep_ds = LeRobotDataset(source_repo, episodes=[ep_idx])
        ep_info = src_meta.episodes[ep_idx]
        task_idx = ep_info.get("task_index", 0)
        if isinstance(task_idx, (list, np.ndarray)):
            task_idx = int(task_idx[0])
        task = src_meta.tasks.index[int(task_idx)]
        task = task if isinstance(task, str) else str(task)

        indices = _compute_resample_indices(src_fps, effective_fps, len(ep_ds))
        frames_written = 0
        skipped_bad_ts = 0
        for pos in indices:
            try:
                raw = ep_ds[pos]
            except FrameTimestampError:
                skipped_bad_ts += 1
                continue
            frame: dict[str, Any] = {}
            for key in ("observation.state", "action"):
                vec = raw.get(key)
                if vec is None:
                    continue
                arr = vec.numpy() if isinstance(vec, torch.Tensor) else np.asarray(vec)
                src_names = src_state_names if key == "observation.state" else src_action_names
                frame[key] = torch.from_numpy(
                    convert_joint_vector(
                        arr,
                        src_names,
                        profile,
                        solver,
                        ee_format,
                        joint_units,
                        norm_scale,
                    )
                )
            for key in cam_keys:
                if key not in raw:
                    continue
                feat = target_features[key]
                if feat.get("dtype") == "depth":
                    frame[key] = _ensure_depth_chw_numpy(raw[key], target_hw=target_image_size)
                else:
                    frame[key] = _ensure_image_hwc_numpy(raw[key], feat, target_hw=target_image_size)
            frame = {k: v for k, v in frame.items() if k in target_keys}
            frame["task"] = task
            out_ds.add_frame(frame)
            frames_written += 1
        if skipped_bad_ts:
            tqdm.write(f"  episode {ep_idx}: skipped {skipped_bad_ts} frames with bad video timestamps")
        if frames_written:
            out_ds.save_episode()
        else:
            out_ds.clear_episode_buffer()

    out_ds.finalize()
    logger.info("EE dataset written to %s", output_root)

    if push_to_hub:
        logger.info("Pushing %s to the Hub…", target_repo_id)
        LeRobotDataset(target_repo_id, root=output_root).push_to_hub()
    return output_root


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(
        description="Convert joint-space LeRobot datasets to end-effector representation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--target-repo-id", required=True)
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        help="Named conversion layout: " + ", ".join(sorted(PROFILES)),
    )
    profile_group.add_argument(
        "--auto-profile",
        action="store_true",
        help="Infer profile from observation.state names (xlerobot / SO-101).",
    )
    parser.add_argument(
        "--solver",
        choices=("classic", "urdf"),
        default="urdf",
        help="FK backend: classic analytic SO-101 or URDF/placo (default: urdf).",
    )
    parser.add_argument(
        "--ee-format",
        choices=[e.value for e in EeFormat],
        default=EeFormat.ROTVEC.value,
        help="EE layout: rotvec (ee.x/wx, LeRobot processor style) or euler (UMI style).",
    )
    parser.add_argument(
        "--joint-units",
        choices=[u.value for u in JointUnits],
        default=JointUnits.DEGREES.value,
        help="Units of arm joint values in the source dataset.",
    )
    parser.add_argument(
        "--norm-scale",
        type=float,
        default=1.8,
        help="When --joint-units=normalized, multiply non-gripper joints by this to get degrees "
        "(default 1.8 ≈ [-100,100] → ±180°).",
    )
    parser.add_argument("--urdf-path", type=Path, default=None, help="SO-101 arm URDF (urdf solver).")
    parser.add_argument(
        "--target-frame-name",
        default="gripper_frame_link",
        help="End-effector link name in the URDF.",
    )
    parser.add_argument("--cameras", nargs="+", default=None, help="Camera short names to keep.")
    parser.add_argument("--keep-depth", action="store_true", help="Keep depth streams.")
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--target-image-size", type=str, default=None, help="HxW, e.g. 360x640")
    parser.add_argument("--robot-type", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--push-to-hub",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
    )
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    ee_format = EeFormat(args.ee_format)
    joint_units = JointUnits(args.joint_units)

    if args.profile:
        profile = PROFILES[args.profile]
    elif args.auto_profile:
        src_meta = LeRobotDataset(args.source_repo).meta
        state_names = _feature_names(src_meta.features, "observation.state")
        inferred = infer_profile_from_names(state_names)
        if inferred is None:
            raise ValueError(
                f"Could not infer profile from state names: {state_names}. Pass --profile explicitly."
            )
        profile = inferred
        logger.info("Auto-detected profile with pass-through=%s", profile.pass_through_names)
    else:
        raise ValueError("Pass --profile or --auto-profile.")

    size = None
    if args.target_image_size:
        parts = args.target_image_size.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"--target-image-size must be HxW, got {args.target_image_size!r}")
        size = (int(parts[0]), int(parts[1]))

    solver = build_solver(
        solver_name=args.solver,
        ee_format=ee_format,
        urdf_path=str(args.urdf_path.expanduser()) if args.urdf_path else None,
        target_frame_name=args.target_frame_name,
    )

    convert_joint_dataset_to_ee(
        source_repo=args.source_repo,
        target_repo_id=args.target_repo_id,
        profile=profile,
        solver=solver,
        ee_format=ee_format,
        joint_units=joint_units,
        norm_scale=args.norm_scale,
        output_root=args.output_root.expanduser() if args.output_root else None,
        cameras=args.cameras,
        drop_depth=not args.keep_depth,
        target_fps=args.target_fps,
        target_image_size=size,
        robot_type=args.robot_type,
        push_to_hub=args.push_to_hub,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
