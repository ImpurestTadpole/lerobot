#!/usr/bin/env python3
"""
Retarget UMI-style end-effector datasets (e.g. IPEC-COMMUNITY/FastUMI_100k_lerobot)
to the xlerobot embodiment.

FastUMI conventions (verified against the released data):
- LeRobot **v2.1** layout, one sub-dataset per task under ``single_arm/<task>`` /
  ``dual_arm/<task>``; 20 fps; wrist fisheye cameras at 720x1280.
- ``observation.state[t]`` is the **absolute EE pose relative to the episode
  start** (meters / radians): ``[x, y, z, roll, pitch, yaw, gripper]`` per arm
  (dual-arm = left 7 + right 7). Gripper is normalized width in [0, 1].
- ``action[t] == state[t+1]`` exactly (next-frame absolute pose).

Two output modes (``--action-space``):

``joint`` (Option A)
    Bakes the EE trajectory into **18-dim xlerobot joint space** using the same
    decomposed IK conventions as the VR teleoperator (``xlerobot_vr``):
    ``shoulder_pan = atan2(lateral, forward)``, 2-link planar IK
    (:class:`~lerobot.model.SO101Robot.SO101Kinematics`) for
    shoulder_lift/elbow_flex, ``wrist_flex = -(lift + elbow) + pitch``,
    ``wrist_roll = roll``, gripper width -> [gripper-range]. Head/base/lift
    dims are constants. Joint names are copied from ``--match-features-from``
    (one of your xlerobot task repos) so the output merges cleanly with
    ``lerobot-cotrain-align``.

``ee`` (Option B)
    Skips IK and writes the **robot-frame EE trajectories** (anchored to the
    arm workspace, meters/radians, gripper in [0, 1]) with FastUMI's motor
    names. Use this to co-train an EE-space policy; at deployment the same
    conversion runs live via ``InverseKinematicsEEToJoints`` (see
    https://huggingface.co/docs/lerobot/en/action_representations).

Both modes share Stage 1 (EE extraction + axis mapping + workspace anchoring),
so switching A -> B later does not change the data conventions.

Usage:
    lerobot-umi-retarget \\
        --source-root ~/fastumi/dual_arm/Clean_Desktop \\
        --target-repo-id Odog16/umi_clean_desktop_joint \\
        --action-space joint \\
        --match-features-from Odog16/tool_pickup \\
        --max-episodes 50

    # Download a task subset first:
    huggingface-cli download IPEC-COMMUNITY/FastUMI_100k_lerobot \\
        --repo-type dataset --include "dual_arm/Clean_Desktop/*" \\
        --local-dir ~/fastumi
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.model.SO101Robot import SO101Kinematics

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Episode-start EE anchor (planar reach / height, meters). The VR teleop rest
# pose (0.1629, 0.1131) sits at r=0.198 of the 0.251 m workspace radius — too
# close to the edge for UMI excursions in both directions. Mid-workspace
# anchoring keeps scaled human trajectories reachable (verified on FastUMI
# Clean_Desktop: 0 clamped frames at scale 0.3 vs 46-87% at the VR anchor).
DEFAULT_ANCHOR_FWD = 0.12
DEFAULT_ANCHOR_UP = 0.03
DEFAULT_SCALE = 0.3

UMI_EE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

XLE_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


# ---------------------------------------------------------------------------
# Stage 1 — EE trajectory extraction (shared by joint and ee modes)
# ---------------------------------------------------------------------------


@dataclass
class AxisMap:
    """Maps UMI (x, y, z) columns to robot (forward, lateral, up) axes."""

    fwd: tuple[int, float]  # (source column, sign)
    lat: tuple[int, float]
    up: tuple[int, float]

    @classmethod
    def parse(cls, spec: str) -> AxisMap:
        """Parse e.g. '+x,+y,+z' as (fwd, lat, up) = (+x, +y, +z)."""
        cols = {"x": 0, "y": 1, "z": 2}
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--axis-map needs 3 comma-separated entries, got {spec!r}")
        out = []
        for p in parts:
            sign = -1.0 if p.startswith("-") else 1.0
            axis = p.lstrip("+-")
            if axis not in cols:
                raise ValueError(f"Bad axis {p!r} in --axis-map (use +x/-x/+y/-y/+z/-z)")
            out.append((cols[axis], sign))
        return cls(fwd=out[0], lat=out[1], up=out[2])

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        """(T, 3) UMI positions -> (T, 3) [forward, lateral, up]."""
        return np.stack(
            [
                xyz[:, self.fwd[0]] * self.fwd[1],
                xyz[:, self.lat[0]] * self.lat[1],
                xyz[:, self.up[0]] * self.up[1],
            ],
            axis=1,
        )


def umi_to_robot_ee(
    vec7: np.ndarray,
    axis_map: AxisMap,
    scale: float,
    anchor_fwd: float,
    anchor_up: float,
) -> np.ndarray:
    """(T, 7) UMI EE (episode-start-relative) -> (T, 7) robot-frame EE.

    Output columns: [forward_m, lateral_m, up_m, roll_rad, pitch_rad, yaw_rad,
    gripper_01], anchored so the episode starts at (anchor_fwd, 0, anchor_up).
    """
    pos = axis_map.apply(vec7[:, :3]) * scale
    pos[:, 0] += anchor_fwd
    pos[:, 2] += anchor_up
    out = vec7.copy().astype(np.float32)
    out[:, :3] = pos
    return out


# ---------------------------------------------------------------------------
# Stage 2 — decomposed IK (VR teleop conventions)
# ---------------------------------------------------------------------------


class ArmRetargeter:
    """EE pose -> 6 xlerobot arm joints, matching xlerobot_vr conventions."""

    def __init__(self, gripper_range: tuple[float, float]):
        self.kin = SO101Kinematics()
        self.g_min, self.g_max = gripper_range
        self.clamped_frames = 0
        self.total_frames = 0

    def __call__(self, ee: np.ndarray) -> np.ndarray:
        """(T, 7) robot-frame EE -> (T, 6) [pan, lift, elbow, wrist_flex, wrist_roll, gripper] deg."""
        n_frames = ee.shape[0]
        joints = np.zeros((n_frames, 6), dtype=np.float32)
        r_max = self.kin.l1 + self.kin.l2
        r_min = abs(self.kin.l1 - self.kin.l2)
        for t in range(n_frames):
            fwd, lat, up, roll, pitch, _yaw, grip = ee[t]
            pan_deg = math.degrees(math.atan2(lat, fwd))
            reach = math.hypot(fwd, lat)  # planar reach in the pan direction
            self.total_frames += 1
            if not (r_min <= math.hypot(reach, up) <= r_max):
                self.clamped_frames += 1
            lift_deg, elbow_deg = self.kin.inverse_kinematics(reach, up)
            wrist_flex = -(lift_deg + elbow_deg) + math.degrees(pitch)
            wrist_roll = float(np.clip(math.degrees(roll), -90.0, 90.0))
            gripper = self.g_min + float(np.clip(grip, 0.0, 1.0)) * (self.g_max - self.g_min)
            joints[t] = [
                float(np.clip(pan_deg, -180.0, 180.0)),
                lift_deg,
                elbow_deg,
                float(np.clip(wrist_flex, -110.0, 110.0)),
                wrist_roll,
                gripper,
            ]
        return joints

    @property
    def clamp_fraction(self) -> float:
        return self.clamped_frames / max(1, self.total_frames)


# ---------------------------------------------------------------------------
# v2.1 source reading
# ---------------------------------------------------------------------------


@dataclass
class UmiSource:
    root: Path
    info: dict
    tasks: dict[int, str]
    episodes: list[dict]
    camera_keys: list[str]
    dual_arm: bool
    fps: int

    @classmethod
    def load(cls, root: Path) -> UmiSource:
        info = json.loads((root / "meta" / "info.json").read_text())
        if not str(info.get("codebase_version", "")).startswith("v2"):
            raise ValueError(
                f"Expected a LeRobot v2.x source, got {info.get('codebase_version')!r} at {root}"
            )
        tasks = {}
        with open(root / "meta" / "tasks.jsonl") as f:
            for line in f:
                d = json.loads(line)
                tasks[d["task_index"]] = d["task"]
        episodes = []
        with open(root / "meta" / "episodes.jsonl") as f:
            for line in f:
                episodes.append(json.loads(line))
        camera_keys = [k for k in info["features"] if k.startswith("observation.images.")]
        action_dim = info["features"]["action"]["shape"][0]
        return cls(
            root=root,
            info=info,
            tasks=tasks,
            episodes=episodes,
            camera_keys=camera_keys,
            dual_arm=action_dim == 14,
            fps=int(info["fps"]),
        )

    def parquet_path(self, ep_idx: int) -> Path:
        chunk = ep_idx // self.info["chunks_size"]
        return self.root / self.info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)

    def video_path(self, ep_idx: int, video_key: str) -> Path:
        chunk = ep_idx // self.info["chunks_size"]
        return self.root / self.info["video_path"].format(
            episode_chunk=chunk, video_key=video_key, episode_index=ep_idx
        )


def read_video_frames(path: Path, target_hw: tuple[int, int]):
    """Yield HWC uint8 RGB frames resized to target_hw."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    th, tw = target_hw
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                return
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame.shape[:2] != (th, tw):
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            yield np.ascontiguousarray(frame)
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


def _video_feature(h: int, w: int, fps: int) -> dict:
    return {
        "dtype": "video",
        "shape": (h, w, 3),
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": h,
            "video.width": w,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def build_joint_features(
    ref_repo: str, ref_root: Path | None, cam_keys: list[str], h: int, w: int, fps: int
) -> tuple[dict, list[str]]:
    """18-dim schema with names copied from an xlerobot reference repo."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(ref_repo, root=ref_root)
    names = list(meta.features["observation.state"]["names"])
    dim = len(names)
    features = {
        "observation.state": {"dtype": "float32", "shape": (dim,), "names": names},
        "action": {"dtype": "float32", "shape": (dim,), "names": names},
    }
    for k in cam_keys:
        features[k] = _video_feature(h, w, fps)
    return features, names


def build_ee_features(
    dual_arm: bool, cam_keys: list[str], h: int, w: int, fps: int
) -> tuple[dict, list[str]]:
    """EE-space schema (Option B): FastUMI motor names, robot-frame values."""
    if dual_arm:
        names = [f"left_{n}" for n in UMI_EE_NAMES] + [f"right_{n}" for n in UMI_EE_NAMES]
    else:
        names = list(UMI_EE_NAMES)
    dim = len(names)
    features = {
        "observation.state": {"dtype": "float32", "shape": (dim,), "names": names},
        "action": {"dtype": "float32", "shape": (dim,), "names": names},
    }
    for k in cam_keys:
        features[k] = _video_feature(h, w, fps)
    return features, names


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def retarget_umi_task(
    source_root: Path,
    target_repo_id: str,
    output_root: Path,
    action_space: str = "joint",
    match_features_from: str | None = None,
    match_features_root: Path | None = None,
    target_arm: str = "right",
    axis_map: AxisMap | None = None,
    scale: float = DEFAULT_SCALE,
    anchor_fwd: float = DEFAULT_ANCHOR_FWD,
    anchor_up: float = DEFAULT_ANCHOR_UP,
    gripper_range: tuple[float, float] = (2.0, 98.0),
    gantry_mm: float = 0.0,
    target_image_size: tuple[int, int] = (360, 640),
    head_fill: str = "black",
    missing_wrist_fill: str = "black",
    max_episodes: int | None = None,
    max_clamp_frac: float = 0.25,
    force_rebuild: bool = False,
) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if axis_map is None:
        axis_map = AxisMap.parse("+x,+y,+z")

    src = UmiSource.load(source_root)
    th, tw = target_image_size
    logger.info(
        "Source: %s (%s-arm, %d eps, %d fps, cams=%s)",
        source_root,
        "dual" if src.dual_arm else "single",
        len(src.episodes),
        src.fps,
        src.camera_keys,
    )

    # Camera remap: UMI wrist cams -> xlerobot wrist cams (+ optional black head)
    if src.dual_arm:
        cam_remap = {
            "observation.images.left_camera_rgb_image": "observation.images.left_wrist",
            "observation.images.right_camera_rgb_image": "observation.images.right_wrist",
        }
    else:
        cam_remap = {"observation.images.camera_rgb_image": f"observation.images.{target_arm}_wrist"}
    out_cam_keys = sorted(cam_remap.values())
    fill_cams: list[str] = []
    if head_fill == "black":
        fill_cams.append("observation.images.head")
    if missing_wrist_fill == "black":
        for side in ("left", "right"):
            k = f"observation.images.{side}_wrist"
            if k not in out_cam_keys:
                fill_cams.append(k)
    all_cam_keys = sorted(out_cam_keys + fill_cams)

    if action_space == "joint":
        if match_features_from is None:
            raise ValueError("--match-features-from is required for --action-space joint")
        features, names = build_joint_features(
            match_features_from, match_features_root, all_cam_keys, th, tw, src.fps
        )
    elif action_space == "ee":
        features, names = build_ee_features(src.dual_arm, all_cam_keys, th, tw, src.fps)
    else:
        raise ValueError(f"--action-space must be 'joint' or 'ee', got {action_space!r}")

    if output_root.exists():
        if not force_rebuild:
            raise FileExistsError(f"{output_root} exists. Pass --force-rebuild to overwrite.")
        shutil.rmtree(output_root)

    dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=src.fps,
        robot_type="xlerobot",
        features=features,
        root=output_root,
        use_videos=True,
        image_writer_threads=4,
    )

    black = np.zeros((th, tw, 3), dtype=np.uint8)
    dim = len(names)
    name_idx = {n: i for i, n in enumerate(names)}

    def assemble_joint_vec(arm_joints: dict[str, np.ndarray], t: int) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for prefix, joints in arm_joints.items():
            for j, jn in enumerate(XLE_ARM_JOINTS):
                key = f"{prefix}_arm_{jn}.pos"
                if key in name_idx:
                    vec[name_idx[key]] = joints[t, j]
        if "gantry.height_mm" in name_idx:
            vec[name_idx["gantry.height_mm"]] = gantry_mm
        return vec

    n_eps = len(src.episodes) if max_episodes is None else min(max_episodes, len(src.episodes))
    skipped: list[tuple[int, float]] = []
    for ep in tqdm(src.episodes[:n_eps], desc=f"Retarget {source_root.name}", unit="ep"):
        ep_idx = ep["episode_index"]
        df = pd.read_parquet(src.parquet_path(ep_idx))
        state = np.stack(df["observation.state"].values).astype(np.float32)
        action = np.stack(df["action"].values).astype(np.float32)
        task_idx = int(df["task_index"].iloc[0]) if "task_index" in df else 0
        task = src.tasks.get(task_idx, source_root.name.replace("_", " "))

        # ---- Stage 1: per-arm robot-frame EE
        arms = ["left", "right"] if src.dual_arm else [target_arm]
        ee_state, ee_action = {}, {}
        for i, arm in enumerate(arms):
            sl = slice(i * 7, i * 7 + 7)
            ee_state[arm] = umi_to_robot_ee(state[:, sl], axis_map, scale, anchor_fwd, anchor_up)
            ee_action[arm] = umi_to_robot_ee(action[:, sl], axis_map, scale, anchor_fwd, anchor_up)

        # ---- Stage 2: vectors in the chosen action space
        if action_space == "joint":
            rt = ArmRetargeter(gripper_range)
            joints_state = {arm: rt(ee_state[arm]) for arm in arms}
            joints_action = {arm: rt(ee_action[arm]) for arm in arms}
            if rt.clamp_fraction > max_clamp_frac:
                skipped.append((ep_idx, rt.clamp_fraction))
                continue
            vec_state = [assemble_joint_vec(joints_state, t) for t in range(len(state))]
            vec_action = [assemble_joint_vec(joints_action, t) for t in range(len(state))]
        else:
            order = ["left", "right"] if src.dual_arm else [target_arm]
            vec_state = [
                np.concatenate([ee_state[a][t] for a in order]).astype(np.float32) for t in range(len(state))
            ]
            vec_action = [
                np.concatenate([ee_action[a][t] for a in order]).astype(np.float32) for t in range(len(state))
            ]

        # ---- Videos
        readers = {
            cam_remap[k]: read_video_frames(src.video_path(ep_idx, k), (th, tw)) for k in src.camera_keys
        }
        n_frames = len(state)
        wrote = 0
        for t in range(n_frames):
            frame = {"observation.state": vec_state[t], "action": vec_action[t], "task": task}
            missing_video = False
            for cam_key, reader in readers.items():
                img = next(reader, None)
                if img is None:
                    missing_video = True
                    break
                frame[cam_key] = img
            if missing_video:
                break
            for k in fill_cams:
                frame[k] = black
            dataset.add_frame(frame)
            wrote += 1

        if wrote > 0:
            dataset.save_episode()
        else:
            dataset.clear_episode_buffer()

    dataset.finalize()
    if skipped:
        logger.warning(
            "Skipped %d/%d episodes over --max-clamp-frac=%.2f (unreachable workspace): %s",
            len(skipped),
            n_eps,
            max_clamp_frac,
            [(e, round(f, 3)) for e, f in skipped[:10]],
        )
    logger.info("Retargeted dataset written to %s", output_root)
    return output_root


def main() -> None:
    p = argparse.ArgumentParser(
        description="Retarget FastUMI (UMI-style EE) tasks to xlerobot joint or EE space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Local root of ONE task, e.g. ~/fastumi/dual_arm/Clean_Desktop",
    )
    p.add_argument("--target-repo-id", required=True)
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to ~/.cache/huggingface/lerobot/<target-repo-id>",
    )
    p.add_argument("--action-space", choices=["joint", "ee"], default="joint")
    p.add_argument(
        "--match-features-from",
        default=None,
        help="xlerobot reference repo for 18-dim joint names (joint mode)",
    )
    p.add_argument("--match-features-root", type=Path, default=None)
    p.add_argument(
        "--target-arm", choices=["left", "right"], default="right", help="Arm used for single_arm tasks"
    )
    p.add_argument(
        "--axis-map",
        default="+x,+y,+z",
        help="UMI axes for (forward, lateral, up), e.g. '+x,+y,+z' or '-z,+x,+y'",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Scale UMI meters into the arm workspace (human reach >> SO-100 reach; "
        "0.25-0.35 keeps FastUMI trajectories reachable)",
    )
    p.add_argument("--anchor-fwd", type=float, default=DEFAULT_ANCHOR_FWD)
    p.add_argument("--anchor-up", type=float, default=DEFAULT_ANCHOR_UP)
    p.add_argument(
        "--gripper-range",
        default="2,98",
        help="'min,max' joint values for gripper width 0..1 (xlerobot uses ~0..100)",
    )
    p.add_argument("--gantry-mm", type=float, default=0.0)
    p.add_argument("--target-image-size", default="360x640")
    p.add_argument(
        "--head-fill",
        choices=["black", "none"],
        default="black",
        help="UMI has no head camera; 'black' keeps the merged schema 3-camera",
    )
    p.add_argument(
        "--missing-wrist-fill",
        choices=["black", "none"],
        default="black",
        help="single_arm tasks have one wrist cam; fill the other or omit it",
    )
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument(
        "--max-clamp-frac",
        type=float,
        default=0.25,
        help="Skip episodes whose EE leaves the arm workspace more than this fraction",
    )
    p.add_argument("--force-rebuild", action="store_true")
    args = p.parse_args()

    h, w = (int(x) for x in args.target_image_size.lower().split("x"))
    gmin, gmax = (float(x) for x in args.gripper_range.split(","))
    output_root = args.output_root
    if output_root is None:
        output_root = Path("~/.cache/huggingface/lerobot").expanduser() / args.target_repo_id

    retarget_umi_task(
        source_root=args.source_root.expanduser(),
        target_repo_id=args.target_repo_id,
        output_root=output_root.expanduser(),
        action_space=args.action_space,
        match_features_from=args.match_features_from,
        match_features_root=args.match_features_root,
        target_arm=args.target_arm,
        axis_map=AxisMap.parse(args.axis_map),
        scale=args.scale,
        anchor_fwd=args.anchor_fwd,
        anchor_up=args.anchor_up,
        gripper_range=(gmin, gmax),
        gantry_mm=args.gantry_mm,
        target_image_size=(h, w),
        head_fill=args.head_fill,
        missing_wrist_fill=args.missing_wrist_fill,
        max_episodes=args.max_episodes,
        max_clamp_frac=args.max_clamp_frac,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
