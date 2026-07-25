#!/usr/bin/env python3
"""Convert a raw ALOHA (ViperX/vx300s) joint-space LeRobot dataset to the same
robot-frame EE-space schema produced by ``lerobot-umi-retarget --action-space ee``
(see ``src/lerobot/data_processing/umi_retarget.py``):

    observation.state / action = [left_x, left_y, left_z, left_roll, left_pitch,
    left_yaw, left_gripper, right_x, ..., right_gripper] (dual arm, 14-dim),
    meters/radians, gripper in [0, 1], pose expressed relative to the episode's
    first frame ("absolute EE pose relative to episode start", matching
    FastUMI's own recording convention).

Forward kinematics uses the real Interbotix vx300s URDF (the arm ALOHA is
built from) via pinocchio, NOT a hand-rolled DH model — verified against
known joint-limit specs and a 90-degree-waist-rotation sanity check before
being wired in here (see docs/upstream_merge_plan.md's sibling note in
memory ee-space-generalist-plan.md for the verification transcript).

ALOHA's per-arm motor order is fixed by convention:
    [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper]
Only the first 6 feed forward kinematics — the gripper joint doesn't move the
end-effector frame in this URDF (finger prismatic joints are downstream of
the fixed ee_gripper_link), so its raw value is passed through unchanged
(already ~[0,1]-normalized by ALOHA's own recording pipeline, not URDF-joint
radians — clipped to [0,1] for safety, not re-derived from FK).

Usage:
    python scripts/convert_joint_to_ee.py \\
        --source-repo lerobot/aloha_static_battery \\
        --target-repo-id Odog16/aloha_static_battery_ee \\
        --urdf scripts/robot_descriptions/vx300s.urdf
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.SO101Robot import SO101Kinematics

ALOHA_ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
XLE_ARM_JOINT_SUFFIXES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
EE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


class ViperXKinematics:
    """Forward kinematics for one vx300s (ALOHA) arm via pinocchio."""

    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("vx300s/ee_gripper_link")
        self.arm_q_idx = [self.model.joints[self.model.getJointId(j)].idx_q for j in ALOHA_ARM_JOINTS]

    def fk(self, arm_joint_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """6 joint angles (rad) -> (position (3,) meters, rotation matrix (3,3))."""
        q = pin.neutral(self.model)
        for idx, val in zip(self.arm_q_idx, arm_joint_values, strict=True):
            q[idx] = val
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.ee_frame_id]
        return placement.translation.copy(), placement.rotation.copy()


def episode_to_ee(
    kin: ViperXKinematics, state_14: np.ndarray, arm_offset: int
) -> np.ndarray:
    """(T, 6-or-7-per-arm slice) raw ALOHA joint state -> (T, 7) EE-space, relative to frame 0."""
    n = state_14.shape[0]
    positions = np.zeros((n, 3), dtype=np.float64)
    rotations = np.zeros((n, 3, 3), dtype=np.float64)
    grippers = np.clip(state_14[:, arm_offset + 6], 0.0, 1.0).astype(np.float32)
    for t in range(n):
        arm_vals = state_14[t, arm_offset : arm_offset + 6]
        pos, rot = kin.fk(arm_vals)
        positions[t] = pos
        rotations[t] = rot

    pos_rel = positions - positions[0]
    rot0_T = rotations[0].T
    out = np.zeros((n, 7), dtype=np.float32)
    out[:, :3] = pos_rel
    for t in range(n):
        rel_rot = rot0_T @ rotations[t]
        roll, pitch, yaw = Rotation.from_matrix(rel_rot).as_euler("XYZ")
        out[t, 3:6] = [roll, pitch, yaw]
    out[:, 6] = grippers
    return out


class XLerobotArmKinematics:
    """Forward kinematics for one xlerobot/SO-101 arm — the exact inverse of
    ``ArmRetargeter.__call__`` in ``umi_retarget.py`` (which goes EE robot-frame
    pose -> 6 joints), reusing the *same* ``SO101Kinematics`` 2-link planar
    solver already used there rather than a new model. This arm only has 2
    independently-actuated orientation DOF (wrist_flex -> pitch, wrist_roll ->
    roll); ``ArmRetargeter`` explicitly discards yaw (``_yaw`` in its unpacking)
    when going the other direction, confirming yaw was never a real recoverable
    quantity for this arm — so it's fixed at 0 here too, not invented.
    """

    def __init__(self, gripper_range: tuple[float, float] = (2.0, 98.0)):
        self.kin = SO101Kinematics()
        self.g_min, self.g_max = gripper_range

    def fk(
        self, pan_deg: float, lift_deg: float, elbow_deg: float, wrist_flex_deg: float, wrist_roll_deg: float, gripper_deg: float
    ) -> tuple[float, float, float, float, float, float]:
        reach, up = self.kin.forward_kinematics(lift_deg, elbow_deg)
        pan_rad = math.radians(pan_deg)
        fwd = reach * math.cos(pan_rad)
        lat = reach * math.sin(pan_rad)
        # Inverts: wrist_flex = -(lift + elbow) + degrees(pitch)
        pitch = math.radians(wrist_flex_deg) + math.radians(lift_deg) + math.radians(elbow_deg)
        roll = math.radians(wrist_roll_deg)
        gripper01 = float(np.clip((gripper_deg - self.g_min) / (self.g_max - self.g_min), 0.0, 1.0))
        return fwd, lat, up, roll, pitch, gripper01


def xlerobot_episode_to_ee(kin: XLerobotArmKinematics, state: np.ndarray, col_idx: dict[str, int]) -> np.ndarray:
    """(T, N) raw xlerobot joint state -> (T, 7) EE-space, relative to frame 0."""
    n = state.shape[0]
    out = np.zeros((n, 7), dtype=np.float32)
    for t in range(n):
        vals = [state[t, col_idx[suffix]] for suffix in XLE_ARM_JOINT_SUFFIXES]
        fwd, lat, up, roll, pitch, grip = kin.fk(*vals)
        out[t] = [fwd, lat, up, roll, pitch, 0.0, grip]
    out[:, 0:3] -= out[0, 0:3].copy()
    out[:, 3:5] -= out[0, 3:5].copy()
    return out


def convert_aloha(
    source_repo: str,
    target_repo_id: str,
    urdf_path: str,
    output_root: Path | None,
    max_episodes: int | None,
    tolerance_s: float = 1e-4,
):
    kin = ViperXKinematics(urdf_path)
    src = LeRobotDataset(source_repo, tolerance_s=tolerance_s)
    state_names_raw = src.meta.features["observation.state"]["names"]
    state_names = state_names_raw["motors"] if isinstance(state_names_raw, dict) else state_names_raw
    dual_arm = len(state_names) == 14
    if not dual_arm:
        raise NotImplementedError(f"{source_repo}: expected 14-dim dual-arm ALOHA state, got {len(state_names)}")

    names = [f"left_{n}" for n in EE_NAMES] + [f"right_{n}" for n in EE_NAMES]
    cam_keys = [k for k in src.meta.features if k.startswith("observation.images.")]
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": names},
        "action": {"dtype": "float32", "shape": (14,), "names": names},
    }
    for k in cam_keys:
        features[k] = dict(src.meta.features[k])

    output_root = output_root or (Path.home() / ".cache/huggingface/lerobot" / target_repo_id)
    if output_root.exists():
        shutil.rmtree(output_root)

    out_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=src.meta.fps,
        robot_type="aloha_vx300s",
        features=features,
        root=output_root,
        use_videos=True,
        image_writer_threads=4,
    )

    n_eps = src.meta.total_episodes if max_episodes is None else min(max_episodes, src.meta.total_episodes)
    n_written = 0
    n_skipped = 0
    for ep_idx in tqdm(range(n_eps), desc=f"Convert {source_repo}", unit="ep"):
        try:
            ep_meta = src.meta.episodes[ep_idx]
            lo, hi = ep_meta["dataset_from_index"], ep_meta["dataset_to_index"]
            raw_frames = [src[i] for i in range(lo, hi)]
            state = np.stack([f["observation.state"].numpy() for f in raw_frames]).astype(np.float32)
            task = raw_frames[0]["task"]

            left_ee = episode_to_ee(kin, state, arm_offset=0)
            right_ee = episode_to_ee(kin, state, arm_offset=7)
            vec_state = np.concatenate([left_ee, right_ee], axis=1)
            vec_action = np.vstack([vec_state[1:], vec_state[-1:]])

            for t, raw in enumerate(raw_frames):
                frame = {"observation.state": vec_state[t], "action": vec_action[t], "task": task}
                for k in cam_keys:
                    # CHW float32 [0,1] torch tensor (LeRobotDataset's decoded form) -> HWC uint8 for writing.
                    chw = raw[k].numpy()
                    frame[k] = np.clip(chw * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                out_ds.add_frame(frame)
            out_ds.save_episode()
            n_written += 1
        except Exception as e:  # noqa: BLE001 - isolated bad episode (corrupt video, timestamp desync) shouldn't kill the whole run
            print(f"WARNING: skipping episode {ep_idx} of {source_repo}: {type(e).__name__}: {e}")
            out_ds.clear_episode_buffer()
            n_skipped += 1

    out_ds.finalize()
    print(f"Wrote {target_repo_id} -> {output_root} ({n_written} episodes, {n_skipped} skipped)")


def convert_xlerobot(
    source_repo: str,
    target_repo_id: str,
    output_root: Path | None,
    max_episodes: int | None,
    gripper_range: tuple[float, float],
    tolerance_s: float = 1e-4,
):
    kin = XLerobotArmKinematics(gripper_range)
    src = LeRobotDataset(source_repo, tolerance_s=tolerance_s)
    state_names_raw = src.meta.features["observation.state"]["names"]
    state_names = state_names_raw["motors"] if isinstance(state_names_raw, dict) else state_names_raw

    left_idx = {suf: state_names.index(f"left_arm_{suf}") for suf in XLE_ARM_JOINT_SUFFIXES}
    right_idx = {suf: state_names.index(f"right_arm_{suf}") for suf in XLE_ARM_JOINT_SUFFIXES}

    names = [f"left_{n}" for n in EE_NAMES] + [f"right_{n}" for n in EE_NAMES]
    cam_keys = [k for k in src.meta.features if k.startswith("observation.images.")]
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": names},
        "action": {"dtype": "float32", "shape": (14,), "names": names},
    }
    for k in cam_keys:
        features[k] = dict(src.meta.features[k])

    output_root = output_root or (Path.home() / ".cache/huggingface/lerobot" / target_repo_id)
    if output_root.exists():
        shutil.rmtree(output_root)

    out_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=src.meta.fps,
        robot_type="xlerobot",
        features=features,
        root=output_root,
        use_videos=True,
        image_writer_threads=4,
    )

    n_eps = src.meta.total_episodes if max_episodes is None else min(max_episodes, src.meta.total_episodes)
    n_written = 0
    n_skipped = 0
    for ep_idx in tqdm(range(n_eps), desc=f"Convert {source_repo}", unit="ep"):
        try:
            ep_meta = src.meta.episodes[ep_idx]
            lo, hi = ep_meta["dataset_from_index"], ep_meta["dataset_to_index"]
            raw_frames = [src[i] for i in range(lo, hi)]
            state = np.stack([f["observation.state"].numpy() for f in raw_frames]).astype(np.float32)
            task = raw_frames[0]["task"]

            left_ee = xlerobot_episode_to_ee(kin, state, left_idx)
            right_ee = xlerobot_episode_to_ee(kin, state, right_idx)
            vec_state = np.concatenate([left_ee, right_ee], axis=1)
            vec_action = np.vstack([vec_state[1:], vec_state[-1:]])

            for t, raw in enumerate(raw_frames):
                frame = {"observation.state": vec_state[t], "action": vec_action[t], "task": task}
                for k in cam_keys:
                    chw = raw[k].numpy()
                    frame[k] = np.clip(chw * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                out_ds.add_frame(frame)
            out_ds.save_episode()
            n_written += 1
        except Exception as e:  # noqa: BLE001 - isolated bad episode shouldn't kill the whole run
            print(f"WARNING: skipping episode {ep_idx} of {source_repo}: {type(e).__name__}: {e}")
            out_ds.clear_episode_buffer()
            n_skipped += 1

    out_ds.finalize()
    print(f"Wrote {target_repo_id} -> {output_root} ({n_written} episodes, {n_skipped} skipped)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-repo", required=True)
    p.add_argument("--target-repo-id", required=True)
    p.add_argument("--robot-type", choices=["aloha", "xlerobot"], default="aloha")
    p.add_argument("--urdf", default="scripts/robot_descriptions/vx300s.urdf")
    p.add_argument("--output-root", default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--gripper-range", default="2.0,98.0", help="xlerobot only: 'min,max' raw gripper.pos range")
    p.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Video frame-timestamp sync tolerance passed to LeRobotDataset (default matches upstream's "
        "default of 1e-4s; some datasets' video encoding has slightly looser real precision and need "
        "e.g. 5e-3 or every episode raises FrameTimestampError)",
    )
    args = p.parse_args()
    output_root = Path(args.output_root) if args.output_root else None
    if args.robot_type == "aloha":
        convert_aloha(
            args.source_repo, args.target_repo_id, args.urdf, output_root, args.max_episodes, args.tolerance_s
        )
    else:
        g_min, g_max = (float(x) for x in args.gripper_range.split(","))
        convert_xlerobot(
            args.source_repo, args.target_repo_id, output_root, args.max_episodes, (g_min, g_max), args.tolerance_s
        )


if __name__ == "__main__":
    main()
