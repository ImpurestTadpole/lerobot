#!/usr/bin/env python3
"""Convert the flex-pi datasets (huggingface.co/flex-pi) to this project's
episode-start-relative EE-space convention (see convert_joint_to_ee.py's and
convert_droid_ee.py's module docstrings for the established convention this
matches).

flex-pi's repos are **genuinely LeRobot v2.1 format** — not just a stale
``codebase_version`` tag (checked: ``LeRobotDataset()`` hard-rejects them
with ``BackwardCompatibilityError`` regardless of ``revision``, since this
codebase's dataset class is v3.0-only). So, like ``convert_droid_ee.py``,
this script works directly on the downloaded raw files (parquet + symlinked
video) rather than through ``LeRobotDataset`` — the only approach that can
touch this data at all in the current codebase.

Two source shapes, two modes:

``yam`` — the 6 flex-pi "yam" (bimanual) teleop datasets
(self_repair_gripper_bc/dagger, soft_bag_zipping, put_plate_on_the_rack,
kitchen_organization, sort_utensils). Their native ``observation.state`` is
already dual-arm EE pose, just in a different representation than this
project's convention: 32-dim = [left_pos(3), left_rot6d(6), right_pos(3),
right_rot6d(6), left_gripper, right_gripper, left_joint(6), right_joint(6)].
No FK needed — only a rotation-representation change (rot6d -> Euler XYZ,
via the same Gram-Schmidt decode used by ``lerobot.policies.groot.utils
.rot6d_to_matrix``) plus dropping the redundant joint columns. Verified
against real data before writing this: action[t] == state[t+1] exactly (the
project's standard "next-frame absolute target" convention already), so
action is re-derived as the one-step shift of the transformed state rather
than independently transformed (equivalent, simpler, avoids double work).
Cameras (cam_high/cam_left_wrist/cam_right_wrist) already match this
project's default camera remap and native image size (360x640) already
matches the generalist merge's target size, so RGB+depth videos are all
symlinked through unchanged — no re-encoding, matching convert_droid_ee.py.

``libero`` — the 4 suites inside flex-pi/libero_mujoco3.3.2_depth
(libero_10/goal/object/spatial, each a nested LeRobot dataset under its own
subdirectory of one repo). Single-arm (Franka Panda — the same embodiment as
DROID), ``observation.states.ee_state`` (6-dim, already Euler XYZ) +
``observation.state[6]`` (gripper; index 7 is a duplicate, discarded) is the
real EE pose+gripper. Native ``action`` is a small per-step DELTA command
(verified against real data, same as DROID's own native action column) —
discarded and re-derived as the one-step shift of the transformed state,
exactly matching convert_droid_ee.py's ``episode_relative`` treatment. Only
2 RGB cameras (no right_wrist) — same as DROID.

Both modes leave the *right* arm / missing RGB camera entirely up to a
LATER merge step's own padding for single-arm sources, matching the DROID
precedent — this script never pads to 14-dim itself.

**Depth is preserved, not dropped.** ``observation.depth_ffv1.*`` video
directories are symlinked through alongside RGB (same full-feature-key
directory naming, verified directly against a real downloaded repo:
``videos/chunk-XXX/<full_feature_key>/*.{mp4,mkv}`` — chunk outer, camera
key inner) and kept as real features in the output dataset's schema/stats,
verbatim — needed downstream for both pretrain and post-train use. Per an
explicit 2026-08-16 decision, these depth-bearing conversions are kept as
their OWN standalone dataset(s), not folded into the shared RGB-only
``generalist_ee_merged`` schema: that merge tool (``lerobot-cotrain-align``)
requires identical ``features`` dicts across every source it aggregates, so
adding a depth key to it would force a zero-filled-depth rebuild of all
~30 already-aligned RGB-only sources (own_tasks/FastUMI/ALOHA) plus the
DROID alignment already running — real, avoidable cost for a feature that
would go unused in the majority of sources.

**Episode metadata is the LEGACY jsonl format** (``meta/episodes.jsonl`` +
``meta/episodes_stats.jsonl``, plain ``{episode_index, tasks, length}``
records with no explicit ``dataset_from_index``/``dataset_to_index`` —
recognized in this codebase as ``LEGACY_EPISODES_PATH``), not the
parquet-episodes directory ``convert_droid_ee.py``'s own DROID data uses.
Since this script performs a lossless 1:1 row transform (same frame count
per episode in and out — only observation.state/action VALUES change, no
frames added/dropped), each episode's real length always equals its source
length, so ``episodes.jsonl`` is copied through unchanged rather than
recomputed from scratch (recomputing would be redundant, not more correct).
A hard assertion still checks this equality against the actual transform
pass before trusting the copy, so a real mismatch would fail loudly rather
than silently mislabel data.

Usage:
    python scripts/convert_flexpi_ee.py yam \\
        --source-repo flex-pi/soft_bag_zipping \\
        --target-repo-id Odog16/flexpi_soft_bag_zipping_ee

    python scripts/convert_flexpi_ee.py libero \\
        --source-repo flex-pi/libero_mujoco3.3.2_depth \\
        --suite libero_10_no_noops_lerobot \\
        --target-repo-id Odog16/flexpi_libero_10_ee
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lerobot.policies.groot.utils import rot6d_to_matrix

EE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
DUAL_EE_NAMES = [f"left_{n}" for n in EE_NAMES] + [f"right_{n}" for n in EE_NAMES]


def _episode_relative_pos_rot(pos: np.ndarray, rot: Rotation) -> tuple[np.ndarray, np.ndarray]:
    """(T,3) positions + (T,) Rotation -> position relative to frame 0, and
    Euler XYZ relative to frame 0's orientation (proper composition, not
    naive angle subtraction — same approach as convert_joint_to_ee.py /
    convert_droid_ee.py)."""
    pos_rel = pos - pos[0]
    rot0_inv = rot[0].inv()
    rel_euler = (rot0_inv * rot).as_euler("XYZ")
    return pos_rel, rel_euler


def _copy_meta_and_videos(source_root: Path, output_root: Path, keep_video_keys: list[str]) -> None:
    """*keep_video_keys* are FULL feature keys (e.g. "observation.images.cam_high",
    "observation.depth_ffv1.cam_high") — on-disk video/depth directories are
    named after the full feature key verbatim, not the stripped camera name
    (verified directly against a real downloaded repo; RGB and depth for the
    same physical camera live in sibling dirs that would otherwise collide
    if stripped, e.g. both "cam_high")."""
    for rel in ["meta/episodes.jsonl", "meta/episodes_stats.jsonl", "meta/tasks.jsonl",
                "meta/tasks.parquet", "meta/camera_intrinsics.json"]:
        src = source_root / rel
        if src.exists():
            dst = output_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    videos_src = source_root / "videos"
    if videos_src.exists():
        videos_dst = output_root / "videos"
        for chunk_dir in videos_src.iterdir():
            if not chunk_dir.is_dir():
                continue
            for cam_dir in chunk_dir.iterdir():
                if not cam_dir.is_dir() or cam_dir.name not in keep_video_keys:
                    continue
                dst_chunk = videos_dst / chunk_dir.name / cam_dir.name
                dst_chunk.mkdir(parents=True, exist_ok=True)
                for f in cam_dir.iterdir():
                    (dst_chunk / f.name).symlink_to(f.resolve())


def _write_info_and_stats(
    output_root: Path,
    source_root: Path,
    info: dict,
    video_keys: list[str],
    state_names: list[str],
    state_dim: int,
    total_episodes: int,
    total_frames: int,
    state_stats: dict,
    action_stats: dict,
) -> None:
    """*video_keys* = RGB camera keys + depth keys combined (both preserved
    verbatim, including their original dtype/shape/video-codec info — only
    observation.state/action are rewritten to this project's EE convention)."""
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["features"] = {
        k: v
        for k, v in info["features"].items()
        if k in video_keys or k in ("observation.state", "action", "timestamp", "frame_index",
                                     "episode_index", "index", "task_index")
    }
    info["features"]["observation.state"] = {
        "dtype": "float32", "shape": [state_dim], "names": state_names
    }
    info["features"]["action"] = {
        "dtype": "float32", "shape": [state_dim], "names": state_names
    }
    (output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # No aggregate meta/stats.json exists upstream for these repos (they ship
    # per-episode episodes_stats.jsonl instead) — write a minimal one with
    # just the two features whose values actually changed. A depth-consuming
    # training pass should compute full stats (incl. depth) itself.
    stats = {"observation.state": state_stats, "action": action_stats}
    (output_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))


def _running_stats(arr: np.ndarray, acc: dict | None) -> dict:
    """Accumulate exact running sum/sq_sum/min/max/count across chunks, return final stats dict."""
    if acc is None:
        acc = {
            "sum": arr.sum(axis=0), "sq_sum": (arr**2).sum(axis=0),
            "min": arr.min(axis=0), "max": arr.max(axis=0), "count": len(arr),
        }
    else:
        acc["sum"] += arr.sum(axis=0)
        acc["sq_sum"] += (arr**2).sum(axis=0)
        acc["min"] = np.minimum(acc["min"], arr.min(axis=0))
        acc["max"] = np.maximum(acc["max"], arr.max(axis=0))
        acc["count"] += len(arr)
    return acc


def _finalize_stats(acc: dict) -> dict:
    mean = acc["sum"] / acc["count"]
    std = np.sqrt(np.maximum(acc["sq_sum"] / acc["count"] - mean**2, 0))
    return {
        "min": acc["min"].tolist(), "max": acc["max"].tolist(),
        "mean": mean.tolist(), "std": std.tolist(), "count": [int(acc["count"])],
    }


def _verify_episode_lengths(output_root: Path, real_lengths: dict[int, int], total_episodes: int) -> int:
    """episodes.jsonl was copied through unchanged (see module docstring: a
    1:1 row transform never changes per-episode length). Verify that claim
    against what the transform pass actually counted, rather than assuming
    it — a real mismatch here means something upstream truncated/reordered
    rows and must not be silently trusted."""
    jsonl_path = output_root / "meta" / "episodes.jsonl"
    cursor = 0
    seen = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ep = rec["episode_index"]
            if ep >= total_episodes:
                continue
            claimed_len = rec["length"]
            real_len = real_lengths.get(ep, 0)
            if claimed_len != real_len:
                raise AssertionError(
                    f"Episode {ep}: episodes.jsonl claims length={claimed_len} but the transform "
                    f"pass counted {real_len} real rows — a 1:1 transform should never disagree; "
                    f"investigate before trusting this source."
                )
            cursor += real_len
            seen.add(ep)
    missing = set(range(total_episodes)) - seen
    if missing:
        raise AssertionError(f"episodes.jsonl is missing entries for episodes {sorted(missing)[:10]}...")
    return cursor


def convert_yam(source_repo: str, target_repo_id: str, output_root: Path | None, max_episodes: int | None):
    print(f"Downloading/locating {source_repo} locally (RGB + depth)...")
    source_root = Path(snapshot_download(source_repo, repo_type="dataset"))
    output_root = output_root or (Path.home() / ".cache/huggingface/lerobot" / target_repo_id)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    info = json.loads((source_root / "meta" / "info.json").read_text())
    src_names = info["features"]["observation.state"]["names"][0]
    idx = {n: i for i, n in enumerate(src_names)}
    cam_keys = [k for k in info["features"] if k.startswith("observation.images.")]
    depth_keys = [k for k in info["features"] if k.startswith("observation.depth_ffv1.")]
    video_keys = cam_keys + depth_keys

    total_episodes = info["total_episodes"] if max_episodes is None else min(max_episodes, info["total_episodes"])
    print(f"Converting {total_episodes} of {info['total_episodes']} episodes "
          f"({len(cam_keys)} RGB + {len(depth_keys)} depth cameras)")

    _copy_meta_and_videos(source_root, output_root, video_keys)

    state_acc = None
    action_acc = None
    real_lengths: dict[int, int] = {}
    data_src = source_root / "data"
    for chunk_dir in tqdm(sorted(data_src.iterdir()), desc=f"Transform {source_repo}", unit="chunk"):
        dst_chunk = output_root / "data" / chunk_dir.name
        dst_chunk.mkdir(parents=True, exist_ok=True)
        for pq_file in sorted(chunk_dir.iterdir()):
            df = pd.read_parquet(pq_file)
            df = df[df["episode_index"] < total_episodes]
            if df.empty:
                continue

            raw_state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
            ep_idx_arr = df["episode_index"].to_numpy()

            new_state = np.zeros((len(df), 14), dtype=np.float64)
            new_action = np.zeros((len(df), 14), dtype=np.float64)
            for ep in np.unique(ep_idx_arr):
                positions = np.where(ep_idx_arr == ep)[0]
                ep_state = raw_state[positions]
                per_arm_out = []
                for pos_slice, rot_slice, grip_key in (
                    (slice(idx["left_pos_x"], idx["left_pos_z"] + 1),
                     slice(idx["left_rot6d_0"], idx["left_rot6d_5"] + 1), "left_gripper"),
                    (slice(idx["right_pos_x"], idx["right_pos_z"] + 1),
                     slice(idx["right_rot6d_0"], idx["right_rot6d_5"] + 1), "right_gripper"),
                ):
                    pos = ep_state[:, pos_slice]
                    mats = np.stack([rot6d_to_matrix(v) for v in ep_state[:, rot_slice]])
                    rot = Rotation.from_matrix(mats)
                    pos_rel, euler_rel = _episode_relative_pos_rot(pos, rot)
                    grip = np.clip(ep_state[:, idx[grip_key]], 0.0, 1.0)
                    per_arm_out.append(np.concatenate([pos_rel, euler_rel, grip[:, None]], axis=1))
                ep_rel_state = np.concatenate(per_arm_out, axis=1)  # (n, 14)
                ep_action = np.vstack([ep_rel_state[1:], ep_rel_state[-1:]])
                new_state[positions] = ep_rel_state
                new_action[positions] = ep_action
                real_lengths[int(ep)] = real_lengths.get(int(ep), 0) + len(positions)

            # Video features (observation.images.*/observation.depth_ffv1.*) are never
            # actual parquet columns — only their video files exist, symlinked above.
            base_cols = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
            out_df = df[[c for c in base_cols if c in df.columns]].copy()
            out_df["observation.state"] = list(new_state.astype(np.float32))
            out_df["action"] = list(new_action.astype(np.float32))
            out_df.to_parquet(dst_chunk / pq_file.name, index=False)

            state_acc = _running_stats(new_state, state_acc)
            action_acc = _running_stats(new_action, action_acc)

    cursor = _verify_episode_lengths(output_root, real_lengths, total_episodes)
    if cursor != state_acc["count"]:
        raise AssertionError(
            f"episodes.jsonl-verified frame total {cursor} != transform pass count "
            f"{state_acc['count']} — must match exactly."
        )

    _write_info_and_stats(
        output_root, source_root, info, video_keys, DUAL_EE_NAMES, 14,
        total_episodes, cursor, _finalize_stats(state_acc), _finalize_stats(action_acc),
    )
    print(f"Wrote {target_repo_id} -> {output_root} ({total_episodes} episodes, {cursor} frames)")


def convert_libero(
    source_repo: str, suite: str, target_repo_id: str, output_root: Path | None, max_episodes: int | None
):
    print(f"Downloading/locating {source_repo} locally (this is a large multi-suite repo, RGB + depth)...")
    repo_root = Path(snapshot_download(source_repo, repo_type="dataset"))
    source_root = repo_root / suite
    if not source_root.is_dir():
        raise FileNotFoundError(f"Suite subdir not found: {source_root}")

    output_root = output_root or (Path.home() / ".cache/huggingface/lerobot" / target_repo_id)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    info = json.loads((source_root / "meta" / "info.json").read_text())
    cam_keys = [k for k in info["features"] if k.startswith("observation.images.")]
    depth_keys = [k for k in info["features"] if k.startswith("observation.depth_ffv1.")]
    video_keys = cam_keys + depth_keys

    total_episodes = info["total_episodes"] if max_episodes is None else min(max_episodes, info["total_episodes"])
    print(f"Converting {total_episodes} of {info['total_episodes']} episodes "
          f"({len(cam_keys)} RGB + {len(depth_keys)} depth cameras)")

    _copy_meta_and_videos(source_root, output_root, video_keys)

    state_acc = None
    action_acc = None
    real_lengths: dict[int, int] = {}
    data_src = source_root / "data"
    for chunk_dir in tqdm(sorted(data_src.iterdir()), desc=f"Transform {suite}", unit="chunk"):
        dst_chunk = output_root / "data" / chunk_dir.name
        dst_chunk.mkdir(parents=True, exist_ok=True)
        for pq_file in sorted(chunk_dir.iterdir()):
            df = pd.read_parquet(pq_file)
            df = df[df["episode_index"] < total_episodes]
            if df.empty:
                continue

            ee_state = np.stack(df["observation.states.ee_state"].to_numpy()).astype(np.float64)  # (n,6)
            gripper = np.stack(df["observation.state"].to_numpy()).astype(np.float64)[:, 6]  # dup at [7]
            raw7 = np.concatenate([ee_state, gripper[:, None]], axis=1)  # (n,7)
            ep_idx_arr = df["episode_index"].to_numpy()

            new_state = np.zeros((len(df), 7), dtype=np.float64)
            new_action = np.zeros((len(df), 7), dtype=np.float64)
            for ep in np.unique(ep_idx_arr):
                positions = np.where(ep_idx_arr == ep)[0]
                ep_raw = raw7[positions]
                rot = Rotation.from_euler("XYZ", ep_raw[:, 3:6])
                pos_rel, euler_rel = _episode_relative_pos_rot(ep_raw[:, 0:3], rot)
                ep_rel_state = np.concatenate(
                    [pos_rel, euler_rel, np.clip(ep_raw[:, 6:7], 0.0, 1.0)], axis=1
                )
                ep_action = np.vstack([ep_rel_state[1:], ep_rel_state[-1:]])
                new_state[positions] = ep_rel_state
                new_action[positions] = ep_action
                real_lengths[int(ep)] = real_lengths.get(int(ep), 0) + len(positions)

            out_df = df[["timestamp", "frame_index", "episode_index", "index", "task_index"]].copy()
            out_df["observation.state"] = list(new_state.astype(np.float32))
            out_df["action"] = list(new_action.astype(np.float32))
            out_df.to_parquet(dst_chunk / pq_file.name, index=False)

            state_acc = _running_stats(new_state, state_acc)
            action_acc = _running_stats(new_action, action_acc)

    cursor = _verify_episode_lengths(output_root, real_lengths, total_episodes)
    if cursor != state_acc["count"]:
        raise AssertionError(
            f"episodes.jsonl-verified frame total {cursor} != transform pass count "
            f"{state_acc['count']} — must match exactly."
        )

    _write_info_and_stats(
        output_root, source_root, info, video_keys, EE_NAMES, 7,
        total_episodes, cursor, _finalize_stats(state_acc), _finalize_stats(action_acc),
    )
    print(f"Wrote {target_repo_id} -> {output_root} ({total_episodes} episodes, {cursor} frames)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    py = sub.add_parser("yam")
    py.add_argument("--source-repo", required=True)
    py.add_argument("--target-repo-id", required=True)
    py.add_argument("--output-root", default=None)
    py.add_argument("--max-episodes", type=int, default=None)

    pl = sub.add_parser("libero")
    pl.add_argument("--source-repo", required=True)
    pl.add_argument("--suite", required=True)
    pl.add_argument("--target-repo-id", required=True)
    pl.add_argument("--output-root", default=None)
    pl.add_argument("--max-episodes", type=int, default=None)

    args = p.parse_args()
    output_root = Path(args.output_root) if args.output_root else None
    if args.mode == "yam":
        convert_yam(args.source_repo, args.target_repo_id, output_root, args.max_episodes)
    else:
        convert_libero(args.source_repo, args.suite, args.target_repo_id, output_root, args.max_episodes)


if __name__ == "__main__":
    main()
