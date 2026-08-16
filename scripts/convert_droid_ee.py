#!/usr/bin/env python3
"""Convert DROID (or any LeRobot dataset whose observation.state is ALREADY
absolute-frame EE pose [x,y,z,roll,pitch,yaw,gripper]) to the same
episode-start-relative EE-space convention used elsewhere in this project
(see convert_joint_to_ee.py's module docstring).

Unlike convert_joint_to_ee.py, this does NOT go through
LeRobotDataset.create()/add_frame() and does NOT touch video files at all —
at DROID's scale (95,658 episodes / 27.6M frames), per-frame video
decode+re-encode would take on the order of two weeks. Since DROID's videos
don't need to change (only observation.state/action need a value transform,
no new camera data, no FK), this script instead:

  1. Symlinks video files through unchanged (instant, no re-encoding).
  2. Rewrites each data/chunk-*/file-*.parquet in place: transforms
     observation.state and action to episode-start-relative EE-space,
     leaves every other column (images excepted, since those aren't in
     these parquet files at all — image *paths* only) untouched.
  3. Copies meta/episodes/*, meta/tasks.parquet unchanged (episode/frame
     boundaries and task strings don't change).
  4. Copies meta/info.json, only updating the observation.state/action
     feature "names" field to the project's EE_NAMES convention.
  5. Copies meta/stats.json, replacing ONLY the observation.state/action
     entries with exact (not sampled) min/max/mean/std/count computed over
     the transformed data — this matters because
     lerobot-cotrain-align's merge step (`aggregate_stats`) trusts each
     source's stats.json rather than recomputing from raw data, so a stale
     stats.json here would corrupt cross-source normalization for
     everything merged in, not just this source.

DROID's own "action" column is a *delta* command (near-zero, not a target
pose — verified: action[0] != state[1]), a different convention from every
other EE-space source in this project (all use action[t] == state[t+1], the
next-frame absolute target). Kept consistent with the rest of the project:
DROID's native action column is discarded and re-derived as the one-step
shift of the (already-relabeled) state, not copied through.

Usage:
    python scripts/convert_droid_ee.py \\
        --source-repo lerobot/droid_1.0.1 \\
        --target-repo-id Odog16/droid_ee \\
        --max-episodes 500   # omit for the full dataset
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

EE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def episode_relative(state: np.ndarray) -> np.ndarray:
    """(T, 7) absolute EE pose -> (T, 7) relative to frame 0, matching the
    same rotation-composition approach as convert_joint_to_ee.py (not naive
    Euler subtraction, which breaks near +/-pi wraparound)."""
    pos_rel = state[:, 0:3] - state[0, 0:3]
    rot = Rotation.from_euler("XYZ", state[:, 3:6])
    rot0_inv = rot[0].inv()
    rel_rot = rot0_inv * rot
    rel_euler = rel_rot.as_euler("XYZ")
    out = state.copy()
    out[:, 0:3] = pos_rel
    out[:, 3:6] = rel_euler
    return out


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Transform observation.state/action in place, per episode, without ever
    reconstructing the dataframe via groupby-apply — pandas' groupby(...).apply()
    can silently drop the grouping column from the result depending on version/
    return shape, which would corrupt every other column (episode_index first)."""
    df = df.copy()
    state_col = list(df["observation.state"].values)
    action_col = list(df["action"].values)
    ep_idx_arr = df["episode_index"].values
    for ep in np.unique(ep_idx_arr):
        positions = np.where(ep_idx_arr == ep)[0]
        state = np.stack([state_col[i] for i in positions]).astype(np.float64)
        rel_state = episode_relative(state)
        rel_action = np.vstack([rel_state[1:], rel_state[-1:]])
        for j, pos in enumerate(positions):
            state_col[pos] = rel_state[j].astype(np.float32)
            action_col[pos] = rel_action[j].astype(np.float32)
    df["observation.state"] = state_col
    df["action"] = action_col
    return df


def convert(source_repo: str, target_repo_id: str, output_root: Path | None, max_episodes: int | None):
    print(f"Downloading/locating {source_repo} locally (this may take a while for large datasets)...")
    source_root = Path(snapshot_download(source_repo, repo_type="dataset"))

    output_root = output_root or (Path.home() / ".cache/huggingface/lerobot" / target_repo_id)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    info = json.loads((source_root / "meta" / "info.json").read_text())
    total_episodes = info["total_episodes"] if max_episodes is None else min(max_episodes, info["total_episodes"])
    print(f"Converting {total_episodes} of {info['total_episodes']} episodes")

    # --- meta: episodes / tasks unchanged (boundaries don't change) ---
    for rel in ["meta/episodes", "meta/tasks.parquet", "meta/tasks.jsonl"]:
        src = source_root / rel
        if src.exists():
            dst = output_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    # --- videos: symlink through, no re-encoding needed ---
    videos_src = source_root / "videos"
    if videos_src.exists():
        videos_dst = output_root / "videos"
        for cam_dir in videos_src.iterdir():
            if not cam_dir.is_dir():
                continue
            for chunk_dir in cam_dir.iterdir():
                if not chunk_dir.is_dir():
                    continue
                dst_chunk = videos_dst / cam_dir.name / chunk_dir.name
                dst_chunk.mkdir(parents=True, exist_ok=True)
                for f in chunk_dir.iterdir():
                    (dst_chunk / f.name).symlink_to(f.resolve())

    # --- data: transform state/action per episode, write new parquet files ---
    data_src = source_root / "data"
    running_sum = None
    running_sq_sum = None
    running_min = None
    running_max = None
    running_count = 0
    action_running_sum = None
    action_running_sq_sum = None
    action_running_min = None
    action_running_max = None
    # Real per-episode row counts, measured directly from the data as we go —
    # NOT trusted from meta/episodes' dataset_from_index/dataset_to_index/length.
    # Verified against lerobot/droid_1.0.1 specifically: those fields are wrong
    # for ~42.6k of 95,658 episodes (everything from episode 52973 on undercounts
    # its real row span, ~12.4M missing rows total) — a genuine upstream
    # metadata bug, confirmed by summing real parquet row counts per episode_index
    # and comparing to the claimed dataset_to_index-dataset_from_index. Using the
    # claimed boundaries as-is would have every downstream episode's frames
    # misaligned relative to where they actually sit in data/*.parquet from
    # episode 52973 onward. Recomputed here from truth instead of copied through.
    real_episode_lengths: dict[int, int] = {}

    chunk_dirs = sorted(data_src.iterdir())
    for chunk_dir in tqdm(chunk_dirs, desc="Transforming data chunks", unit="chunk"):
        dst_chunk = output_root / "data" / chunk_dir.name
        dst_chunk.mkdir(parents=True, exist_ok=True)
        for pq_file in sorted(chunk_dir.iterdir()):
            df = pd.read_parquet(pq_file)
            # episode_index is a global identifier (0..total_original_episodes-1),
            # not per-chunk-relative, so this directly selects "the first N episodes".
            df = df[df["episode_index"] < total_episodes]
            if df.empty:
                continue

            transformed = transform_dataframe(df)
            transformed.to_parquet(dst_chunk / pq_file.name, index=False)

            for ep, n in transformed["episode_index"].value_counts().items():
                real_episode_lengths[int(ep)] = real_episode_lengths.get(int(ep), 0) + int(n)

            state_arr = np.stack(transformed["observation.state"].values).astype(np.float64)
            action_arr = np.stack(transformed["action"].values).astype(np.float64)
            if running_sum is None:
                running_sum = state_arr.sum(axis=0)
                running_sq_sum = (state_arr**2).sum(axis=0)
                running_min = state_arr.min(axis=0)
                running_max = state_arr.max(axis=0)
                action_running_sum = action_arr.sum(axis=0)
                action_running_sq_sum = (action_arr**2).sum(axis=0)
                action_running_min = action_arr.min(axis=0)
                action_running_max = action_arr.max(axis=0)
            else:
                running_sum += state_arr.sum(axis=0)
                running_sq_sum += (state_arr**2).sum(axis=0)
                running_min = np.minimum(running_min, state_arr.min(axis=0))
                running_max = np.maximum(running_max, state_arr.max(axis=0))
                action_running_sum += action_arr.sum(axis=0)
                action_running_sq_sum += (action_arr**2).sum(axis=0)
                action_running_min = np.minimum(action_running_min, action_arr.min(axis=0))
                action_running_max = np.maximum(action_running_max, action_arr.max(axis=0))
            running_count += len(state_arr)

    mean = running_sum / running_count
    std = np.sqrt(np.maximum(running_sq_sum / running_count - mean**2, 0))
    action_mean = action_running_sum / running_count
    action_std = np.sqrt(np.maximum(action_running_sq_sum / running_count - action_mean**2, 0))

    # --- meta/episodes: overwrite dataset_from_index/dataset_to_index/length with
    # real values recomputed from the data (see comment above real_episode_lengths;
    # every other column — tasks, video chunk/file/timestamp mapping, per-episode
    # stats/* — is untouched, since only the row-count bookkeeping was wrong upstream,
    # not which physical file/video segment an episode's content lives in). ---
    ep_meta_files = sorted((output_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    cursor = 0
    for f in ep_meta_files:
        eps_df = pd.read_parquet(f)
        eps_df = eps_df[eps_df["episode_index"] < total_episodes].sort_values("episode_index")
        new_from, new_to, new_len = [], [], []
        for ep in eps_df["episode_index"]:
            real_len = real_episode_lengths.get(int(ep), 0)
            new_from.append(cursor)
            cursor += real_len
            new_to.append(cursor)
            new_len.append(real_len)
        eps_df["dataset_from_index"] = new_from
        eps_df["dataset_to_index"] = new_to
        eps_df["length"] = new_len
        eps_df.to_parquet(f, index=False)
    if cursor != running_count:
        raise AssertionError(
            f"Recomputed episode boundaries sum to {cursor} frames but the data transform pass "
            f"counted {running_count} — these must match exactly. Investigate before trusting output."
        )

    # --- meta/info.json: same shape, EE_NAMES instead of motor_N, REAL total_frames
    # (source's own total_frames field is the same stale number the broken
    # per-episode indices summed to — see real_episode_lengths comment) ---
    info["total_episodes"] = total_episodes
    info["total_frames"] = running_count
    info["features"]["observation.state"]["names"] = EE_NAMES
    info["features"]["action"]["names"] = EE_NAMES
    (output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # --- meta/stats.json: copy through, replace only state/action ---
    stats = json.loads((source_root / "meta" / "stats.json").read_text())
    stats["observation.state"] = {
        "min": running_min.tolist(),
        "max": running_max.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [int(running_count)],
    }
    stats["action"] = {
        "min": action_running_min.tolist(),
        "max": action_running_max.tolist(),
        "mean": action_mean.tolist(),
        "std": action_std.tolist(),
        "count": [int(running_count)],
    }
    (output_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"Wrote {target_repo_id} -> {output_root} ({total_episodes} episodes, {running_count} frames)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-repo", required=True)
    p.add_argument("--target-repo-id", required=True)
    p.add_argument("--output-root", default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()
    convert(
        args.source_repo,
        args.target_repo_id,
        Path(args.output_root) if args.output_root else None,
        args.max_episodes,
    )


if __name__ == "__main__":
    main()
