#!/usr/bin/env python3
"""Add a JET-colourmap RGB rendering of ``observation.images.head_depth`` to
``Odog16/keys_into_bowl_ob15`` as a new video camera key
(``observation.images.head_depth_rgb``), so a depth-blind policy (SmolVLA,
ACT — neither has a native depth head in this codebase) can still see coarse
depth structure through its ordinary RGB vision encoder.

Depth is read through the normal ``LeRobotDataset`` decode path, not
hand-rolled: ``observation.images.head_depth`` is tagged
``info.is_depth_map=True`` with ``depth_unit=mm`` (see
``meta/info.json``), which routes it through the pyav depth decoder
(``video_utils.decode_video_frames_pyav(..., is_depth=True)``) and returns
real millimetre values (verified on episode 0: range ~10-6900mm), not
raw 12-bit pixel codes — so no separate log/shift decode is needed here.

The output dataset does NOT re-include the original ``head_depth`` feature:
``add_frame()``'s per-frame image dispatch picks its encoder config from
``dtype == "depth"`` literally (see ``DatasetWriter.add_frame``), but this
source's depth feature is tagged ``dtype: "video"`` + ``info.is_depth_map``
instead — re-writing it through that path would silently drop the lossless
12-bit gray12le encoding instead of preserving it. Training only needs the
derived RGB colourmap, not the raw stream, so it's simplest and safest to
just not touch the original depth video at all here.

Per-episode percentile clipping (2nd/98th) rather than a fixed global range
or per-frame min/max: a fixed global range would be dominated by outlier
background pixels (episode 0 already spans 10-6900mm), and per-frame min/max
would make the colour scale flicker frame-to-frame. Percentile clipping is
computed once per episode (from all its frames) for a stable-but-local scale.

Usage:
    python scripts/preprocess_depth_ob15.py \\
        --source-repo Odog16/keys_into_bowl_ob15 \\
        --target-repo-id Odog16/keys_into_bowl_ob15_depth_rgb \\
        --push-to-hub
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEPTH_KEY = "observation.images.head_depth"
DEPTH_RGB_KEY = "observation.images.head_depth_rgb"


def depth_frame_to_rgb(depth_chw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """(1,H,W) float32 mm -> (H,W,3) uint8 JET colourmap, clipped to [lo, hi]."""
    d = depth_chw[0]
    span = max(hi - lo, 1e-6)
    d_norm = np.clip((d - lo) / span * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-repo", default="Odog16/keys_into_bowl_ob15")
    ap.add_argument("--target-repo-id", default="Odog16/keys_into_bowl_ob15_depth_rgb")
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    src = LeRobotDataset(args.source_repo)
    if DEPTH_KEY not in src.meta.features:
        raise ValueError(f"{args.source_repo} has no {DEPTH_KEY} feature")

    cam_keys = [k for k in src.meta.features if k.startswith("observation.images.") and k != DEPTH_KEY]
    depth_shape = src.meta.features[DEPTH_KEY]["shape"]  # (H, W, 1)

    features = {}
    for k in ("observation.state", "action"):
        features[k] = dict(src.meta.features[k])
    for k in cam_keys:
        features[k] = dict(src.meta.features[k])
    features[DEPTH_RGB_KEY] = {
        "dtype": "video",
        "shape": (depth_shape[0], depth_shape[1], 3),
        "names": ["height", "width", "channels"],
    }

    output_root = args.output_root or (Path.home() / ".cache/huggingface/lerobot" / args.target_repo_id)
    if output_root.exists():
        if not args.force_rebuild:
            raise FileExistsError(f"{output_root} already exists; pass --force-rebuild to overwrite")
        shutil.rmtree(output_root)

    out_ds = LeRobotDataset.create(
        repo_id=args.target_repo_id,
        fps=src.meta.fps,
        robot_type=src.meta.robot_type,
        features=features,
        root=output_root,
        use_videos=True,
        image_writer_threads=4,
    )

    n_eps = src.meta.total_episodes if args.max_episodes is None else min(args.max_episodes, src.meta.total_episodes)
    n_written = 0
    n_skipped = 0
    for ep_idx in tqdm(range(n_eps), desc="Adding depth_rgb", unit="ep"):
        try:
            ep_meta = src.meta.episodes[ep_idx]
            lo_idx, hi_idx = ep_meta["dataset_from_index"], ep_meta["dataset_to_index"]
            raw_frames = [src[i] for i in range(lo_idx, hi_idx)]
            task = raw_frames[0]["task"]

            depth_stack = np.stack([f[DEPTH_KEY].numpy() for f in raw_frames])  # (T, 1, H, W)
            lo, hi = np.percentile(depth_stack, [2, 98])

            for raw in raw_frames:
                frame = {
                    "observation.state": raw["observation.state"].numpy(),
                    "action": raw["action"].numpy(),
                    "task": task,
                }
                for k in cam_keys:
                    chw = raw[k].numpy()
                    frame[k] = np.clip(chw * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                frame[DEPTH_RGB_KEY] = depth_frame_to_rgb(raw[DEPTH_KEY].numpy(), lo, hi)
                out_ds.add_frame(frame)
            out_ds.save_episode()
            n_written += 1
        except Exception as e:  # noqa: BLE001 - isolated bad episode shouldn't kill the whole run
            print(f"WARNING: skipping episode {ep_idx}: {type(e).__name__}: {e}")
            out_ds.clear_episode_buffer()
            n_skipped += 1

    out_ds.finalize()
    print(f"Wrote {args.target_repo_id} -> {output_root} ({n_written} episodes, {n_skipped} skipped)")

    if args.push_to_hub:
        out_ds.push_to_hub()
        print(f"Pushed to https://huggingface.co/datasets/{args.target_repo_id}")


if __name__ == "__main__":
    main()
