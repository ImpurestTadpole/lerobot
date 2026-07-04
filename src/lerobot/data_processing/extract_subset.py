#!/usr/bin/env python3
"""
Extract a reduced-DOF / RGB-only subset from a "master" dataset.

The master-dataset strategy: record teleop sessions ONCE with everything on —
all 18 xlerobot DOF and RGB-D (``use_depth=True`` on the head RealSense) — then
derive cheaper views of that data for specific training runs. This script does
the derivation: select state/action dims **by joint name**, drop (or keep)
depth streams, optionally resample fps and resize frames, and write a new
LeRobot dataset whose tensor shapes line up with open-source bimanual datasets
for co-training (see COTRAINING.md).

Profiles (``--profile``):
    bimanual12   left+right arm joints (12) — matches two 6-DOF-arm datasets
                 (ALOHA-style bimanual manipulation, converted UMI pairs)
    arms_head14  bimanual12 + head_pan/head_tilt (14)
    full18       every xlerobot dim (use with --drop-depth to strip depth only)

Or pass an explicit ``--keep-names`` list. Names must exist in the source.

Usage:
    lerobot-extract-subset \\
        --source-repo Odog16/master_home_v1 \\
        --target-repo-id Odog16/master_home_v1_bimanual12 \\
        --profile bimanual12 \\
        --push-to-hub false

The output keeps per-episode task strings, so the subset drops straight into
``lerobot-cotrain-align`` as a source (its 12 names then match a bimanual
reference schema instead of needing 6 dims padded).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from lerobot.data_processing.co_training_utils import (
    _compute_resample_indices,
    _ensure_depth_chw_numpy,
    _ensure_image_hwc_numpy,
    _remap_vector_by_names,
    logger,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import FrameTimestampError

XLEROBOT_ARM_NAMES = [
    "left_arm_shoulder_pan.pos", "left_arm_shoulder_lift.pos", "left_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos", "left_arm_wrist_roll.pos", "left_arm_gripper.pos",
    "right_arm_shoulder_pan.pos", "right_arm_shoulder_lift.pos", "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos", "right_arm_wrist_roll.pos", "right_arm_gripper.pos",
]
XLEROBOT_FULL_NAMES = XLEROBOT_ARM_NAMES + [
    "head_pan.pos", "head_tilt.pos", "x.vel", "y.vel", "theta.vel", "gantry.height_mm",
]

PROFILES: dict[str, list[str]] = {
    "bimanual12": XLEROBOT_ARM_NAMES,
    "arms_head14": XLEROBOT_ARM_NAMES + ["head_pan.pos", "head_tilt.pos"],
    "full18": XLEROBOT_FULL_NAMES,
}


def _feature_names(meta_features: dict, key: str) -> list[str]:
    names = (meta_features.get(key) or {}).get("names") or []
    return [str(n) for n in names]


def build_subset_features(
    src_features: dict[str, Any],
    keep_names: list[str],
    cameras: list[str] | None,
    drop_depth: bool,
    target_image_size: tuple[int, int] | None,
    effective_fps: int,
) -> tuple[dict[str, Any], list[str]]:
    """Target feature dict for the subset: reduced state/action + filtered cameras.

    Depth streams (dtype ``depth`` or a ``*_depth`` camera key, e.g. the
    ``observation.images.head_depth`` written by xlerobot with
    ``use_depth=True``) are removed when *drop_depth* — that is what turns an
    18-DOF RGB-D master into an RGB-only subset whose shapes match external
    bimanual datasets. Returns ``(features, dropped_camera_names)``.
    """
    target_features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (len(keep_names),),
                              "names": list(keep_names)},
        "action": {"dtype": "float32", "shape": (len(keep_names),),
                   "names": list(keep_names)},
    }
    dropped_cams: list[str] = []
    for key, feat in src_features.items():
        if not key.startswith("observation.images."):
            continue
        cam = key.removeprefix("observation.images.")
        is_depth = feat.get("dtype") == "depth" or cam.endswith("_depth")
        if drop_depth and is_depth:
            dropped_cams.append(cam)
            continue
        if cameras is not None and cam not in cameras:
            dropped_cams.append(cam)
            continue
        feat = json.loads(json.dumps(feat))  # deep copy without torch types
        if target_image_size is not None and not is_depth:
            th, tw = target_image_size
            feat["shape"] = (th, tw, 3)
            if "info" in feat:
                feat["info"]["video.height"] = th
                feat["info"]["video.width"] = tw
        if "info" in feat:
            feat["info"]["video.fps"] = effective_fps
        feat.pop("video_info", None)
        target_features[key] = feat
    return target_features, dropped_cams


def extract_subset(
    source_repo: str,
    target_repo_id: str,
    keep_names: list[str],
    output_root: Path | None = None,
    cameras: list[str] | None = None,
    drop_depth: bool = True,
    target_fps: int | None = None,
    target_image_size: tuple[int, int] | None = None,
    robot_type: str | None = None,
    push_to_hub: bool = False,
    force_rebuild: bool = False,
) -> Path:
    """Write a dim/camera subset of *source_repo* as a new dataset.

    *keep_names* selects state AND action dims by joint name (order defines the
    output order). *cameras* restricts ``observation.images.*`` keys (short
    names, e.g. ``head``); default keeps all. *drop_depth* removes depth-typed
    features and any camera key ending in ``_depth``.
    """
    src_ds = LeRobotDataset(source_repo)
    src_meta = src_ds.meta
    src_fps = src_meta.fps
    effective_fps = min(src_fps, target_fps) if target_fps else src_fps

    src_state_names = _feature_names(src_meta.features, "observation.state")
    src_action_names = _feature_names(src_meta.features, "action")
    for label, src_names in (("state", src_state_names), ("action", src_action_names)):
        missing = [n for n in keep_names if n not in src_names]
        if missing:
            raise ValueError(
                f"--keep-names not present in source {label} names: {missing}. "
                f"Source has: {src_names}"
            )

    target_features, dropped_cams = build_subset_features(
        src_meta.features, keep_names, cameras, drop_depth, target_image_size, effective_fps
    )
    cam_keys = [k for k in target_features if k.startswith("observation.images.")]
    if not cam_keys:
        raise ValueError("No cameras left after --cameras/--drop-depth filtering.")

    if output_root is None:
        output_root = Path(f"~/.cache/huggingface/lerobot/{target_repo_id}").expanduser()
    if output_root.exists():
        if not force_rebuild:
            raise FileExistsError(
                f"{output_root} exists. Pass --force-rebuild to overwrite it."
            )
        logger.warning("Force rebuild: removing %s", output_root)
        shutil.rmtree(output_root)

    logger.info(
        "Extracting %s → %s: dims %d→%d (%s), cameras %s (dropped %s), fps %d→%d%s",
        source_repo, target_repo_id,
        len(src_state_names), len(keep_names),
        ", ".join(keep_names[:4]) + ("…" if len(keep_names) > 4 else ""),
        [k.removeprefix("observation.images.") for k in cam_keys],
        dropped_cams or "none",
        src_fps, effective_fps,
        f", resized to {target_image_size[0]}×{target_image_size[1]}" if target_image_size else "",
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
    for ep_idx in tqdm(range(num_episodes), desc="Extract episodes", unit="ep", dynamic_ncols=True):
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
                    _remap_vector_by_names(arr, src_names, keep_names)
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
    logger.info("Subset written to %s", output_root)

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
        description="Extract a reduced-DOF / RGB-only subset from a master dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--target-repo-id", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile", choices=sorted(PROFILES),
                       help="Named joint subset: " + ", ".join(sorted(PROFILES)))
    group.add_argument("--keep-names", nargs="+",
                       help="Explicit joint names to keep (order = output order).")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera short names to keep (default: all RGB cameras).")
    parser.add_argument("--keep-depth", action="store_true",
                        help="Keep *_depth streams (dropped by default — subsets are "
                             "meant to shape-match RGB-only external datasets).")
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--target-image-size", type=str, default=None, help="HxW, e.g. 360x640")
    parser.add_argument("--robot-type", default=None,
                        help="Override robot_type in the subset metadata (default: source's).")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--push-to-hub", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=False)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    keep_names = PROFILES[args.profile] if args.profile else args.keep_names

    size = None
    if args.target_image_size:
        parts = args.target_image_size.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"--target-image-size must be HxW, got {args.target_image_size!r}")
        size = (int(parts[0]), int(parts[1]))

    extract_subset(
        source_repo=args.source_repo,
        target_repo_id=args.target_repo_id,
        keep_names=keep_names,
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
