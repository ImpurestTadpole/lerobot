#!/usr/bin/env python3
"""
Cross-embodiment co-training dataset alignment and merge utilities.

Aligns one or more LeRobot-format datasets (ALOHA, Open-X, other robots, mixed xlerobot
tasks) into a **shared schema**, then **optionally** physically merges them with
``aggregate_datasets()`` and **optionally** uploads to the Hugging Face Hub.

Per-source alignment
---------------------
1. Temporal resample to a common effective fps (nearest-neighbour when downsampling).
2. State/action: pad or truncate; with ``--match-features-from``, remap by joint **names**
   (drops extras, e.g. ``gantry.vel``, when the reference has 18 names).
3. Cameras: rename via ``--camera-remap`` (defaults map ALOHA/Open-X names → head /
   left_wrist / right_wrist).
4. Write each source to ``<output-root.parent>/_align_tmp_<safe_repo_id>/`` (cached when
   ``meta/info.json`` still matches fps, joint names, and image shapes).

Multi-source merge (``aggregate_datasets``)
--------------------------------------------
Merged shards must have **identical** ``features`` (keys, shapes, video ``info``, fps).
The script builds one canonical visual schema when ``len(sources) > 1``:

- **Camera keys**: *intersection* of remapped ``observation.images.*`` keys (e.g. depth
  only on one dataset is omitted from the merge).
- **Resolution**: minimum H×W over those keys, unless ``--target-image-size HxW`` is set.
- **Fps**: ``min(min(source.fps), target_fps)`` for every shard.

Always use ``--match-features-from`` (and optional ``--match-features-root``) when
merging xlerobot-style tasks so ``observation.state`` / ``action`` **names** match;
``aggregate_datasets`` compares full feature dicts, not only vector length.

Hub upload
----------
``--push-to-hub true`` calls ``LeRobotDataset(..., root=merged_root).push_to_hub()``
after merge. Requires ``huggingface-cli login`` and a valid ``meta/tasks.parquet`` under
the merged root. If merge used ``push_to_hub false``, push later::

    from pathlib import Path
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rid = "Odog16/my_merged_dataset"
    root = Path("~/.cache/huggingface/lerobot").expanduser() / rid
    LeRobotDataset(rid, root=root).push_to_hub()

Use ``--inspect-only`` to print fps / state / action dims / cameras per source.
``--skip-missing-sources`` continues if a repo id is missing on Hub or locally.
``--force-rebuild`` wipes ``_align_tmp_*`` and the output tree when caches are stale.
By default, sources whose ``meta`` already matches the merge schema (fps, ``robot_type``,
full ``features`` dict including canonical video ``info``) are **not** re-encoded; they
are merged in-place from their cache root. Use ``--realign-all-sources`` to always align
via ``_align_tmp_*``.

**Note:** uploading **training checkpoints** (policies) is separate — use
``--policy.push_to_hub`` in ``lerobot-train`` or ``python -m lerobot.upload_checkpoints``;
this script only handles **datasets** (``repo_type="dataset"``).

Usage (CLI):
    python src/lerobot/data_processing/co_training_utils.py \\
        --source-repos lerobot/aloha_sim_insertion_human lerobot/aloha_mobile_cabinet \\
        --target-repo-id Odog16/aloha_aligned_for_cotrain \\
        --target-fps 30 \\
        --target-state-dim 18 \\
        --target-action-dim 18 \\
        --match-features-from Odog16/tool_pickup \\
        --camera-remap "top:head,cam_high:head,cam_left_wrist:left_wrist,cam_right_wrist:right_wrist" \\
        --output-root ~/.cache/huggingface/lerobot/Odog16/aloha_aligned_for_cotrain \\
        --push-to-hub false

Usage (Python API):
    from lerobot.data_processing.co_training_utils import align_datasets_for_cotraining

    align_datasets_for_cotraining(
        source_repos=["lerobot/aloha_sim_insertion_human"],
        target_repo_id="Odog16/aloha_aligned_for_cotrain",
        target_fps=30,
        target_state_dim=18,
        target_action_dim=18,
        camera_remap={"top": "head", "cam_high": "head",
                      "cam_left_wrist": "left_wrist",
                      "cam_right_wrist": "right_wrist"},
        output_root=Path("~/.cache/huggingface/lerobot/Odog16/aloha_aligned").expanduser(),
        match_features_from="Odog16/tool_pickup",
        push_to_hub=False,
    )
"""

from __future__ import annotations

import sys

# ``conda run`` / non-TTY stdout is often fully buffered — tell the user we started.
print(
    "co_training_utils: loading torch, OpenCV, LeRobot (30s–few min on cold start)…",
    flush=True,
)

import argparse
import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from huggingface_hub.errors import RepositoryNotFoundError

from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_TASKS_PATH
from lerobot.datasets.video_utils import FrameTimestampError

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _compute_resample_indices(src_fps: int, target_fps: int, n_src_frames: int) -> list[int]:
    """Nearest-neighbour frame index list for resampling *src_fps* → *target_fps*.

    For each target frame slot k (0 … n_target-1) we pick the source frame
    whose timestamp is closest to k / target_fps.  That source index is
    ``min(round(k * src_fps / target_fps), n_src_frames - 1)``.

    Unlike simple integer subsampling (keep_every = round(src/tgt)), this
    preserves the true *target_fps* timing.  For example, 50 → 30 fps gives
    the non-uniform pattern  0, 2, 3, 5, 7, 8, 10, …  rather than the
    uniform  0, 2, 4, 6, …  (which is actually 25 fps).

    When target_fps >= src_fps every source frame is kept (no upsampling).
    """
    if target_fps >= src_fps:
        return list(range(n_src_frames))
    n_target = max(1, round(n_src_frames * target_fps / src_fps))
    return [min(round(k * src_fps / target_fps), n_src_frames - 1) for k in range(n_target)]


def _pad_vector(
    vec: np.ndarray, target_dim: int, fill_values: np.ndarray | None = None
) -> np.ndarray:
    """Resize *vec* along its last axis to *target_dim* (truncate or pad).

    Padded components take *fill_values* (a length-*target_dim* vector) when
    given, else 0. If the source is longer (e.g. an extra ``gantry.vel``),
    keeps the leading *target_dim* components — prefer
    :func:`_remap_vector_by_names` when metadata lists ``names`` so the correct
    components are kept.
    """
    vec = np.asarray(vec, dtype=np.float32)
    current_dim = int(vec.shape[-1])
    if current_dim > target_dim:
        return vec[..., :target_dim].copy()
    if current_dim == target_dim:
        return vec.copy()
    pad_width = [(0, 0)] * (vec.ndim - 1) + [(0, target_dim - current_dim)]
    out = np.pad(vec, pad_width, mode="constant", constant_values=0.0)
    if fill_values is not None and vec.ndim == 1:
        out[current_dim:] = np.asarray(fill_values, dtype=np.float32)[current_dim:]
    return out


def _remap_vector_by_names(
    vec: np.ndarray,
    src_names: list[str] | None,
    ref_names: list[str],
    fill_values: np.ndarray | None = None,
) -> np.ndarray:
    """Build a *len(ref_names)* vector by copying each ``ref_names[i]`` from *src_names*.

    Components absent from *src_names* take *fill_values* (a length-
    ``len(ref_names)`` vector) when given, else 0. Drops source-only components
    (e.g. ``gantry.vel`` when the reference schema stops at
    ``gantry.height_mm``). Falls back to :func:`_pad_vector` if *src_names* is
    missing or its length does not match *vec*.
    """
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if src_names is None or len(src_names) != len(vec):
        return _pad_vector(vec, len(ref_names), fill_values=fill_values)
    by_name: dict[str, int] = {}
    for i, n in enumerate(src_names):
        if n not in by_name:
            by_name[n] = i
    if fill_values is not None:
        out = np.asarray(fill_values, dtype=np.float32).copy()
    else:
        out = np.zeros(len(ref_names), dtype=np.float32)
    for i, name in enumerate(ref_names):
        j = by_name.get(name)
        if j is not None:
            out[i] = vec[j]
    return out


# ---------------------------------------------------------------------------
# Padding fill policies (see COTRAINING.md — co-training bias mitigation)
# ---------------------------------------------------------------------------

PAD_FILL_MODES = ("zero", "ref-mean", "state-copy")

# Joint-name tokens that denote velocity dims; for those, 0 is semantically
# correct ("stopped") and is kept in every fill mode.
_VEL_NAME_TOKENS = (".vel", "_vel", "velocity")


def _is_velocity_name(name: str) -> bool:
    return any(tok in name for tok in _VEL_NAME_TOKENS)


def _missing_dims(src_names: list[str] | None, src_dim: int, ref_names: list[str]) -> list[int]:
    """Indices of *ref_names* that a source cannot provide (they get filled).

    Without usable source names, alignment falls back to positional padding,
    so the missing dims are the tail beyond the source vector length.
    """
    if src_names is None or len(src_names) != src_dim:
        return list(range(min(src_dim, len(ref_names)), len(ref_names)))
    src = set(src_names)
    return [i for i, n in enumerate(ref_names) if n not in src]


def _reference_fill_values(
    ref_stats: dict[str, Any] | None,
    feature_key: str,
    ref_names: list[str],
    pad_fill_mode: str,
) -> np.ndarray:
    """Fill vector for padded dims: reference per-dim mean, except velocity dims → 0.

    Filling with the reference dataset's mean keeps the merged normalization
    stats centred on the real joint distribution instead of dragging them
    toward 0 (which can put the robot's true home/extreme positions far in the
    normalized tails). Returns zeros for ``pad_fill_mode="zero"`` or when
    reference stats are unavailable.
    """
    fill = np.zeros(len(ref_names), dtype=np.float32)
    if pad_fill_mode == "zero" or ref_stats is None:
        return fill
    mean = (ref_stats.get(feature_key) or {}).get("mean")
    if mean is None:
        logger.warning(
            "pad-fill-mode=%s: reference stats have no mean for %s; padded dims fall back to 0.",
            pad_fill_mode, feature_key,
        )
        return fill
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    if len(mean) != len(ref_names):
        logger.warning(
            "pad-fill-mode=%s: reference %s mean has %d dims, expected %d; padded dims fall back to 0.",
            pad_fill_mode, feature_key, len(mean), len(ref_names),
        )
        return fill
    for i, name in enumerate(ref_names):
        if not _is_velocity_name(name):
            fill[i] = mean[i]
    return fill


def _load_reference_stats(
    repo_id: str, root: Path | str | None
) -> dict[str, Any] | None:
    """Per-feature stats (mean/std/min/max arrays) of the reference dataset."""
    root_path = Path(root).expanduser() if root else None
    meta = LeRobotDatasetMetadata(repo_id, root=root_path)
    stats = meta.stats
    if not stats:
        logger.warning("Reference dataset %s has no stats; fill/override falls back to 0/no-op.", repo_id)
        return None
    return stats


_ALIGN_OPTIONS_FILENAME = "cotrain_align_options.json"


def _write_align_options(aligned_root: Path, pad_fill_mode: str) -> None:
    opts_path = aligned_root / "meta" / _ALIGN_OPTIONS_FILENAME
    opts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(opts_path, "w", encoding="utf-8") as f:
        json.dump({"pad_fill_mode": pad_fill_mode}, f, indent=2)


def _cached_pad_fill_mode(meta_dir: Path) -> str:
    """Fill mode a cached aligned dataset was written with (pre-existing caches → zero)."""
    opts_path = meta_dir / _ALIGN_OPTIONS_FILENAME
    if not opts_path.is_file():
        return "zero"
    try:
        with open(opts_path, encoding="utf-8") as f:
            return json.load(f).get("pad_fill_mode", "zero")
    except (json.JSONDecodeError, OSError):
        return "zero"


def _ensure_image_hwc_numpy(
    value: Any,
    feature: dict[str, Any],
    target_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert a decoded frame to HWC uint8 ``numpy``, optionally resizing.

    :meth:`LeRobotDataset.__getitem__` returns video frames as ``(C, H, W)``
    float tensors in [0, 1].  ``DatasetWriter.add_frame`` expects either
    ``(H, W, C)`` or ``(C, H, W)`` numpy arrays with uint8 dtype for video
    features.  We always normalise to ``(H, W, C)`` uint8 so the writer and
    codec have a consistent input, and optionally resize to *target_hw*.
    """
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    if arr.ndim != 3 or feature.get("dtype") not in ("image", "video"):
        return arr

    # CHW → HWC
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    # float [0,1] → uint8 [0,255]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    # Resize if a target height/width was requested and differs from current
    if target_hw is not None:
        h, w = arr.shape[:2]
        th, tw = target_hw
        if (h, w) != (th, tw):
            arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(arr)


def _ensure_depth_chw_numpy(
    value: Any,
    target_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Normalise depth to ``(1, H, W)`` uint16 mm for :func:`write_image` with ``is_depth=True``."""
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim != 3:
        return arr
    # HWC single channel → CHW
    if arr.shape[-1] == 1 and arr.shape[0] != 1:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.shape[0] != 1:
        return arr
    if arr.dtype == np.int16:
        arr = np.where(arr < 0, 0, arr).astype(np.uint16)
    elif arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr * 1000.0, 0, 65535).astype(np.uint16)
    elif arr.dtype != np.uint16:
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    if target_hw is not None:
        th, tw = target_hw
        _, h, w = arr.shape
        if (h, w) != (th, tw):
            plane = cv2.resize(arr[0], (tw, th), interpolation=cv2.INTER_NEAREST)
            arr = plane[np.newaxis, ...]
    return np.ascontiguousarray(arr)


def _remap_camera_keys(
    sample: dict[str, Any],
    camera_remap: dict[str, str],
) -> dict[str, Any]:
    """Rename observation.images.<src> → observation.images.<dst> in *sample*.

    If multiple source cameras map to the same destination name, only the first
    present source key is kept; duplicates are dropped.
    """
    out: dict[str, Any] = {}
    used_dst: set[str] = set()
    for k, v in sample.items():
        if not k.startswith("observation.images."):
            out[k] = v
            continue
        src_cam = k.removeprefix("observation.images.")
        dst_cam = camera_remap.get(src_cam, src_cam)
        dst_key = f"observation.images.{dst_cam}"
        if dst_key in used_dst:
            continue
        out[dst_key] = v
        used_dst.add(dst_key)
    return out


def _state_action_names_from_reference(
    repo_id: str,
    root: Path | str | None,
    target_state_dim: int,
    target_action_dim: int,
) -> tuple[list[str], list[str]]:
    """Load ``observation.state`` / ``action`` ``names`` from a reference dataset.

    ``aggregate_datasets()`` requires *identical* ``features`` dicts, including
    the string ``names`` list for each vector feature — not just matching
    shapes.  Use this with one of your xlerobot task repos so aligned ALOHA
    data can be merged and co-trained without metadata mismatches.
    """
    root_path = Path(root).expanduser() if root else None
    meta = LeRobotDatasetMetadata(repo_id, root=root_path)
    st = meta.features.get("observation.state") or {}
    ac = meta.features.get("action") or {}
    sn, an = st.get("names"), ac.get("names")
    if not isinstance(sn, (list, tuple)) or not isinstance(an, (list, tuple)):
        raise ValueError(
            f"Reference dataset {repo_id!r} has no observation.state/action "
            f'"names" lists in meta/features.'
        )
    sn, an = list(sn), list(an)
    if len(sn) != target_state_dim or len(an) != target_action_dim:
        raise ValueError(
            f"Reference {repo_id!r} has len(names) state={len(sn)}, action={len(an)}; "
            f"expected {target_state_dim} and {target_action_dim}. "
            "Adjust --target-state-dim / --target-action-dim or use another reference."
        )
    return sn, an


def _alignment_artifacts_match(
    info_json: Path,
    target_features: dict[str, Any],
    effective_fps: int,
    pad_fill_mode: str = "zero",
) -> bool:
    """True if *info_json* matches the schema we would write with *target_features*."""
    if not info_json.is_file():
        return False
    tasks_parquet = info_json.parent / Path(DEFAULT_TASKS_PATH).name
    if not tasks_parquet.is_file():
        return False
    if _cached_pad_fill_mode(info_json.parent) != pad_fill_mode:
        return False
    with open(info_json, encoding="utf-8") as f:
        cached = json.load(f)
    if int(cached.get("fps", -1)) != int(effective_fps):
        return False
    cf = cached.get("features") or {}
    for key in ("observation.state", "action"):
        want = target_features.get(key) or {}
        wnames = want.get("names")
        cnames = (cf.get(key) or {}).get("names")
        if wnames is None or cnames is None:
            return False
        if list(cnames) != list(wnames):
            return False
    for k, feat in target_features.items():
        if not k.startswith("observation.images."):
            continue
        tshape = list(feat.get("shape", ()))
        cshape = (cf.get(k) or {}).get("shape")
        if cshape is None or list(cshape) != tshape:
            return False
    return True


def _source_matches_merge_schema(
    src_meta: LeRobotDatasetMetadata,
    target_features: dict[str, Any],
    merge_fps: int,
    robot_type: str,
) -> bool:
    """True if *src_meta* already matches the merge target (no align pass needed).

    ``aggregate_datasets()`` requires identical ``features`` dicts across shards;
    this compares the on-disk metadata to *target_features* from
    ``_build_target_features`` (including canonical video ``info`` when merging).
    """
    root = Path(src_meta.root)
    tasks_path = root / Path(DEFAULT_TASKS_PATH)
    if not tasks_path.is_file():
        return False
    if int(src_meta.fps) != int(merge_fps):
        return False
    if (src_meta.robot_type or "") != (robot_type or ""):
        return False
    return src_meta.features == target_features


# ---------------------------------------------------------------------------
# Feature schema builders
# ---------------------------------------------------------------------------

def _build_target_features(
    src_meta: LeRobotDatasetMetadata,
    target_state_dim: int,
    target_action_dim: int,
    camera_remap: dict[str, str],
    target_image_size: tuple[int, int] | None = None,
    target_fps: int | None = None,
    state_names: list[str] | None = None,
    action_names: list[str] | None = None,
    canonical_visual: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return (target_features_dict, list_of_target_camera_keys).

    Camera features are derived from the source after applying *camera_remap*.
    If *target_image_size* is given as ``(H, W)`` the video feature shapes and
    ``video.height``/``video.width`` info fields are rewritten to match.
    If *target_fps* is given the ``video.fps`` info field is updated accordingly.

    If *canonical_visual* is set (multi-source merge), copy those entries so every
    aligned shard shares identical visual metadata (keys, shapes, codec info).
    """
    target_features: dict[str, Any] = {}

    if state_names is None:
        state_names = [f"joint_{i}" for i in range(target_state_dim)]
    elif len(state_names) != target_state_dim:
        raise ValueError(
            f"state_names length {len(state_names)} != target_state_dim {target_state_dim}"
        )
    if action_names is None:
        action_names = [f"joint_{i}" for i in range(target_action_dim)]
    elif len(action_names) != target_action_dim:
        raise ValueError(
            f"action_names length {len(action_names)} != target_action_dim {target_action_dim}"
        )

    target_features["observation.state"] = {
        "dtype": "float32",
        "shape": (target_state_dim,),
        "names": state_names,
    }
    target_features["action"] = {
        "dtype": "float32",
        "shape": (target_action_dim,),
        "names": action_names,
    }

    if canonical_visual is not None:
        target_cam_keys = sorted(canonical_visual.keys())
        for dst_key in target_cam_keys:
            target_features[dst_key] = copy.deepcopy(canonical_visual[dst_key])
        return target_features, target_cam_keys

    src_cam_keys = [k for k in src_meta.features if k.startswith("observation.images.")]
    used_dst_cams: set[str] = set()
    target_cam_keys: list[str] = []
    for src_key in src_cam_keys:
        src_cam = src_key.removeprefix("observation.images.")
        dst_cam = camera_remap.get(src_cam, src_cam)
        dst_key = f"observation.images.{dst_cam}"
        if dst_key in used_dst_cams:
            continue
        used_dst_cams.add(dst_key)
        target_cam_keys.append(dst_key)
        feat = copy.deepcopy(src_meta.features[src_key])
        if target_image_size is not None:
            th, tw = target_image_size
            # Update the shape tuple stored in the feature
            feat["shape"] = (th, tw, 3)
            if "info" in feat:
                feat["info"]["video.height"] = th
                feat["info"]["video.width"] = tw
            if "video_info" in feat:
                feat["video_info"]["video.height"] = th
                feat["video_info"]["video.width"] = tw
        if target_fps is not None and "info" in feat:
            feat["info"]["video.fps"] = target_fps
        # Remove the stale video_info block that can carry the original
        # source fps; the canonical fps lives in dataset info.json and
        # feat["info"] above.
        feat.pop("video_info", None)
        target_features[dst_key] = feat

    return target_features, target_cam_keys


def _remapped_dst_keys_for_meta(
    meta: LeRobotDatasetMetadata, camera_remap: dict[str, str]
) -> set[str]:
    """Destination ``observation.images.*`` keys for *meta* after *camera_remap* (first wins)."""
    used_dst: set[str] = set()
    out: set[str] = set()
    for k in meta.features:
        if not k.startswith("observation.images."):
            continue
        src_cam = k.removeprefix("observation.images.")
        dst_key = f"observation.images.{camera_remap.get(src_cam, src_cam)}"
        if dst_key in used_dst:
            continue
        used_dst.add(dst_key)
        out.add(dst_key)
    return out


def _intersection_remapped_camera_keys(
    metas: list[LeRobotDatasetMetadata], camera_remap: dict[str, str]
) -> list[str]:
    """Camera keys present under the same remapped name in every dataset (merge-safe)."""
    if not metas:
        return []
    sets = [_remapped_dst_keys_for_meta(m, camera_remap) for m in metas]
    common = set.intersection(*sets)
    return sorted(common)


def _min_hw_for_merge(
    metas: list[LeRobotDatasetMetadata],
    camera_remap: dict[str, str],
    canonical_keys: set[str],
) -> tuple[int, int]:
    """Smallest H×W among *canonical_keys* across *metas* (after remap)."""
    min_h, min_w = 10**9, 10**9
    found = False
    for meta in metas:
        used_dst: set[str] = set()
        for src_key, feat in meta.features.items():
            if not src_key.startswith("observation.images."):
                continue
            src_cam = src_key.removeprefix("observation.images.")
            dst_key = f"observation.images.{camera_remap.get(src_cam, src_cam)}"
            if dst_key in used_dst:
                continue
            used_dst.add(dst_key)
            if dst_key not in canonical_keys:
                continue
            shp = feat.get("shape")
            if not shp or len(shp) < 2:
                continue
            h, w = int(shp[0]), int(shp[1])
            min_h = min(min_h, h)
            min_w = min(min_w, w)
            found = True
    if not found:
        return 480, 640
    return min_h, min_w


def _merge_effective_fps(
    metas: list[LeRobotDatasetMetadata], target_fps: int
) -> int:
    """Single fps for every aligned shard so aggregate metadata checks pass."""
    return min(min(m.fps, target_fps) for m in metas)


def _find_src_feature_for_dst(
    metas: list[LeRobotDatasetMetadata],
    camera_remap: dict[str, str],
    dst_key: str,
) -> dict[str, Any]:
    for meta in metas:
        used_dst: set[str] = set()
        for src_key, feat in meta.features.items():
            if not src_key.startswith("observation.images."):
                continue
            src_cam = src_key.removeprefix("observation.images.")
            dk = f"observation.images.{camera_remap.get(src_cam, src_cam)}"
            if dk in used_dst:
                continue
            used_dst.add(dk)
            if dk == dst_key:
                return feat
    raise KeyError(dst_key)


def _canonical_visual_feature_dict(
    metas: list[LeRobotDatasetMetadata],
    camera_remap: dict[str, str],
    canonical_keys: list[str],
    th: int,
    tw: int,
    fps: int,
) -> dict[str, Any]:
    """One identical visual feature spec per key for all merged datasets (codec, shape, fps)."""
    out: dict[str, Any] = {}
    for dst_key in canonical_keys:
        feat = copy.deepcopy(_find_src_feature_for_dst(metas, camera_remap, dst_key))
        dt = feat.get("dtype")
        if dt == "video":
            feat["shape"] = (th, tw, 3)
            inf = feat.setdefault("info", {})
            inf["video.height"] = th
            inf["video.width"] = tw
            inf["video.fps"] = fps
            inf["video.codec"] = "av1"
            inf["video.pix_fmt"] = "yuv420p"
            inf["video.is_depth_map"] = False
            inf["video.channels"] = 3
            inf.setdefault("has_audio", False)
        elif dt == "depth":
            feat["shape"] = (th, tw, 1)
        elif dt == "image":
            feat["shape"] = (th, tw, 3)
        feat.pop("video_info", None)
        out[dst_key] = feat
    return out


# ---------------------------------------------------------------------------
# Per-dataset alignment
# ---------------------------------------------------------------------------

def align_single_dataset(
    source_repo: str,
    target_repo_id: str,
    target_fps: int,
    target_state_dim: int,
    target_action_dim: int,
    camera_remap: dict[str, str],
    output_root: Path,
    robot_type: str = "xlerobot",
    target_image_size: tuple[int, int] | None = None,
    match_features_from: str | None = None,
    match_features_root: Path | str | None = None,
    force_rebuild: bool = False,
    canonical_visual: dict[str, Any] | None = None,
    forced_effective_fps: int | None = None,
    pad_fill_mode: str = "zero",
) -> Path:
    """Download *source_repo* and write a feature-aligned copy to *output_root*.

    *target_image_size* is an optional ``(height, width)`` tuple.  When given,
    every video frame is resized to that resolution during the copy.  Use this
    to match the resolution of your primary training datasets so that
    ``aggregate_datasets()`` and JSON-list co-training both work without shape
    errors.

    *match_features_from* is an optional reference repo id whose
    ``observation.state`` / ``action`` ``names`` are copied into the aligned
    metadata (required for ``aggregate_datasets()`` equality checks).

    *force_rebuild* removes *output_root* before writing even if a prior run exists.

    *canonical_visual* / *forced_effective_fps* are used by multi-source merge to
    force one shared schema across shards.

    *pad_fill_mode* controls the values written into state/action dims the
    source cannot provide (see ``PAD_FILL_MODES``):

    - ``"zero"``      — constant 0 (legacy behaviour).
    - ``"ref-mean"``  — the reference dataset's per-dim mean (velocity dims
      stay 0), so padded dims sit at the centre of the real distribution
      instead of dragging normalization stats toward 0.
    - ``"state-copy"``— like ``ref-mean``, but each padded non-velocity
      *action* dim copies the aligned *state* value of the same joint name
      ("hold position" identity action) when that name exists in the state
      schema.

    Non-zero modes require *match_features_from* (the reference supplies the
    joint names and stats).

    Returns the path to the aligned dataset root.
    """
    if pad_fill_mode not in PAD_FILL_MODES:
        raise ValueError(f"pad_fill_mode must be one of {PAD_FILL_MODES}, got {pad_fill_mode!r}")
    if pad_fill_mode != "zero" and match_features_from is None:
        raise ValueError(
            f"pad_fill_mode={pad_fill_mode!r} needs --match-features-from (reference names + stats)."
        )

    logger.info("Loading source dataset: %s", source_repo)
    src_ds = LeRobotDataset(source_repo)
    src_fps = src_ds.meta.fps
    src_meta = src_ds.meta

    state_names: list[str] | None = None
    action_names: list[str] | None = None
    if match_features_from is not None:
        state_names, action_names = _state_action_names_from_reference(
            match_features_from,
            match_features_root,
            target_state_dim,
            target_action_dim,
        )
        logger.info(
            "Copying state/action joint names from reference dataset %s",
            match_features_from,
        )

    # Fill vectors for padded state/action dims (zeros for pad_fill_mode="zero").
    state_fill: np.ndarray | None = None
    action_fill: np.ndarray | None = None
    action_state_copy: list[tuple[int, int]] = []  # (action_dim, state_dim) pairs
    if pad_fill_mode != "zero" and state_names is not None and action_names is not None:
        ref_stats = _load_reference_stats(match_features_from, match_features_root)
        state_fill = _reference_fill_values(
            ref_stats, "observation.state", state_names, pad_fill_mode
        )
        action_fill = _reference_fill_values(ref_stats, "action", action_names, pad_fill_mode)
        if pad_fill_mode == "state-copy":
            src_ac_names = (src_meta.features.get("action") or {}).get("names")
            src_ac_dim = int(src_meta.features["action"]["shape"][0])
            state_pos = {n: i for i, n in enumerate(state_names)}
            for i in _missing_dims(src_ac_names, src_ac_dim, action_names):
                name = action_names[i]
                if not _is_velocity_name(name) and name in state_pos:
                    action_state_copy.append((i, state_pos[name]))
        logger.info(
            "pad-fill-mode=%s: padded dims take reference means (vel dims stay 0)%s.",
            pad_fill_mode,
            f"; {len(action_state_copy)} action dims copy same-name state" if action_state_copy else "",
        )

    if forced_effective_fps is not None:
        effective_fps = forced_effective_fps
    elif src_fps < target_fps:
        logger.warning(
            "Source fps (%d) < target fps (%d). Frames will NOT be "
            "upsampled; the aligned dataset keeps source fps (%d).",
            src_fps, target_fps, src_fps,
        )
        effective_fps = src_fps
    else:
        effective_fps = target_fps

    target_features, target_cam_keys = _build_target_features(
        src_meta,
        target_state_dim,
        target_action_dim,
        camera_remap,
        target_image_size=target_image_size,
        target_fps=effective_fps,
        state_names=state_names,
        action_names=action_names,
        canonical_visual=canonical_visual,
    )

    resize_info = (
        f", images resized to {target_image_size[0]}×{target_image_size[1]}"
        if target_image_size is not None
        else ""
    )
    logger.info(
        "Aligning %s: fps %d→%d (nearest-neighbour resample), "
        "state %d→%d, action %d→%d, cameras %s%s",
        source_repo,
        src_fps, effective_fps,
        src_meta.features["observation.state"]["shape"][0], target_state_dim,
        src_meta.features["action"]["shape"][0], target_action_dim,
        target_cam_keys,
        resize_info,
    )

    # LeRobotDataset.create() uses mkdir(..., exist_ok=False) on the dataset root.
    # A crashed run often leaves a partial directory → FileExistsError on retry.
    if output_root.exists():
        info_json = output_root / "meta" / "info.json"
        if force_rebuild:
            logger.warning("Force rebuild: removing %s", output_root)
            shutil.rmtree(output_root)
        elif info_json.is_file() and _alignment_artifacts_match(
            info_json, target_features, effective_fps, pad_fill_mode
        ):
            logger.info(
                "Reusing completed aligned dataset at %s (metadata matches options).",
                output_root,
            )
            return output_root
        elif info_json.is_file():
            logger.warning(
                "Removing stale aligned dataset (options changed, e.g. "
                "--match-features-from or resolution): %s",
                output_root,
            )
            shutil.rmtree(output_root)
        else:
            logger.warning(
                "Removing incomplete aligned dataset directory from a prior run: %s",
                output_root,
            )
            shutil.rmtree(output_root)

    # Do not mkdir(output_root) here: LeRobotDatasetMetadata.create() requires
    # root to not exist yet (mkdir(..., exist_ok=False)).

    aligned_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=effective_fps,
        robot_type=robot_type,
        features=target_features,
        root=output_root,
        use_videos=len(target_cam_keys) > 0,
        image_writer_threads=4,
    )

    # Keys auto-populated by DatasetWriter; must NOT be in add_frame() dict.
    _auto_keys = {
        "timestamp", "frame_index", "episode_index", "index",
        "task_index", "next.done",
    }
    # The set of feature keys we actually want to write (from target schema).
    target_feature_keys = set(target_features.keys())

    num_episodes = src_meta.total_episodes
    _ds_label = source_repo if len(source_repo) <= 52 else f"…{source_repo[-49:]}"
    ep_pbar = tqdm(
        range(num_episodes),
        desc=f"Align {_ds_label}",
        unit="ep",
        dynamic_ncols=True,
        leave=True,
    )
    for ep_idx in ep_pbar:
        ep_pbar.set_postfix_str(f"episode={ep_idx}/{num_episodes - 1}", refresh=False)
        # Load this episode as a filtered view of the source dataset.
        ep_ds = LeRobotDataset(source_repo, episodes=[ep_idx])
        ep_info = src_meta.episodes[ep_idx]

        # Resolve task description string for this episode.
        task_idx = ep_info.get("task_index", 0)
        if isinstance(task_idx, (list, np.ndarray)):
            task_idx = int(task_idx[0])
        task_description = src_meta.tasks.index[int(task_idx)]
        if not isinstance(task_description, str):
            task_description = str(task_description)

        resample_indices = _compute_resample_indices(src_fps, effective_fps, len(ep_ds))

        frame_count = 0
        skipped_bad_ts = 0
        for frame_pos in tqdm(
            resample_indices,
            desc=f"  frames ep {ep_idx}",
            leave=False,
            unit="fr",
            dynamic_ncols=True,
        ):
            try:
                raw_frame = ep_ds[frame_pos]
            except FrameTimestampError as err:
                # Hub datasets sometimes have one extra parquet row or a last timestamp
                # that rounds past the last decodable MP4 frame (metadata vs file mismatch).
                skipped_bad_ts += 1
                if skipped_bad_ts == 1:
                    logger.warning(
                        "Episode %d: skipping frame(s) with out-of-range video timestamps "
                        "in %s (source metadata vs MP4 length). First index=%d. %s",
                        ep_idx,
                        source_repo,
                        frame_pos,
                        err,
                    )
                continue

            # Remap camera keys first.
            remapped = _remap_camera_keys(raw_frame, camera_remap)

            # State / action: project by joint names when a reference schema is
            # given (drops extras like gantry.vel); else pad or truncate.
            state = remapped.get("observation.state")
            if state is not None:
                arr = state.numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
                src_st = src_meta.features.get("observation.state") or {}
                src_st_names = src_st.get("names")
                if state_names is not None:
                    aligned_st = _remap_vector_by_names(
                        arr, src_st_names, state_names, fill_values=state_fill
                    )
                else:
                    aligned_st = _pad_vector(arr.astype(np.float32), target_state_dim)
                remapped["observation.state"] = torch.from_numpy(aligned_st)

            action = remapped.get("action")
            if action is not None:
                arr = action.numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
                src_ac = src_meta.features.get("action") or {}
                src_ac_names = src_ac.get("names")
                if action_names is not None:
                    aligned_ac = _remap_vector_by_names(
                        arr, src_ac_names, action_names, fill_values=action_fill
                    )
                else:
                    aligned_ac = _pad_vector(arr.astype(np.float32), target_action_dim)
                if action_state_copy:
                    st = remapped.get("observation.state")
                    if st is not None:
                        st_arr = st.numpy() if isinstance(st, torch.Tensor) else np.asarray(st)
                        for i_ac, i_st in action_state_copy:
                            aligned_ac[i_ac] = st_arr[i_st]
                remapped["action"] = torch.from_numpy(aligned_ac)

            # Keep only keys that belong to the target feature schema;
            # strip auto-populated keys and any leftover source-specific keys.
            frame: dict[str, Any] = {
                k: v for k, v in remapped.items()
                if k in target_feature_keys and k not in _auto_keys
            }
            for img_key in list(frame.keys()):
                feat = target_features.get(img_key, {})
                dt = feat.get("dtype")
                if dt in ("image", "video"):
                    frame[img_key] = _ensure_image_hwc_numpy(
                        frame[img_key], feat, target_hw=target_image_size
                    )
                elif dt == "depth":
                    frame[img_key] = _ensure_depth_chw_numpy(
                        frame[img_key], target_hw=target_image_size
                    )
            frame["task"] = task_description

            aligned_ds.add_frame(frame)
            frame_count += 1

        if skipped_bad_ts > 1:
            tqdm.write(
                f"  episode {ep_idx} ({source_repo}): skipped {skipped_bad_ts}/"
                f"{len(resample_indices)} resampled frames (bad timestamps vs MP4)"
            )

        if frame_count > 0:
            aligned_ds.save_episode()
        else:
            aligned_ds.clear_episode_buffer()

        ep_pbar.set_postfix_str(
            f"episode={ep_idx}/{num_episodes - 1} saved_frames={frame_count}",
            refresh=True,
        )

    ep_pbar.close()
    aligned_ds.finalize()
    _write_align_options(output_root, pad_fill_mode)
    logger.info("Aligned dataset written to: %s", output_root)
    return output_root


# ---------------------------------------------------------------------------
# Co-train source manifest + padded-stats override
# ---------------------------------------------------------------------------

COTRAIN_SOURCES_FILENAME = "cotrain_sources.json"


def _episodes_in_root(root: Path) -> int:
    with open(root / "meta" / "info.json", encoding="utf-8") as f:
        return int(json.load(f)["total_episodes"])


def _write_cotrain_sources_manifest(
    merged_root: Path,
    entries: list[dict[str, Any]],
    reference_repo: str | None,
    pad_fill_mode: str,
    state_names: list[str] | None,
    action_names: list[str] | None,
) -> Path:
    """Write ``meta/cotrain_sources.json`` mapping merged episode ranges → sources.

    ``aggregate_datasets()`` concatenates shards in order, so each source owns a
    contiguous ``[episode_start, episode_end)`` range in the merged dataset.
    The manifest records which state/action dims were padded per source; it is
    consumed by the ``source`` sample weighter (down-weight external episodes
    during training) and by :func:`_override_padded_stats`.
    """
    start = 0
    for e in entries:
        e["episode_start"] = start
        e["episode_end"] = start + e.pop("num_episodes")
        start = e["episode_end"]
    manifest = {
        "reference_repo": reference_repo,
        "pad_fill_mode": pad_fill_mode,
        "state_names": state_names,
        "action_names": action_names,
        "total_episodes": start,
        "sources": entries,
    }
    path = merged_root / "meta" / COTRAIN_SOURCES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote co-train source manifest: %s", path)
    return path


def _override_padded_stats(
    merged_root: Path,
    entries: list[dict[str, Any]],
    reference_repo: str,
    reference_root: Path | str | None,
    state_names: list[str],
    action_names: list[str],
) -> None:
    """Overwrite merged stats for padded dims with the reference dataset's stats.

    Padded external data contributes constants to the padded dims, dragging the
    merged mean/std/min/max away from the robot's real motion range — which
    normalizes real motion into outliers at train time. This replaces those
    per-dim entries with stats computed from your own embodiment's data only.
    """
    ref_stats = _load_reference_stats(reference_repo, reference_root)
    if ref_stats is None:
        logger.warning("--override-padded-stats skipped: reference has no stats.")
        return
    merged_stats = load_stats(merged_root)
    if merged_stats is None:
        logger.warning("--override-padded-stats skipped: merged dataset has no meta/stats.json.")
        return

    for feature_key, names, dims_key in (
        ("observation.state", state_names, "padded_state_dims"),
        ("action", action_names, "padded_action_dims"),
    ):
        padded: set[int] = set()
        for e in entries:
            padded.update(e.get(dims_key) or [])
        if not padded:
            continue
        ref_feat = ref_stats.get(feature_key) or {}
        merged_feat = merged_stats.get(feature_key) or {}
        overridden: list[str] = []
        for stat_key in ("mean", "std", "min", "max"):
            ref_arr = ref_feat.get(stat_key)
            dst_arr = merged_feat.get(stat_key)
            if ref_arr is None or dst_arr is None:
                continue
            ref_arr = np.asarray(ref_arr, dtype=np.float64).reshape(-1)
            dst_arr = np.asarray(dst_arr, dtype=np.float64).reshape(-1)
            if len(ref_arr) != len(names) or len(dst_arr) != len(names):
                logger.warning(
                    "--override-padded-stats: %s/%s dim mismatch (ref=%d merged=%d names=%d); skipped.",
                    feature_key, stat_key, len(ref_arr), len(dst_arr), len(names),
                )
                continue
            for i in sorted(padded):
                dst_arr[i] = ref_arr[i]
            merged_feat[stat_key] = dst_arr
            overridden.append(stat_key)
        if overridden:
            logger.info(
                "Overrode merged %s stats (%s) for padded dims %s with reference %s.",
                feature_key,
                "/".join(overridden),
                [names[i] for i in sorted(padded)],
                reference_repo,
            )
    write_stats(merged_stats, merged_root)


# ---------------------------------------------------------------------------
# Multi-source entry point
# ---------------------------------------------------------------------------

# Default camera remapping covers ALOHA and Open-X naming conventions.
_DEFAULT_CAMERA_REMAP: dict[str, str] = {
    # ALOHA
    "top": "head",
    "cam_high": "head",
    "cam_low": "head",
    "image": "head",
    "obs_image": "head",
    "rgb_image": "head",
    "cam_left_wrist": "left_wrist",
    "left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
    "right_wrist": "right_wrist",
    "wrist_image": "left_wrist",
    # Open-X
    "rgb_images.front": "head",
    "rgb_images.left": "left_wrist",
    "rgb_images.right": "right_wrist",
}


def align_datasets_for_cotraining(
    source_repos: list[str],
    target_repo_id: str,
    target_fps: int = 30,
    target_state_dim: int = 18,
    target_action_dim: int = 18,
    camera_remap: dict[str, str] | None = None,
    output_root: Path | None = None,
    push_to_hub: bool = False,
    robot_type: str = "xlerobot",
    target_image_size: tuple[int, int] | None = None,
    match_features_from: str | None = None,
    match_features_root: Path | str | None = None,
    force_rebuild: bool = False,
    skip_missing_sources: bool = False,
    realign_all_sources: bool = False,
    pad_fill_mode: str = "zero",
    override_padded_stats: bool = False,
) -> Path:
    """Align all *source_repos* and merge into one local dataset at *output_root*.

    Each source is either **passed through** (if its on-disk ``meta`` already
    equals the merge schema: fps, ``robot_type``, full ``features`` dict) or
    converted under ``_align_tmp_*``, then all shards are merged via
    ``aggregate_datasets()``.  Pass *realign_all_sources=True* to disable
    pass-through and always re-encode into ``_align_tmp_*``.

    If *push_to_hub* is True the merged dataset is pushed to *target_repo_id*.

    Pass *match_features_from* (e.g. ``Odog16/tool_pickup``) so state/action
    ``names`` match your xlerobot datasets; otherwise only shapes match and
    ``aggregate_datasets`` can still fail on metadata inequality.

    Cached ``_align_tmp_*`` directories are reused only when ``meta/info.json``
    matches the current options (fps, joint names, image shapes).  Use
    *force_rebuild* to delete caches and rebuild from scratch.

    If *skip_missing_sources* is True, repos that are missing locally and return
    Hub 404 are logged and skipped; otherwise loading metadata raises immediately.

    When several sources are merged, camera keys are the **intersection** of
    remapped ``observation.images.*`` keys (so e.g. depth-only streams are dropped
    unless every dataset has them). Resolution is the minimum H×W across sources
    unless *target_image_size* is set. All shards share one fps and identical
    video ``info`` (including codec) so ``aggregate_datasets()`` metadata checks pass.

    *pad_fill_mode* selects the constant written into state/action dims a
    source cannot provide (``zero`` / ``ref-mean`` / ``state-copy``, see
    :func:`align_single_dataset`). *override_padded_stats* rewrites the merged
    ``meta/stats.json`` so padded dims keep the **reference repo's** stats
    (your embodiment's real motion range) instead of stats polluted by
    constant fills. A ``meta/cotrain_sources.json`` manifest (episode ranges +
    padded dims per source) is always written to the merged dataset; the
    ``source`` sample weighter consumes it at train time.

    Returns the path to the merged dataset root.
    """
    if pad_fill_mode not in PAD_FILL_MODES:
        raise ValueError(f"pad_fill_mode must be one of {PAD_FILL_MODES}, got {pad_fill_mode!r}")
    if (pad_fill_mode != "zero" or override_padded_stats) and match_features_from is None:
        raise ValueError(
            "--pad-fill-mode ref-mean/state-copy and --override-padded-stats require "
            "--match-features-from (the reference supplies joint names and stats)."
        )

    if camera_remap is None:
        camera_remap = _DEFAULT_CAMERA_REMAP

    if output_root is None:
        output_root = Path(
            f"~/.cache/huggingface/lerobot/{target_repo_id}"
        ).expanduser()

    aligned_roots: list[Path] = []
    aligned_repo_ids: list[str] = []
    manifest_entries: list[dict[str, Any]] = []

    mfeat_root = (
        Path(match_features_root).expanduser() if match_features_root else None
    )

    resolved: list[tuple[str, LeRobotDatasetMetadata]] = []
    for src_repo in source_repos:
        try:
            resolved.append((src_repo, LeRobotDatasetMetadata(src_repo)))
        except (FileNotFoundError, RepositoryNotFoundError) as err:
            if skip_missing_sources:
                logger.warning("Skipping unavailable source %s: %s", src_repo, err)
                continue
            raise RuntimeError(
                f"Cannot load source dataset {src_repo!r}. "
                f"Fix the repo id, download it locally, log in for private data, or pass "
                f"--skip-missing-sources to continue without it."
            ) from err

    if not resolved:
        raise ValueError(
            "No source datasets were loaded (all missing from Hub/cache or skipped). "
            "Check --source-repos."
        )

    all_metas = [m for _, m in resolved]
    state_names: list[str] | None = None
    action_names: list[str] | None = None
    if match_features_from is not None:
        state_names, action_names = _state_action_names_from_reference(
            match_features_from,
            mfeat_root,
            target_state_dim,
            target_action_dim,
        )

    merge_fps = _merge_effective_fps(all_metas, target_fps)
    canonical_cam_keys = _intersection_remapped_camera_keys(all_metas, camera_remap)
    if not canonical_cam_keys:
        raise ValueError(
            "No observation.images.* keys are shared by all sources after --camera-remap; "
            "cannot merge. Adjust --camera-remap or --source-repos."
        )

    auto_hw = _min_hw_for_merge(all_metas, camera_remap, set(canonical_cam_keys))
    unified_hw = target_image_size if target_image_size is not None else auto_hw

    canonical_visual: dict[str, Any] | None = None
    if len(resolved) > 1:
        canonical_visual = _canonical_visual_feature_dict(
            all_metas,
            camera_remap,
            canonical_cam_keys,
            unified_hw[0],
            unified_hw[1],
            merge_fps,
        )
        logger.info(
            "Merge-safe visual schema: keys=%s, size=%s, fps=%d (intersection across sources).",
            canonical_cam_keys,
            unified_hw,
            merge_fps,
        )

    _src_pbar = tqdm(
        enumerate(resolved),
        total=len(resolved),
        desc="Co-train align sources",
        unit="src",
        dynamic_ncols=True,
        leave=True,
    )
    for i, (src_repo, src_meta) in _src_pbar:
        _src_pbar.set_postfix_str(
            f"{i + 1}/{len(resolved)} {src_repo}",
            refresh=False,
        )
        padded_state_dims: list[int] = []
        padded_action_dims: list[int] = []
        if state_names is not None and action_names is not None:
            src_st = src_meta.features.get("observation.state") or {}
            src_ac = src_meta.features.get("action") or {}
            padded_state_dims = _missing_dims(
                src_st.get("names"), int((src_st.get("shape") or (0,))[0]), state_names
            )
            padded_action_dims = _missing_dims(
                src_ac.get("names"), int((src_ac.get("shape") or (0,))[0]), action_names
            )
        safe_name = src_repo.replace("/", "__")
        aligned_repo_id = f"{target_repo_id}_src{i}_{safe_name}"
        aligned_root = output_root.parent / f"_align_tmp_{safe_name}"
        info_json = aligned_root / "meta" / "info.json"

        target_features, _ = _build_target_features(
            src_meta,
            target_state_dim,
            target_action_dim,
            camera_remap,
            target_image_size=unified_hw,
            target_fps=merge_fps,
            state_names=state_names,
            action_names=action_names,
            canonical_visual=canonical_visual,
        )

        if (
            not realign_all_sources
            and _source_matches_merge_schema(
                src_meta, target_features, merge_fps, robot_type
            )
        ):
            logger.info(
                "Skipping alignment (already merge-ready): %s → %s",
                src_repo,
                src_meta.root,
            )
            aligned_roots.append(Path(src_meta.root))
            aligned_repo_ids.append(src_repo)
            manifest_entries.append({
                "repo_id": src_repo,
                "native": not (padded_state_dims or padded_action_dims),
                "num_episodes": int(src_meta.total_episodes),
                "padded_state_dims": padded_state_dims,
                "padded_action_dims": padded_action_dims,
            })
            continue

        if force_rebuild and aligned_root.exists():
            logger.warning("Force rebuild: removing align cache %s", aligned_root)
            shutil.rmtree(aligned_root)

        cache_matches = info_json.is_file() and _alignment_artifacts_match(
            info_json, target_features, merge_fps, pad_fill_mode
        )
        if not force_rebuild and cache_matches:
            logger.info("Reusing already-aligned cache: %s", aligned_root)
        else:
            if aligned_root.exists():
                logger.warning(
                    "Rebuilding align cache %s (stale vs current options).",
                    aligned_root,
                )
                shutil.rmtree(aligned_root)
            align_single_dataset(
                source_repo=src_repo,
                target_repo_id=aligned_repo_id,
                target_fps=target_fps,
                target_state_dim=target_state_dim,
                target_action_dim=target_action_dim,
                camera_remap=camera_remap,
                output_root=aligned_root,
                robot_type=robot_type,
                target_image_size=unified_hw,
                match_features_from=match_features_from,
                match_features_root=mfeat_root,
                force_rebuild=False,
                canonical_visual=canonical_visual,
                forced_effective_fps=merge_fps if canonical_visual is not None else None,
                pad_fill_mode=pad_fill_mode,
            )

        aligned_roots.append(aligned_root)
        aligned_repo_ids.append(aligned_repo_id)
        manifest_entries.append({
            "repo_id": src_repo,
            "native": not (padded_state_dims or padded_action_dims),
            "num_episodes": _episodes_in_root(aligned_root),
            "padded_state_dims": padded_state_dims,
            "padded_action_dims": padded_action_dims,
        })

    if not aligned_repo_ids:
        raise ValueError(
            "No source datasets were loaded (all missing from Hub/cache or skipped). "
            "Check --source-repos."
        )

    if len(aligned_repo_ids) == 1:
        # Single source: just rename the tmp directory to the final output.
        if aligned_roots[0] != output_root:
            if output_root.exists():
                shutil.rmtree(output_root)
            shutil.copytree(aligned_roots[0], output_root)
        merged_root = output_root
    else:
        _merged_info = output_root / "meta" / "info.json"
        if output_root.exists() and _merged_info.is_file() and not force_rebuild:
            logger.info(
                "Output %s already exists — skipping aggregation (use --force-rebuild to regenerate).",
                output_root,
            )
        else:
            if output_root.exists():
                shutil.rmtree(output_root)
            logger.info(
                "Merging %d aligned datasets with aggregate_datasets() …",
                len(aligned_repo_ids),
            )
            aggregate_datasets(
                repo_ids=aligned_repo_ids,
                aggr_repo_id=target_repo_id,
                roots=aligned_roots,
                aggr_root=output_root,
            )
        merged_root = output_root

    _write_cotrain_sources_manifest(
        merged_root,
        manifest_entries,
        reference_repo=match_features_from,
        pad_fill_mode=pad_fill_mode,
        state_names=state_names,
        action_names=action_names,
    )

    if override_padded_stats and state_names is not None and action_names is not None:
        _override_padded_stats(
            merged_root,
            manifest_entries,
            reference_repo=match_features_from,
            reference_root=mfeat_root,
            state_names=state_names,
            action_names=action_names,
        )

    if push_to_hub:
        tasks_path = merged_root / DEFAULT_TASKS_PATH
        if not tasks_path.is_file():
            raise FileNotFoundError(
                f"Merged dataset is missing {tasks_path} (incomplete alignment or a stale "
                f"_align_tmp_* cache). Re-run with --force-rebuild to regenerate, then push."
            )
        logger.info("Pushing merged dataset to Hub: %s", target_repo_id)
        merged_ds = LeRobotDataset(target_repo_id, root=merged_root)
        merged_ds.push_to_hub()

    logger.info("Done.  Merged dataset at: %s", merged_root)
    return merged_root


# ---------------------------------------------------------------------------
# Compatibility inspection helper
# ---------------------------------------------------------------------------

def inspect_feature_compatibility(repo_ids: list[str]) -> None:
    """Print a feature comparison table for *repo_ids*.

    Run this before attempting a JSON-list co-training run (Strategy A) or
    before deciding whether alignment (Strategy B) is needed.
    """
    print(f"\n{'REPO':<55} {'ACTION':>10} {'STATE':>10} {'FPS':>5}  CAMERAS")
    print("-" * 120)
    for repo in repo_ids:
        try:
            meta = LeRobotDatasetMetadata(repo)
        except Exception as exc:
            print(f"{repo:<55}  ERROR: {exc}")
            continue
        action_dim = meta.features.get("action", {}).get("shape", ("?",))[0]
        state_dim = meta.features.get(
            "observation.state", {}
        ).get("shape", ("?",))[0]
        cams = [
            k.removeprefix("observation.images.")
            for k in meta.features
            if k.startswith("observation.images.")
        ]
        print(
            f"{repo:<55} {str(action_dim):>10} {str(state_dim):>10}"
            f" {meta.fps:>5}  {', '.join(cams)}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_camera_remap(s: str) -> dict[str, str]:
    """Parse 'src1:dst1,src2:dst2' into a dict."""
    result: dict[str, str] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"Invalid camera-remap pair '{pair}'. Expected 'src:dst'."
            )
        src, dst = pair.split(":", 1)
        result[src.strip()] = dst.strip()
    return result


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Align external datasets (ALOHA / Open-X) for co-training "
            "with xlerobot tasks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-repos", nargs="+", required=True,
        help="HF Hub repo IDs to align.",
    )
    parser.add_argument(
        "--target-repo-id", required=True,
        help="Output repo ID for the aligned/merged dataset.",
    )
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--target-state-dim", type=int, default=18)
    parser.add_argument("--target-action-dim", type=int, default=18)
    parser.add_argument(
        "--camera-remap", type=str,
        default=(
            "top:head,cam_high:head,cam_low:head,image:head,obs_image:head,"
            "cam_left_wrist:left_wrist,left_wrist:left_wrist,"
            "wrist_image:left_wrist,"
            "cam_right_wrist:right_wrist,right_wrist:right_wrist,"
            "rgb_images.front:head,rgb_images.left:left_wrist,"
            "rgb_images.right:right_wrist"
        ),
        help="Comma-separated 'src_cam:dst_cam' mappings.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help=(
            "Local directory for the aligned dataset. "
            "Defaults to the HF cache path."
        ),
    )
    parser.add_argument(
        "--robot-type", default="xlerobot",
        help="robot_type string written into the dataset metadata.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Push merged dataset to HF Hub after alignment.",
    )
    parser.add_argument(
        "--target-image-size",
        type=str,
        default=None,
        help=(
            "Resize all video frames to HxW before writing, e.g. '360x640'. "
            "Use this to match the resolution of your primary xlerobot datasets "
            "so the aligned dataset can be physically merged or used in "
            "JSON-list co-training without shape errors."
        ),
    )
    parser.add_argument(
        "--match-features-from",
        type=str,
        default=None,
        help=(
            "Reference dataset repo id (e.g. Odog16/tool_pickup).  Copies "
            "observation.state and action 'names' from its meta so the aligned "
            "dataset matches xlerobot metadata; required for aggregate_datasets() "
            "which compares full feature dicts (not only shapes)."
        ),
    )
    parser.add_argument(
        "--match-features-root",
        type=Path,
        default=None,
        help=(
            "Optional local root for --match-features-from (same as LeRobotDataset root)."
        ),
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "Delete _align_tmp_* caches and the output copy even if they exist. "
            "Normally stale caches are detected automatically when options change."
        ),
    )
    parser.add_argument(
        "--inspect-only", action="store_true",
        help=(
            "Print feature compatibility table for --source-repos and exit."
        ),
    )
    parser.add_argument(
        "--skip-missing-sources",
        "--skip-missing-source",
        action="store_true",
        help=(
            "If a --source-repos id is not on the Hub and not in the local cache, "
            "log a warning and continue with the remaining sources."
        ),
    )
    parser.add_argument(
        "--realign-all-sources",
        action="store_true",
        help=(
            "Always write each source via _align_tmp_* (re-encode / resample). "
            "Default: sources whose meta already matches the merge schema are "
            "used in-place and only copied during aggregate_datasets()."
        ),
    )
    parser.add_argument(
        "--pad-fill-mode",
        choices=PAD_FILL_MODES,
        default="zero",
        help=(
            "Value written into state/action dims a source cannot provide. "
            "'zero': constant 0 (legacy). 'ref-mean': reference dataset's per-dim "
            "mean (velocity dims stay 0) so padded dims sit at the centre of your "
            "real joint distribution. 'state-copy': like ref-mean, plus padded "
            "non-velocity action dims copy the same-name state value (identity "
            "'hold position' action). Non-zero modes require --match-features-from."
        ),
    )
    parser.add_argument(
        "--override-padded-stats",
        action="store_true",
        help=(
            "After merging, rewrite meta/stats.json so dims padded in any source "
            "keep the --match-features-from repo's mean/std/min/max (your "
            "embodiment's real motion range) instead of stats diluted by the "
            "constant fills. Recommended whenever external sources pad dims."
        ),
    )

    args = parser.parse_args()

    if args.inspect_only:
        inspect_feature_compatibility(args.source_repos)
        return

    target_image_size: tuple[int, int] | None = None
    if args.target_image_size:
        parts = args.target_image_size.lower().split("x")
        if len(parts) != 2:
            raise ValueError(
                f"--target-image-size must be 'HxW', got: {args.target_image_size!r}"
            )
        target_image_size = (int(parts[0]), int(parts[1]))

    camera_remap = _parse_camera_remap(args.camera_remap)
    output_root = args.output_root
    if output_root is not None:
        output_root = output_root.expanduser()

    mroot = args.match_features_root
    if mroot is not None:
        mroot = mroot.expanduser()

    align_datasets_for_cotraining(
        source_repos=args.source_repos,
        target_repo_id=args.target_repo_id,
        target_fps=args.target_fps,
        target_state_dim=args.target_state_dim,
        target_action_dim=args.target_action_dim,
        camera_remap=camera_remap,
        output_root=output_root,
        push_to_hub=args.push_to_hub,
        robot_type=args.robot_type,
        target_image_size=target_image_size,
        match_features_from=args.match_features_from,
        match_features_root=mroot,
        force_rebuild=args.force_rebuild,
        skip_missing_sources=args.skip_missing_sources,
        realign_all_sources=args.realign_all_sources,
        pad_fill_mode=args.pad_fill_mode,
        override_padded_stats=args.override_padded_stats,
    )


if __name__ == "__main__":
    main()
