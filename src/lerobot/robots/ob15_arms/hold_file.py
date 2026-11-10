"""The "dataset remembers the hold pose" file for ``ob15_arms``.

Recording writes the resolved head/lift hold pose to ``<dataset_root>/meta/ob15_hold.json``; training
copies it to ``<checkpoint>/pretrained_model/ob15_hold.json``; at inference ``--robot.hold_from=<dataset
or policy>`` moves the head/lift back to that pose. Format::

    {
        "version": 1,
        "robot_type": "ob15_arms",
        "hold": {"gantry.height_mm": 220.0, "head_pan.pos": 0.0, "head_tilt.pos": -35.0},
        "source": {
            "station": null,
            "hold_from": null,
            "overrides": {"head_pan": null, "head_tilt": -35.0, "lift_height_mm": 220.0},
            "hold_on_connect": true,
        },
        "recorded_at": "<ISO UTC>",
    }
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import huggingface_hub

logger = logging.getLogger(__name__)

HOLD_FILENAME = "ob15_hold.json"
DATASET_HOLD_PATH = "meta/ob15_hold.json"
HOLD_FILE_VERSION = 1
_PRETRAINED_HOLD_PATH = f"pretrained_model/{HOLD_FILENAME}"


def build_hold_metadata(robot_type: str, hold: dict[str, float], source: dict[str, Any]) -> dict[str, Any]:
    """The JSON document written next to a dataset."""
    return {
        "version": HOLD_FILE_VERSION,
        "robot_type": robot_type,
        "hold": {k: float(v) for k, v in sorted(hold.items())},
        "source": source,
        "recorded_at": datetime.now(UTC).isoformat(),
    }


def holds_match(a: dict[str, float], b: dict[str, float], head_tol: float, lift_tol: float) -> bool:
    """True if both holds have the same keys and every value agrees within its tolerance."""
    if set(a) != set(b):
        return False
    return all(abs(a[k] - b[k]) <= (lift_tol if k.endswith(".height_mm") else head_tol) for k in a)


def _validate_hold(hold: Any, where: str) -> dict[str, float]:
    if not isinstance(hold, dict) or not hold:
        raise ValueError(f"{where}: 'hold' must be a non-empty object of key -> number.")
    out: dict[str, float] = {}
    for key, value in hold.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: hold value for '{key}' must be a number, got {value!r}.")
        out[key] = float(value)
    return out


def _parse(path: Path) -> dict[str, float]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Hold file {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Hold file {path} must contain a JSON object.")
    return _validate_hold(data.get("hold"), f"Hold file {path}")


def read_hold_file(source: str | os.PathLike) -> dict[str, float]:
    """Load the hold pose from a file, a dataset/checkpoint directory or a Hub repo id.

    Search order: the path itself if it is a file; if it is a directory,
    ``meta/ob15_hold.json``, ``ob15_hold.json`` and ``pretrained_model/ob15_hold.json``; otherwise
    the Hub (dataset repo ``meta/ob15_hold.json``, then model repo ``ob15_hold.json``).
    """
    source = str(source)
    path = Path(source).expanduser()
    tried: list[str] = []

    if path.is_file():
        logger.info(f"Hold pose read from file {path}")
        return _parse(path)
    if path.is_dir():
        for rel in (DATASET_HOLD_PATH, HOLD_FILENAME, _PRETRAINED_HOLD_PATH):
            candidate = path / rel
            tried.append(str(candidate))
            if candidate.is_file():
                logger.info(f"Hold pose read from {candidate}")
                return _parse(candidate)
    else:
        tried.append(f"{path} (local path does not exist)")

    # A non-existent local path falls through to the Hub.
    for filename, repo_type in ((DATASET_HOLD_PATH, "dataset"), (HOLD_FILENAME, "model")):
        label = f"hub:{repo_type}:{source}/{filename}"
        tried.append(label)
        try:
            downloaded = huggingface_hub.hf_hub_download(
                repo_id=source, filename=filename, repo_type=repo_type
            )
        except Exception as e:  # not found, bad repo id, offline, auth...
            logger.debug(f"{label}: {type(e).__name__}: {e}")
            continue
        logger.info(f"Hold pose read from {label}")
        return _parse(Path(downloaded))

    raise FileNotFoundError(
        f"No {HOLD_FILENAME} found for hold_from={source!r}. Tried:\n  " + "\n  ".join(tried)
    )


def _atomic_write(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)


def write_hold_file(
    dataset_root: str | os.PathLike,
    meta: dict[str, Any],
    *,
    overwrite: bool = False,
    head_tol: float,
    lift_tol: float,
) -> Path:
    """Write ``meta/ob15_hold.json`` under ``dataset_root``.

    An existing file is kept unless ``overwrite`` is set; if its pose differs from the new one a
    warning names both poses and the ``--dataset.overwrite_hold=true`` flag.
    """
    path = Path(dataset_root) / DATASET_HOLD_PATH
    if path.exists() and not overwrite:
        try:
            existing = _parse(path)
        except ValueError as e:
            logger.warning(f"Existing hold file is unreadable ({e}); replacing it.")
        else:
            if not holds_match(existing, meta["hold"], head_tol, lift_tol):
                logger.warning(
                    f"Dataset already has a hold pose {existing} but this session holds {meta['hold']}. "
                    f"Keeping the existing file {path}; pass --dataset.overwrite_hold=true to replace it."
                )
            return path
    _atomic_write(path, meta)
    logger.info(f"Wrote hold pose {meta['hold']} to {path}")
    return path


def copy_hold_file_to_checkpoint(dataset_root: str | os.PathLike, pretrained_dir: str | os.PathLike) -> bool:
    """Copy the dataset's hold file into a checkpoint's ``pretrained_model`` dir. False if absent."""
    src = Path(dataset_root) / DATASET_HOLD_PATH
    if not src.is_file():
        return False
    dst = Path(pretrained_dir) / HOLD_FILENAME
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def copy_dataset_hold_to_checkpoint(dataset: Any, pretrained_dir: str | os.PathLike) -> bool:
    """Training hook: copy the training dataset's hold file into ``pretrained_dir``.

    No-op (False) for datasets without a hold file, i.e. everything that is not OB15 arms-only.
    With several candidate datasets, copies only when exactly one has a hold file.
    """
    candidates = list(getattr(dataset, "datasets", None) or [dataset])
    roots: list[Path] = []
    for ds in candidates:
        root = getattr(ds, "root", None) or getattr(getattr(ds, "meta", None), "root", None)
        if root is not None and (Path(root) / DATASET_HOLD_PATH).is_file():
            roots.append(Path(root))
    if not roots:
        return False
    if len(roots) > 1:
        logger.info(f"{len(roots)} datasets have an {HOLD_FILENAME}; not copying any to the checkpoint.")
        return False
    return copy_hold_file_to_checkpoint(roots[0], pretrained_dir)
