"""Everything the arms-only OB15 robots (``ob15_arms`` / ``ob15_arms_client``) share, in one place.

1. Keys and commands: the 12 arm/gripper actions, the command sent to the robot (arms + hold pose, no
   base command).
2. Hold pose: resolve the head/lift targets (flags > ``hold_from`` > current), check drift.
3. Hold file: ``<dataset>/meta/ob15_hold.json`` written when recording, copied into the checkpoint
   when training, read back by ``--robot.hold_from``. Format::

       {
           "version": 1,
           "robot_type": "ob15_arms",
           "hold": {"gantry.height_mm": 220.0, "head_pan.pos": 0.0, "head_tilt.pos": -35.0},
           "source": {
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

# --- 1. Keys and commands ---------------------------------------------------------------------

ARM_KEYS: tuple[str, ...] = (
    "left_arm_shoulder_pan.pos",
    "left_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos",
    "left_arm_gripper.pos",
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
)
HEAD_KEYS: tuple[str, ...] = ("head_pan.pos", "head_tilt.pos")
BASE_KEYS: tuple[str, ...] = ("x.vel", "y.vel", "theta.vel")

# Defaults for "has the head/lift reached the hold pose".
# Camera angled down at the table; positive tilt = down (verified on the robot). head_tilt.pos is in normalized
# units, -100..+100 across the calibrated tilt range (degrees only when use_degrees=True). On this robot (lift at the
# top) 47 shows a third wall, 60-70 shows the whole table with the grippers, 80+ loses the far edge. 62 = 65 nudged up. The robot keeps
# defaults HEAD_TILT_MARGIN inside the end of the range.
DEFAULT_HEAD_TILT_DEG: float = 62.0
HEAD_TILT_MARGIN_DEG = 2.0  # margin when the tilt is in degrees
HEAD_TILT_MARGIN_UNITS = 5.0  # margin when the tilt is in -100..100 units
HEAD_TOL_DEG = 2.0
LIFT_TOL_MM = 5.0


def lift_key(lift_name: str) -> str:
    return f"{lift_name}.height_mm"


def arms_action_features() -> dict[str, type]:
    return dict.fromkeys(ARM_KEYS, float)


def arms_state_features(full_state_ft: dict[str, Any]) -> dict[str, Any]:
    """Drop the base-velocity keys from a full-body state feature dict, keeping order."""
    return {k: v for k, v in full_state_ft.items() if k not in BASE_KEYS}


def hold_keys(lift_name: str | None) -> tuple[str, ...]:
    return (*HEAD_KEYS, lift_key(lift_name)) if lift_name else HEAD_KEYS


def build_arms_command(action: dict[str, Any], hold: dict[str, float]) -> dict[str, float]:
    """Arm keys from ``action`` + the hold targets. No base command is included.

    Any base/head/lift keys in ``action`` are ignored. ``OB15.send_action`` reads a missing base key as
    0.0 every tick (the smoothed base velocity decays to zero), so leaving the base out keeps the
    wheels at zero velocity without arms-only code driving the base.
    """
    command: dict[str, float] = {k: float(action[k]) for k in ARM_KEYS if k in action}
    command.update(hold)
    return command


def arms_only_result(sent: dict[str, Any]) -> dict[str, Any]:
    """Keep only the 12 arm keys from a send_action result."""
    return {k: sent[k] for k in ARM_KEYS if k in sent}


# --- 2. Hold pose -----------------------------------------------------------------------------


def resolve_hold_targets(
    config: Any,
    current_obs: dict[str, Any],
    lift_name: str | None,
    default_head_tilt: float | None = None,
    default_lift_mm: float | None = None,
) -> dict[str, float]:
    """Head/lift targets to hold.

    Precedence: explicit overrides > ``hold_from`` file > the defaults (``default_head_tilt``, ``default_lift_mm``)
    > current obs.
    """
    keys = hold_keys(lift_name)
    targets: dict[str, float] = {}
    for key in keys:
        if key in current_obs:
            targets[key] = float(current_obs[key])
    if default_head_tilt is not None and "head_tilt.pos" in keys:
        targets["head_tilt.pos"] = float(default_head_tilt)
    if default_lift_mm is not None and lift_name and lift_key(lift_name) in keys:
        targets[lift_key(lift_name)] = float(default_lift_mm)

    hold_from = getattr(config, "hold_from", None)
    if hold_from:
        saved = read_hold_file(hold_from)
        extra = sorted(k for k in saved if k not in keys)
        if extra:
            logger.warning(f"hold_from={hold_from!r} has non-hold keys {extra}; ignoring them.")
        for key in keys:
            if key in saved:
                targets[key] = saved[key]
        missing = [k for k in keys if k not in saved]
        if missing:
            logger.warning(
                f"hold_from={hold_from!r} has no value for {missing}; using the current position for them."
            )

    overrides = {
        "head_pan.pos": getattr(config, "head_pan", None),
        "head_tilt.pos": getattr(config, "head_tilt", None),
    }
    if lift_name:
        overrides[lift_key(lift_name)] = getattr(config, "lift_height_mm", None)
    for key, value in overrides.items():
        if value is not None and key in keys:
            targets[key] = float(value)

    logger.info(
        "Hold targets %s (sources: flags=%s, hold_from=%s)",
        targets,
        sorted(k for k, v in overrides.items() if v is not None and k in keys),
        hold_from,
    )
    missing = [k for k in keys if k not in targets]
    if missing:
        raise ValueError(
            f"No hold target for {missing}: not in the current observation, hold_from or an override."
        )
    return targets


def hold_source(config: Any) -> dict[str, Any]:
    """How the hold pose was requested, for the metadata written next to a dataset."""
    return {
        "hold_from": getattr(config, "hold_from", None),
        "overrides": {
            "head_pan": getattr(config, "head_pan", None),
            "head_tilt": getattr(config, "head_tilt", None),
            "lift_height_mm": getattr(config, "lift_height_mm", None),
        },
        "hold_on_connect": bool(getattr(config, "hold_on_connect", True)),
    }


def hold_status(
    obs: dict[str, Any],
    hold: dict[str, float],
    head_tol: float = HEAD_TOL_DEG,
    lift_tol: float = LIFT_TOL_MM,
) -> dict[str, dict[str, float | bool]]:
    """Compare the observed head/lift against the hold targets.

    Returns ``{key: {"target", "actual", "error", "tol", "ok"}}`` for every hold key present in
    the observation. A key missing from the observation is reported with ``actual=None, ok=False``.
    """
    status: dict[str, dict[str, float | bool]] = {}
    for key, target in hold.items():
        tol = lift_tol if key.endswith(".height_mm") else head_tol
        if key not in obs:
            status[key] = {"target": target, "actual": None, "error": None, "tol": tol, "ok": False}
            continue
        actual = float(obs[key])
        err = actual - target
        status[key] = {"target": target, "actual": actual, "error": err, "tol": tol, "ok": abs(err) <= tol}
    return status


def hold_reached(status: dict[str, dict[str, float | bool]]) -> bool:
    return all(entry["ok"] for entry in status.values())


# --- 3. Hold file -----------------------------------------------------------------------------

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
