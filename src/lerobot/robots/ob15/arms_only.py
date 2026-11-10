"""Shared key helpers for the arms-only OB15 robot types (``ob15_arms`` / ``ob15_arms_client``).

The arms-only robots expose 12 actions (both arms + grippers). Head pan/tilt and the lift are
moved to a hold pose and then held, and the base always gets zero velocity, so those keys are
never part of the action space even though the head/lift stay in the observation.
"""

from __future__ import annotations

import logging
from typing import Any

from .hold_file import read_hold_file
from .stations import load_station

logger = logging.getLogger(__name__)

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


def resolve_hold_targets(config: Any, current_obs: dict[str, Any], lift_name: str | None) -> dict[str, float]:
    """Head/lift targets to hold.

    Precedence: explicit overrides > ``hold_from`` file > named station > current obs.
    """
    keys = hold_keys(lift_name)
    targets: dict[str, float] = {}
    for key in keys:
        if key in current_obs:
            targets[key] = float(current_obs[key])

    station = getattr(config, "station", None)
    if station:
        station_values = load_station(config.stations_file, station)
        for key in keys:
            if key in station_values:
                targets[key] = station_values[key]
        missing = [k for k in keys if k not in station_values]
        if missing:
            # Not fatal: the current position is held for keys the station doesn't define.
            logger.warning(
                f"Station '{station}' has no value for {missing}; holding their current positions."
            )

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
                f"hold_from={hold_from!r} has no value for {missing}; using station/current for them."
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
        "Hold targets %s (sources: flags=%s, hold_from=%s, station=%s)",
        targets,
        sorted(k for k, v in overrides.items() if v is not None and k in keys),
        hold_from,
        station,
    )
    missing = [k for k in keys if k not in targets]
    if missing:
        raise ValueError(
            f"No hold target for {missing}: not in the current observation, a station or an override."
        )
    return targets


def hold_source(config: Any) -> dict[str, Any]:
    """How the hold pose was requested, for the metadata written next to a dataset."""
    return {
        "station": getattr(config, "station", None),
        "hold_from": getattr(config, "hold_from", None),
        "overrides": {
            "head_pan": getattr(config, "head_pan", None),
            "head_tilt": getattr(config, "head_tilt", None),
            "lift_height_mm": getattr(config, "lift_height_mm", None),
        },
        "hold_on_connect": bool(getattr(config, "hold_on_connect", True)),
    }


def rest_pose_command(config: Any) -> dict[str, float] | None:
    """Arm/gripper targets to send before the head/lift move, or None if no ``rest_pose``."""
    rest = getattr(config, "rest_pose", None)
    if not rest:
        return None
    bad = sorted(k for k in rest if k not in ARM_KEYS)
    if bad:
        raise ValueError(f"rest_pose may only contain arm/gripper keys {list(ARM_KEYS)}; got {bad}.")
    return {k: float(v) for k, v in rest.items()}


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


def build_arms_command(action: dict[str, Any], hold: dict[str, float]) -> dict[str, float]:
    """Arm keys from ``action`` + the hold targets + explicit zero base velocity.

    Any base/head/lift keys in ``action`` are ignored.
    """
    command: dict[str, float] = {k: float(action[k]) for k in ARM_KEYS if k in action}
    command.update(hold)
    command.update(dict.fromkeys(BASE_KEYS, 0.0))
    return command


def arms_only_result(sent: dict[str, Any]) -> dict[str, Any]:
    """Keep only the 12 arm keys from a send_action result."""
    return {k: sent[k] for k in ARM_KEYS if k in sent}
