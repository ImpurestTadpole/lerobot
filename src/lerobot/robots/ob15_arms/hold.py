"""Head/lift hold pose for ``ob15_arms``: resolve the targets, describe their source, check them."""

from __future__ import annotations

import logging
from typing import Any

from .arms_only import HEAD_TOL_DEG, LIFT_TOL_MM, hold_keys, lift_key
from .hold_file import read_hold_file
from .stations import load_station

logger = logging.getLogger(__name__)


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
