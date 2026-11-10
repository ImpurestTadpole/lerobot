"""Named head/lift "station" poses for arms-only OB15 control.

A station is a saved head pan/tilt and lift height, e.g. the pose the robot should hold while
the arms do a task at a kitchen counter. Stored as JSON::

    {"kitchen_counter": {"head_pan.pos": 0.0, "head_tilt.pos": -35.0, "gantry.height_mm": 220}}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_STATIONS_FILE = "~/.ob15/stations.json"


def _resolve(path: str | os.PathLike) -> Path:
    return Path(path).expanduser()


def _read(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Station file not found: {path}. Create it (see lerobot.robots.ob15.stations) or pass "
            "explicit --robot.head_pan/--robot.head_tilt/--robot.lift_height_mm values."
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Station file {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Station file {path} must contain a JSON object of station name -> pose.")
    return data


def load_station(path: str | os.PathLike, name: str) -> dict[str, float]:
    """Return the head/lift values of station ``name`` as ``{key: float}``."""
    path = _resolve(path)
    data = _read(path)
    if name not in data:
        raise KeyError(f"Station '{name}' not found in {path}. Available: {sorted(data) or 'none'}.")
    station = data[name]
    if not isinstance(station, dict) or not station:
        raise ValueError(f"Station '{name}' in {path} must be a non-empty object of key -> number.")
    out: dict[str, float] = {}
    for key, value in station.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Station '{name}' key '{key}' must be a number, got {value!r}.")
        out[key] = float(value)
    return out


def save_station(path: str | os.PathLike, name: str, values: dict[str, float]) -> None:
    """Create or overwrite station ``name``, keeping the other stations in the file."""
    path = _resolve(path)
    try:
        data = _read(path)
    except FileNotFoundError:
        data = {}
    data[name] = {k: float(v) for k, v in values.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
