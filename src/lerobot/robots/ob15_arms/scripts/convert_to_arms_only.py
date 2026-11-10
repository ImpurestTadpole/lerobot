#!/usr/bin/env python
"""Convert a full-body OB15 dataset to the arms-only layout, or report head/lift/base motion.

Arms-only layout (see ``ob15_arms``): ``action`` keeps the 12 arm/gripper keys; ``observation.state``
drops the base velocities. Cameras, depth, tasks and episodes are unchanged and the source dataset
is never modified.

    # Which episodes moved the head/lift/base? (also shows where each episode ended vs started)
    python -m lerobot.robots.ob15_arms.scripts.convert_to_arms_only --src-repo-id me/ob15_full --report

    # Convert into a new dataset
    python -m lerobot.robots.ob15_arms.scripts.convert_to_arms_only \
        --src-repo-id me/ob15_full --dst-repo-id me/ob15_arms
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.robots.ob15_arms.arms_only import ARM_KEYS, BASE_KEYS, HEAD_KEYS, HEAD_TOL_DEG, LIFT_TOL_MM

STATE = "observation.state"
ACTION = "action"


def _names(info: dict, feature: str) -> list[str]:
    names = info["features"][feature].get("names")
    if not names:
        raise ValueError(f"Feature '{feature}' has no per-dimension names; cannot select by key.")
    return list(names)


def _data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data files under {root / 'data'}")
    return files


def _stack(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in df[column].to_numpy()])


def report(root: Path, head_tol: float = HEAD_TOL_DEG, lift_tol: float = LIFT_TOL_MM) -> list[dict]:
    """Per-episode motion of head, lift and base, plus start/end values of head and lift."""
    info = json.loads((root / "meta" / "info.json").read_text())
    names = _names(info, STATE)
    lift_name = next((n for n in names if n.endswith(".height_mm")), None)
    track = [k for k in (*HEAD_KEYS, lift_name) if k and k in names]
    base = [k for k in BASE_KEYS if k in names]

    frames = pd.concat(
        [pd.read_parquet(f, columns=[STATE, "episode_index", "frame_index"]) for f in _data_files(root)]
    )
    rows = []
    for ep, df in frames.groupby("episode_index"):
        df = df.sort_values("frame_index")
        state = _stack(df, STATE)
        row: dict = {"episode": int(ep), "frames": len(df), "flags": []}
        for key in track:
            col = state[:, names.index(key)]
            tol = lift_tol if key.endswith(".height_mm") else head_tol
            row[key] = {
                "start": float(col[0]),
                "end": float(col[-1]),
                "drift": float(col[-1] - col[0]),
                "range": float(col.max() - col.min()),
            }
            if row[key]["range"] > tol:
                row["flags"].append(f"{key} moved {row[key]['range']:.1f}")
        if base:
            idx = [names.index(k) for k in base]
            row["base_max_abs_vel"] = float(np.abs(state[:, idx]).max())
            if row["base_max_abs_vel"] > 1e-3:
                row["flags"].append(f"base moved (max |vel| {row['base_max_abs_vel']:.3f})")
        rows.append(row)
    return rows


def print_report(rows: list[dict]) -> None:
    flagged = [r for r in rows if r["flags"]]
    print(f"{len(rows)} episodes, {len(flagged)} with head/lift/base motion")
    for r in rows:
        parts = [f"ep {r['episode']:>4} ({r['frames']:>5} fr)"]
        for key, val in r.items():
            if isinstance(val, dict):
                parts.append(f"{key}: {val['start']:.1f}->{val['end']:.1f} (range {val['range']:.1f})")
        if "base_max_abs_vel" in r:
            parts.append(f"base |v|max {r['base_max_abs_vel']:.3f}")
        print("  " + " | ".join(parts) + ("  <-- " + "; ".join(r["flags"]) if r["flags"] else ""))


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy a dataset; videos are hard-linked (never rewritten) and everything else is copied."""

    def copy(s: str, d: str) -> None:
        if "videos" in Path(s).parts:
            try:
                os.link(s, d)
                return
            except OSError:
                pass
        shutil.copy2(s, d)

    shutil.copytree(src, dst, copy_function=copy)


def convert(src_root: Path, dst_root: Path, dst_repo_id: str) -> Path:
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_tools import _write_parquet, recompute_stats

    if dst_root.exists():
        raise FileExistsError(f"{dst_root} already exists; refusing to overwrite.")
    info = json.loads((src_root / "meta" / "info.json").read_text())
    action_names, state_names = _names(info, ACTION), _names(info, STATE)
    missing = [k for k in ARM_KEYS if k not in action_names]
    if missing:
        raise ValueError(f"Source dataset action is missing arm keys {missing}.")
    a_idx = [action_names.index(k) for k in ARM_KEYS]
    s_keep = [i for i, n in enumerate(state_names) if n not in BASE_KEYS]

    _copy_tree(src_root, dst_root)

    new_info = json.loads((dst_root / "meta" / "info.json").read_text())
    new_info["features"][ACTION]["names"] = [action_names[i] for i in a_idx]
    new_info["features"][ACTION]["shape"] = [len(a_idx)]
    new_info["features"][STATE]["names"] = [state_names[i] for i in s_keep]
    new_info["features"][STATE]["shape"] = [len(s_keep)]
    (dst_root / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

    meta = LeRobotDatasetMetadata(dst_repo_id, root=dst_root)
    for path in _data_files(dst_root):
        df = pd.read_parquet(path)
        df[ACTION] = list(_stack(df, ACTION)[:, a_idx].astype(np.float32))
        df[STATE] = list(_stack(df, STATE)[:, s_keep].astype(np.float32))
        _write_parquet(df, path, meta)

    dataset = LeRobotDataset(dst_repo_id, root=dst_root)
    recompute_stats(dataset)
    return dst_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src-repo-id", help="Source dataset repo id (downloaded if not local).")
    parser.add_argument("--src-root", type=Path, help="Local path of the source dataset.")
    parser.add_argument("--dst-repo-id", help="Repo id of the converted dataset.")
    parser.add_argument(
        "--dst-root", type=Path, help="Output directory (default: $HF_LEROBOT_HOME/<dst-repo-id>)."
    )
    parser.add_argument(
        "--report", action="store_true", help="Only report head/lift/base motion per episode."
    )
    parser.add_argument(
        "--head-tol",
        type=float,
        default=HEAD_TOL_DEG,
        help="Head range (deg) above which an episode is flagged.",
    )
    parser.add_argument(
        "--lift-tol",
        type=float,
        default=LIFT_TOL_MM,
        help="Lift range (mm) above which an episode is flagged.",
    )
    args = parser.parse_args()

    if args.src_root is None and args.src_repo_id is None:
        parser.error("give --src-root or --src-repo-id")
    src_root = args.src_root
    if src_root is None:
        from lerobot.datasets import LeRobotDataset

        src_root = Path(LeRobotDataset(args.src_repo_id).root)

    if args.report:
        print_report(report(src_root, args.head_tol, args.lift_tol))
        return

    if not args.dst_repo_id:
        parser.error("--dst-repo-id is required unless --report is given")
    from lerobot.utils.constants import HF_LEROBOT_HOME

    dst_root = args.dst_root or HF_LEROBOT_HOME / args.dst_repo_id
    out = convert(src_root, dst_root, args.dst_repo_id)
    print(f"Wrote arms-only dataset to {out}")


if __name__ == "__main__":
    main()
