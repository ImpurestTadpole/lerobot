#!/usr/bin/env python3
"""Convert community-contributed xLeRobot (SO-101-arm-based, dual-arm mobile
base) datasets on the Hub to this project's episode-start-relative EE-space
convention. Reuses the *same*, already-verified ``XLerobotArmKinematics`` /
``xlerobot_episode_to_ee`` FK (2-link planar solver via ``SO101Kinematics``,
identical arm to this project's own ``own_tasks`` xlerobot data) from
``convert_joint_to_ee.py`` rather than re-deriving anything — this is the
same physical arm, not a new embodiment.

Two things genuinely differ across these community repos and are handled,
not guessed:

1. **Joint-name style.** Two conventions observed on the Hub:
   ``left_arm_shoulder_pan.pos`` (zonglin11, Keith-Luo, Grigorij) vs
   ``left_shoulder_pan.pos`` (siyulw2025, yihao-brain-bot) — auto-detected
   per source by checking which prefix is actually present, not assumed.

2. **Gripper calibration.** Verified against real data before writing this:
   raw ``gripper.pos`` ranges differ *by contributor*, not just by a
   documented constant — e.g. zonglin11/Grigorij's grippers sit in a small
   ~1-8 range (their own recording pipeline apparently doesn't rescale to
   the LeRobot-standard ~0-100 calibration ``convert_joint_to_ee.py``'s
   default ``(2.0, 98.0)`` assumes), while siyulw2025's sit around 50-95
   (much closer to that default, but not identical). Silently reusing one
   hardcoded range would have driven at least the first group's gripper
   channel to ~0 (always "closed") regardless of its real state — the same
   class of silent-corruption risk this project's own docs flag for a wrong
   FK. Instead of guessing per-contributor constants, gripper range is
   **auto-calibrated per source repo**: a full pass over that repo's own
   raw gripper values finds its true observed min/max, which becomes the
   ``gripper_range`` fed to the existing FK's ``(g - g_min) / (g_max -
   g_min)`` normalization. Works for either convention without per-source
   special-casing.

Single-arm sources (e.g. Grigorij/XLeRobot_arms, right arm only) are
detected automatically and written as this project's native 7-dim
single-arm EE_NAMES schema — the missing arm is left for the merge step's
own ``--pad-fill-mode ref-mean`` padding, matching the DROID/LIBERO
precedent exactly, not invented here.

Loads with ``revision="main"`` unconditionally: several of these repos lack
a Hub version tag matching their own ``info.json`` codebase_version (a
publishing gap on the contributor's side, not this project's), which makes
the default tag-based revision resolution fail outright; a couple of others
claim v3.0 in info.json but are actually still v2.1 on the Hub. Datasets
that fail to load even with this workaround (checked, not assumed) are
skipped and reported, not silently dropped.

Usage:
    python scripts/convert_xlerobot_community.py \\
        --source-repo siyulw2025/kitchen_fridge_orange_xlerobot \\
        --target-repo-id Odog16/xlc_kitchen_fridge_orange_ee
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from convert_joint_to_ee import (  # noqa: E402
    EE_NAMES,
    XLE_ARM_JOINT_SUFFIXES,
    XLerobotArmKinematics,
    xlerobot_episode_to_ee,
)

DUAL_EE_NAMES = [f"left_{n}" for n in EE_NAMES] + [f"right_{n}" for n in EE_NAMES]


def _detect_prefix(state_names: list[str], side: str) -> str | None:
    """Return the column-name prefix ("{side}_arm_" or "{side}_") actually
    present in *state_names* for this source, or None if that side's arm
    isn't present at all (single-arm source)."""
    for prefix in (f"{side}_arm_", f"{side}_"):
        if all(f"{prefix}{suf}" in state_names for suf in XLE_ARM_JOINT_SUFFIXES):
            return prefix
    return None


def _gripper_range(src: LeRobotDataset, col_idx: int, n_episodes: int) -> tuple[float, float]:
    """Empirical (min, max) of raw gripper.pos over the whole dataset (or a
    representative episode subsample for large ones) — see module docstring
    for why this is computed, not assumed.

    ``src[i]`` decodes that row's video features too (as part of building
    the full frame dict), even though only ``observation.state`` is read
    here — so this is exposed to the same real video-decode/frame-boundary
    bugs as the main conversion loop (verified: hit the exact
    ``FrameTimestampError``/``IndexError`` class already documented for
    DROID's video-split bug, on ``yihao-brain-bot/xlerobot-get``). Per-row,
    not just per-episode, error isolation here — a bad row can't take down
    calibration for an otherwise-fine episode."""
    g_min, g_max = float("inf"), float("-inf")
    sample_eps = range(n_episodes) if n_episodes <= 30 else range(0, n_episodes, max(1, n_episodes // 30))
    n_ok = 0
    n_bad = 0
    for ep_idx in sample_eps:
        ep_meta = src.meta.episodes[ep_idx]
        lo, hi = ep_meta["dataset_from_index"], ep_meta["dataset_to_index"]
        for i in range(lo, hi):
            try:
                v = src[i]["observation.state"][col_idx].item()
            except Exception as e:  # noqa: BLE001 - isolated bad row (video decode) shouldn't kill calibration
                n_bad += 1
                continue
            g_min = min(g_min, v)
            g_max = max(g_max, v)
            n_ok += 1
    if n_bad:
        print(f"  ({n_bad} of {n_ok + n_bad} sampled rows failed to decode, skipped for calibration)")
    if n_ok == 0:
        raise ValueError("Every sampled row failed to decode — cannot calibrate gripper range for this source.")
    return g_min, g_max


def convert(
    source_repo: str,
    target_repo_id: str,
    output_root: Path | None,
    max_episodes: int | None,
    tolerance_s: float,
):
    src = LeRobotDataset(source_repo, revision="main", tolerance_s=tolerance_s)
    state_names_raw = src.meta.features["observation.state"]["names"]
    state_names = state_names_raw["motors"] if isinstance(state_names_raw, dict) else state_names_raw

    left_prefix = _detect_prefix(state_names, "left")
    right_prefix = _detect_prefix(state_names, "right")
    if left_prefix is None and right_prefix is None:
        raise ValueError(f"{source_repo}: no left_* or right_* arm joint columns found in {state_names}")
    dual_arm = left_prefix is not None and right_prefix is not None
    print(f"{source_repo}: dual_arm={dual_arm} left_prefix={left_prefix!r} right_prefix={right_prefix!r}")

    n_eps_total = src.meta.total_episodes
    n_eps = n_eps_total if max_episodes is None else min(max_episodes, n_eps_total)

    kins = {}
    for side, prefix in (("left", left_prefix), ("right", right_prefix)):
        if prefix is None:
            continue
        g_idx = state_names.index(f"{prefix}gripper.pos")
        g_min, g_max = _gripper_range(src, g_idx, n_eps)
        if g_max - g_min < 1e-6:
            print(f"  WARNING: {side} gripper never moves in sampled episodes ({g_min:.3f}) — "
                  f"gripper channel will be constant 0 for this source.")
            g_max = g_min + 1.0
        print(f"  {side} gripper auto-calibrated range: [{g_min:.3f}, {g_max:.3f}]")
        kins[side] = (XLerobotArmKinematics((g_min, g_max)), {
            suf: state_names.index(f"{prefix}{suf}") for suf in XLE_ARM_JOINT_SUFFIXES
        })

    names = DUAL_EE_NAMES if dual_arm else EE_NAMES
    state_dim = 14 if dual_arm else 7
    cam_keys = [k for k in src.meta.features if k.startswith("observation.images.")]
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": names},
        "action": {"dtype": "float32", "shape": (state_dim,), "names": names},
    }
    for k in cam_keys:
        features[k] = dict(src.meta.features[k])

    output_root = output_root or (Path.home() / ".cache/huggingface/lerobot" / target_repo_id)
    if output_root.exists():
        shutil.rmtree(output_root)

    out_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=src.meta.fps,
        robot_type="xlerobot",
        features=features,
        root=output_root,
        use_videos=True,
        image_writer_threads=4,
    )

    n_written = 0
    n_skipped = 0
    for ep_idx in tqdm(range(n_eps), desc=f"Convert {source_repo}", unit="ep"):
        try:
            ep_meta = src.meta.episodes[ep_idx]
            lo, hi = ep_meta["dataset_from_index"], ep_meta["dataset_to_index"]
            raw_frames = [src[i] for i in range(lo, hi)]
            state = np.stack([f["observation.state"].numpy() for f in raw_frames]).astype(np.float32)
            task = raw_frames[0]["task"]

            arm_out = []
            if "left" in kins:
                kin, idx = kins["left"]
                arm_out.append(xlerobot_episode_to_ee(kin, state, idx))
            if "right" in kins:
                kin, idx = kins["right"]
                arm_out.append(xlerobot_episode_to_ee(kin, state, idx))
            vec_state = np.concatenate(arm_out, axis=1)
            vec_action = np.vstack([vec_state[1:], vec_state[-1:]])

            for t, raw in enumerate(raw_frames):
                frame = {"observation.state": vec_state[t], "action": vec_action[t], "task": task}
                for k in cam_keys:
                    chw = raw[k].numpy()
                    frame[k] = np.clip(chw * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                out_ds.add_frame(frame)
            out_ds.save_episode()
            n_written += 1
        except Exception as e:  # noqa: BLE001 - isolated bad episode shouldn't kill the whole run
            print(f"WARNING: skipping episode {ep_idx} of {source_repo}: {type(e).__name__}: {e}")
            out_ds.clear_episode_buffer()
            n_skipped += 1

    out_ds.finalize()
    print(f"Wrote {target_repo_id} -> {output_root} ({n_written} episodes, {n_skipped} skipped)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-repo", required=True)
    p.add_argument("--target-repo-id", required=True)
    p.add_argument("--output-root", default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--tolerance-s", type=float, default=1e-4)
    args = p.parse_args()
    output_root = Path(args.output_root) if args.output_root else None
    convert(args.source_repo, args.target_repo_id, output_root, args.max_episodes, args.tolerance_s)


if __name__ == "__main__":
    main()
