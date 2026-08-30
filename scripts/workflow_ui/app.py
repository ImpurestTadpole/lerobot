#!/usr/bin/env python3
"""
Co-training workflow UI — lay out a training paradigm as ordered stages
(merge → pretrain → co-train fine-tunes → custom), generate runnable .sh
scripts for each stage, and execute them sequentially with live logs.

Stdlib only (http.server + subprocess + threading); no new dependencies.

Usage:
    uv run python scripts/workflow_ui/app.py [--port 7799]

Then open http://127.0.0.1:7799 .

Layout on disk (repo root):
    workflows/<name>/workflow.json     — the workflow definition (editable in UI)
    workflows/<name>/scripts/NN_*.sh   — generated stage scripts (re-runnable by hand)
    workflows/<name>/logs/NN_*.log     — stage run logs
    workflows/<name>/logs/run_state.json — last run status (shown after page reload)

Stage types:
    merge  — lerobot-cotrain-align (datasets → one merged co-train dataset,
             with pad-fill-mode / override-padded-stats mitigation flags)
    train  — lerobot-train (used for both generalist pretrain and per-task
             fine-tune; supports RA-BC and source-based sample weighting)
    custom — raw bash body (eval, rollout, upload, anything)

Workflow-level variables are substituted into every string field as {{NAME}}
at script-generation time, so one workflow can be re-pointed at new datasets
or step counts without editing each stage.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / "workflows"

# lerobot-* commands run inside this conda env, NOT `uv run` — Guide.sh has
# said "activate conda only, do not activate .venv" from the start of this
# project, and the repo's separate uv-managed .venv can drift onto an
# incompatible Python (e.g. draccus's argparse integration crashes on
# Python 3.14 for any `X | None`-typed config field). Override with
# --conda-env if your env has a different name.
CONDA_ENV_NAME = "lerobot"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_VAR_RE = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

def _subst(value: str, variables: dict[str, str]) -> str:
    """Replace {{VAR}} placeholders; unknown variables raise so typos fail fast."""

    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in variables:
            raise ValueError(f"Unknown workflow variable {{{{{key}}}}} (defined: {sorted(variables)})")
        return str(variables[key])

    return _VAR_RE.sub(repl, value)


_UNQUOTED_RE = re.compile(r"^[A-Za-z0-9_@%+=:,./~^${}-]+$")


def _shq(s: str) -> str:
    """Shell-quote *s* but keep ``$VAR`` expandable (double quotes, not single).

    Generated scripts are meant to be read and re-run by hand, and fields like
    dataset roots legitimately use ``$HOME``.
    """
    if _UNQUOTED_RE.match(s):
        return s
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{escaped}"'


def _flag(args: list[str], name: str, value, quote: bool = True) -> None:
    """Append --name=value when value is a non-empty string / not None."""
    if value is None:
        return
    s = str(value).strip()
    if not s:
        return
    args.append(f"{name}={_shq(s) if quote else s}")


def _bool_str(v) -> str:
    return "true" if v in (True, "true", "True", 1, "1") else "false"


def _merge_stage_command(p: dict) -> str:
    src = [r for r in re.split(r"[\s,]+", p.get("source_repos", "")) if r]
    if not src:
        raise ValueError("merge stage: 'source_repos' is empty")
    if not p.get("target_repo_id"):
        raise ValueError("merge stage: 'target_repo_id' is required")
    args = ["lerobot-cotrain-align"]
    args.append("--source-repos " + " ".join(shlex.quote(r) for r in src))
    _flag(args, "--target-repo-id", p.get("target_repo_id"))
    _flag(args, "--target-fps", p.get("target_fps"))
    _flag(args, "--target-image-size", p.get("target_image_size"))
    _flag(args, "--target-state-dim", p.get("target_state_dim"))
    _flag(args, "--target-action-dim", p.get("target_action_dim"))
    _flag(args, "--match-features-from", p.get("match_features_from"))
    _flag(args, "--pad-fill-mode", p.get("pad_fill_mode"))
    if p.get("override_padded_stats"):
        args.append("--override-padded-stats")
    if p.get("force_rebuild"):
        args.append("--force-rebuild")
    if p.get("skip_missing_sources"):
        args.append("--skip-missing-sources")
    args.append(f"--push-to-hub {_bool_str(p.get('push_to_hub', False))}")
    extra = (p.get("extra_args") or "").strip()
    if extra:
        args.append(extra)
    return " \\\n    ".join(args)


def _train_stage_command(p: dict) -> str:
    if not p.get("dataset_repo_id"):
        raise ValueError("train stage: 'dataset_repo_id' is required")
    if not p.get("output_dir"):
        raise ValueError("train stage: 'output_dir' is required")
    args = ["lerobot-train"]
    _flag(args, "--dataset.repo_id", p.get("dataset_repo_id"))
    _flag(args, "--dataset.root", p.get("dataset_root"))
    _flag(args, "--policy.type", p.get("policy_type"))
    _flag(args, "--policy.pretrained_path", p.get("pretrained_path"))
    # PreTrainedConfig.push_to_hub defaults to True upstream, so cfg.validate()
    # raises "'repo_id' argument missing" unless a repo id is given — must be
    # explicit either way, since omitting the flag leaves that True default in
    # effect (silently, for every intermediate pipeline stage).
    if p.get("push_to_hub"):
        if not p.get("policy_repo_id"):
            raise ValueError(
                "train stage: 'policy_repo_id' is required when 'Push policy to Hub' is on"
            )
        args.append("--policy.push_to_hub=true")
        _flag(args, "--policy.repo_id", p.get("policy_repo_id"))
    else:
        args.append("--policy.push_to_hub=false")
    if p.get("train_state_proj"):
        # SmolVLA-only field (src/lerobot/policies/smolvla/configuration_smolvla.py)
        # — sending it for any other policy type raises an unrecognized-argument
        # error, the same class of bug as the scheduler fields above.
        if (p.get("policy_type") or "").strip() != "smolvla":
            raise ValueError(
                "train stage: 'Train state projection' is SmolVLA-only — uncheck it "
                f"or set Policy type to smolvla (got {p.get('policy_type')!r})"
            )
        args.append("--policy.train_state_proj=true")
    if p.get("use_amp"):
        args.append("--policy.use_amp=true")
    _flag(args, "--batch_size", p.get("batch_size"))
    _flag(args, "--steps", p.get("steps"))
    _flag(args, "--save_freq", p.get("save_freq"))
    _flag(args, "--output_dir", p.get("output_dir"))
    if p.get("scheduler_type"):
        # cfg.validate() unconditionally overwrites self.scheduler (and
        # self.optimizer) from the policy's own get_scheduler_preset() when
        # use_policy_training_preset is true (the default) — silently
        # discarding any --scheduler.* CLI overrides, no error raised. A
        # custom scheduler here only takes effect with the preset disabled.
        args.append("--use_policy_training_preset=false")
        _flag(args, "--scheduler.type", p.get("scheduler_type"))
        _flag(args, "--scheduler.peak_lr", p.get("peak_lr"))
        if p.get("scheduler_type") == "cosine_decay_with_warmup":
            # This scheduler has no field defaults (num_warmup_steps,
            # num_decay_steps, decay_lr are all required) — draccus rejects
            # peak_lr alone with "Missing required field(s)".
            _flag(args, "--scheduler.num_warmup_steps", p.get("scheduler_warmup_steps") or "1000")
            _flag(args, "--scheduler.num_decay_steps",
                  p.get("scheduler_decay_steps") or p.get("steps") or "20000")
            _flag(args, "--scheduler.decay_lr", p.get("scheduler_decay_lr") or "1e-6")
        # use_policy_training_preset also supplies the optimizer; without it
        # draccus requires --optimizer.type explicitly too (validate() raises
        # "Optimizer and Scheduler must be set when the policy presets are
        # not used" otherwise). Default to the adamw preset SmolVLA itself
        # uses so behavior matches except for the scheduler override.
        _flag(args, "--optimizer.type", p.get("optimizer_type") or "adamw")
        _flag(args, "--optimizer.lr", p.get("peak_lr"))
    elif p.get("peak_lr"):
        _flag(args, "--scheduler.peak_lr", p.get("peak_lr"))
    weighting = p.get("weighting") or "none"
    if weighting == "rabc":
        args.append("--use_rabc=true")
        _flag(args, "--rabc_head_mode", p.get("rabc_head_mode") or "sparse")
        _flag(args, "--rabc_kappa", p.get("rabc_kappa") or "0.01")
    elif weighting == "source":
        args.append("--sample_weighting.type=source")
        _flag(args, "--sample_weighting.external_weight", p.get("external_weight") or "0.3")
    if p.get("wandb_enable"):
        args.append("--wandb.enable=true")
        _flag(args, "--wandb.project", p.get("wandb_project") or "lerobot")
    if p.get("resume"):
        args.append("--resume=true")
    extra = (p.get("extra_args") or "").strip()
    if extra:
        args.append(extra)
    return " \\\n    ".join(args)


def _extract_stage_command(p: dict) -> str:
    if not p.get("source_repo") or not p.get("target_repo_id"):
        raise ValueError("extract stage: 'source_repo' and 'target_repo_id' are required")
    args = ["lerobot-extract-subset"]
    _flag(args, "--source-repo", p.get("source_repo"))
    _flag(args, "--target-repo-id", p.get("target_repo_id"))
    if p.get("keep_names"):
        args.append("--keep-names " + " ".join(
            _shq(n) for n in re.split(r"[\s,]+", p["keep_names"]) if n))
    else:
        _flag(args, "--profile", p.get("profile") or "bimanual12")
    if p.get("cameras"):
        args.append("--cameras " + " ".join(
            _shq(c) for c in re.split(r"[\s,]+", p["cameras"]) if c))
    if p.get("keep_depth"):
        args.append("--keep-depth")
    _flag(args, "--target-fps", p.get("target_fps"))
    _flag(args, "--target-image-size", p.get("target_image_size"))
    _flag(args, "--robot-type", p.get("robot_type"))
    if p.get("force_rebuild"):
        args.append("--force-rebuild")
    args.append(f"--push-to-hub {_bool_str(p.get('push_to_hub', False))}")
    extra = (p.get("extra_args") or "").strip()
    if extra:
        args.append(extra)
    return " \\\n    ".join(args)


def _sarm_stage_command(p: dict) -> str:
    """SARM progress annotation → sarm_progress.parquet, consumed by RA-BC
    (train with weighting: rabc). No console entry point exists, so invoke the
    module directly."""
    if not p.get("dataset_repo_id"):
        raise ValueError("sarm stage: 'dataset_repo_id' is required")
    if not p.get("reward_model_path"):
        raise ValueError("sarm stage: 'reward_model_path' is required")
    args = ["python -m lerobot.rewards.sarm.compute_rabc_weights"]
    _flag(args, "--dataset-repo-id", p.get("dataset_repo_id"))
    _flag(args, "--reward-model-path", p.get("reward_model_path"))
    _flag(args, "--output-path", p.get("output_path"))
    _flag(args, "--head-mode", p.get("head_mode") or "sparse")
    _flag(args, "--device", p.get("device") or "cuda")
    _flag(args, "--stride", p.get("stride"))
    _flag(args, "--tolerance-s", p.get("tolerance_s"))
    if p.get("push_to_hub"):
        args.append("--push-to-hub")
    extra = (p.get("extra_args") or "").strip()
    if extra:
        args.append(extra)
    return " \\\n    ".join(args)


def render_stage_script(workflow: dict, index: int) -> str:
    """Render stage *index* of *workflow* into a standalone bash script."""
    stage = workflow["stages"][index]
    variables = {str(k): str(v) for k, v in (workflow.get("variables") or {}).items()}
    params = {
        k: (_subst(v, variables) if isinstance(v, str) else v)
        for k, v in (stage.get("params") or {}).items()
    }
    stype = stage.get("type", "custom")
    if stype == "merge":
        body = _merge_stage_command(params)
    elif stype == "train":
        body = _train_stage_command(params)
    elif stype == "extract":
        body = _extract_stage_command(params)
    elif stype == "sarm":
        body = _sarm_stage_command(params)
    elif stype == "custom":
        body = (params.get("script") or "").strip()
        if not body:
            raise ValueError("custom stage: 'script' is empty")
    else:
        raise ValueError(f"Unknown stage type: {stype!r}")

    name = stage.get("name", f"stage_{index}")
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by scripts/workflow_ui — workflow: {workflow.get('name', '?')}, "
        f"stage {index + 1}: {name}",
        "# Regenerate from the UI after editing the workflow; manual edits are overwritten.",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        'export PYTHONPATH=""  # keep external site-packages (e.g. isaacsim) out of the venv',
        "",
    ]
    if stype == "custom":
        # Raw user-authored bash — left untouched; activate conda yourself if
        # your command needs it (see the commented example in the master preset).
        lines.append(body)
    else:
        lines += [
            "# lerobot-* CLIs need the conda env (GPU torch, matching Python) — NOT",
            '# `uv run`, which resolves to this repo\'s separate uv-managed .venv and',
            "# can silently be on a different, incompatible Python (see GUIDE.md).",
            'CONDA_BASE="$(conda info --base)"',
            'source "$CONDA_BASE/etc/profile.d/conda.sh"',
            f"conda activate {shlex.quote(CONDA_ENV_NAME)}",
            "",
            body,
        ]
    lines.append("")
    return "\n".join(lines)


def _stage_slug(stage: dict, index: int) -> str:
    raw = stage.get("name", "") or stage.get("type", "stage")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower() or "stage"
    return f"{index + 1:02d}_{slug}"


def generate_scripts(workflow: dict) -> list[dict]:
    """Write one .sh per enabled stage; return [{index, name, path, script}]."""
    wf_dir = WORKFLOWS_DIR / workflow["name"]
    scripts_dir = wf_dir / "scripts"
    if scripts_dir.exists():
        shutil.rmtree(scripts_dir)
    scripts_dir.mkdir(parents=True)
    out = []
    for i, stage in enumerate(workflow.get("stages", [])):
        if not stage.get("enabled", True):
            continue
        script = render_stage_script(workflow, i)
        path = scripts_dir / f"{_stage_slug(stage, i)}.sh"
        path.write_text(script)
        path.chmod(0o755)
        out.append({
            "index": i,
            "name": stage.get("name", f"stage {i + 1}"),
            "path": str(path.relative_to(REPO_ROOT)),
            "script": script,
        })
    return out


# ---------------------------------------------------------------------------
# Repo option registries (policy/robot/scheduler type names) for UI dropdowns
# ---------------------------------------------------------------------------

# Matches e.g. @PreTrainedConfig.register_subclass("smolvla") across policies/
# robots/schedulers — every draccus.ChoiceRegistry subclass in this fork uses
# this exact decorator to register its type string.
_REGISTER_SUBCLASS_RE = re.compile(r'register_subclass\(\s*["\']([\w.\-]+)["\']\s*\)')


def _scan_registered_choices(glob_patterns: list[str]) -> list[str]:
    """Type names passed to register_subclass(...) under matching source files.

    A static regex scan, not an import: many policy/robot modules pull in
    torch or optional hardware SDKs at import time (CLAUDE.md: "optional
    dependencies ... must be lazy"), which is unnecessary cost/risk just to
    list their registered names, and the workflow UI server otherwise never
    imports torch/cv2 so it can answer requests instantly.
    """
    names: set[str] = set()
    lerobot_src = REPO_ROOT / "src" / "lerobot"
    for pattern in glob_patterns:
        for path in lerobot_src.glob(pattern):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            names.update(_REGISTER_SUBCLASS_RE.findall(text))
    return sorted(names)


_registry_cache: dict[str, list[str]] | None = None


def get_repo_registry() -> dict[str, list[str]]:
    """Policy/robot/scheduler type names actually registered in this checkout.

    Cached for the life of the server process — the source tree doesn't
    change while it's running; restart the UI after adding a new policy/robot/
    scheduler to pick it up.
    """
    global _registry_cache
    if _registry_cache is None:
        _registry_cache = {
            "policies": _scan_registered_choices(["policies/*/configuration_*.py"]),
            "robots": _scan_registered_choices(["robots/*/config_*.py"]),
            "schedulers": _scan_registered_choices(["optim/schedulers.py"]),
        }
    return _registry_cache


# ---------------------------------------------------------------------------
# Dataset comparison (padding-bias preflight for lerobot-cotrain-align)
# ---------------------------------------------------------------------------

HF_LEROBOT_CACHE = Path("~/.cache/huggingface/lerobot").expanduser()
VISUALIZER_BASE = "https://lerobot-visualize-dataset.hf.space"
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

# Keep in sync with _DEFAULT_CAMERA_REMAP in
# src/lerobot/data_processing/co_training_utils.py (duplicated so this server
# never imports torch/cv2 — comparisons must answer in milliseconds).
DEFAULT_CAMERA_REMAP: dict[str, str] = {
    "top": "head", "cam_high": "head", "cam_low": "head", "image": "head",
    "obs_image": "head", "rgb_image": "head",
    "cam_left_wrist": "left_wrist", "left_wrist": "left_wrist",
    "wrist_image": "left_wrist",
    "cam_right_wrist": "right_wrist", "right_wrist": "right_wrist",
    "rgb_images.front": "head", "rgb_images.left": "left_wrist",
    "rgb_images.right": "right_wrist",
}
_VEL_NAME_TOKENS = (".vel", "_vel", "velocity")

_meta_cache: dict[str, dict] = {}
_meta_cache_lock = threading.Lock()


def _fetch_hub_json(repo: str, filename: str) -> dict | None:
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/meta/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None


def _load_dataset_meta(repo: str) -> dict:
    """info.json (+stats.json when available) from the local HF cache or the Hub."""
    if not _REPO_ID_RE.match(repo):
        raise ValueError(f"Repo id must look like 'user/name', got {repo!r}")
    with _meta_cache_lock:
        if repo in _meta_cache:
            return _meta_cache[repo]
    local_meta = HF_LEROBOT_CACHE / repo / "meta"
    if (local_meta / "info.json").is_file():
        info = json.loads((local_meta / "info.json").read_text())
        stats = None
        if (local_meta / "stats.json").is_file():
            stats = json.loads((local_meta / "stats.json").read_text())
        entry = {"repo": repo, "source": "local", "info": info, "stats": stats}
    else:
        info = _fetch_hub_json(repo, "info.json")
        if info is None:
            raise FileNotFoundError(
                f"{repo}: no meta/info.json in the local cache ({local_meta}) or on the Hub "
                "(private repo? typo? run `huggingface-cli login` for private data)."
            )
        entry = {"repo": repo, "source": "hub", "info": info,
                 "stats": _fetch_hub_json(repo, "stats.json")}
    with _meta_cache_lock:
        _meta_cache[repo] = entry
    return entry


def _is_vel(name: str) -> bool:
    return any(tok in name for tok in _VEL_NAME_TOKENS)


def _names_list(feat: dict) -> list[str]:
    """Flatten a feature's names — v2 datasets nest them (e.g. {"motors": [...]})."""
    names = feat.get("names")
    if isinstance(names, dict):
        flat: list[str] = []
        for v in names.values():
            if isinstance(v, list):
                flat.extend(str(n) for n in v)
        return flat
    return [str(n) for n in (names or [])]


def _feature_rows(ref_meta: dict, cand_meta: dict, key: str) -> tuple[list[dict], list[str]]:
    """Per-reference-dim mapping rows for 'observation.state' / 'action'."""
    ref_feat = (ref_meta["info"].get("features") or {}).get(key) or {}
    cand_feat = (cand_meta["info"].get("features") or {}).get(key) or {}
    ref_names = _names_list(ref_feat)
    cand_names = set(_names_list(cand_feat))
    ref_stats = ((ref_meta.get("stats") or {}).get(key) or {})
    means, stds = ref_stats.get("mean"), ref_stats.get("std")
    rows = []
    for i, name in enumerate(ref_names):
        mean = means[i] if isinstance(means, list) and i < len(means) else None
        std = stds[i] if isinstance(stds, list) and i < len(stds) else None
        zero_sigma = None
        if mean is not None and std:
            zero_sigma = round(abs(mean) / std, 1) if std > 1e-9 else None
        rows.append({
            "dim": i, "name": name,
            "mapped": name in cand_names,
            "vel": _is_vel(name),
            "ref_mean": round(mean, 3) if mean is not None else None,
            "ref_std": round(std, 3) if std is not None else None,
            "zero_sigma": zero_sigma,   # how many σ the 0-fill sits from the ref mean
        })
    dropped = [n for n in _names_list(cand_feat) if n not in set(ref_names)]
    return rows, dropped


def _camera_rows(ref_meta: dict, cand_meta: dict) -> list[dict]:
    def cams(meta):
        feats = meta["info"].get("features") or {}
        return {k.removeprefix("observation.images."): (v.get("shape") or [])
                for k, v in feats.items() if k.startswith("observation.images.")}
    ref_cams, cand_cams = cams(ref_meta), cams(cand_meta)
    rows = []
    seen_dst = set()
    for cam, shape in cand_cams.items():
        dst = DEFAULT_CAMERA_REMAP.get(cam, cam)
        dup = dst in seen_dst
        seen_dst.add(dst)
        rows.append({"cand_cam": cam, "maps_to": dst, "shape": list(shape),
                     "in_ref": dst in ref_cams, "duplicate": dup})
    for cam, shape in ref_cams.items():
        if cam not in seen_dst:
            rows.append({"cand_cam": None, "maps_to": cam, "shape": list(shape),
                         "in_ref": True, "duplicate": False})
    return rows


def dataset_schema(repo: str) -> dict:
    """Real joint/camera names for *repo* — powers the extract stage's
    dimension/camera picker so "keep_names"/"cameras"/"keep_depth" can be
    filled in by ticking checkboxes instead of typing exact strings."""
    meta = _load_dataset_meta(repo)
    feats = meta["info"].get("features") or {}
    state_names = _names_list(feats.get("observation.state") or {})
    action_names = _names_list(feats.get("action") or {})
    cameras = []
    for key, feat in feats.items():
        if not key.startswith("observation.images."):
            continue
        cam = key.removeprefix("observation.images.")
        is_depth = feat.get("dtype") == "depth" or cam.endswith("_depth")
        cameras.append({"name": cam, "is_depth": is_depth, "shape": list(feat.get("shape") or [])})
    return {
        "repo": repo,
        "source": meta["source"],
        "state_names": state_names,
        "action_names": action_names,
        "cameras": cameras,
        "fps": meta["info"].get("fps"),
        "episodes": meta["info"].get("total_episodes"),
    }


def compare_datasets(ref_repo: str, cand_repo: str) -> dict:
    """Preflight report: what lerobot-cotrain-align would pad/drop, and how bad
    zero-filling would be given the reference stats."""
    ref, cand = _load_dataset_meta(ref_repo), _load_dataset_meta(cand_repo)
    state_rows, state_dropped = _feature_rows(ref, cand, "observation.state")
    action_rows, action_dropped = _feature_rows(ref, cand, "action")
    cameras = _camera_rows(ref, cand)

    issues: list[dict] = []
    rec = {"pad_fill_mode": "zero", "override_padded_stats": False,
           "weighting": "none", "external_weight": None}

    padded_actions = [r for r in action_rows if not r["mapped"]]
    padded_pos = [r for r in padded_actions if not r["vel"]]
    if action_rows and not any(r["mapped"] for r in action_rows):
        issues.append({
            "severity": "high",
            "title": "No joint names match — convert this source first",
            "text": (f"None of {cand_repo}'s joint names appear in the {ref_repo} schema, so "
                     "by-name remapping would pad every dim (the data would contribute nothing "
                     "but constants). Run the DATA_CONVERSION.md pipeline (or "
                     "lerobot-umi-retarget for UMI data) to rename/retarget joints into the "
                     "xlerobot schema, then compare the converted repo instead."),
        })
    if padded_pos:
        frac = len(padded_actions) / max(1, len(action_rows))
        rec.update({"pad_fill_mode": "ref-mean", "override_padded_stats": True,
                    "weighting": "source",
                    "external_weight": 0.2 if frac > 0.5 else (0.3 if frac > 0.25 else 0.5)})
        issues.append({
            "severity": "high",
            "title": f"{len(padded_actions)}/{len(action_rows)} action dims will be padded",
            "text": (f"{cand_repo} cannot supply "
                     f"{', '.join(r['name'] for r in padded_pos[:6])}"
                     f"{'…' if len(padded_pos) > 6 else ''}. Zero-filling teaches the policy "
                     "to hold these joints frozen and biases the merged normalization stats. "
                     "Fix: --pad-fill-mode ref-mean (+ state-copy for identity actions), "
                     "--override-padded-stats, and train with --sample_weighting.type=source."),
        })
        worst = sorted((r for r in padded_pos if r["zero_sigma"]),
                       key=lambda r: -r["zero_sigma"])[:3]
        if worst and worst[0]["zero_sigma"] >= 2:
            listing = ", ".join(f"{r['name']} ({r['zero_sigma']}σ)" for r in worst)
            issues.append({
                "severity": "high",
                "title": "Zero-fill lands far outside your real motion range",
                "text": (f"A constant 0 sits {listing} away from the {ref_repo} mean. "
                         "Under merged normalization your real motion on these dims becomes "
                         "outliers. --pad-fill-mode ref-mean puts fills at the distribution "
                         "centre; --override-padded-stats keeps your stats authoritative."),
            })
    elif padded_actions:
        issues.append({
            "severity": "warn",
            "title": "Only velocity dims are padded",
            "text": "Zero is semantically correct ('stopped') for velocity dims — no fill-mode "
                    "change needed for this source.",
        })
    else:
        issues.append({"severity": "ok", "title": "All action dims map by name",
                       "text": f"{cand_repo} covers the full {ref_repo} action schema — "
                               "native-quality source, weight 1.0."})

    if action_dropped:
        issues.append({
            "severity": "warn",
            "title": f"{len(action_dropped)} candidate dims will be dropped",
            "text": f"Not in the reference schema: {', '.join(action_dropped[:8])}"
                    f"{'…' if len(action_dropped) > 8 else ''}.",
        })

    shared_cams = [c for c in cameras if c["cand_cam"] and c["in_ref"] and not c["duplicate"]]
    if not shared_cams:
        issues.append({
            "severity": "high",
            "title": "No shared cameras after default remap",
            "text": "aggregate_datasets() needs at least one common observation.images.* key. "
                    "Pass a custom --camera-remap mapping the candidate's cameras onto "
                    "head / left_wrist / right_wrist.",
        })
    unmapped = [c for c in cameras if c["cand_cam"] and not c["in_ref"]]
    if unmapped and shared_cams:
        issues.append({
            "severity": "warn",
            "title": "Some candidate cameras won't survive the merge",
            "text": f"{', '.join(c['cand_cam'] for c in unmapped)} have no reference "
                    "counterpart; merged cameras are the intersection across sources.",
        })

    ref_fps, cand_fps = ref["info"].get("fps"), cand["info"].get("fps")
    if ref_fps and cand_fps and ref_fps != cand_fps:
        issues.append({
            "severity": "warn" if cand_fps > ref_fps else "high",
            "title": f"fps mismatch: {cand_fps} vs {ref_fps}",
            "text": ("Downsampled automatically (nearest-neighbour) during alignment."
                     if cand_fps > ref_fps else
                     "Candidate is slower than the target fps — frames are NOT upsampled and "
                     "the whole merge drops to the lowest fps. Consider excluding this source."),
        })

    ref_eps = ref["info"].get("total_episodes") or 0
    cand_eps = cand["info"].get("total_episodes") or 0
    if ref_eps and cand_eps > 3 * ref_eps:
        issues.append({
            "severity": "warn",
            "title": f"Candidate dwarfs your data ({cand_eps} vs {ref_eps} episodes)",
            "text": "Keep external data a minority of merged frames: convert fewer episodes "
                    "(--max-episodes during conversion) or lean harder on source weighting "
                    "(external_weight 0.2).",
        })

    if rec["pad_fill_mode"] != "zero" and not ref.get("stats"):
        issues.append({
            "severity": "warn",
            "title": "Reference has no stats.json",
            "text": "ref-mean fill and stats override need the reference repo's stats; "
                    "they will fall back to zero-fill until stats exist.",
        })

    order = {"high": 0, "warn": 1, "ok": 2}
    issues.sort(key=lambda i: order.get(i["severity"], 3))

    def summary(m):
        info = m["info"]
        return {
            "repo": m["repo"], "source": m["source"],
            "robot_type": info.get("robot_type"), "fps": info.get("fps"),
            "episodes": info.get("total_episodes"), "frames": info.get("total_frames"),
            "state_dim": ((info.get("features") or {}).get("observation.state") or {})
                .get("shape", [None])[0],
            "action_dim": ((info.get("features") or {}).get("action") or {})
                .get("shape", [None])[0],
            "has_stats": bool(m.get("stats")),
            "visualizer_url": f"{VISUALIZER_BASE}/{m['repo']}/episode_0",
        }

    return {
        "ref": summary(ref), "cand": summary(cand),
        "state": state_rows, "action": action_rows,
        "state_dropped": state_dropped, "action_dropped": action_dropped,
        "cameras": cameras, "issues": issues, "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# Run engine
# ---------------------------------------------------------------------------

class WorkflowRun:
    """Sequential execution of a workflow's generated stage scripts."""

    def __init__(self, workflow: dict, only_stage: int | None = None):
        self.workflow = workflow
        self.name = workflow["name"]
        self.only_stage = only_stage
        self.lock = threading.Lock()
        self.status = "pending"          # pending | running | done | failed | stopped
        self.stages: list[dict] = []     # {index, name, status, log, returncode}
        self.proc: subprocess.Popen | None = None
        self.stop_requested = False
        self.error: str | None = None
        self.started_at = time.time()

    # -- state ----------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "error": self.error,
                "started_at": self.started_at,
                "stages": [dict(s) for s in self.stages],
            }

    def _persist(self) -> None:
        state_path = WORKFLOWS_DIR / self.name / "logs" / "run_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(self.snapshot(), indent=2))

    # -- execution --------------------------------------------------------------
    def run(self) -> None:
        try:
            entries = generate_scripts(self.workflow)
        except ValueError as err:
            with self.lock:
                self.status = "failed"
                self.error = str(err)
            self._persist()
            return

        if self.only_stage is not None:
            entries = [e for e in entries if e["index"] == self.only_stage]
            if not entries:
                with self.lock:
                    self.status = "failed"
                    self.error = f"Stage {self.only_stage} is disabled or does not exist."
                self._persist()
                return

        logs_dir = WORKFLOWS_DIR / self.name / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.status = "running"
            self.stages = [
                {
                    "index": e["index"],
                    "name": e["name"],
                    "status": "pending",
                    "log": str((logs_dir / (Path(e["path"]).stem + ".log")).relative_to(REPO_ROOT)),
                    "returncode": None,
                }
                for e in entries
            ]
        self._persist()

        for pos, entry in enumerate(entries):
            if self.stop_requested:
                break
            log_path = REPO_ROOT / self.stages[pos]["log"]
            with self.lock:
                self.stages[pos]["status"] = "running"
            self._persist()
            with open(log_path, "w") as log_f:
                log_f.write(f"$ bash {entry['path']}\n\n")
                log_f.flush()
                proc = subprocess.Popen(
                    ["bash", str(REPO_ROOT / entry["path"])],
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=REPO_ROOT,
                    start_new_session=True,  # own process group → clean stop of children
                )
                with self.lock:
                    self.proc = proc
                    # Persisted so a later process (e.g. after this server
                    # restarts and loses the Popen handle) can tell a
                    # genuinely-still-running stage apart from one whose
                    # subprocess already exited orphaned — see
                    # _workflow_status_snapshot's liveness reconciliation.
                    self.stages[pos]["pid"] = proc.pid
                self._persist()
                rc = proc.wait()
            with self.lock:
                self.proc = None
                self.stages[pos]["returncode"] = rc
                if self.stop_requested:
                    self.stages[pos]["status"] = "stopped"
                else:
                    self.stages[pos]["status"] = "done" if rc == 0 else "failed"
            self._persist()
            if rc != 0 and not self.stop_requested:
                with self.lock:
                    self.status = "failed"
                    self.error = f"Stage '{entry['name']}' exited with code {rc} — see its log."
                self._persist()
                return

        with self.lock:
            self.status = "stopped" if self.stop_requested else "done"
        self._persist()

    def stop(self) -> None:
        with self.lock:
            self.stop_requested = True
            proc = self.proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


_runs: dict[str, WorkflowRun] = {}
_runs_lock = threading.Lock()


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else — treat as alive
    return True


def _workflow_status_snapshot(name: str) -> dict:
    """Live status if this process still has the WorkflowRun in memory,
    else fall back to the last-persisted run_state.json — status survives a
    workflow_ui server restart even though live run tracking doesn't (the
    underlying stage subprocess keeps running orphaned; only this process's
    bookkeeping of it is lost).

    A disk-sourced "running" status is only trustworthy if the recorded pid
    is still alive. If the server itself was restarted (losing its Popen
    handle) while a stage's subprocess was mid-run, that subprocess keeps
    running to completion on its own (start_new_session=True) but nothing is
    left to catch its exit code — run_state.json then says "running" forever
    even after the process is long gone. Detect that here and self-heal by
    persisting the corrected status, rather than reporting a stale "running"
    indefinitely."""
    with _runs_lock:
        run = _runs.get(name)
    if run is not None:
        return {"live": run.status == "running", **run.snapshot()}
    state_path = WORKFLOWS_DIR / name / "logs" / "run_state.json"
    if state_path.is_file():
        try:
            data = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            data = None
        if data is not None:
            if data.get("status") == "running":
                running_stage = next(
                    (s for s in data.get("stages", []) if s.get("status") == "running"), None
                )
                if running_stage is not None and not _pid_alive(running_stage.get("pid")):
                    running_stage["status"] = "orphaned"
                    data["status"] = "orphaned"
                    data["error"] = (
                        f"Stage '{running_stage['name']}' subprocess (pid {running_stage.get('pid')}) "
                        "is no longer running, but the workflow_ui server lost track of it before "
                        "recording an exit code (likely a server restart mid-run). Check its log "
                        "under logs/ directly to see whether it actually finished or crashed."
                    )
                    with contextlib.suppress(OSError):
                        state_path.write_text(json.dumps(data, indent=2))
            return {"live": False, **data}
    return {"live": False, "status": "idle", "stages": []}


# ---------------------------------------------------------------------------
# Generalist merge status — the ad-hoc, externally-launched EE-space merge
# (own tasks + FastUMI + ALOHA + DROID → Odog16/generalist_ee_merged). This
# job predates the WorkflowRun engine above and is a single long-running
# `lerobot-cotrain-align` invocation launched directly with nohup, not
# through this server — so it's tracked by parsing its log + a process-table
# lookup instead of the run_state.json mechanism used for UI-launched runs.
# ---------------------------------------------------------------------------

GENERALIST_MERGE_LOG = WORKFLOWS_DIR / "ee_pretrain" / "logs" / "14_generalist_merge.log"
GENERALIST_MERGE_RESUME_DOC = REPO_ROOT / "docs" / "generalist_merge_resume.md"
GENERALIST_MERGE_PID_SIGNATURE = "Odog16/generalist_ee_merged"

# (repo_id, group) in the exact order passed to --source-repos.
GENERALIST_MERGE_SOURCES: list[tuple[str, str]] = [
    ("Odog16/tool_pickup_ee", "own_tasks"),
    ("Odog16/trash_pickup_merged_ee", "own_tasks"),
    ("Odog16/block_sorting_single_ee", "own_tasks"),
    ("Odog16/block_sorting_clean_ee", "own_tasks"),
    ("Odog16/making_coffee_v1_ee", "own_tasks"),
    ("Odog16/ob15_general_dataset_v1_ee", "own_tasks"),
    ("Odog16/ob15_packing_box_filtered_ee", "own_tasks"),
    ("Odog16/umi_ee_pretrain_full_clean_desktop", "fastumi_ee"),
    ("Odog16/umi_ee_pretrain_full_dispose_of_desktop_debris", "fastumi_ee"),
    ("Odog16/umi_ee_pretrain_full_put_books_into_schoolbag", "fastumi_ee"),
    ("Odog16/umi_ee_pretrain_full_pack_skincare_products", "fastumi_ee"),
    ("Odog16/umi_ee_pretrain_full_take_bottle_and_place_on_coaster", "fastumi_ee"),
    ("Odog16/aloha_sim_insertion_human_ee", "aloha_ee"),
    ("Odog16/aloha_sim_transfer_cube_human_ee", "aloha_ee"),
    ("Odog16/aloha_sim_transfer_cube_scripted_ee", "aloha_ee"),
    ("Odog16/aloha_static_coffee_ee", "aloha_ee"),
    ("Odog16/aloha_mobile_cabinet_ee", "aloha_ee"),
    ("Odog16/aloha_mobile_wash_pan_ee", "aloha_ee"),
    ("Odog16/aloha_static_screw_driver_ee", "aloha_ee"),
    ("Odog16/aloha_static_candy_ee", "aloha_ee"),
    ("Odog16/aloha_mobile_wipe_wine_ee", "aloha_ee"),
    ("Odog16/aloha_static_towel_ee", "aloha_ee"),
    ("Odog16/aloha_static_vinh_cup_ee", "aloha_ee"),
    ("Odog16/aloha_static_vinh_cup_left_ee", "aloha_ee"),
    ("Odog16/aloha_static_ziploc_slide_ee", "aloha_ee"),
    ("Odog16/aloha_static_coffee_new_ee", "aloha_ee"),
    ("Odog16/aloha_static_cups_open_ee", "aloha_ee"),
    ("Odog16/aloha_static_pingpong_test_ee", "aloha_ee"),
    ("Odog16/aloha_static_pro_pencil_ee", "aloha_ee"),
    ("Odog16/aloha_sim_insertion_scripted_ee", "aloha_ee"),
    ("Odog16/droid_ee", "droid_ee"),
]

# tqdm line: "Align Odog16/foo_ee:  56%|█████▌    | 56/100 [08:02<06:24,  8.75s/ep, ...]"
# Captures: repo, percent, episode, episode_total, elapsed, remaining, rate value+unit.
# The "remaining" field is tqdm's own ETA for the current source — reused directly
# below rather than re-derived, since it's already exactly what we'd compute.
_ALIGN_PROGRESS_RE = re.compile(
    r"^Align ([^\s:]+):\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*"
    r"\[([\d:]+)<([\d:]+),\s*([\d.]+)(s/ep|ep/s|s/it|it/s)",
    re.MULTILINE,
)

# Rough historical per-episode alignment cost per source group (seconds),
# from real timings observed this session — own_tasks/fastumi_ee already
# match the merge schema (no camera remap/resample) so re-encode fastest;
# aloha_ee additionally resamples 50fps->30fps and remaps cameras; droid_ee
# is a rewritten-elsewhere-in-this-project fast parquet-level path. Only
# used to project an ETA for sources that haven't started yet — the CURRENT
# source's own ETA below comes from tqdm's live rate, not this table.
_GROUP_AVG_SECONDS_PER_EPISODE = {
    "own_tasks": 9.0,
    "fastumi_ee": 9.0,
    "aloha_ee": 12.0,
    "droid_ee": 4.3,
}


def _source_total_episodes(repo: str) -> int | None:
    """Real episode count for *repo*, read directly from its local
    meta/info.json — no torch/cv2 import needed, just a small JSON read."""
    info_path = HF_LEROBOT_CACHE / repo / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        return int(json.loads(info_path.read_text()).get("total_episodes") or 0)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _source_done_on_disk(repo: str) -> bool:
    """Whether *repo*'s alignment is genuinely complete, checked directly on
    disk (survives log truncation/rotation across a stop+relaunch — a plain
    log-line scan alone can't see progress from before the log was last
    overwritten). Mirrors the same episode-count comparison
    `_alignment_artifacts_match` uses in co_training_utils.py, reading only
    two small JSON files — no torch/cv2 import needed."""
    tmp_info = (
        HF_LEROBOT_CACHE / "Odog16" / f"_align_tmp_{repo.replace('/', '__')}" / "meta" / "info.json"
    )
    expected = _source_total_episodes(repo)
    if not expected or not tmp_info.is_file():
        return False
    try:
        cached = int(json.loads(tmp_info.read_text()).get("total_episodes") or -1)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return cached == expected


def _parse_tqdm_duration(s: str) -> int | None:
    """"H:MM:SS" or "MM:SS" (tqdm's own bracketed duration format) -> seconds."""
    try:
        parts = [int(p) for p in s.split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def _format_eta(seconds: int) -> str:
    if seconds <= 0:
        return "any moment"
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"~{d}d {h}h"
    if h:
        return f"~{h}h {m}m"
    return f"~{m}m"


def _merge_pid_alive() -> int | None:
    """Real PID of the running merge process, found by command-line signature
    rather than a hand-maintained pid file — survives this UI server being
    restarted, and a relaunch under a new PID after a stop/reboot."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", GENERALIST_MERGE_PID_SIGNATURE],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
    return pids[0] if pids else None


def generalist_merge_status() -> dict:
    """Live status of the multi-day EE-space generalist merge, parsed
    straight from its log."""
    log_path = GENERALIST_MERGE_LOG
    if not log_path.is_file():
        return {"found": False}

    # Which sources have a completion marker anywhere in the log. grep does
    # the heavy lifting on a log that grows to many MB over a multi-day run —
    # Python never reads the whole file, just this short, filtered result.
    try:
        grep = subprocess.run(
            ["grep", "-aE",
             r"already merge-ready\)|Aligned dataset written to|"
             r"Reusing already-aligned cache|Reusing completed aligned dataset at|"
             r"Wrote co-train source manifest",
             str(log_path)],
            capture_output=True, text=True, timeout=20,
        )
        done_lines = grep.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        done_lines = []

    merge_complete = any("Wrote co-train source manifest" in line for line in done_lines)
    done_repos: set[str] = set()
    for repo, _group in GENERALIST_MERGE_SOURCES:
        marker_tmp = f"_align_tmp_{repo.replace('/', '__')}"
        marker_native = f"already merge-ready): {repo} "
        from_log = any(marker_native in line or marker_tmp in line for line in done_lines)
        if from_log or _source_done_on_disk(repo):
            done_repos.add(repo)

    # Current in-progress source + episode progress: only the log tail matters.
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 32_000))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        tail = ""
    tail = tail.replace("\r", "\n")
    current_repo = current_pct = current_ep = current_ep_total = None
    current_elapsed_s = current_remaining_s = current_rate_s_per_ep = None
    for m in _ALIGN_PROGRESS_RE.finditer(tail):
        current_repo, current_pct, current_ep, current_ep_total = (
            m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)),
        )
        current_elapsed_s = _parse_tqdm_duration(m.group(5))
        current_remaining_s = _parse_tqdm_duration(m.group(6))
        rate_val, rate_unit = float(m.group(7)), m.group(8)
        current_rate_s_per_ep = rate_val if rate_unit in ("s/ep", "s/it") else (
            1.0 / rate_val if rate_val > 0 else None
        )

    error_line = None
    for line in tail.splitlines():
        if any(tok in line for tok in ("Traceback", "CRITICAL", "Killed", "OOM")):
            error_line = line.strip()

    pid = _merge_pid_alive()
    sources = []
    total_all_episodes = 0
    total_done_episodes = 0
    pending_eta_s = 0
    for repo, group in GENERALIST_MERGE_SOURCES:
        ep_total = _source_total_episodes(repo)
        if repo in done_repos:
            status = "done"
        elif repo == current_repo:
            status = "current"
        else:
            status = "pending"
        entry = {"repo": repo, "group": group, "status": status, "episode_total": ep_total}
        if repo == current_repo:
            entry.update(
                percent=current_pct, episode=current_ep,
                eta_s=current_remaining_s, elapsed_s=current_elapsed_s,
                rate_s_per_ep=(
                    round(current_rate_s_per_ep, 2) if current_rate_s_per_ep is not None else None
                ),
            )
        if status == "done" and ep_total:
            total_done_episodes += ep_total
        if ep_total:
            total_all_episodes += ep_total
        # ETA contribution: current source uses tqdm's own live remaining-time;
        # any source after it (not yet started) uses the group's historical
        # average rate against its real episode count (rough estimate — see
        # _GROUP_AVG_SECONDS_PER_EPISODE's docstring).
        if status == "current" and current_remaining_s is not None:
            pending_eta_s += current_remaining_s
        elif status == "pending" and ep_total:
            pending_eta_s += ep_total * _GROUP_AVG_SECONDS_PER_EPISODE.get(group, 10.0)
        sources.append(entry)

    total_eta_s = int(pending_eta_s)

    return {
        "found": True,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "pid": pid,
        "alive": pid is not None,
        "merge_complete": merge_complete,
        "total_sources": len(GENERALIST_MERGE_SOURCES),
        "done_count": len(done_repos),
        "current_repo": current_repo,
        "error_line": error_line,
        "sources": sources,
        "total_all_episodes": total_all_episodes,
        "total_done_episodes": total_done_episodes,
        "eta_seconds": total_eta_s,
        "eta_human": _format_eta(total_eta_s) if not merge_complete else None,
    }


def generalist_merge_resume() -> dict:
    """Raw text of the resume/status doc — narrative facts (decisions,
    pending steps) live in one file so this UI and future sessions read the
    same source of truth instead of duplicating it in this server."""
    if not GENERALIST_MERGE_RESUME_DOC.is_file():
        return {"found": False, "text": ""}
    return {"found": True, "text": GENERALIST_MERGE_RESUME_DOC.read_text(encoding="utf-8")}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def home_tasks_preset() -> dict:
    """The COTRAINING.md recipe as an editable workflow: merge → generalist
    pre-train → per-task specialist fine-tunes, with the padding-bias
    mitigations (ref-mean fill, stats override, source down-weighting) on."""
    return {
        "name": "home_tasks_cotrain",
        "variables": {
            "HF_USER": "Odog16",
            "MERGED_REPO": "Odog16/xlerobot_cotrain_v1",
            "REFERENCE_REPO": "Odog16/tool_pickup",
            "TASK_A": "Odog16/trash_pickup",
            "TASK_B": "Odog16/tool_pickup",
            "EXTERNAL_REPOS": "Odog16/aloha_aligned_for_cotrain",
            "GENERALIST_STEPS": "60000",
            "FINETUNE_STEPS": "20000",
            "FINETUNE_LR": "5e-5",
            "BATCH_SIZE": "16",
        },
        "stages": [
            {
                "name": "Merge co-train dataset",
                "type": "merge",
                "enabled": True,
                "params": {
                    "source_repos": "{{TASK_A}} {{TASK_B}} {{EXTERNAL_REPOS}}",
                    "target_repo_id": "{{MERGED_REPO}}",
                    "target_fps": "30",
                    "target_image_size": "360x640",
                    "target_state_dim": "18",
                    "target_action_dim": "18",
                    "match_features_from": "{{REFERENCE_REPO}}",
                    "pad_fill_mode": "ref-mean",
                    "override_padded_stats": True,
                    "push_to_hub": False,
                },
            },
            {
                "name": "Generalist pre-train (SmolVLA)",
                "type": "train",
                "enabled": True,
                "params": {
                    "dataset_repo_id": "{{MERGED_REPO}}",
                    "dataset_root": "$HOME/.cache/huggingface/lerobot/{{MERGED_REPO}}",
                    "policy_type": "smolvla",
                    "pretrained_path": "lerobot/smolvla_base",
                    "train_state_proj": True,
                    "use_amp": True,
                    "batch_size": "{{BATCH_SIZE}}",
                    "steps": "{{GENERALIST_STEPS}}",
                    "save_freq": "10000",
                    "output_dir": "outputs/train/generalist_cotrain",
                    "weighting": "source",
                    "external_weight": "0.3",
                    "wandb_enable": True,
                    "wandb_project": "lerobot",
                },
            },
            {
                "name": "Fine-tune specialist: TASK_A",
                "type": "train",
                "enabled": True,
                "params": {
                    "dataset_repo_id": "{{TASK_A}}",
                    "policy_type": "smolvla",
                    "pretrained_path":
                        "outputs/train/generalist_cotrain/checkpoints/last/pretrained_model",
                    "scheduler_type": "cosine_decay_with_warmup",
                    "peak_lr": "{{FINETUNE_LR}}",
                    "batch_size": "{{BATCH_SIZE}}",
                    "steps": "{{FINETUNE_STEPS}}",
                    "output_dir": "outputs/train/task_a_specialist",
                    "weighting": "none",
                    "wandb_enable": True,
                    "wandb_project": "lerobot",
                },
            },
            {
                "name": "Fine-tune specialist: TASK_B",
                "type": "train",
                "enabled": True,
                "params": {
                    "dataset_repo_id": "{{TASK_B}}",
                    "policy_type": "smolvla",
                    "pretrained_path":
                        "outputs/train/generalist_cotrain/checkpoints/last/pretrained_model",
                    "scheduler_type": "cosine_decay_with_warmup",
                    "peak_lr": "{{FINETUNE_LR}}",
                    "batch_size": "{{BATCH_SIZE}}",
                    "steps": "{{FINETUNE_STEPS}}",
                    "output_dir": "outputs/train/task_b_specialist",
                    "weighting": "none",
                    "wandb_enable": True,
                    "wandb_project": "lerobot",
                },
            },
        ],
    }


def master_umi_preset() -> dict:
    """Master-dataset strategy at UMI scale: record once with all 18 DOF +
    RGB-D, extract a 12-DOF RGB-only subset whose tensor shapes match
    open-source bimanual data, pre-train in two stages (UMI/bimanual-heavy
    12-DOF merge first, then native 18-DOF merge), and fine-tune specialists.
    SARM annotation stages feed RA-BC weighting for the fine-tunes."""
    return {
        "name": "master_rgbd_umi_cotrain",
        "variables": {
            "MASTER_REPO": "Odog16/master_home_v1",
            "SUBSET_REPO": "Odog16/master_home_v1_bimanual12",
            "REFERENCE_REPO": "Odog16/tool_pickup",
            "TASK_REPOS": "Odog16/trash_pickup Odog16/tool_pickup",
            "TASK_A": "Odog16/trash_pickup",
            "BIMANUAL_REPOS": "Odog16/aloha_aligned_for_cotrain",
            "UMI_REPOS": "Odog16/umi_pretrain_clean_desktop",
            "MERGED_12DOF": "Odog16/cotrain_umi_bimanual_12dof",
            "MERGED_18DOF": "Odog16/cotrain_native_18dof",
            "SARM_MODEL": "Odog16/sarm_xlerobot",
            "STAGE1_STEPS": "40000",
            "STAGE2_STEPS": "30000",
            "FINETUNE_STEPS": "20000",
            "BATCH_SIZE": "16",
        },
        "stages": [
            {
                "name": "Record master dataset (run on robot — reference)",
                "type": "custom",
                "enabled": False,
                "params": {"script":
                    "# Record with EVERYTHING on: all 18 DOF (lift axis enabled) + RGB-D head\n"
                    "# camera (use_depth=True is now the default in config_xlerobot.py).\n"
                    "# One master recording feeds every derived subset below — never\n"
                    "# re-collect for a schema change.\n"
                    "#\n"
                    "# lerobot-record needs the conda env (GPU, cameras) — non-custom stages\n"
                    "# activate it automatically; custom stages don't, so do it yourself:\n"
                    "CONDA_BASE=\"$(conda info --base)\"\n"
                    "source \"$CONDA_BASE/etc/profile.d/conda.sh\"\n"
                    "conda activate lerobot\n"
                    "\n"
                    "lerobot-record \\\n"
                    "    --robot.type=xlerobot \\\n"
                    "    --robot.lift_axis.enabled=true \\\n"
                    "    --teleop.type=xlerobot_vr \\\n"
                    "    --dataset.repo_id={{MASTER_REPO}} \\\n"
                    "    --dataset.num_episodes=50 \\\n"
                    "    --dataset.single_task=\"Describe the task here\""},
            },
            {
                "name": "Extract 12-DOF RGB subset (shape-matches bimanual data)",
                "type": "extract",
                "enabled": True,
                "params": {
                    "source_repo": "{{MASTER_REPO}}",
                    "target_repo_id": "{{SUBSET_REPO}}",
                    "profile": "bimanual12",
                    "target_image_size": "360x640",
                    "force_rebuild": False,
                    "push_to_hub": False,
                },
            },
            {
                "name": "Merge 12-DOF: subset + UMI + bimanual",
                "type": "merge",
                "enabled": True,
                "params": {
                    "source_repos": "{{SUBSET_REPO}} {{UMI_REPOS}} {{BIMANUAL_REPOS}}",
                    "target_repo_id": "{{MERGED_12DOF}}",
                    "target_fps": "30",
                    "target_image_size": "360x640",
                    "target_state_dim": "12",
                    "target_action_dim": "12",
                    "match_features_from": "{{SUBSET_REPO}}",
                    "pad_fill_mode": "ref-mean",
                    "override_padded_stats": True,
                    "push_to_hub": False,
                },
            },
            {
                "name": "Stage-1 pre-train (UMI-heavy, 12-DOF)",
                "type": "train",
                "enabled": True,
                "params": {
                    "dataset_repo_id": "{{MERGED_12DOF}}",
                    "policy_type": "smolvla",
                    "pretrained_path": "lerobot/smolvla_base",
                    "train_state_proj": True, "use_amp": True,
                    "batch_size": "{{BATCH_SIZE}}", "steps": "{{STAGE1_STEPS}}",
                    "save_freq": "10000",
                    "output_dir": "outputs/train/stage1_umi_12dof",
                    "weighting": "source", "external_weight": "0.3",
                    "wandb_enable": True, "wandb_project": "lerobot",
                },
            },
            {
                "name": "Merge 18-DOF: native task repos",
                "type": "merge",
                "enabled": True,
                "params": {
                    "source_repos": "{{TASK_REPOS}}",
                    "target_repo_id": "{{MERGED_18DOF}}",
                    "target_fps": "30",
                    "target_image_size": "360x640",
                    "target_state_dim": "18",
                    "target_action_dim": "18",
                    "match_features_from": "{{REFERENCE_REPO}}",
                    "pad_fill_mode": "ref-mean",
                    "override_padded_stats": True,
                    "push_to_hub": False,
                },
            },
            {
                "name": "Stage-2 pre-train (native 18-DOF, from stage-1)",
                "type": "train",
                "enabled": True,
                "params": {
                    "dataset_repo_id": "{{MERGED_18DOF}}",
                    "policy_type": "smolvla",
                    "pretrained_path":
                        "outputs/train/stage1_umi_12dof/checkpoints/last/pretrained_model",
                    "train_state_proj": True, "use_amp": True,
                    "batch_size": "{{BATCH_SIZE}}", "steps": "{{STAGE2_STEPS}}",
                    "save_freq": "10000",
                    "output_dir": "outputs/train/stage2_native_18dof",
                    "weighting": "none",
                    "wandb_enable": True, "wandb_project": "lerobot",
                },
            },
            {
                "name": "SARM-annotate task repo (for RA-BC fine-tune)",
                "type": "sarm",
                "enabled": False,
                "params": {
                    "dataset_repo_id": "{{TASK_A}}",
                    "reward_model_path": "{{SARM_MODEL}}",
                    "head_mode": "sparse",
                    "device": "cuda",
                },
            },
            {
                "name": "Fine-tune specialist: TASK_A (RA-BC if annotated)",
                "type": "train",
                "enabled": True,
                "params": {
                    "dataset_repo_id": "{{TASK_A}}",
                    "policy_type": "smolvla",
                    "pretrained_path":
                        "outputs/train/stage2_native_18dof/checkpoints/last/pretrained_model",
                    "scheduler_type": "cosine_decay_with_warmup", "peak_lr": "5e-5",
                    "batch_size": "{{BATCH_SIZE}}", "steps": "{{FINETUNE_STEPS}}",
                    "output_dir": "outputs/train/task_a_specialist_v2",
                    "weighting": "none",
                    "wandb_enable": True, "wandb_project": "lerobot",
                },
            },
        ],
    }


def _model_comparison_train_stages(dataset_var: str, batch_var: str, steps_var: str) -> list[dict]:
    """One train stage per policy architecture, sharing a dataset/batch/steps
    variable set — "different levels and types of generalist policy". Only
    SmolVLA is enabled by default: Run all would otherwise chain four full
    pretrains back to back (each potentially many hours). Enable the others
    to compare; each was smoke-tested for real (steps=1) against a local
    dataset before being added here, since silently-wrong or silently-crashing
    per-policy fields are a real risk (see cotrain-workflow-ui memory notes).
    """
    common = {
        "dataset_repo_id": f"{{{{{dataset_var}}}}}",
        "batch_size": f"{{{{{batch_var}}}}}",
        "steps": f"{{{{{steps_var}}}}}",
        "weighting": "source", "external_weight": "0.3",
        "push_to_hub": False,
        "wandb_enable": True, "wandb_project": "lerobot",
    }
    return [
        {
            "name": "Pretrain: SmolVLA (VLA, ~450M)",
            "type": "train", "enabled": True,
            "params": {**common,
                "policy_type": "smolvla", "pretrained_path": "lerobot/smolvla_base",
                "train_state_proj": True, "use_amp": True,
                "output_dir": "outputs/train/max_data_smolvla",
            },
        },
        {
            "name": "Pretrain: SmolVLA-large (bigger VLM capacity)",
            "type": "train", "enabled": False,
            "params": {**common,
                "policy_type": "smolvla", "pretrained_path": "lerobot/smolvla_base",
                "train_state_proj": True, "use_amp": True,
                "output_dir": "outputs/train/max_data_smolvla_large",
                "extra_args": "--policy.num_vlm_layers=32 --policy.expert_width_multiplier=1.0",
            },
        },
        {
            "name": "Pretrain: ACT (transformer BC baseline, ~52M)",
            "type": "train", "enabled": False,
            "params": {**common, "policy_type": "act", "use_amp": True,
                "output_dir": "outputs/train/max_data_act"},
        },
        {
            "name": "Pretrain: Diffusion Policy (~293M)",
            "type": "train", "enabled": False,
            "params": {**common, "policy_type": "diffusion", "use_amp": True,
                "output_dir": "outputs/train/max_data_diffusion"},
        },
    ]


def max_data_18dof_preset() -> dict:
    """Maximize native 18-DOF xlerobot data + every locally-available bimanual
    external source into one merge, then pretrain a generalist — with a choice
    of policy architecture/scale to compare (SmolVLA, SmolVLA-large, ACT,
    Diffusion). Source lists were picked from an actual audit of this
    machine's local dataset cache (episode/frame counts, robot_type, state
    dim), deliberately excluding test/smoke recordings and superseded
    duplicate variants (e.g. block_sorting raw vs. _clean, trash_pickup vs.
    _merged, making_coffee 19-dim vs. _v1 18-dim, ob15_packing_box vs.
    _filtered) so the merge doesn't double-count near-identical episodes."""
    return {
        "name": "max_data_18dof_pretrain",
        "variables": {
            "REFERENCE_REPO": "Odog16/tool_pickup",
            "TASK_REPOS": (
                "Odog16/tool_pickup Odog16/trash_pickup_merged "
                "Odog16/block_sorting_single Odog16/block_sorting_clean "
                "Odog16/making_coffee_v1 Odog16/ob15_general_dataset_v1 "
                "Odog16/ob15_packing_box_filtered Odog16/test_transfer_block"
            ),
            "BIMANUAL_REPOS": (
                "lerobot/aloha_sim_insertion_human lerobot/aloha_sim_transfer_cube_human "
                "lerobot/aloha_sim_transfer_cube_scripted lerobot/aloha_mobile_cabinet "
                "lerobot/aloha_mobile_wash_pan lerobot/aloha_static_battery "
                "lerobot/aloha_static_coffee"
            ),
            "MERGED_REPO": "Odog16/max_data_18dof_v1",
            "PRETRAIN_STEPS": "60000",
            "BATCH_SIZE": "16",
        },
        "stages": [
            {
                "name": "Merge: all native 18-DOF task data + bimanual external",
                "type": "merge", "enabled": True,
                "params": {
                    "source_repos": "{{TASK_REPOS}} {{BIMANUAL_REPOS}}",
                    "target_repo_id": "{{MERGED_REPO}}",
                    "target_fps": "30", "target_image_size": "360x640",
                    "target_state_dim": "18", "target_action_dim": "18",
                    "match_features_from": "{{REFERENCE_REPO}}",
                    "pad_fill_mode": "ref-mean", "override_padded_stats": True,
                    "push_to_hub": False,
                },
            },
            *_model_comparison_train_stages("MERGED_REPO", "BATCH_SIZE", "PRETRAIN_STEPS"),
        ],
    }


def max_data_12dof_preset() -> dict:
    """Same idea as the 18-DOF max-data preset, projected to the 12-DOF
    (both-arms-only) schema: an extract stage derives a 12-dim reference
    schema from an existing real task repo (no master RGB-D recording
    required), then every native task repo is merged into that 12-dim target
    — by-name remap keeps the 12 shared arm-joint names and drops
    head/base/gantry automatically, no per-repo extract stage needed."""
    return {
        "name": "max_data_12dof_pretrain",
        "variables": {
            "REFERENCE_REPO": "Odog16/tool_pickup",
            "BIMANUAL12_REF": "Odog16/tool_pickup_bimanual12",
            "TASK_REPOS": (
                "Odog16/tool_pickup Odog16/trash_pickup_merged "
                "Odog16/block_sorting_single Odog16/block_sorting_clean "
                "Odog16/making_coffee_v1 Odog16/ob15_general_dataset_v1 "
                "Odog16/ob15_packing_box_filtered Odog16/test_transfer_block"
            ),
            "BIMANUAL_REPOS": (
                "lerobot/aloha_sim_insertion_human lerobot/aloha_sim_transfer_cube_human "
                "lerobot/aloha_sim_transfer_cube_scripted lerobot/aloha_mobile_cabinet "
                "lerobot/aloha_mobile_wash_pan lerobot/aloha_static_battery "
                "lerobot/aloha_static_coffee"
            ),
            "MERGED_REPO": "Odog16/max_data_12dof_v1",
            "PRETRAIN_STEPS": "60000",
            "BATCH_SIZE": "16",
        },
        "stages": [
            {
                "name": "Extract 12-DOF reference schema from an existing task repo",
                "type": "extract", "enabled": True,
                "params": {
                    "source_repo": "{{REFERENCE_REPO}}",
                    "target_repo_id": "{{BIMANUAL12_REF}}",
                    "profile": "bimanual12",
                    "target_image_size": "360x640",
                },
            },
            {
                "name": "Merge: all task data projected to 12-DOF + bimanual external",
                "type": "merge", "enabled": True,
                "params": {
                    "source_repos": "{{TASK_REPOS}} {{BIMANUAL_REPOS}}",
                    "target_repo_id": "{{MERGED_REPO}}",
                    "target_fps": "30", "target_image_size": "360x640",
                    "target_state_dim": "12", "target_action_dim": "12",
                    "match_features_from": "{{BIMANUAL12_REF}}",
                    "pad_fill_mode": "ref-mean", "override_padded_stats": True,
                    "push_to_hub": False,
                },
            },
            *_model_comparison_train_stages("MERGED_REPO", "BATCH_SIZE", "PRETRAIN_STEPS"),
        ],
    }


PRESETS = {
    "home_tasks": home_tasks_preset,
    "master_umi": master_umi_preset,
    "max_data_18dof": max_data_18dof_preset,
    "max_data_12dof": max_data_12dof_preset,
}


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

def _check_name(name: str) -> str:
    if not name or not _SAFE_NAME_RE.match(name):
        raise ValueError("Workflow name must match [A-Za-z0-9._-]+")
    return name


def _load_workflow(name: str) -> dict:
    path = WORKFLOWS_DIR / _check_name(name) / "workflow.json"
    if not path.is_file():
        raise FileNotFoundError(f"No workflow named {name!r}")
    return json.loads(path.read_text())


def _save_workflow(workflow: dict) -> None:
    name = _check_name(workflow.get("name", ""))
    wf_dir = WORKFLOWS_DIR / name
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "workflow.json").write_text(json.dumps(workflow, indent=2))


class Handler(BaseHTTPRequestHandler):
    server_version = "CotrainWorkflowUI/1.0"

    # -- helpers ---------------------------------------------------------------
    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, msg: str, status: int = 400) -> None:
        self._json({"error": msg}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- GET ---------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            if parsed.path in ("/", "/index.html"):
                html = (STATIC_DIR / "index.html").read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            elif parsed.path == "/api/workflows":
                names = sorted(
                    p.parent.name for p in WORKFLOWS_DIR.glob("*/workflow.json")
                ) if WORKFLOWS_DIR.is_dir() else []
                self._json({"workflows": names})
            elif parsed.path == "/api/workflow":
                self._json(_load_workflow(q["name"]))
            elif parsed.path == "/api/registry":
                self._json(get_repo_registry())
            elif parsed.path == "/api/preset":
                which = q.get("which", "home_tasks")
                if which not in PRESETS:
                    raise ValueError(f"Unknown preset {which!r}. Available: {sorted(PRESETS)}")
                self._json(PRESETS[which]())
            elif parsed.path == "/api/status":
                self._status(q["name"])
            elif parsed.path == "/api/running":
                names = sorted(
                    p.parent.name for p in WORKFLOWS_DIR.glob("*/workflow.json")
                ) if WORKFLOWS_DIR.is_dir() else []
                running = [n for n in names if _workflow_status_snapshot(n).get("status") == "running"]
                self._json({"running": running})
            elif parsed.path == "/api/log":
                self._log(q)
            elif parsed.path == "/api/compare":
                self._json(compare_datasets(q["ref"], q["cand"]))
            elif parsed.path == "/api/dataset-schema":
                self._json(dataset_schema(q["repo"]))
            elif parsed.path == "/api/generalist-merge/status":
                self._json(generalist_merge_status())
            elif parsed.path == "/api/generalist-merge/resume":
                self._json(generalist_merge_resume())
            elif parsed.path == "/api/processes":
                self._json(running_processes_status())
            else:
                self._error("Not found", 404)
        except (KeyError, ValueError, FileNotFoundError) as err:
            self._error(str(err), 404 if isinstance(err, FileNotFoundError) else 400)

    def _status(self, name: str) -> None:
        _check_name(name)
        self._json(_workflow_status_snapshot(name))

    def _log(self, q: dict) -> None:
        rel = q.get("path", "")
        offset = int(q.get("offset", 0))
        path = (REPO_ROOT / rel).resolve()
        if not str(path).startswith(str((WORKFLOWS_DIR).resolve())) or path.suffix != ".log":
            self._error("Log path must be a workflows/*.log file")
            return
        if not path.is_file():
            self._json({"data": "", "offset": offset})
            return
        size = path.stat().st_size
        if offset > size:
            offset = 0
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(512 * 1024)
        self._json({
            "data": data.decode("utf-8", errors="replace"),
            "offset": offset + len(data),
        })

    # -- POST ---------------------------------------------------------------
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/workflow":
                _save_workflow(body)
                self._json({"ok": True})
            elif parsed.path == "/api/workflow/delete":
                name = _check_name(body["name"])
                with _runs_lock:
                    run = _runs.get(name)
                if run is not None and run.status == "running":
                    self._error("Stop the running workflow before deleting it.", 409)
                    return
                shutil.rmtree(WORKFLOWS_DIR / name, ignore_errors=True)
                self._json({"ok": True})
            elif parsed.path == "/api/generate":
                workflow = _load_workflow(body["name"])
                self._json({"scripts": generate_scripts(workflow)})
            elif parsed.path == "/api/run":
                self._run(body)
            elif parsed.path == "/api/stop":
                name = _check_name(body["name"])
                with _runs_lock:
                    run = _runs.get(name)
                if run is None or run.status != "running":
                    self._error("Nothing is running for this workflow.", 409)
                    return
                run.stop()
                self._json({"ok": True})
            elif parsed.path == "/api/processes/stop":
                pid = int(body["pid"])
                result = stop_running_process(pid)
                if not result["ok"]:
                    self._error(result["error"], 400)
                    return
                self._json(result)
            else:
                self._error("Not found", 404)
        except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as err:
            self._error(str(err))

    def _run(self, body: dict) -> None:
        name = _check_name(body["name"])
        workflow = _load_workflow(name)
        only_stage = body.get("stage")
        with _runs_lock:
            existing = _runs.get(name)
            if existing is not None and existing.status == "running":
                self._error("This workflow is already running — stop it first.", 409)
                return
            run = WorkflowRun(workflow, only_stage=only_stage)
            _runs[name] = run
        threading.Thread(target=run.run, daemon=True, name=f"run-{name}").start()
        self._json({"ok": True})


# ---------------------------------------------------------------------------
# Running-processes panel — any lerobot-* job on this machine, not just the
# generalist merge above (training runs, other conversions/merges started by
# hand or via a previous workflow run). Answers "what's actually going on on
# this machine right now" from any device, without needing SSH access.
# ---------------------------------------------------------------------------

# (label, exact argv-token to match) — checked against each token of a
# process's real argv (from /proc/<pid>/cmdline, never truncated), not a
# substring search over a joined command string: a `--foo=/path/workflow_ui`
# argument to some unrelated command must not false-positive-match the
# "workflow-ui" entry the way a loose substring/regex search over the whole
# command line could.
_PROCESS_TOKENS: list[tuple[str, str]] = [
    ("train", "lerobot-train"),
    ("merge", "lerobot-cotrain-align"),
    ("umi-retarget", "lerobot-umi-retarget"),
    ("record", "lerobot-record"),
    ("eval", "lerobot-eval"),
    ("extract-subset", "lerobot-extract-subset"),
    ("sarm-annotate", "compute_rabc_weights"),
    ("workflow-ui", "workflow_ui/app.py"),
]
_PROCESS_BINARY_NAMES = {tok for _label, tok in _PROCESS_TOKENS}

# Flags worth pulling out per job label, so the panel shows a short human
# summary instead of a multi-hundred-argument raw command line.
_SUMMARY_FLAGS: dict[str, list[str]] = {
    "train": ["--dataset.repo_id", "--policy.type", "--output_dir", "--steps"],
    "merge": ["--target-repo-id"],
    "umi-retarget": ["--target-repo-id"],
    "record": ["--dataset.repo_id"],
    "eval": ["--policy.path", "--env.type"],
    "extract-subset": ["--target-repo-id"],
    "sarm-annotate": ["--dataset-repo-id"],
}

_CLK_TCK = os.sysconf("SC_CLK_TCK")


def _proc_cmdline(pid: int) -> list[str] | None:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    if not raw:
        return None  # empty cmdline = kernel thread
    return raw.decode("utf-8", errors="replace").rstrip("\x00").split("\x00")


def _proc_elapsed_seconds(pid: int) -> int | None:
    """Seconds since *pid* started, via /proc/<pid>/stat's starttime field
    (in clock ticks since boot) against /proc/uptime — avoids relying on
    `ps`'s etimes column, which needs its own subprocess call per query."""
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        after_comm = stat.rsplit(")", 1)[1].split()  # comm can contain "(" ")" " "
        starttime_ticks = int(after_comm[19])  # field 22 overall, index 19 after comm
        uptime_s = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, IndexError, ValueError):
        return None
    return int(uptime_s - starttime_ticks / _CLK_TCK)


def _format_elapsed(seconds: int) -> str:
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h or d:
        parts.append(f"{h}h")
    parts.append(f"{m}m")
    return "".join(parts)


def _extract_flag(argv: list[str], flag: str) -> str | None:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def running_processes_status() -> dict:
    """Every distinct lerobot-* job currently running, deduplicated so a
    job's `conda run` wrapper shell and its video-encoder worker forks (which
    share the job's exact argv tail — see align_single_dataset's
    ProcessPoolExecutor) don't show up as separate entries alongside the
    real invocation."""
    groups: dict[tuple[str, ...], list[tuple[int, list[str], bool]]] = {}
    try:
        pids = [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()]
    except OSError:
        pids = []

    for pid in pids:
        argv = _proc_cmdline(pid)
        if not argv:
            continue
        # Index of the first token that's one of our known binaries/scripts —
        # matched by exact equality or a path ending in it (argv[0] is often
        # a full interpreter path like ".../bin/lerobot-cotrain-align").
        match_idx = None
        for i, tok in enumerate(argv):
            if tok in _PROCESS_BINARY_NAMES or any(tok.endswith("/" + b) for b in _PROCESS_BINARY_NAMES):
                match_idx = i
                break
        if match_idx is None:
            continue
        binary = next(
            b for b in _PROCESS_BINARY_NAMES
            if argv[match_idx] == b or argv[match_idx].endswith("/" + b)
        )
        is_wrapper = "conda" in argv[0] and "run" in argv[:match_idx]
        key = (binary, *argv[match_idx + 1:])  # tail args identify the logical job
        groups.setdefault(key, []).append((pid, argv, is_wrapper))

    processes = []
    for (binary, *_tail), entries in groups.items():
        label = next(lbl for lbl, tok in _PROCESS_TOKENS if tok == binary)
        # Prefer the real interpreter invocation over the thin `conda run`
        # wrapper shell as the entry whose pid/elapsed we report.
        entries.sort(key=lambda e: e[2])  # non-wrapper (False) sorts first
        primary_pid, primary_argv, _ = entries[0]
        elapsed_s = _proc_elapsed_seconds(primary_pid) or 0
        summary_parts = [
            f"{flag.lstrip('-')}={val}"
            for flag in _SUMMARY_FLAGS.get(label, [])
            if (val := _extract_flag(primary_argv, flag))
        ]
        entry = {
            "label": label,
            "pid": primary_pid,
            "worker_count": len(entries) - 1,
            "elapsed": _format_elapsed(elapsed_s),
            "elapsed_s": elapsed_s,
            "summary": ", ".join(summary_parts) or " ".join(primary_argv)[:160],
        }
        # Only a known, fixed log path is exposed — never derived from argv,
        # so a process can't smuggle an arbitrary path into the log viewer.
        if label == "merge" and GENERALIST_MERGE_LOG.is_file():
            entry["log_path"] = str(GENERALIST_MERGE_LOG.relative_to(REPO_ROOT))
        processes.append(entry)
    processes.sort(key=lambda p: -p["elapsed_s"])
    return {"processes": processes}


def stop_running_process(pid: int) -> dict:
    """Send SIGTERM to *pid*'s whole process group, but only after
    re-confirming (right now, not from a stale list) that it's still one of
    our recognized lerobot-* jobs — this server has no auth and is reachable
    on the LAN, so this must never become a generic "kill any PID" endpoint.
    """
    argv = _proc_cmdline(pid)
    if not argv:
        return {"ok": False, "error": f"pid {pid} not found (already exited?)"}
    is_known = any(
        tok in _PROCESS_BINARY_NAMES or any(tok.endswith("/" + b) for b in _PROCESS_BINARY_NAMES)
        for tok in argv
    )
    if not is_known:
        return {"ok": False, "error": "refusing to stop a process that isn't a recognized lerobot job"}
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return {"ok": False, "error": "process already exited"}
    except PermissionError:
        return {"ok": False, "error": "permission denied (owned by another user?)"}
    return {"ok": True}


def _detect_lan_ip() -> str | None:
    """This machine's LAN-facing IP, for the startup banner's remote-access hint.

    Opens a UDP "connection" (no packet actually sent — UDP has no handshake)
    to a public address purely so the OS picks a real outbound interface/IP;
    works even fully offline since nothing is transmitted.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def main() -> None:
    global CONDA_ENV_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7799)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (keep loopback unless you trust the LAN — no auth).")
    parser.add_argument("--conda-env", default=CONDA_ENV_NAME,
                        help=f"Conda env generated lerobot-* stage scripts activate "
                             f"(default: {CONDA_ENV_NAME!r}).")
    args = parser.parse_args()

    CONDA_ENV_NAME = args.conda_env

    WORKFLOWS_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Co-training workflow UI → http://{args.host}:{args.port}")
    print(f"Workflows dir: {WORKFLOWS_DIR}")
    if args.host != "127.0.0.1":
        lan_ip = _detect_lan_ip()
        if lan_ip:
            print(f"Bound beyond loopback — reachable on your LAN at http://{lan_ip}:{args.port}")
            print("No auth on this server: only do this on a network you fully trust "
                  "(see GUIDE.md \"Remote access\" for phone/Tailscale setup).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
