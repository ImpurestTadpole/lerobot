"""Join config/train/models/<MODEL>.yaml and config/train/stages/<STAGE>.yaml into lerobot-train flags.

Stage values override model values. ``${VAR}`` is expanded from the environment, and
``INIT_CKPT`` (if set) replaces ``policy.path``. Prints one ``--flag=value`` per line.
"""

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return os.path.expandvars(value)
    return str(value)


def main() -> None:
    model, stage = os.environ["MODEL"], os.environ["STAGE"]
    flags: dict = {}
    for path in (ROOT / "config/train/models" / f"{model}.yaml", ROOT / "config/train/stages" / f"{stage}.yaml"):
        flags.update(yaml.safe_load(path.read_text()) or {})
    if os.environ.get("INIT_CKPT"):
        flags["policy.path"] = os.environ["INIT_CKPT"]
    flags["dataset.repo_id"] = os.environ["DATASET"]
    if os.environ.get("DATASET_ROOT"):
        flags["dataset.root"] = os.environ["DATASET_ROOT"]
    flags["output_dir"] = f"outputs/{stage}_{model}"
    flags["job_name"] = f"{stage}_{model}"
    if "${DATASET_ROOT}" in str(flags.get("dim_masking.sources_path", "")) and not os.environ.get("DATASET_ROOT"):
        sys.exit("stage needs DATASET_ROOT (the merged dataset's local root)")
    for key, value in flags.items():
        print(f"--{key}={_fmt(value)}")


if __name__ == "__main__":
    main()
