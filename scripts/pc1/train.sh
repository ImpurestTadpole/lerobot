#!/usr/bin/env bash
# Usage: MODEL=pi05 STAGE=probe DATASET=Odog16/generalist_ee_merged DATASET_ROOT=... scripts/pc1/train.sh [extra flags]
# Stage values override model values; extra flags are appended last and win.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${MODEL:?set MODEL (see config/train/models)}" "${STAGE:?set STAGE (see config/train/stages)}" "${DATASET:?set DATASET repo id}"
mapfile -t ARGS < <(env -u PYTHONPATH uv run python -I scripts/pc1/build_train_args.py)
exec env -u PYTHONPATH uv run lerobot-train "${ARGS[@]}" "$@"
