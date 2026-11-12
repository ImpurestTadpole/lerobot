#!/usr/bin/env bash
# Smoke-test driver for LeRobot: trains a tiny ACT policy on a real HF dataset
# against the PushT simulator, checkpoints it, evaluates it in-sim (producing
# an MP4 rollout), then re-runs standalone eval against the saved checkpoint.
#
# This exercises the same CLI surface (`lerobot-train`, `lerobot-eval`) and
# extras (`lerobot[pusht]`) as `make test-act-ete-train`/`test-act-ete-eval`
# in the Makefile, but swaps env.type=aloha -> env.type=pusht so it only
# needs the lightweight `gym-pusht` extra (no mujoco/gym-aloha build).
#
# Usage: bash .claude/skills/run-lerobot/smoke.sh [output_dir]
# Run from the repo root with the project venv activated.
set -euo pipefail

OUT_DIR="${1:-tests/outputs/act_smoke}"
DEVICE="${DEVICE:-cpu}"

rm -rf "$OUT_DIR"

echo "==> [1/2] lerobot-train: 4 steps of ACT on lerobot/pusht, eval every 2 steps"
lerobot-train \
  --policy.type=act \
  --policy.dim_model=64 \
  --policy.n_action_steps=20 \
  --policy.chunk_size=20 \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --env.type=pusht \
  --env.episode_length=5 \
  --dataset.repo_id=lerobot/pusht \
  --dataset.image_transforms.enable=true \
  --dataset.episodes="[0]" \
  --batch_size=2 \
  --steps=4 \
  --env_eval_freq=2 \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --save_freq=2 \
  --save_checkpoint=true \
  --log_freq=1 \
  --wandb.enable=false \
  --output_dir="$OUT_DIR/"

CKPT="$OUT_DIR/checkpoints/000004/pretrained_model"
VIDEO="$OUT_DIR/eval/videos_step_000004/pusht_0/eval_episode_0.mp4"

test -f "$CKPT/model.safetensors" || { echo "FAIL: no checkpoint at $CKPT"; exit 1; }
test -f "$VIDEO" || { echo "FAIL: no eval video at $VIDEO"; exit 1; }
echo "==> checkpoint OK: $CKPT"
echo "==> rollout video OK: $VIDEO"

echo "==> [2/2] lerobot-eval: standalone eval against the saved checkpoint"
lerobot-eval \
  --policy.path="$CKPT" \
  --policy.device="$DEVICE" \
  --env.type=pusht \
  --env.episode_length=5 \
  --eval.n_episodes=1 \
  --eval.batch_size=1

echo "==> SMOKE TEST PASSED"
