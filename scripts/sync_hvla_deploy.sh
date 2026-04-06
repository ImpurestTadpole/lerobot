#!/usr/bin/env bash
# Sync HVLA inference artifacts to PC2 (S2 VLM) and Jetson (S1 + robot).
#
# Prerequisites:
#   - SSH key auth works:  ssh PC2_USER@192.168.0.205 true
#                           ssh JETSON_USER@192.168.0.207 true
#   - Jetson default user is "jetson" (NVIDIA images). PC2 often matches your desktop login.
#
# Usage (from this machine, repo root optional):
#   ./scripts/sync_hvla_deploy.sh
#   JETSON_USER=owen ./scripts/sync_hvla_deploy.sh   # if your Jetson uses a different account

set -euo pipefail

PC2_IP="${PC2_IP:-192.168.0.205}"
JETSON_IP="${JETSON_IP:-192.168.0.207}"
PC2_USER="${PC2_USER:-owen}"
JETSON_USER="${JETSON_USER:-jetson}"

# Source paths on *this* machine (edit if your cache/output dirs differ)
S2_SRC="${S2_SRC:-$HOME/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch}"
S1_SRC="${S1_SRC:-$HOME/lerobot/outputs/train/tool_pickup_hvla_flow_v1/checkpoints/last}"
ROBOT_JSON="${ROBOT_JSON:-$HOME/.config/lerobot/robots/xlerobot.json}"

SSH_OPTS=(-o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)
# rsync remote receiver only mkdirs the final segment; parent ~/hvla_weights must exist.
remote_mkdir() {
  local user_host="$1"
  local dir="$2"
  ssh "${SSH_OPTS[@]}" "$user_host" "mkdir -p \"$dir\""
}

rsync_run() {
  rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" "$@"
}

echo "=== PC2 ($PC2_USER@$PC2_IP): S2 converted VLM (~7 GB) → ~/hvla_weights/s2/ ==="
if [[ ! -f "$S2_SRC/model.safetensors" ]]; then
  echo "ERROR: missing $S2_SRC/model.safetensors" >&2
  exit 1
fi
remote_mkdir "${PC2_USER}@${PC2_IP}" '$HOME/hvla_weights/s2'
rsync_run "$S2_SRC/" "${PC2_USER}@${PC2_IP}:~/hvla_weights/s2/"

echo ""
echo "=== Jetson ($JETSON_USER@$JETSON_IP): S1 checkpoint → ~/hvla_weights/s1/ ==="
if [[ ! -d "$S1_SRC/pretrained_model" ]]; then
  echo "ERROR: missing $S1_SRC/pretrained_model" >&2
  exit 1
fi
remote_mkdir "${JETSON_USER}@${JETSON_IP}" '$HOME/hvla_weights/s1'
rsync_run "$S1_SRC/" "${JETSON_USER}@${JETSON_IP}:~/hvla_weights/s1/"

if [[ -f "$ROBOT_JSON" ]]; then
  echo ""
  echo "=== Jetson: robot profile → ~/.config/lerobot/robots/ ==="
  ssh "${SSH_OPTS[@]}" "${JETSON_USER}@${JETSON_IP}" "mkdir -p ~/.config/lerobot/robots"
  rsync_run "$ROBOT_JSON" "${JETSON_USER}@${JETSON_IP}:~/.config/lerobot/robots/xlerobot.json"
else
  echo ""
  echo "SKIP: no local $ROBOT_JSON (create on Jetson or set ROBOT_JSON=...)"
fi

echo ""
echo "Done."
echo ""
echo "On PC2, S2 checkpoint path:"
echo "  ~/hvla_weights/s2/model.safetensors"
echo "On Jetson, S1 checkpoint path:"
echo "  ~/hvla_weights/s1   (pass as --s1-checkpoint ~/hvla_weights/s1)"
echo ""
echo "Set ZMQ IPs in Guide.sh Mode B:"
echo "  GPU_IP=192.168.0.205"
echo "  JET_IP=192.168.0.207"
