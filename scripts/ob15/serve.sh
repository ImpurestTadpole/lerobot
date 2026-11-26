#!/bin/bash
# PC1: start one block-passing policy server (stops any running one first).
# usage: ~/ob15_experiments/serve.sh <pi05|jepa_none|jepa_future>   (pinned-state checkpoints)
#        PERTURB=dim ~/ob15_experiments/serve.sh ...   stress test (see lerobot-policy-server-rec.py for the conditions)
#        add "orig" as a second argument for the original, unpinned checkpoint
M=${1:?usage: serve.sh <pi05|jepa_none|jepa_future> [orig]}
CFG=/home/owen/ob15_experiments/configs/server_blockpass_${M}_pinned.yaml
[ "$2" = orig ] && CFG=/home/owen/ob15_experiments/configs/server_blockpass_${M}.yaml
[ -f "$CFG" ] || { echo "no config $CFG"; exit 1; }
pkill -f lerobot-policy-server && sleep 3
cd /home/owen/lerobot_main || exit 1
echo "serving $CFG"
exec env -u PYTHONPATH LOG_RAW_STATE=1 PERTURB="${PERTURB:-}" PYTHONPATH=/home/owen/lerobot_main/src \
  /home/owen/lerobot/.venv-new/bin/python /home/owen/ob15_experiments/lerobot-policy-server-rec.py --config_path="$CFG"
