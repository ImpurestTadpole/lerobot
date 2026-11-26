#!/bin/bash
# Remote week: train each run in runs.json in order, evaluate every saved checkpoint offline, write its report,
# then start the next run. A failed or collapsed run is logged and skipped; the queue never stops on one run.
set -u
E=/home/owen/ob15_experiments/ob15_eval; R=${QUEUE_OUT:-/home/owen/goodview_runs/remote_week}; CODE=/home/owen/lerobot_rw
LOG=$E/queue.log; REG=$E/registry.csv
PY="env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH PYTHONPATH=$CODE/src HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/owen/lerobot/.venv-new/bin"
JCFG=/home/owen/goodview_runs/gv_blockpass_none_c30/checkpoints/last/pretrained_model/train_config.json
PCFG=/home/owen/pi05_block_passing/checkpoints/last/pretrained_model/train_config.json
EP=$(python3 -c "import json;print(json.dumps(json.load(open('$E/manifests/bp45_v1.json'))['train']).replace(' ',''))")
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
[ -f $REG ] || echo "run_id,start,end,status,git_commit,config_sha,manifest,seed,episodes,chunk,steps,checkpoint,train_hours,args" > $REG
busy() { pgrep -f "[l]erobot-policy-server|[b]in/lerobot-train" >/dev/null; }

run_one() {  # run_id kind seed args...
  local id=$1 kind=$2 seed=$3; shift 3; local args="$*" O=$R/$id
  if [ -e $O ]; then say "$id: $O exists, skipping"; return; fi
  while busy; do sleep 120; done
  local df=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); if [ "$df" -lt 40 ]; then say "$id: only ${df} GB free, skipping"; return; fi
  local cfg=$JCFG steps=40000 save=10000 extra=""
  if [ $kind = pi05 ]; then cfg=$PCFG; steps=20000; save=4000; extra="--sample_weighting=null"; fi
  if [ -n "${QUEUE_TEST:-}" ]; then steps=20; save=10; fi
  local start=$(date '+%F %T'); say "$id: training ($kind, seed $seed) $args"
  ( cd $CODE && $PY/lerobot-train --config_path=$cfg --dataset.episodes=$EP --output_dir=$O --resume=false --seed=$seed \
      --steps=$steps --save_freq=$save --log_freq=50 --job_name=$id $extra $args \
      --trackio.enable=true --trackio.project=ob15 --trackio.run_name=$id ) > $O.train.log 2>&1
  local rc=$? status=done
  if [ $rc != 0 ] || ! [ -d $O/checkpoints/last ]; then status=failed; say "$id: FAILED (rc $rc), see $O.train.log"; fi
  # collapse guard: a non-finite or exploding loss in the last logged steps
  if [ $status = done ] && tr '\r' '\n' < $O.train.log | grep -aoE " loss:[^ ]+" | tail -20 | grep -qiE "nan|inf"; then status=collapsed; say "$id: loss went non-finite"; fi
  local sha=$(python3 -c "import hashlib,sys;print(hashlib.sha1(open('$O/checkpoints/last/pretrained_model/train_config.json','rb').read()).hexdigest()[:10])" 2>/dev/null)
  echo "$id,$start,$(date '+%F %T'),$status,$(git -C $CODE rev-parse --short HEAD),$sha,bp45_v1,$seed,45,30,$steps,$O/checkpoints/last,$(python3 -c "import datetime as d;print(round((d.datetime.now()-d.datetime.fromisoformat('$start')).total_seconds()/3600,2))"),\"$args\"" >> $REG
  [ $status = done ] || return
  # keep weights of every checkpoint, optimizer state only for the last
  for c in $O/checkpoints/0*; do [ "$(readlink -f $O/checkpoints/last)" = "$(readlink -f $c)" ] || rm -rf $c/training_state; done
  for c in $O/checkpoints/0*; do
    say "$id: evaluating $(basename $c)"
    ( cd $CODE && $PY/python -W ignore $E/eval_run.py $c/pretrained_model $id --tag $(basename $c) ) > $E/results/${id}__$(basename $c).log 2>&1 || say "$id: eval of $(basename $c) failed"
  done
  ( cd $CODE && $PY/python $E/report.py $id ) >> $LOG 2>&1 || say "$id: report failed"
  say "$id: finished"
}

say "queue start, code $(git -C $CODE rev-parse --short HEAD)"
python3 -c "import json;[print(r['id'],r.get('kind','jepa'),' '.join(r['args'])) for r in json.load(open('$E/runs.json'))]" | while read -r id kind args; do
  if [ "$kind" = decide ]; then
    read -r id kind seed args < <(cd $CODE && $PY/python $E/report.py --decide-last)
    say "run 11 decided: $id ($kind) seed $seed $args"
    run_one $id $kind $seed $args < /dev/null
  else
    run_one $id $kind 1000 $args < /dev/null
  fi
  [ -n "${QUEUE_TEST:-}" ] && break
done
( cd $CODE && $PY/python $E/report.py --summary ) >> $LOG 2>&1
say "queue done"
