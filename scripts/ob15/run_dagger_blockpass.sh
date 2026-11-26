#!/bin/bash
# Jetson: DAgger on block passing against the policy server on PC1. Records only your corrections, locally. VR: RIGHT B = take over / hand back, LEFT MENU = stop.
# continue an existing correction dataset: DAGGER_NAME=<dataset name> ~/run_dagger_blockpass.sh <mode> <episodes to add>
# usage: ~/run_dagger_blockpass.sh <pi05|jepa> [num_corrections]
#   pi05 -> trained RTC (rtc_trained); jepa (Good-View / VLA-JEPA) -> plain chunks, append merge (JEPA_MERGE=aligned to override; no RTC support)
FAM=${1:?first argument: pi05 or jepa}; N=${2:-10}; NAME=Odog16/${DAGGER_NAME:-rollout_block_passing_dagger_${1}_$(date +%Y%m%d_%H%M)}; RESUME=""; [ -n "${DAGGER_NAME:-}" ] && RESUME="--resume=true"; BUTTONS='{"left.menu":"stop_session","right.b":"toggle_intervention"}'
TASK="picking block up with the right arm and passing it to the left and setting it down"
case "$FAM" in
  pi05) RENAME='{"observation.images.head":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}'
        INF="--inference.mode=rtc_trained" ;;
  pi05-smooth) RENAME='{"observation.images.head":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}'
        # plain chunks, blended over BLEND steps at each join (blending is not available with trained RTC)
        INF="--inference.mode=chunk --inference.chunk_merge=aligned --inference.blend_steps=${BLEND:-6} --inference.blend_components=[left_arm_shoulder_pan.pos,left_arm_shoulder_lift.pos,left_arm_elbow_flex.pos,left_arm_wrist_flex.pos,left_arm_wrist_roll.pos,left_arm_gripper.pos,right_arm_shoulder_pan.pos,right_arm_shoulder_lift.pos,right_arm_elbow_flex.pos,right_arm_wrist_flex.pos,right_arm_wrist_roll.pos,right_arm_gripper.pos]" ;;
  jepa) RENAME='{"observation.images.head":"observation.images.exterior_1_left","observation.images.right_wrist":"observation.images.exterior_2_left"}'
        INF="--inference.mode=chunk --inference.chunk_merge=${JEPA_MERGE:-append}" ;;
  jepa-smooth) RENAME='{"observation.images.head":"observation.images.exterior_1_left","observation.images.right_wrist":"observation.images.exterior_2_left"}'
        # aligned chunks, each join blended over BLEND steps (default 3)
        INF="--inference.mode=chunk --inference.chunk_merge=aligned --inference.blend_steps=${BLEND:-3} --inference.blend_components=[left_arm_shoulder_pan.pos,left_arm_shoulder_lift.pos,left_arm_elbow_flex.pos,left_arm_wrist_flex.pos,left_arm_wrist_roll.pos,left_arm_gripper.pos,right_arm_shoulder_pan.pos,right_arm_shoulder_lift.pos,right_arm_elbow_flex.pos,right_arm_wrist_flex.pos,right_arm_wrist_roll.pos,right_arm_gripper.pos]" ;;
  xvla) RENAME='{"observation.images.head":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}'
        INF="--inference.mode=chunk --inference.chunk_merge=aligned --inference.blend_steps=${BLEND:-6} --inference.blend_components=[left_arm_shoulder_pan.pos,left_arm_shoulder_lift.pos,left_arm_elbow_flex.pos,left_arm_wrist_flex.pos,left_arm_wrist_roll.pos,left_arm_gripper.pos,right_arm_shoulder_pan.pos,right_arm_shoulder_lift.pos,right_arm_elbow_flex.pos,right_arm_wrist_flex.pos,right_arm_wrist_roll.pos,right_arm_gripper.pos]" ;;
  molmoact2) RENAME='{}'
        INF="--inference.mode=chunk --inference.chunk_merge=aligned --inference.blend_steps=${BLEND:-6} --inference.blend_components=[left_arm_shoulder_pan.pos,left_arm_shoulder_lift.pos,left_arm_elbow_flex.pos,left_arm_wrist_flex.pos,left_arm_wrist_roll.pos,left_arm_gripper.pos,right_arm_shoulder_pan.pos,right_arm_shoulder_lift.pos,right_arm_elbow_flex.pos,right_arm_wrist_flex.pos,right_arm_wrist_roll.pos,right_arm_gripper.pos]" ;;
  *) echo "first argument must be pi05, pi05-smooth, jepa, jepa-smooth, xvla or molmoact2"; exit 1 ;;
esac
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate lerobot
cd "$HOME/lerobot"
echo "Camera preflight (about 12 s)..."; python "$HOME/cam_check.py" tune || { echo "Aborting: fix the cameras first (nothing was moved)."; exit 1; }
LOG="$HOME/dagger_blockpass_${FAM}_$(date +%H%M).log"; echo "DAgger block passing | $FAM | corrections -> $NAME (local, not pushed) | log $LOG"
lerobot-rollout --strategy.type=dagger --strategy.input_device=teleop --strategy.smooth_handover=true --strategy.record_autonomous=false --strategy.num_episodes="$N" \
  --teleop.type=xlerobot_vr --teleop.stream_cameras_to_vr=false --teleop.button_map="$BUTTONS" --fps=30 --interactive=false \
  --dataset.repo_id="$NAME" --dataset.root="$HOME/.cache/huggingface/lerobot/$NAME" --dataset.no_stamp=true --dataset.single_task="$TASK" --dataset.fps=30 --dataset.push_to_hub=false $RESUME \
  --robot.type=ob15_arms --robot.lift_height_mm=25 --robot.head_tilt=60 --robot.head_pan=0 --robot.max_relative_target=10 \
  --robot.camera_read_mode=latest --robot.camera_fresh_wait_ms=8 \
  --rename_map="$RENAME" \
  --inference.type=remote --inference.endpoint=${POLICY_ENDPOINT:-tcp/10.0.0.122:7447} --inference.deployment=ob15arms \
  --inference.semantics=ob15-arms-v1 $INF --inference.encoding=jpeg ${EXTRA_ARGS} \
  --return_to_initial_position=false 2>&1 | tee "$LOG"
