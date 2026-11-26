#!/bin/bash
# Jetson: block-passing rollout against the policy server on PC1 (start the matching server first).
# usage: ~/run_blockpass.sh <pi05|jepa> [duration_s]
#   pi05 -> trained RTC (rtc_trained); jepa (Good-View / VLA-JEPA) -> plain chunks, append merge (JEPA_MERGE=aligned to override; no RTC support)
FAM=${1:?first argument: pi05 or jepa}; DUR=${2:-60}
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
LOG="$HOME/rollout_blockpass_${FAM}_$(date +%H%M).log"; echo "block passing | $FAM | ${DUR}s | log $LOG"
lerobot-rollout --strategy.type=base --interactive=true --fps=30 --duration="$DUR" --task="$TASK" \
  --robot.type=ob15_arms --robot.lift_height_mm=25 --robot.head_tilt=60 --robot.head_pan=0 --robot.max_relative_target=10 \
  --robot.camera_read_mode=latest --robot.camera_fresh_wait_ms=8 \
  --rename_map="$RENAME" \
  --inference.type=remote --inference.endpoint=${POLICY_ENDPOINT:-tcp/10.0.0.122:7447} --inference.deployment=ob15arms \
  --inference.semantics=ob15-arms-v1 $INF --inference.encoding=jpeg ${EXTRA_ARGS} \
  --return_to_initial_position=false 2>&1 | tee "$LOG"
