# =============================================================================
# SETUP
# =============================================================================
# LeRobot requires Python >=3.12. Use a single conda environment (no venv).

# ONE-TIME: Create conda env with Python 3.12 and install lerobot
# conda create -n lerobot python=3.12 -y
# conda activate lerobot
# cd /home/owen/lerobot   # or your path
# pip install -e ".[dev]"

# Every session: activate conda only (do not activate .venv)
conda activate lerobot

# GPU required for inference on this device (training is done elsewhere).
# If lerobot-info shows "CUDA support?: False" but nvidia-smi works, run once (with lerobot active):
#   bash scripts/setup_cuda_env.sh
# Then open a NEW terminal (or run conda deactivate, then conda activate lerobot) so the hook runs.
# Verify: conda activate lerobot && lerobot-info  → should show CUDA True and GPU model.
# See also: docs/source/torch_accelerators.mdx (CUDA troubleshooting).

# Set USB permissions (run every time after reboot or USB reconnect)
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1

# =============================================================================
# CAMERA SETUP (ONE-TIME SETUP + RUN AFTER REBOOT)
# =============================================================================
# CRITICAL: Fix camera bandwidth issue by configuring MJPG format
# Without this, cameras default to YUYV (uncompressed) which causes:
#   - Control rate drops from 30 Hz to 12-25 Hz
#   - Severe USB bandwidth saturation
#   - Unstable robot control
#
# RUN THIS ONCE TO INSTALL UDEV RULE (persists across reboots):
./setup_camera_formats.sh
#
# OR MANUAL SETUP (run after every reboot if you don't install udev rule):
# v4l2-ctl -d /dev/video6 --set-fmt-video=width=640,height=480,pixelformat=MJPG
# v4l2-ctl -d /dev/video8 --set-fmt-video=width=640,height=480,pixelformat=MJPG
#
# Verify camera formats:
# v4l2-ctl -d /dev/video6 --get-fmt-video | grep "Pixel Format"
# v4l2-ctl -d /dev/video8 --get-fmt-video | grep "Pixel Format"
# v4l2-ctl -d /dev/video4 --get-fmt-video | grep "Pixel Format"
#
# Expected output:
#   video6: MJPG (Motion-JPEG)
#   video8: MJPG (Motion-JPEG)
#   video4: YUYV (RealSense uses native SDK, not V4L2) 

# =============================================================================
# GIT WORKFLOW - SYNC WITH GITHUB REPO
# =============================================================================
# Navigate to lerobot directory
cd /home/jetson/lerobot

# ONE-TIME SETUP (if not already done):
# git config --global user.name "ImpurestTadpole"
# git config --global user.email "your-email@example.com"
# git remote add origin https://github.com/ImpurestTadpole/lerobot.git

# UPDATE FROM GITHUB (pull latest changes):
# First, check if you have uncommitted changes:
git status

# OPTION A: Commit your local changes first, then pull:
git add .
git commit -m "Save local changes before pulling"
git config pull.rebase false  # Set merge strategy (one-time)
git pull origin main          # Merge remote changes into local

# OPTION B: Stash your changes, pull, then reapply:
git stash                      # Temporarily save your changes
git config pull.rebase false  # Set merge strategy (one-time)
git pull origin main          # Merge remote changes into local
git stash pop                 # Reapply your stashed changes

# If merge conflicts occur after pull:
# 1. Resolve conflicts in the files git lists (look for <<<<<<< markers)
# 2. git add .
# 3. git commit -m "Resolve merge conflicts"
# 4. git push origin main

# PUSH LOCAL CHANGES TO GITHUB:
git add .                     # Stage all changes
git commit -m "Your commit message here"
git push origin main          # Push to GitHub

# CHECK STATUS:
git status                    # See what files have changed
git log --oneline            # See commit history
git remote -v                 # Verify remote URL

# =============================================================================
# TELEOPERATION
# =============================================================================
# ON JETSON:
lerobot-teleoperate \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --display_data=true


# -----------------------------------------------------------------------------
# LIFT AXIS (gantry) - activation, calibration, control
# -----------------------------------------------------------------------------
# Activation: Set robot.lift_axis.enabled=true (CLI above or in config).
# Hardware:   Lift motor must be on bus2 (port2) at motor_id=9 (default).
#
# Calibration: When enabled, during robot calibrate() you are prompted to home
#              the lift. Homing drives the axis down until stall, then sets
#              that position as 0 mm. You can skip with 's' if not ready.
#              (The lift motor is now registered on the bus at construction
#              so bus operations like disable_torque and homing work.)
#
# Control:     - Target height (mm): action["gantry.height_mm"] = target_mm
#              - Direct velocity:    action["gantry.vel"] = velocity (raw)
#              VR: Left controller thumbstick UP/DOWN = lift up/down.
#
# Recording:   Observations: gantry.height_mm, gantry.vel
#              Actions:      gantry.height_mm and/or gantry.vel (same keys)

# =============================================================================
# REMOTE VISUALIZATION (ON EXTERNAL PC)
# =============================================================================

# OPTION 1: Direct connection (recommended - best performance)
rerun --serve-web --web-viewer-port 9090 --connect "rerun+http://172.20.10.2:9876/proxy"

rerun --serve-web --web-viewer-port 9090 --connect "rerun+http://192.168.0.205:9876/proxy"

rerun --serve-web --web-viewer-port 9090 --connect "rerun+http://10.249.40.136:9876/proxy"

# OPTION 2: Via SSH tunnel
# Terminal 1:
ssh -L 9876:localhost:9876 jetson@192.168.0.104
# Terminal 2:
rerun --serve-web --web-viewer-port 9090 --connect "rerun+http://localhost:9876/proxy"

# Then open in browser: http://localhost:9090

# NOTE: Install rerun via pip (not snap):
pip3 install rerun-sdk

# Make Rerun streaming low-latency / live (optional; defaults are already tuned):
#   export RERUN_FLUSH_TICK_SECS=0.008   # 8ms flush (default). Use 0.002 for minimal latency.
#   export RERUN_LOG_FREQUENCY=1        # Log every frame (default). Do not increase if you want live view.
#   export RERUN_DOWNSAMPLE_FACTOR=0.33 # Faster encode on motion-heavy frames (320x180 -> 213x120). Optional.
# Rerun compression is now always on in code; for record/teleop you can still pass:
#   --display_compressed_images=true   # Redundant but explicit; avoids raw frames if logic ever changes.




# =============================================================================
# HUGGINGFACE SETUP (ONE-TIME)
# =============================================================================
# Login to HuggingFace (will prompt for token)
huggingface-cli login

# =============================================================================
# RECORDING
# =============================================================================
# VR CONTROLLER CONTROLS:
# Thumbstick UP    → Stop recording
# Thumbstick LEFT  → Re-record current episode
# Thumbstick RIGHT → Save episode & move to next
# Thumbstick DOWN  → Reset robot position

# RECORDING WITH AUTO-PUSH TO HUB (recommended):
# NOTE: Remove --resume=true for new datasets or when robot configuration has changed
# Use --resume=true only when continuing an existing compatible dataset
lerobot-record \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/test_transfer_block \
    --dataset.single_task="transfer the block" \
    --dataset.num_episodes=5 \
    --dataset.fps=30 \
    --display_data=true \
    --dataset.push_to_hub=true
    #--resume=true 

lerobot-record \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/making_coffee  \
    --dataset.single_task="take coffee from blue place it in the machine, then place it on the yellow." \
    --dataset.num_episodes=15 \
    --dataset.fps=30 \
    --display_data=true \
    --dataset.push_to_hub=true \
    --resume=true 




lerobot-record \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/trash_pickup  \
    --dataset.single_task="pick up the can and place it in the trash bin" \
    --dataset.num_episodes=10 \
    --dataset.fps=30 \
    --display_data=true \
    --dataset.push_to_hub=false \
    --resume=true


# Rerun live-view tuning — set before ANY lerobot-record call.
# server_memory_limit is now 200MB in code (was 55% = ~4.4GB on 8GB Jetson).
# A large buffer causes the viewer to replay gigabytes of old data when it
# (re)connects, appearing frozen/laggy. 200MB ≈ last ~1 min of compressed frames.
# RERUN_LOG_FREQUENCY=2  → log every other frame (halves viz CPU load)
# RERUN_DOWNSAMPLE_FACTOR=0.2 → 72×128px images sent over WiFi instead of 360×640
# RERUN_JPEG_QUALITY=55  → faster encode, smaller network payload
export RERUN_LOG_FREQUENCY=2
export RERUN_DOWNSAMPLE_FACTOR=0.2
export RERUN_JPEG_QUALITY=55

lerobot-record \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/block_sorting  \
    --dataset.single_task="sort the blocks by color" \
    --dataset.num_episodes=20 \
    --dataset.fps=30 \
    --display_data=true \
    --dataset.push_to_hub=false \
    --resume=true 

# Session 1: Red block (62 episodes) X4 sessions
lerobot-record \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/block_sorting_single \
    --dataset.single_task="Pick up the red block and place it in the red bowl" \
    --dataset.num_episodes=14 \
    --display_data=true \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --resume=true 

# Session 2: Blue block (62 episodes) X4 sessions
lerobot-record \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/block_sorting_single \
    --dataset.single_task="Pick up the blue block and place it in the blue bowl" \
    --dataset.num_episodes=16 \
    --display_data=true \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --resume=true 


# Session 3: Green block (62 episodes) X4 sessions
lerobot-record \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/block_sorting_single \
    --dataset.single_task="Pick up the green block and place it in the green bowl" \
    --dataset.num_episodes=16 \
    --display_data=true \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --resume=true 
# Session 4: Yellow block (62 episodes) X4 sessions
lerobot-record \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/block_sorting_single \
    --dataset.single_task="Pick up the yellow block and place it in the yellow bowl" \
    --dataset.num_episodes=16 \
    --display_data=true \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --resume=true 


# RECORDING LOCAL ONLY (push manually later):
lerobot-record \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/tool_pickup \
    --dataset.single_task="pick up the tools from the table and place it in the red bin" \
    --dataset.num_episodes=5 \
    --dataset.fps=30 \
    --display_data=true \
    --dataset.push_to_hub=false \
    --resume=true

# Manually push to hub after recording:
# HF_DATASETS_CACHE=/tmp bypasses any corrupt Arrow cache from a previous interrupted load
HF_DATASETS_CACHE=/tmp/hf_datasets_tmp python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
    dataset = LeRobotDataset('Odog16/trash_pickup'); \
    dataset.push_to_hub()"

# Local data location: ~/.cache/huggingface/lerobot/Odog16/ob15_test_1/
# Parquet files: data/chunk-000/episode_*.parquet
# Videos: videos/chunk-000/observation.images.*/episode_*.mp4
# Push edited local dataset (set HF_TOKEN in env or `huggingface-cli login`; never commit tokens):
# conda run -n lerobot bash -c 'export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN first}"; python -c "
# from pathlib import Path
# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# ds = LeRobotDataset(\"Odog16/trash_pickup\", root=Path.home() / \".cache/huggingface/lerobot/Odog16/trash_pickup\")
# ds.push_to_hub()
# "'

# Delete local dataset (if you want to start fresh):
rm -rf ~/.cache/huggingface/lerobot/Odog16/tool_pickup

# =============================================================================
# PERFORMANCE TIPS
# =============================================================================
# Camera resolution is set to 320x240 for ~30Hz control rate
# To change resolution, edit: src/lerobot/robots/xlerobot/config_xlerobot.py
# - 320x240: ~30 Hz (faster control)
# - 640x480: ~15 Hz (higher quality)
#
# VR CONTROLLER EVENTS (Left Controller):
# - Thumbstick RIGHT: Save episode & move to next
# - Thumbstick LEFT: Re-record current episode
# - Thumbstick UP: Stop recording completely
# - Thumbstick DOWN: Reset robot to zero position

# =============================================================================
# MERGE & PUSH DATASETS (co_training_utils.py)
# =============================================================================
#
# End-to-end: each source either passes through (meta already matches merge schema) or is
# aligned into _align_tmp_<repo> (cached), then aggregate_datasets() writes the merged
# tree to --output-root (default: ~/.cache/huggingface/lerobot/<target-repo-id> if omitted).
#
# Hugging Face (datasets are repo_type=dataset):
#   huggingface-cli login
#
# Before a big merge, sanity-check sources:
#   ... same python command ... --inspect-only
#
# Multi-source merge rules (see module docstring in co_training_utils.py):
#   - Cameras: intersection of remapped observation.images.* (depth etc. only if all have it).
#   - Resolution: min H×W per key, or --target-image-size HxW.
#   - Use --match-features-from (and optional --match-features-root) so state/action
#     joint names match; aggregate_datasets compares full feature metadata.
#
# Common flags:
#   --skip-missing-sources     skip Hub-404 / missing local cache repos
#   --force-rebuild            wipe _align_tmp_* and output when options changed
#   --realign-all-sources      always re-encode into _align_tmp_* (default: pass through
#                              sources whose meta already matches the merge schema)
#   --push-to-hub true|false   upload merged dataset after merge (needs meta/tasks.parquet)
#
# If you merged with --push-to-hub false, push later from Python (same root as output):
#   python -c "from pathlib import Path; from lerobot.datasets.lerobot_dataset import LeRobotDataset; r='Odog16/making_coffee_v1'; LeRobotDataset(r, root=Path('~/.cache/huggingface/lerobot').expanduser()/r).push_to_hub()"
#
# Policy checkpoints are not datasets: use lerobot-train --policy.push_to_hub or upload_checkpoints.py.
#

# All Odog16 xlerobot task folders under ~/.cache/huggingface/lerobot/Odog16 (no ALOHA,
# no _align_tmp_*). Target name is arbitrary; tune --source-repos if you want to drop
# near-duplicates (e.g. making_coffee vs making_coffee_v1, trash_pickup vs trash_pickup_merged).
python src/lerobot/data_processing/co_training_utils.py \
    --source-repos \
        Odog16/2_23_test_1 \
        Odog16/block_sorting \
        Odog16/block_sorting_single \
        Odog16/making_coffee \
        Odog16/making_coffee_v1 \
        Odog16/ob15_packing_box \
        Odog16/test_transfer_block \
        Odog16/tool_pickup \
        Odog16/trash_pickup \
        Odog16/trash_pickup_merged \
        Odog16/trash_pickup_with_rabc \
    --target-repo-id Odog16/xlerobot_generalist_v1 \
    --target-fps 30 \
    --target-state-dim 18 \
    --target-action-dim 18 \
    --match-features-from Odog16/trash_pickup \
    --camera-remap "head:head,left_wrist:left_wrist,right_wrist:right_wrist" \
    --output-root $HOME/.cache/huggingface/lerobot/Odog16/xlerobot_generalist_v1 \
    --robot-type xlerobot \
    --skip-missing-sources \
    --push-to-hub true

# Odog16/ob15_general_dataset_v1 — merge eight xlerobot task datasets, then push:
#   - Joint names from Odog16/trash_pickup (--match-features-from).
#   - Identity camera remap (already head / left_wrist / right_wrist).
#   - Multi-source rules: shared cams only, min H×W, one fps (see co_training_utils docstring).
#   - Sources that already match that merge schema skip _align_tmp_* (no re-encode); add
#     --realign-all-sources to force full re-align for every repo.
#   - Missing Hub/local repos are skipped (--skip-missing-sources). Packing task: use
#     Odog16/ob15_packing_box or Odog16/packing_box depending on which repo exists.
#   - Add --force-rebuild if you changed remap, fps, or sources and need a clean merge.
    python src/lerobot/data_processing/co_training_utils.py \
        --source-repos \
            Odog16/trash_pickup \
            Odog16/tool_pickup \
            Odog16/block_sorting_single \
            Odog16/test_transfer_block \
            Odog16/making_coffee \
            Odog16/block_sorting \
            Odog16/2_23_test_1 \
            Odog16/ob15_packing_box \
        --target-repo-id Odog16/ob15_general_dataset_v1 \
        --target-fps 30 \
        --target-state-dim 18 \
        --target-action-dim 18 \
        --match-features-from Odog16/trash_pickup \
        --camera-remap "head:head,left_wrist:left_wrist,right_wrist:right_wrist" \
        --output-root $HOME/.cache/huggingface/lerobot/Odog16/ob15_general_dataset_v1 \
        --robot-type xlerobot \
        --skip-missing-sources \
        --realign-all-sources \
        --force-rebuild \
        --push-to-hub true

# Single-task realign + upload dataset to Hub:
python src/lerobot/data_processing/co_training_utils.py \
    --source-repos Odog16/making_coffee \
    --target-repo-id Odog16/making_coffee_v1 \
    --target-fps 30 \
    --target-state-dim 18 \
    --target-action-dim 18 \
    --match-features-from Odog16/trash_pickup \
    --camera-remap "head:head,left_wrist:left_wrist,right_wrist:right_wrist" \
    --output-root $HOME/.cache/huggingface/lerobot/Odog16/making_coffee_v1 \
    --robot-type xlerobot \
    --push-to-hub true

# =============================================================================
# MAKING COFFEE V1 — SmolVLA
# =============================================================================
# Train on the aligned dataset (xlerobot schema: 3 cams, 18-D state/action).
# Local root (matches default cache layout for this repo_id).
# Use $HOME (not ~) in --dataset.root: draccus passes the string through to Python,
# which does not expand tilde unless you use Path.expanduser() (lerobot does that now).
#   $HOME/.cache/huggingface/lerobot/Odog16/making_coffee_v1
# Warm-start from lerobot/smolvla_base; swap --policy.pretrained_path for a Hub
# checkpoint (e.g. Odog16/block_sorting_SmolVLA_SARM_v4_15k) if you prefer.
# Training uses --tolerance_s=0.02 by default (parquet time vs video PTS); merged AV1
# datasets often drift a few ms — if you still see FrameTimestampError, try 0.04.
unset PYTHONPATH

lerobot-train \
    --dataset.repo_id=Odog16/making_coffee_v1 \
    --dataset.root=$HOME/.cache/huggingface/lerobot/Odog16/making_coffee_v1 \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/making_coffee_v1_SmolVLA_v1 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=35 \
    --policy.n_action_steps=35 \
    --output_dir=outputs/train/making_coffee_v1_SmolVLA_v1 \
    --job_name=making_coffee_v1_SmolVLA_v1 \
    --batch_size=16 \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=1e-4 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=18000 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot


# =============================================================================
# TRAINING POLICY (Run on PC with GPU)
# =============================================================================
lerobot-train \
    --dataset.repo_id=Odog16/test_transfer_block \
    --policy.type=act \
    --output_dir=outputs/train/act_test_transfer_block \
    --job_name=act_test_transfer_block \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id=Odog16/test_transfer_block_policy

# Train excluding depth: same ACT inputs as normal (multiple RGB images + robot state;
# see https://huggingface.co/docs/lerobot/act) but omit observation.images.head_depth:
lerobot-train \
    --dataset.repo_id=Odog16/test_transfer_block \
    --policy.type=act \
    --policy.input_features='{"observation.images.head": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.images.left_wrist": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.images.right_wrist": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.state": {"type": "STATE", "shape": [17]}}' \
    --output_dir=outputs/train/act_test_transfer_block_no_depth \
    --job_name=act_test_transfer_block_no_depth \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id=Odog16/test_transfer_block_policy_no_depth

# Resume training from checkpoint:
lerobot-train \
    --config_path=outputs/train/act_xlerobot/checkpoints/last/pretrained_model/train_config.json \
    --resume=true

unset PYTHONPATH


# NEXT: Step 2 (optional) then Step 3. Re-failures only: add --episodes … --skip-existing. If GPU OOM: --extract-fps 1, stop other GPU users, or PYTORCH_ALLOC_CONF=expandable_segments:True.
#
# 2) Train SARM reward model (dual annotations)
 lerobot-train \
     --dataset.repo_id=Odog16/trash_pickup \
     --policy.type=sarm \
     --policy.annotation_mode=dual \
     --policy.image_key=observation.images.head \
     --policy.state_key=observation.state \
     --policy.n_obs_steps=8 \
     --policy.frame_gap=10 \
     --policy.drop_n_last_frames=200 \
     --output_dir=outputs/train/trash_pickup_sarm_dual_v4 \
     --batch_size=32 \
     --steps=5000 \
     --save_freq=2500 \
     --log_freq=100 \
     --num_workers=1 \
     --wandb.enable=true \
     --wandb.project=lerobot \
     --policy.push_to_hub=false \
     --policy.repo_id=Odog16/trash_pickup_sarm_dual_v4


# NEXT: Step 4 (optional) then Step 5a. Reward path for RA-BC: .../checkpoints/005000/pretrained_model
#
# Optional Step 4 — SARM prediction curves (not VLM subtask PNGs)
 python src/lerobot/policies/sarm/compute_rabc_weights.py \
     --dataset-repo-id Odog16/trash_pickup \
     --reward-model-path outputs/train/trash_pickup_sarm_dual_v2/checkpoints/005000/pretrained_model \
     --tolerance-s 0.02 \
     --visualize-only \
     --num-visualizations 5 \
     --head-mode both \
     --output-dir ./trash_pickup_sarm_predictions
# NEXT: Step 5a.
#
# 3) Step 5a — precompute SARM progress (RA-BC weights; use 0.02 tolerance for this dataset)
# python src/lerobot/policies/sarm/compute_rabc_weights.py \
#     --dataset-repo-id Odog16/trash_pickup \
#     --reward-model-path outputs/train/trash_pickup_sarm_dual_v2/checkpoints/005000/pretrained_model \
#     --tolerance-s 0.02 \
#     --head-mode both \
#     --num-visualizations 0 \
#     --output-dir ./trash_pickup_sarm_predictions \
#     --stride 5
# NEXT: Step 5b. Parquet: <dataset.root>/sarm_progress.parquet (see comment after Step 5a blocks below).
#
# 4) Step 5b — SmolVLA + RA-BC (--rabc_head_mode sparse matches dual sparse head in parquet)
# lerobot-train \
#     --dataset.repo_id=Odog16/trash_pickup \
#     --tolerance_s=0.02 \
#     --policy.type=smolvla \
#     --policy.repo_id=Odog16/trash_pickup_SmolVLA_v3rabc \
#     --policy.push_to_hub=false \
#     --policy.pretrained_path=lerobot/smolvla_base \
#     --policy.train_expert_only=true \
#     --policy.train_state_proj=true \
#     --policy.use_amp=true \
#     --policy.num_vlm_layers=16 \
#     --policy.chunk_size=42 \
#     --policy.n_action_steps=42 \
#     --use_rabc=true \
#     --rabc_head_mode=sparse \
#     --rabc_kappa=0.01 \
#     --output_dir=outputs/train/trash_pickup_SmolVLA_v3rabc \
#     --job_name=trash_pickup_SmolVLA_v3rabc \
#     --batch_size=16 \
#     --steps=20000 \
#     --save_freq=5000 \
#     --log_freq=200 \
#     --scheduler.type=cosine_decay_with_warmup \
#     --scheduler.peak_lr=1e-4 \
#     --scheduler.decay_lr=2.5e-6 \
#     --scheduler.num_warmup_steps=1000 \
#     --scheduler.num_decay_steps=18000 \
#     --wandb.enable=true \
#     --wandb.project=lerobot
# If output_dir exists: --config_path=.../checkpoints/last/pretrained_model/train_config.json --resume=true
# If you have a task SmolVLA v1 on the Hub, swap pretrained_path; else keep lerobot/smolvla_base.
#
# -----------------------------------------------------------------------------
# Step 1 — Subtask annotation (HF doc “Step 1: Subtask Annotation”)
# -----------------------------------------------------------------------------
#
# Generic dual example from the doc (swap repo, subtasks, video-key):
#   python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
#     --repo-id your-username/your-dataset \
#     --sparse-subtasks "Bring arms up from starting position,Fold the towel (3 folds in total)" \
#     --dense-subtasks "Bring robot arms up from starting position,Grab near side and do 1st fold,Grab side and do 2nd fold,Grab side and do 3rd fold to finish folding" \
#     --video-key observation.images.base \
#     --num-workers 4 \
#     --push-to-hub
#
# trash_pickup_merged — dual (sparse + dense). Add --push-to-hub to sync labels to the Hub repo.
python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
    --repo-id Odog16/trash_pickup \
    --sparse-subtasks "navigate to object,descend gantry,grasp object,lift object,navigate to bin,place in bin" \
    --dense-subtasks "turn to object,drive toward object,align in front of object,descend gantry to grasp height,grasp object,lift object clear,navigate toward bin,position over bin,place object in bin" \
    --video-key observation.images.head \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --extract-fps 2 \
    --max-retries 6 \
    --max-new-tokens 4096 \
    --sample \
    --num-visualizations 5 \
    --output-dir ./subtask_viz

python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
    --repo-id Odog16/tool_pickup \
    --sparse-subtasks "turn and drive to table,stop at table, pick up the tool left arm, pick up the box with the right arm, place the tool in the red bin" \
    --dense-subtasks "turn to the table, drive toward the table, align in front of the table, pick up the tool left arm, pick up the box with the right arm, place the tool in the red bin" \
    --video-key observation.images.head \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --extract-fps 2 \
    --max-retries 6 \
    --max-new-tokens 4096 \
    --sample \
    --num-visualizations 5 \
    --output-dir ./tool_pickup_subtask_viz

python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
    --repo-id Odog16/block_sorting_single \
    --sparse-subtasks "drive to the table, pick up the block, put the block in the same color bowl" \
    --dense-subtasks "turn to the table, drive toward the table, align in front of the table, find the block, pick up the block, find the bowl with the same color as the block, place the block in the bowl" \
    --video-key observation.images.base \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --num-workers 4 \
    --extract-fps 2 \
    --max-retries 6 \
    --max-new-tokens 4096 \
    --num-visualizations 5 \
    --output-dir ./block_sorting_single_subtask_viz


# -----------------------------------------------------------------------------
# Step 2 — Verify annotations (HF doc “Step 2: Verify Annotations”)
# -----------------------------------------------------------------------------
python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
    --repo-id Odog16/tool_pickup\
    --visualize-only \
    --visualize-type both \
    --num-visualizations 5 \
    --video-key observation.images.head \
    --output-dir ./tool_pickup_subtask_viz

python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
    --repo-id Odog16/trash_pickup\
    --visualize-only \
    --visualize-type both \
    --num-visualizations 5 \
    --video-key observation.images.head \
    --output-dir ./trash_pickup_subtask_viz



python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
    --repo-id Odog16/tool_pickup\
    --visualize-type both \
    --num-visualizations 5 \
    --video-key observation.images.head \
    --output-dir ./tool_pickup_subtask_viz




# Optional: inspect dataset-level priors (after Step 1):
python3 -c "
import json, pathlib
root = pathlib.Path.home() / '.cache/huggingface/lerobot/hub/datasets--Odog16--trash_pickup/snapshots/6cb7090be810934cc427a30e86d5f4638c346410/meta'
for name in ('temporal_proportions_sparse.json', 'temporal_proportions_dense.json'):
    p = root / name
    if not p.is_file():
        print(name, ': (missing)')
        continue
    d = json.loads(p.read_text())
    vals = list(d.values())
    print(name, '| stages:', list(d.keys()))
    print('  proportions:', [round(v, 3) for v in vals], '| sum', round(sum(vals), 4))
"

# -----------------------------------------------------------------------------
# Step 3 — Train SARM (HF doc “Step 3: Train SARM”, annotation_mode=dual)
# -----------------------------------------------------------------------------
lerobot-train \
    --dataset.repo_id=Odog16/block_sorting_single \
    --policy.type=sarm \
    --policy.annotation_mode=dual \
    --policy.image_key=observation.images.head \
    --policy.state_key=observation.state \
    --policy.n_obs_steps=8 \
    --policy.frame_gap=10 \
    --policy.drop_n_last_frames=200 \
    --output_dir=outputs/train/block_sorting_single_sarm_dual \
    --batch_size=32 \
    --steps=5000 \
    --save_freq=2500 \
    --log_freq=100 \
    --num_workers=1 \
    --wandb.enable=true \
    --wandb.project=lerobot \
    --policy.push_to_hub=false \
    --policy.repo_id=Odog16/block_sorting_single_sarm_dual

lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=sarm \
    --policy.annotation_mode=dual \
    --policy.image_key=observation.images.head \
    --policy.state_key=observation.state \
    --policy.n_obs_steps=8 \
    --policy.frame_gap=10 \
    --policy.drop_n_last_frames=200 \
    --output_dir=outputs/train/trash_pickup_sarm_dual_v2 \
    --batch_size=32 \
    --steps=5000 \
    --save_freq=2500 \
    --log_freq=100 \
    --num_workers=1 \
    --wandb.enable=true \
    --wandb.project=lerobot \
    --policy.push_to_hub=false \
    --policy.repo_id=Odog16/trash_pickup_sarm_dual_v2


# Alternative — sparse_only labels: --policy.annotation_mode=sparse_only,
#   --output_dir=outputs/train/tool_pickup_sarm, --policy.repo_id=Odog16/tool_pickup_sarm
#   Point Steps 4–5a --reward-model-path at that run’s .../005000/pretrained_model.
# trash_pickup RA-BC chain: keep Steps 4–5a–5b aligned with the dual SARM run at
#   outputs/train/trash_pickup_sarm_dual_v2/checkpoints/005000/pretrained_model (not an older ..._dual/ path).

# -----------------------------------------------------------------------------
# Step 4 — Visualize predictions (HF doc “Step 4: Visualize Predictions”)
# -----------------------------------------------------------------------------
# matplotlib: if kiwisolver._cext fails → unset PYTHONPATH (Isaac Sim PYTHONPATH).
python src/lerobot/policies/sarm/compute_rabc_weights.py \
    --dataset-repo-id Odog16/tool_pickup \
    --reward-model-path outputs/train/trash_pickup_sarm_dual_v2/checkpoints/005000/pretrained_model \
    --tolerance-s 0.02 \
    --visualize-only \
    --num-visualizations 5 \
    --head-mode both \
    --output-dir ./sarm_predictions

# trash_pickup — reward model: trash_pickup_sarm_dual_v2 (Step 3 .../005000/pretrained_model).
python src/lerobot/policies/sarm/compute_rabc_weights.py \
    --dataset-repo-id Odog16/trash_pickup \
    --reward-model-path outputs/train/block_sorting_single_sarm_dual/checkpoints/005000/pretrained_model \
    --tolerance-s 0.02 \
    --visualize-only \
    --num-visualizations 5 \
    --head-mode both \
    --output-dir ./block_sorting_single_sarm_predictions



# -----------------------------------------------------------------------------
# Step 5 (optional) — Train policy with RA-BC (HF doc “Step 5”)
# -----------------------------------------------------------------------------

# Step 5a — Compute SARM progress values (HF “Step 5a: Compute SARM Progress Values”)
# Writes sarm_progress.parquet; uploads to dataset repo by default (--push-to-hub).
python src/lerobot/policies/sarm/compute_rabc_weights.py \
    --dataset-repo-id Odog16/tool_pickup \
    --reward-model-path outputs/train/tool_pickup_sarm_dual/checkpoints/005000/pretrained_model \
    --tolerance-s 0.02 \
    --head-mode both \
    --num-visualizations 0 \
    --output-dir ./sarm_predictions \
    --stride 5

python src/lerobot/policies/sarm/compute_rabc_weights.py \
    --dataset-repo-id Odog16/trash_pickup \
    --reward-model-path outputs/train/block_sorting_single_sarm_dual/checkpoints/005000/pretrained_model \
    --tolerance-s 0.02 \
    --num-visualizations 0 \
    --head-mode both \
    --output-dir ./block_sorting_single_sarm_predictions \
    --stride 5

# Local sarm_progress (after Step 5a): <dataset.root>/sarm_progress.parquet — usually
#   ~/.cache/huggingface/lerobot/hub/datasets--Odog16--<repo_name>/snapshots/<hash>/sarm_progress.parquet
# trash_pickup Step 5a must use --reward-model-path .../trash_pickup_sarm_dual_v2/... so RA-BC matches that SARM.
# Hub: hf://datasets/<repo>/sarm_progress.parquet (after --push-to-hub)

# Step 5b — Train policy with RA-BC (HF “Step 5b: Train Policy with RA-BC”)
# --rabc_head_mode: sparse or dense (must exist in parquet when using dual SARM).
lerobot-train \
    --dataset.repo_id=Odog16/tool_pickup \
    --tolerance_s=0.02 \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/tool_pickup_SmolVLA_v3rabc\
    --policy.push_to_hub=false \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=42 \
    --policy.n_action_steps=42 \
    --use_rabc=true \
    --rabc_head_mode=sparse \
    --rabc_kappa=0.01 \
    --output_dir=outputs/train/tool_pickup_SmolVLA_v3rabc \
    --job_name=tool_pickup_SmolVLA_v3rabc \
    --batch_size=16 \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=200 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=1e-4 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=18000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# trash_pickup SmolVLA RA-BC: uses sarm_progress from Step 5a (reward: trash_pickup_sarm_dual_v2).
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --tolerance_s=0.02 \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/trash_pickup_SmolVLA_v3rabc\
    --policy.push_to_hub=false \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=42 \
    --policy.n_action_steps=42 \
    --use_rabc=true \
    --rabc_head_mode=sparse \
    --rabc_kappa=0.01 \
    --output_dir=outputs/train/trash_pickup_SmolVLA_v3rabc \
    --job_name=trash_pickup_SmolVLA_v3rabc \
    --batch_size=16 \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=200 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=1e-4 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=18000 \
    --wandb.enable=true \
    --wandb.project=lerobot



# Push SmolVLA checkpoint (import path is lerobot.policies.smolvla, not lerobot.common):
# python3 -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; \
# p=SmolVLAPolicy.from_pretrained('outputs/train/trash_pickup_SmolVLA_v2_rabc/checkpoints/020000/pretrained_model'); \
# p.push_to_hub('Odog16/trash_pickup_SmolVLA_v2_rabc_20k')"

# Resume Step 5b: lerobot-train --config_path=outputs/train/trash_pickup_SmolVLA_v2_rabc/checkpoints/last/pretrained_model/train_config.json --resume=true

# =============================================================================
# TRASH PICKUP — base policy training (not numbered on HF SARM page)
# =============================================================================
# Train a vanilla policy first if Step 5b uses --policy.pretrained_path (e.g. SmolVLA v1).
# Dataset: 3 cameras (head/left_wrist/right_wrist @ 640x360), 18-DOF state/action, 1 task
# Hardware: RTX 3090 Ti 24 GB
# Steps math: 85K frames / batch_size=16 ≈ 5.3K steps/epoch → 15K ≈ 2.8 epochs
#
# IMPORTANT: IsaacSim + ROS pollute PYTHONPATH with Python 3.11 pydantic/pydantic_core
# which breaks wandb import in the Python 3.12 conda env. Always clear it first:
unset PYTHONPATH

# ── OPTION A: SmolVLA (recommended — VLM-based, language-conditioned) ──────────
# ~3.5 h on 3090 Ti (same speed as block_sorting at 1.6 step/s)
# Warm-start from block_sorting checkpoint so the robot dynamics transfer.
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=smolvla \
    --policy.pretrained_path=Odog16/block_sorting_SmolVLA_SARM_v4_15k \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.push_to_hub=true \
    --policy.chunk_size=35 \
    --policy.n_action_steps=35 \
    --policy.repo_id=Odog16/trash_pickup_SmolVLA_SARM_dual_v5 \
    --output_dir=outputs/train/trash_pickup_SmolVLA_SARM_dual_v5 \
    --job_name=trash_pickup_SmolVLA_SARM_dual_v5 \
    --batch_size=16 \
    --steps=15000 \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=1e-4 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=13000 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# ── OPTION A2: SmolVLA — train from scratch (no warm-start) ────────────────────
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/trash_pickup_SmolVLA_scratch \
    --policy.push_to_hub=false \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --output_dir=outputs/train/trash_pickup_SmolVLA_scratch \
    --job_name=trash_pickup_SmolVLA_scratch \
    --batch_size=16 \
    --steps=15000 \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=1e-4 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=13000 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# ── OPTION B: ACT (fast, deterministic, good for 72 episodes) ──────────────────
# ~1–2 h on 3090 Ti. Larger batch fits easily in 24 GB.
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=act \
    --policy.repo_id=Odog16/trash_pickup_ACT_v1 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --output_dir=outputs/train/trash_pickup_ACT_v1 \
    --job_name=trash_pickup_ACT_v1 \
    --batch_size=64 \
    --steps=80000 \
    --save_freq=20000 \
    --log_freq=200 \
    --num_workers=4 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# ── OPTION C: Diffusion (highest quality, slowest inference) ───────────────────
# ~2–3 h training on 3090 Ti. Use async inference server for 30 Hz on Jetson.
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=diffusion \
    --policy.repo_id=Odog16/trash_pickup_diffusion_v1 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --output_dir=outputs/train/trash_pickup_diffusion_v1 \
    --job_name=trash_pickup_diffusion_v1 \
    --batch_size=32 \
    --steps=60000 \
    --save_freq=20000 \
    --log_freq=200 \
    --num_workers=4 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# Resume any run:
# lerobot-train \
#     --config_path=outputs/train/trash_pickup_SmolVLA_v1/checkpoints/last/pretrained_model/train_config.json \
#     --resume=true

# Upload SmolVLA checkpoints after training (edit upload_checkpoints.py for each run):
# python upload_checkpoints.py

# ── MULTI-DATASET: combine trash_pickup with older datasets ────────────────────
# (lerobot supports comma-separated repo_ids or use a local merged dataset)
# Example — combine trash_pickup + block_sorting for a generalist SmolVLA:
lerobot-train \
    --dataset.repo_id='["Odog16/trash_pickup","Odog16/block_sorting_single"]' \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/multi_task_SmolVLA_v1 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=Odog16/block_sorting_SmolVLA_v5_15k \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --output_dir=outputs/train/multi_task_SmolVLA_v1 \
    --job_name=multi_task_SmolVLA_v1 \
    --batch_size=16 \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=1e-4 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=18000 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# =============================================================================
# CO-TRAINING WITH LARGE EXTERNAL DATASETS (Open-X-Embodiment / ALOHA)
# =============================================================================
# Goal: leverage massive open datasets to improve generalization and success rate
# on trash_pickup, tool_pickup, block_sorting, and block_transfer tasks.
#
# The robot uses 18-DOF state/action and 3 cameras; external datasets typically
# have a different DOF count and different camera names.  Two strategies are
# provided depending on how different the source data is:
#
#   Strategy A — Direct JSON list co-training (same robot schema).
#     Use when co-training datasets are already Odog16/* repos recorded on the
#     same xlerobot platform.  LeRobot loads them interleaved in a single run.
#     Works TODAY with no extra tooling.
#
#   Strategy B — Feature-aligned merge then train (cross-embodiment).
#     Use when pulling ALOHA / Open-X data whose action/state dim differs.
#     Run co_training_utils.py to pad+remap features into a shared schema,
#     push the aligned dataset to Hub, then train on the merged repo.
#
# IMPORTANT: Always clear PYTHONPATH before any training command.
unset PYTHONPATH

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY A — All-tasks joint training (same-schema xlerobot datasets)
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 of 2: Generalist co-training on ALL four tasks simultaneously.
# Warm-starts from the block_sorting checkpoint (strongest prior for arm dynamics).
# Steps math: ~4 datasets × ~85-100 K frames each = ~360 K total frames
#             with batch_size=16 → 22 K steps ≈ ~1 epoch across all tasks.
# Time: ~5-7 h on RTX 3090 Ti.
lerobot-train \
    --dataset.repo_id='["Odog16/trash_pickup","Odog16/tool_pickup","Odog16/block_sorting_single","Odog16/block_transfer"]' \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/all_tasks_generalist_SmolVLA_v1 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=Odog16/block_sorting_SmolVLA_v5_15k \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=42 \
    --policy.n_action_steps=42 \
    --output_dir=outputs/train/all_tasks_generalist_SmolVLA_v1 \
    --job_name=all_tasks_generalist_v1 \
    --batch_size=16 \
    --steps=40000 \
    --save_freq=10000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=1e-4 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=2000 \
    --scheduler.num_decay_steps=36000 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# Phase 2 of 2: Fine-tune the generalist on each individual task.
# Load the generalist checkpoint and fine-tune for ~10-15 K steps per task.
# Lower learning rate (5e-5) preserves generalist features while specialising.

# ── trash_pickup specialist ─────────────────────────────────────────────────
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/trash_pickup_SmolVLA_cotrain_v1 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=outputs/train/all_tasks_generalist_SmolVLA_v1/checkpoints/last/pretrained_model \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=42 \
    --policy.n_action_steps=42 \
    --output_dir=outputs/train/trash_pickup_SmolVLA_cotrain_v1 \
    --job_name=trash_pickup_cotrain_v1 \
    --batch_size=16 \
    --steps=15000 \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=5e-5 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=500 \
    --scheduler.num_decay_steps=13500 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# ── tool_pickup specialist ───────────────────────────────────────────────────
lerobot-train \
    --dataset.repo_id=Odog16/tool_pickup \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/tool_pickup_SmolVLA_cotrain_v1 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=outputs/train/all_tasks_generalist_SmolVLA_v1/checkpoints/last/pretrained_model \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=42 \
    --policy.n_action_steps=42 \
    --output_dir=outputs/train/tool_pickup_SmolVLA_cotrain_v1 \
    --job_name=tool_pickup_cotrain_v1 \
    --batch_size=16 \
    --steps=15000 \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=5e-5 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=500 \
    --scheduler.num_decay_steps=13500 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# ── block_sorting specialist ─────────────────────────────────────────────────
lerobot-train \
    --dataset.repo_id=Odog16/block_sorting_single \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/block_sorting_SmolVLA_cotrain_v1 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=outputs/train/all_tasks_generalist_SmolVLA_v1/checkpoints/last/pretrained_model \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=42 \
    --policy.n_action_steps=42 \
    --output_dir=outputs/train/block_sorting_SmolVLA_cotrain_v1 \
    --job_name=block_sorting_cotrain_v1 \
    --batch_size=16 \
    --steps=15000 \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=5e-5 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=500 \
    --scheduler.num_decay_steps=13500 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# ── block_transfer specialist ────────────────────────────────────────────────
lerobot-train \
    --dataset.repo_id=Odog16/block_transfer \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/block_transfer_SmolVLA_cotrain_v1 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=outputs/train/all_tasks_generalist_SmolVLA_v1/checkpoints/last/pretrained_model \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=42 \
    --policy.n_action_steps=42 \
    --output_dir=outputs/train/block_transfer_SmolVLA_cotrain_v1 \
    --job_name=block_transfer_cotrain_v1 \
    --batch_size=16 \
    --steps=15000 \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=5e-5 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=500 \
    --scheduler.num_decay_steps=13500 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY B — Cross-embodiment co-training (ALOHA / Open-X-Embodiment)
# ─────────────────────────────────────────────────────────────────────────────
# External datasets (ALOHA, Open-X) have different action/state dimensions and
# different camera names.  aggregate_datasets() in lerobot requires identical
# features, so we first run co_training_utils.py to produce feature-aligned
# intermediate datasets that share the xlerobot schema (18-DOF state+action,
# head/left_wrist/right_wrist cameras).
#
# The script pads shorter action/state vectors with zeros and renames cameras
# to match the xlerobot naming convention, then calls aggregate_datasets()
# to physically merge everything into a single local dataset.
#
# Checklist (auth, intersection cameras, Hub push, manual push): see
# "MERGE & PUSH DATASETS (co_training_utils.py)" near the top of this file.
#
# Step B-0 (optional): Smoke test — real-world ALOHA only (not sim).
# Sim repos (aloha_sim_*) are MuJoCo; mobile/static repos are physical robots.
# Inspect camera key names before aligning (they differ from sim in some repos):
#   python src/lerobot/data_processing/co_training_utils.py \
#       --source-repos lerobot/aloha_static_battery \
#       --target-repo-id Odog16/dummy \
#       --inspect-only
# Then align (defaults include top/cam_high → head; override --camera-remap if inspect shows other names).
# Use --target-fps 30 (nearest-neighbour resampling from 50fps gives true 30fps output).
# Add --target-image-size 360x640 to match xlerobot camera resolution (source is 480×640):
# python src/lerobot/data_processing/co_training_utils.py \
#     --source-repos lerobot/aloha_static_battery \
#     --target-repo-id Odog16/aloha_aligned_smoke_rw \
#     --target-fps 30 \
#     --target-image-size 360x640 \
#     --match-features-from Odog16/tool_pickup \
#     --target-state-dim 18 \
#     --target-action-dim 18 \
#     --output-root /tmp/aloha_aligned_smoke_rw \
#     --push-to-hub false
#
# Step B-1: Align and merge ALOHA datasets into a local xlerobot-compatible repo.
#   Available ALOHA repos on HF Hub (14-DOF bimanual; sim = single top view, real often cam_high + wrists):
#     lerobot/aloha_sim_insertion_human
#     lerobot/aloha_sim_insertion_scripted
#     lerobot/aloha_sim_transfer_cube_human
#     lerobot/aloha_sim_transfer_cube_scripted
#     lerobot/aloha_mobile_cabinet
#     lerobot/aloha_mobile_wash_pan
#     lerobot/aloha_mobile_elevator
#     lerobot/aloha_static_battery
#     lerobot/aloha_static_cups_open
#   NOTE: ALOHA is 50 fps; --target-fps 30 uses nearest-neighbour temporal
#         resampling (frame pattern 0,2,3,5,7,8,10,…) to give true 30 fps output.
#         This matches the xlerobot datasets exactly, so both physical merging
#         (aggregate_datasets, which enforces fps==) and JSON-list co-training work.
#         --target-image-size 360x640 resizes the 480×640 ALOHA frames to match
#         the xlerobot camera resolution so physical merging works.
#         --match-features-from Odog16/tool_pickup copies observation.state/action
#         joint name strings from your xlerobot dataset; aggregate_datasets() compares
#         full feature dicts (names must match, not only shape [18]).
#         Stale _align_tmp_* caches are auto-rebuilt when fps/names/resolution change;
#         use --force-rebuild to wipe caches and output copy unconditionally.
python src/lerobot/data_processing/co_training_utils.py \
    --source-repos \
        lerobot/aloha_sim_insertion_human \
        lerobot/aloha_sim_transfer_cube_human \
        lerobot/aloha_mobile_cabinet \
        lerobot/aloha_static_battery \
    --target-repo-id Odog16/aloha_aligned_for_cotrain \
    --target-fps 30 \
    --target-image-size 360x640 \
    --match-features-from Odog16/tool_pickup \
    --target-state-dim 18 \
    --target-action-dim 18 \
    --camera-remap "top:head,cam_high:head,cam_left_wrist:left_wrist,cam_right_wrist:right_wrist" \
    --output-root ~/.cache/huggingface/lerobot/Odog16/aloha_aligned_for_cotrain \
    --push-to-hub false

python src/lerobot/data_processing/co_training_utils.py \
    --source-repos lerobot/aloha_static_battery \
    --target-repo-id Odog16/aloha_aligned_for_cotrain \
    --target-fps 30 \
    --target-image-size 360x640 \
    --match-features-from Odog16/tool_pickup \
    --target-state-dim 18 \
    --target-action-dim 18 \
    --output-root ~/.cache/huggingface/lerobot/Odog16/aloha_aligned_smoke_rw \
    --push-to-hub false




# Step B-2 (optional): Push aligned dataset to Hub for reuse across machines.
# Requires huggingface-cli login and meta/tasks.parquet under root. push_to_hub()
# uses the dataset repo_id from the constructor (no positional repo string).
python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
r = 'Odog16/aloha_aligned_for_cotrain'
root = Path.home() / '.cache/huggingface/lerobot' / r
LeRobotDataset(r, root=root).push_to_hub()
"

# Step B-3: Phase 1 — Generalist pre-training on xlerobot tasks + aligned ALOHA.
# Mixing ratio is proportional to dataset sizes (lerobot samples uniformly across episodes).
# ALOHA adds ~3,000-5,000 diverse manipulation episodes → better grasp primitives.
lerobot-train \
    --dataset.repo_id='["Odog16/trash_pickup","Odog16/tool_pickup","Odog16/block_sorting_single","Odog16/block_transfer","Odog16/aloha_aligned_for_cotrain"]' \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/xlerobot_aloha_generalist_v1 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=42 \
    --policy.n_action_steps=42 \
    --output_dir=outputs/train/xlerobot_aloha_generalist_v1 \
    --job_name=xlerobot_aloha_generalist_v1 \
    --batch_size=16 \
    --steps=60000 \
    --save_freq=10000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=1e-4 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=3000 \
    --scheduler.num_decay_steps=55000 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# Step B-4: Phase 2 — Per-task fine-tuning from generalist checkpoint.
# Same commands as Strategy A Phase 2 above; just change --policy.pretrained_path:
#   --policy.pretrained_path=outputs/train/xlerobot_aloha_generalist_v1/checkpoints/last/pretrained_model
# (Replace the generalist path in each specialist command above.)

# ─────────────────────────────────────────────────────────────────────────────
# CO-TRAINING WITH RA-BC (combine generalist init + reward-weighted training)
# ─────────────────────────────────────────────────────────────────────────────
# Best of both worlds: start from a generalist co-trained checkpoint, then
# apply RA-BC to focus on high-reward (high-quality) transitions.
# Prerequisite: complete SARM Steps 1-5a for the task first (see SARM section).

# ── trash_pickup: generalist init + RA-BC ────────────────────────────────────
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --tolerance_s=0.02 \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/trash_pickup_SmolVLA_cotrain_rabc_v1 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=outputs/train/all_tasks_generalist_SmolVLA_v1/checkpoints/last/pretrained_model \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=42 \
    --policy.n_action_steps=42 \
    --use_rabc=true \
    --rabc_head_mode=sparse \
    --rabc_kappa=0.01 \
    --output_dir=outputs/train/trash_pickup_SmolVLA_cotrain_rabc_v1 \
    --job_name=trash_pickup_cotrain_rabc_v1 \
    --batch_size=16 \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=5e-5 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=500 \
    --scheduler.num_decay_steps=18000 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# ── tool_pickup: generalist init + RA-BC ─────────────────────────────────────
lerobot-train \
    --dataset.repo_id=Odog16/tool_pickup \
    --tolerance_s=0.02 \
    --policy.type=smolvla \
    --policy.repo_id=Odog16/tool_pickup_SmolVLA_cotrain_rabc_v1 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=outputs/train/all_tasks_generalist_SmolVLA_v1/checkpoints/last/pretrained_model \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --policy.num_vlm_layers=16 \
    --policy.chunk_size=42 \
    --policy.n_action_steps=42 \
    --use_rabc=true \
    --rabc_head_mode=sparse \
    --rabc_kappa=0.01 \
    --output_dir=outputs/train/tool_pickup_SmolVLA_cotrain_rabc_v1 \
    --job_name=tool_pickup_cotrain_rabc_v1 \
    --batch_size=16 \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=5e-5 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_warmup_steps=500 \
    --scheduler.num_decay_steps=18000 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=lerobot

# ─────────────────────────────────────────────────────────────────────────────
# VERIFY DATASET ALIGNMENT (run before training to catch feature mismatches)
# ─────────────────────────────────────────────────────────────────────────────
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import json

repos = [
    'Odog16/trash_pickup',
    'Odog16/tool_pickup',
    'Odog16/block_sorting_single',
    'Odog16/block_transfer',
]
for repo in repos:
    meta = LeRobotDatasetMetadata(repo)
    action_shape = meta.features['action']['shape']
    state_shape  = meta.features['observation.state']['shape']
    cameras      = [k for k in meta.features if k.startswith('observation.images')]
    print(f'{repo}: action={action_shape}, state={state_shape}, cameras={cameras}')
"

# ─────────────────────────────────────────────────────────────────────────────
# RESUME CO-TRAINING RUN
# ─────────────────────────────────────────────────────────────────────────────
# lerobot-train \
#     --config_path=outputs/train/all_tasks_generalist_SmolVLA_v1/checkpoints/last/pretrained_model/train_config.json \
#     --resume=true

# =============================================================================
# ASYNC INFERENCE (policy on external GPU, robot on Jetson)
# =============================================================================
# Use when the policy is too large for Jetson or you want inference on a stronger GPU.
# Policy runs on the external machine; Jetson runs the robot client over the network.
#
# Step 1 — ON EXTERNAL GPU MACHINE (PC with CUDA): start the policy server

# If you see "stack expects a non-empty TensorList" (e.g. diffusion on first frame):
# 1) Sync lerobot code to the GPU machine: the fix is in src/lerobot/policies/diffusion/modeling_diffusion.py
#    (predict_action_chunk handles empty queues). Copy from Jetson or: git pull / same branch on both.
# 2) Optionally try: --obs_queue_timeout=5.0
# Depth: if the policy was not trained with depth, head_depth is already dropped (server uses only policy keys).
#
# Step 2 — ON JETSON (robot + cameras): run the robot client
#          Set server_address to the external GPU machine's IP (same LAN as Jetson).
# ACT example (use the _policy repo, not the dataset repo):
python -m lerobot.async_inference.robot_client \
    --server_address=10.249.34.90:8080 \
    --robot.type=xlerobot \
    --task="pick up the yellow block and place it in the yellow bowl" \
    --policy_type=smolvla \
    --pretrained_name_or_path=Odog16/block_sorting_SmolVLA_v5_15k\
    --policy_device=cuda \
    --actions_per_chunk=35 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name="average" \
    --fps=30

###

python -m lerobot.async_inference.robot_client \
    --server_address=10.249.34.90:8080 \
    --robot.type=xlerobot \
    --task="pick up the red block and place it in the blue bowl" \
    --policy_type=smolvla \
    --pretrained_name_or_path=Odog16/block_sorting_SmolVLA_8k \
    --policy_device=cuda \
    --actions_per_chunk=30 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=average \
    --fps=30

python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=1.0



python -m lerobot.async_inference.robot_client \
    --server_address=10.249.36.224:8080 \
    --robot.type=xlerobot \
    --task="pick up the tools from the table and place it in the red bin" \
    --policy_type=act \
    --pretrained_name_or_path=Odog16/tool_pickup_ACT_policy_B\
    --policy_device=cuda \
    --actions_per_chunk=40\
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name="average" \
    --fps=30



# Diffusion example:
python -m lerobot.async_inference.robot_client \
    --server_address=10.249.36.224:8080 \
    --robot.type=xlerobot \
    --task="pick up the tools from the table and place it in the red bin" \
    --policy_type=diffusion \
    --pretrained_name_or_path=Odog16/tool_pickup_diffusion_1obs_policy \
    --policy_device=cuda \
    --actions_per_chunk=10 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name="average" \
    --fps=30


python -m lerobot.async_inference.robot_client \
    --server_address=10.249.40.136:8080 \
    --robot.type=xlerobot \
    --task="pick up the plastic bottle and place it in the trash bin" \
    --policy_type=smolvla \
    --pretrained_name_or_path=Odog16/trash_pickup_SmolVLA_v2.1_rabc_15k\
    --policy_device=cuda \
    --actions_per_chunk=42 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --fps=30


# aggregate_fn_name options (when overlapping action chunks are merged):
#   weighted_average  0.3*old + 0.7*new  (default, favours newer)
#   latest_only       use new only       (ignore previous at same timestep)
#   average           0.5*old + 0.5*new  (equal blend)
#   conservative      0.7*old + 0.3*new (favours older, smoother)
#
# PERFORMANCE: Diffusion policies automatically use 10 inference steps (instead of default 100)
# for ~10x speedup with minimal quality loss (~2000ms → ~200ms). If inference is still slow:
#   - Check GPU usage: nvidia-smi (should see python process using GPU)
#   - Reduce image resolution in robot config (e.g., 320x240 instead of 640x480)
#   - Disable depth cameras if not needed for the task
#   - Consider using a faster policy type (e.g., ACT is typically faster than diffusion)

    --robot.port1=/dev/ttyACM1 \
    --robot.port2=/dev/ttyACM2 \
    --robot.port3=/dev/ttyACM0 \
    --robot.cameras="{head: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: MJPG}, left_wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG}, right_wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \



# OPTION 2: Direct evaluation (may hit CUDA OOM on Jetson)
# 
# MODEL SIZE LIMITS FOR JETSON ORIN NANO (8GB):
# - Total system RAM: 8GB (shared with GPU)
# - Available GPU memory: ~4-5GB (after OS overhead)
# - Model size limits:
#   * Small models (<500MB): Should work fine
#   * Medium models (500MB-1GB): May work with optimizations (mixed precision, smaller batch)
#   * Large models (>1GB): Likely to hit OOM (like your 1.2GB SmolVLA model)
# 
# TROUBLESHOOTING CUDA OOM:
# If you get "NvMapMemAllocInternalTagged: error 12" or "CUDACachingAllocator" errors,
# the model (1.2GB) is too large for Jetson GPU memory. Use Option 1 (async) instead.
# 
# For larger Jetson models:
# - Jetson Orin AGX (32GB/64GB): Can handle models up to ~10-20GB
# - Jetson Orin NX (16GB): Can handle models up to ~5-8GB
# 
# DOCKER PERMISSIONS:
# If you get "permission denied" or "unknown server OS" errors, add user to docker group:
#   sudo usermod -aG docker $USER
#   newgrp docker  # or logout/login
# The "unknown server OS" error is often a misleading message when Docker can't connect due to permissions.
# Verify docker access: docker version (should work without sudo after adding to docker group)
docker run --runtime nvidia --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -it --rm --network host --shm-size=4g \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  --volume /etc/enctune.conf:/etc/enctune.conf \
  --volume /etc/nv_tegra_release:/etc/nv_tegra_release \
  --volume /tmp/nv_jetson_model:/tmp/nv_jetson_model \
  --volume /var/run/dbus:/var/run/dbus \
  --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
  --device /dev/snd -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  -v /run/user/1000/pulse:/run/user/1000/pulse \
  --device /dev/bus/usb \
  --device /dev/video0 --device /dev/video1 --device /dev/video2 \
  --device /dev/video3 --device /dev/video4 --device /dev/video5 \
  --device /dev/ttyACM0 --device /dev/ttyACM1 --device /dev/ttyACM2 \
  -v /home/jetson/lerobot:/opt/lerobot -w /opt/lerobot \
  dustynv/lerobot:r36.4.0 \
  bash -c "set +H && cd /opt/lerobot && pip install --index-url https://pypi.org/simple --force-reinstall 'numpy<2' && pip install --index-url https://pypi.org/simple 'datasets>=4.0.0,<4.2.0' 'diffusers>=0.27.2,<0.36.0' 'huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0' 'accelerate>=1.10.0,<2.0.0' 'setuptools>=71.0.0,<81.0.0' 'cmake>=3.29.0.1,<4.2.0' 'einops>=0.8.0,<0.9.0' 'opencv-python-headless>=4.9.0,<4.13.0' 'av>=15.0.0,<16.0.0' 'jsonlines>=4.0.0,<5.0.0' 'packaging>=24.2,<26.0' 'pynput>=1.7.7,<1.9.0' 'pyserial>=3.5,<4.0' 'wandb>=0.20.0,<0.22.0' 'draccus==0.10.0' 'gymnasium>=1.1.1,<2.0.0' 'rerun-sdk>=0.24.0,<0.27.0' 'deepdiff>=7.0.1,<9.0.0' 'imageio[ffmpeg]>=2.34.0,<3.0.0' 'termcolor>=2.4.0,<4.0.0' 'transformers>=4.53.0,<5.0.0' 'num2words>=0.5.14,<0.6.0' 'safetensors>=0.4.3,<1.0.0' 'feetech-servo-sdk>=1.0.0,<2.0.0' && pip install --index-url https://pypi.org/simple --no-deps -e . && pip install --index-url https://pypi.org/simple --force-reinstall 'numpy<2' && python -c 'import torch; torch.cuda.empty_cache(); assert torch.cuda.is_available(), \"CUDA not available\"; print(f\"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\")' && python -m lerobot.scripts.lerobot_record --robot.type=xlerobot --dataset.repo_id=Odog16/eval_ob15_packing_box --dataset.single_task=packing_box --dataset.num_episodes=5 --display_data=false --policy.path=Odog16/smolvla_ob15_packing_box_policy_1"




# Split dataset and push to hub
# NOTE: Split names must use underscores, not spaces (Hugging Face repo ID requirement)
# If you get "FileExistsError", delete existing split directories first:
# rm -rf ~/.cache/huggingface/lerobot/Odog16/ob15_packing_box_place_blocks_in_box ~/.cache/huggingface/lerobot/Odog16/ob15_packing_box_pick_box_and_place

lerobot-edit-dataset \
    --repo_id Odog16/ob15_packing_box \
    --operation.type split \
    --operation.splits '{"place_blocks_in_box": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74], "pick_box_and_place_blocks_in_box": [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]}' \
    --push_to_hub true

# This creates two datasets on Hugging Face Hub:
# - Odog16/ob15_packing_box_place_blocks_in_box (75 episodes)
# - Odog16/ob15_packing_box_pick_box_and_place (40 episodes)

# =============================================================================
# DATASET CLEANUP - Delete Unwanted Episodes
# =============================================================================
# Delete specific episodes and keep only the ones you want
# Example: Keep only episodes 740, 1-15 (delete all others)

# First, inspect the dataset to see total episodes:
# python -c "from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata; meta = LeRobotDatasetMetadata('Odog16/packing_box'); print(f'Total episodes: {meta.total_episodes}')"

# Then delete unwanted episodes (save to new dataset to preserve original):
# NOTE: You need to specify ALL episodes to DELETE, not the ones to keep
# If you want to keep episodes [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 740],
# you need to delete all other episodes (e.g., if total is 741, delete [0, 16-739])

# Example: Delete episodes 0, 16-739 (keeping 1-15 and 740)
# lerobot-edit-dataset \
#     --repo_id Odog16/packing_box \
#     --new_repo_id Odog16/packing_box_cleaned \
#     --operation.type delete_episodes \
#     --operation.episode_indices "[0, 16, 17, 18, ...]" \
#     --push_to_hub true

# Or delete episodes and overwrite original (WARNING: permanent deletion):
 lerobot-edit-dataset \
     --repo_id Odog16/test_transfer_block \
     --operation.type delete_episodes \
     --operation.episode_indices "[35, 69]" \
     --push_to_hub true


# =============================================================================
# GIT WORKFLOW
# =============================================================================
cd /home/jetson/lerobot

# 1. Stage your changes
git add .

# 2. Commit your changes
git commit -m "Your commit message"

# 3. Pull latest changes from remote (merge if needed)
git pull origin main

# 4. Push your changes to remote
# NOTE: First time pushing requires authentication:
#   - Create Personal Access Token: https://github.com/settings/tokens
#   - Select scope: "repo" (full control)
#   - When prompted:
#     * Username: your GitHub username
#     * Password: paste your Personal Access Token (NOT your GitHub password)
#   - Credentials will be saved for future pushes
git push origin main

# If you get authentication errors, ensure remote is HTTPS:
#   git remote set-url origin https://github.com/ImpurestTadpole/lerobot.git
