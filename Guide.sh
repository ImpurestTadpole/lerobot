# =============================================================================
# SETUP
# =============================================================================
# Activate conda environment
conda activate lerobot

# Set USB permissions (run every time after reboot or USB reconnect)
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1 

# =============================================================================
# GIT WORKFLOW - SYNC WITH GITHUB REPO
# =============================================================================
# Navigate to lerobot directory
cd /home/jetson/lerobot

# ONE-TIME SETUP (if not already done):
# git config --global user.name "ImpurestTadpole"
# git config --global user.email "your-email@example.com"
# git remote add origin https://github.com/ImpurestTadpole/lerobot.git

# COMPLETE SYNC WORKFLOW (pull remote changes + push local changes):
# Step 1: Check current status
git status

# Step 2: Pull remote changes first (to get latest from GitHub)
git config pull.rebase false  # Set merge strategy (one-time, if not already set)
git pull origin main

# Step 3: If merge conflicts occur, resolve them:
#   - Edit files with conflicts (look for <<<<<<< markers)
#   - git add .
#   - git commit -m "Resolve merge conflicts"

# Step 4: Stage and commit your local changes
git add .                     # Stage all changes
git commit -m "Your commit message here"

# Step 5: Push your local changes to GitHub
git push origin main

# -----------------------------------------------------------------------------
# ALTERNATIVE: If you have uncommitted changes when pulling:
# -----------------------------------------------------------------------------
# OPTION A: Commit your local changes first, then pull:
git add .
git commit -m "Save local changes before pulling"
git pull origin main

# OPTION B: Stash your changes, pull, then reapply:
git stash                      # Temporarily save your changes
git pull origin main          # Get remote changes
git stash pop                 # Reapply your stashed changes
# Then commit and push:
git add .
git commit -m "Your commit message here"
git push origin main

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
rerun --serve-web --web-viewer-port 9090 --connect "rerun+http://172.20.10.13:9876/proxy"

# OPTION 2: Via SSH tunnel
# Terminal 1:
ssh -L 9876:localhost:9876 jetson@192.168.0.104
# Terminal 2:
rerun --serve-web --web-viewer-port 9090 --connect "rerun+http://localhost:9876/proxy"

# Then open in browser: http://localhost:9090

# NOTE: Install rerun via pip (not snap):
pip3 install rerun-sdk




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
    --dataset.repo_id=Odog16/test_transfer_block_2  \
    --dataset.single_task="transfer the block" \
    --dataset.num_episodes=15 \
    --dataset.fps=30 \
    --display_data=true \
    --dataset.push_to_hub=true 
 #   --resume=true 



# RECORDING LOCAL ONLY (push manually later):
lerobot-record \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/ob15_test_1 \
    --dataset.single_task="place clothes in bin" \
    --dataset.num_episodes=10 \
    --dataset.fps=30 \
    --display_data=true

# Manually push to hub after recording:
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
    dataset = LeRobotDataset('Odog16/test_transfer_block_2'); \
    dataset.push_to_hub()"

# Local data location: ~/.cache/huggingface/lerobot/Odog16/ob15_test_1/
# Parquet files: data/chunk-000/episode_*.parquet
# Videos: videos/chunk-000/observation.images.*/episode_*.mp4

# Delete local dataset (if you want to start fresh):
rm -rf ~/.cache/huggingface/lerobot/Odog16/making_coffee

# =============================================================================
# REPLAY EPISODE
# =============================================================================
# Replay an episode from a dataset on the robot (change --dataset.episode for other episodes)
lerobot-replay \
    --robot.type=xlerobot \
    --dataset.repo_id=Odog16/test_transfer_block_2 \
    --dataset.episode=0

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

# =============================================================================
# ASYNC INFERENCE (policy on external GPU, robot on Jetson)
# =============================================================================
# Use when the policy is too large for Jetson or you want inference on a stronger GPU.
# Policy runs on the external machine; Jetson runs the robot client over the network.
#
# Step 1 — ON EXTERNAL GPU MACHINE (PC with CUDA): start the policy server
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=1
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
    --server_address=10.249.39.211:8080 \
    --robot.type=xlerobot \
    --task="take coffee from blue place it in the machine, then place it on the yellow." \
    --policy_type=act \
    --pretrained_name_or_path=Odog16/act_making_coffee_policy_60k\
    --policy_device=cuda \
    --actions_per_chunk=80\
    --chunk_size_threshold=0.25 \
    --aggregate_fn_name="weighted_average" \
    --fps=30





# Diffusion example:
python -m lerobot.async_inference.robot_client \
    --server_address=10.249.43.224:8080 \
    --robot.type=xlerobot \
    --task="transfer the block" \
    --policy_type=diffusion \
    --pretrained_name_or_path=Odog16/test_transfer_block_diffusion \
    --policy_device=cuda \
    --actions_per_chunk=80 \
    --chunk_size_threshold=0.3 \
    --aggregate_fn_name="weighted_average" \
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
