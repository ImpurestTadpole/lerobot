# Hierarchical VLA (HVLA)

A dual-system VLA for bimanual robot control. S2 (VLM) provides scene understanding, S1 (action policy) generates actions conditioned on S2's latent.

Inspired by [Helix](https://www.figure.ai/news/helix), [OpenHelix](https://arxiv.org/abs/2505.03912), and [Dual Process VLA](https://arxiv.org/abs/2410.15549).

```
S2 Process (~4-15Hz)                shared memory           S1 Process (~22-30Hz)
┌──────────────────────┐           ┌───────────┐          ┌───────────────────────────┐
│ 4 cameras → SigLIP   │           │ [2048]    │          │ 2-4 cameras → DINOv2/ResNet│
│ task text → Gemma 2B │──latent──→│ + age(s)  │←──read──│ state → state token        │
│ mean-pool → [2048]   │           └───────────┘          │ latent + age → S2 token    │
│ (VLM-only, no action │                                  │ → ACT encoder/decoder      │
│  expert)             │                                  │ → action chunk (14-DOF)    │
└──────────────────────┘                                  └───────────────────────────┘
```

## Quick Start

### 0. Convert S2 checkpoint (JAX → PyTorch, one-time)

S2 loads **VLM-only** weights from a Pi0.5 checkpoint in PyTorch `safetensors` form. Use OpenPI’s official converter (`examples/convert_jax_model_to_pytorch.py` — not `scripts/run_conversion.py`).

A HuggingFace download should look like: `params/` (Orbax), `assets/` (norm stats), metadata — same layout OpenPI expects.

```bash
# One-time: clone OpenPI and install with uv (see OpenPI README)
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git ~/src/openpi
cd ~/src/openpi
# Install `uv` if needed (e.g. `pip install uv` in a Python ≥3.11 conda env). Do not use only
# `pip install -e .`: pip ignores OpenPI’s `[tool.uv.sources]` and unpins `lerobot` from PyPI,
# which often ends in `resolution-too-deep` during dependency resolution.
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# One-time: patch the venv’s `transformers` with OpenPI’s forked modules (required for JAX→PyTorch convert):
DST="$(uv run python -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')"
cp -r src/openpi/models_pytorch/transformers_replace/* "$DST/"

# Run the converter with `uv run python ...` from this repo (uses `.venv`). Plain `conda`/`python`
# without that venv causes `ModuleNotFoundError: safetensors` if OpenPI was not installed into conda.

# Download JAX checkpoint
hf download KeWangRobotics/soarm-pi05-state-11997 \
    --local-dir ~/.cache/openpi/checkpoints/soarm-pi05-state-11997

# Convert (`--config-name` must be a TrainConfig whose model is Pi0Config — not Pi05Config)
uv run python examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir ~/.cache/openpi/checkpoints/soarm-pi05-state-11997 \
    --config-name pi05_aloha \
    --output-path ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch
```

The [HF README](https://huggingface.co/KeWangRobotics/soarm-pi05-state-11997) lists train config `soarm_pi05_flow`; the stock converter only accepts configs backed by `Pi0Config`. `pi05_aloha` is `Pi0Config(pi05=True)` with default `action_horizon=50`, which matches typical Pi0.5 SOARM settings. If conversion warns on missing/unexpected keys, try `--inspect-only` first, or another `pi05_*` config whose `action_horizon` matches your checkpoint.

The converter copies `assets` from `checkpoint_dir/../assets`; HF layouts put `assets` **inside** the checkpoint folder. HVLA latent extraction only needs `model.safetensors`; optionally run `cp -r .../soarm-pi05-state-11997/assets .../soarm-pi05-state-11997-pytorch/`.

Output: `~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors`.

Available checkpoints:
- `KeWangRobotics/soarm-pi05-state-11997` — Pi0.5 base (state-based)
- `KeWangRobotics/soarm-pi05-fast-7998` — Pi0.5 with FAST tokens

### 1. Extract S2 latents (offline, one-time)

No OpenPI dependency — uses the HVLA S2 model directly:

```bash
python scripts/extract_s2_latents_hvla.py \
    --checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --dataset thewisp/cylinder_ring_assembly \
    --prompt "assemble cylinder into ring" \
    --output ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents.npy \
    --image-keys observation.images.front,observation.images.top,observation.images.left_wrist,observation.images.right_wrist \
    --batch-size 8
```

Produces `s2_latents.npy` shape `[N_frames, 2048]`. ~81ms/frame on GPU.

### 2a. Train S1 — ACT (default, CVAE)

```bash
python scripts/train_act_vlm.py \
    --dataset-repo-id thewisp/cylinder_ring_assembly \
    --s2-latent-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents_pt_11997.npy \
    --output-dir outputs/act_vlm_hvla \
    --steps 100000 \
    --batch-size 16 \
    --save-freq 20000 \
    --use-dino-backbone \
    --resize-images 224x224 \
    --num-workers 8 \
    --max-delay 0.15
```

`--max-delay 0.15` enables delay augmentation: each training sample randomly shifts the S2 latent backward by 0-150ms (0-5 frames at 30fps), within episode boundaries. The age (seconds of staleness) is injected as a learned embedding — same scalar used at inference.

### 2b. Train S1 — Flow Matching with Training-Time RTC

```bash
# With S2 conditioning:
python -u -m lerobot.policies.hvla.s1.flow_matching.train \
    --dataset-repo-id thewisp/cylinder_ring_assembly \
    --s2-latent-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents_pt_11997.npy \
    --output-dir outputs/flow_s1_hvla_v6 \
    --steps 50000 \
    --batch-size 64 \
    --save-freq 10000 \
    --num-workers 16 \
    --resize-images 224x224

# Without S2 (images + state only):
python -u -m lerobot.policies.hvla.s1.flow_matching.train \
    --dataset-repo-id thewisp/cylinder_ring_assembly \
    --output-dir outputs/flow_s1_no_s2_v1 \
    --steps 50000 \
    --batch-size 64 \
    --save-freq 10000 \
    --num-workers 16 \
    --resize-images 224x224
```

Training logs are automatically saved to `{output_dir}/train.log`.

Flow matching S1 implements [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964) (Mees et al., 2025): simulated inference delay during training, with ground-truth action prefix inpainting. No architecture changes vs standard flow matching — just masking. At inference, actually-executed actions replace the prefix positions at each denoising step.

- 97M trainable params (DINOv2 ViT-S finetuned + 75M decoder)
- LR 2.5e-5 with cosine decay to 2.5e-6 (matching Pi0)
- bf16 autocast + TF32 matmul
- Bidirectional action decoder attention (Pi0-style, all positions attend to all others)
- 15 denoising steps (Euler integration), rtc_max_delay=6
- No S2 latent delay augmentation or age embedding (matching old ACT setup that worked)
- v6: curated dataset + bidirectional attention (v5 used causal)

### 3. Inference on the robot

**Required to run the policy:** a trained S1 checkpoint, a converted S2 Pi0.5 `model.safetensors` (unless S2 is already running via `s2_standalone` and S1 attaches over shared memory), a `--task` string, and a connected robot. `launch` uses your default robot JSON unless you pass `--robot-config`.

**Teleoperation is optional.** With none of the flags below, the stack is fully autonomous: S2 reads cameras + task text, S1 outputs action chunks, the robot executes.

| Mode | What you add | Purpose |
|------|----------------|--------|
| Autonomous only | (nothing extra) | Deploy trained policy |
| Autonomous + logging | `--record-dataset REPO_ID` | Save every frame (eval, analysis, more training data) |
| Human in the loop | `--teleop-config` (+ optional `--intervention-dataset`) | Leader-arm mirror / SPACE toggle; interventions for RLT or corrections |

**BC demonstrations** to train S1 are recorded with `lerobot-record` (or your usual pipeline), not via `launch` teleop flags.

**xlerobot:** pass `--hvla-preset xlerobot` so shared-memory image size and camera naming match the platform (see `Guide.sh`).

### 3a. Inference — ACT (default)

```bash
python -m lerobot.policies.hvla.launch \
    --s1-checkpoint outputs/act_vlm_hvla/checkpoint-100000 \
    --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring" \
    --resize-images 224x224 \
    --temporal-ensemble-coeff 0.07
```

### 3b. Inference — Flow Matching with RTC

```bash
python -m lerobot.policies.hvla.launch \
    --s1-type flow \
    --s1-checkpoint outputs/flow_s1_hvla_v6/checkpoint-40000 \
    --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring" \
    --resize-images 224x224
```

No `--temporal-ensemble-coeff` needed — RTC provides chunk continuity natively. Key flags:
- `--denoise-steps N`: override denoising steps (default 10, min usable ~5, max quality ~15)
- `--s1-query-interval N`: re-query every N actions (default 2, ~67ms). Lower = fresher chunks, higher = more of each chunk executed
- `--no-compile-s1`: disable torch.compile if needed (on by default, ~2ms savings)
- `--save-grip-drops DIR`: save observations when gripper drops detected (debugging)
- `--record-dataset REPO_ID`: record inference episode to LeRobotDataset (e.g. `user/hvla_ep1`). Saves obs+actions every frame, commits on shutdown. Compatible with GUI, Rerun, and further training.
- `--teleop-config PATH`: path to teleop profile JSON for intervention / inverse follow. When set, the leader arm mirrors the robot during policy mode; press SPACE to toggle human control.
- `--intervention-dataset REPO_ID`: record intervention fragments to a separate dataset (e.g. `user/hvla_interventions`). Each human takeover segment becomes a separate episode.

S2 auto-discovery: the launcher first tries to attach to an existing S2 process via shared memory. If found, S1 starts instantly (no 45s cold start). If not found, spawns S2 from `--s2-checkpoint`.

### 3c. Persistent S2 (recommended for iteration)

Start S2 once in a separate terminal — it stays hot between S1 restarts:
```bash
# Terminal 1: start S2 (one-time, stays running)
python -m lerobot.policies.hvla.s2_standalone \
    --checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring"
```

```bash
# Terminal 2: start/restart S1 freely (connects to S2 instantly)
python -m lerobot.policies.hvla.launch \
    --s1-type flow \
    --s1-checkpoint outputs/flow_s1_hvla_v6/checkpoint-40000 \
    --task "assemble cylinder into ring" \
    --resize-images 224x224
```

S1 auto-discovers the running S2 via well-known shared memory names. No `--s2-checkpoint` needed when S2 is already running.

### 3d. S1 + S2 on GPU PC, robot on Jetson (ZMQ robot bridge)

`launch.py` always runs the control loop in the process that owns `Robot`. To run **both** S2 and S1 on a single GPU machine while motors stay on the Jetson, use **`xlerobot_client`** on the PC and **`xlerobot_host`** on the Jetson (ports **5555** cmd, **5556** observations — Jetson binds, PC connects).

**Do not** pass `--zmq-latent-host` here; that mode moves only S2↔S1 images/latents. For “all inference on PC2”, use normal shared-memory HVLA on one machine + network robot.

**Jetson** (USB robot, firewall open to PC2):

```bash
python -m lerobot.robots.xlerobot.xlerobot_host \
  --robot-config ~/.config/lerobot/robots/xlerobot.json \
  --port-cmd 5555 --port-observations 5556
```

**PC2** (copy and edit `examples/xlerobot/xlerobot_client_pc2_remote.example.json`; set `remote_ip` to the Jetson):

One process (spawns S2 + runs S1 against the remote robot):

```bash
python -m lerobot.policies.hvla.launch \
  --hvla-preset xlerobot \
  --s1-type flow \
  --s1-checkpoint /path/to/s1/checkpoints/last \
  --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
  --robot-config /path/to/xlerobot_client_pc2_remote.example.json \
  --task "pick up the tools from the table and place it in the red bin" \
  --resize-images 224x224
```

Or keep S2 hot in a first terminal (`s2_standalone`), then run the same `launch` **without** `--s2-checkpoint` so S1 attaches over shared memory.

Match **`lift_axis`** and **camera names / resolutions** between host JSON and client JSON so `observation.state` matches the trained policy (e.g. 18-DOF with head + lift).

## Module Structure

```
src/lerobot/policies/hvla/
├── s2/
│   ├── config.py           S2 VLM config (PaliGemma dims, LoRA settings)
│   ├── model.py            VLM-only model: extract_prefix_latent, AR subtask decode
│   ├── preprocessing.py    Camera image resize + normalize to [-1,1]
│   └── tokenizer.py        SentencePiece tokenizer for PaliGemma prompt format
├── s1/
│   ├── protocol.py        S1Policy interface (supports_rtc, needs_temporal_ensemble)
│   ├── __init__.py         Load helpers: load_act_policy / load_flow_matching_policy
│   └── flow_matching/
│       ├── config.py       FlowMatchingS1Config
│       ├── model.py        Flow matching decoder with RTC prefix conditioning
│       └── train.py        Training script with delay augmentation + RTC prefix
├── ipc.py                  SharedLatentCache: CPU shared memory between processes
├── s2_process.py           S2 process entry point (extraction loop)
├── s1_process.py           S1 process entry point (robot control loop)
├── launch.py               Spawns S2 process + runs S1
└── README.md               This file
```

## Design Decisions

### S2 is VLM-only
S2 uses PaliGemma (SigLIP + Gemma 2B) for scene understanding. The Pi0.5 action expert is NOT loaded — S1 generates all actions. This saves ~40% GPU memory and decouples the action policy from the VLM. S2 loads from any Pi0.5 safetensors checkpoint — action expert keys are filtered out at load time.

### Latent interface: 2048-dim, single token

S2 → S1 communication is a single [2048] float32 vector (PaliGemma's hidden dim), mean-pooled from the VLM's prefix features. S1 projects it down via MLP: 2048→1024→512.

| System | Latent dim | Projection |
|--------|-----------|------------|
| OpenHelix | 512 | Linear from 4096 |
| Dual Process VLA | 4096 | MLP |
| RoboDual | 256 + 8 tokens | Perceiver |
| **Ours** | **2048** | MLP: 2048→1024→512 in S1 |

### Action decoder: bidirectional attention (v6)
Pi0's action expert uses bidirectional self-attention so all chunk positions "negotiate" and commit to one mode. Our v5 (causal) showed within-chunk gripper oscillation. v6 switches to bidirectional (`tgt_mask=None`), matching Pi0. Requires retraining — not a drop-in change.

### S1 training: pre-extract S2 latents

S2 latents are pre-extracted offline rather than computed during S1 training:
- S2 inference (50ms/frame) on the same GPU as S1 training causes contention
- DataLoader workers can't efficiently share a GPU model
- Extraction is one-time: ~2.5h for 186k frames, produces ~1.5GB .npy file
- If S2 is retrained, latents must be re-extracted before S1 retraining

### Delay augmentation: tried and dropped (v5+)

v1-v4 shifted the S2 latent backward by k frames (k ~ Uniform(0, 15) at 30fps = 0-500ms) to simulate S2 staleness, with an age embedding MLP so S1 knows how stale the latent is. In practice, this caused S1 to learn to ignore the latent entirely. v5+ trains with aligned latents (k=0, no age embedding), matching the original ACT setup that worked. The age embedding MLP still exists in the model but is disabled (`use_s2_age_embedding=False`).

### Shared memory IPC
`SharedLatentCache` uses `torch.Tensor.share_memory_()` backed by `/dev/shm`. Each process manages its own CUDA context — no GPU contention from IPC. The [2048] float32 latent (8KB) transfers in <0.1ms.

## Performance

### S1 Flow Matching (v6, 10 denoise steps, compiled)

| Component | Latency | Notes |
|-----------|---------|-------|
| S2 latent extraction | ~81ms | VLM-only, no action expert |
| S1 obs prep (4 cameras) | ~6ms | cv2.resize on CPU, then GPU transfer |
| S1 inference (10 steps) | 22-42ms | bf16 + KV cache + batched DINOv2 + compile |
| IPC round-trip | <0.1ms | CPU shared memory |
| Robot send_action | <1ms | Feetech serial, non-blocking |

Typical S1 loop interval: ~34ms (~29Hz). Inference spikes to ~59ms under S2 GPU contention.

### Optimization stack (cumulative)
| Optimization | Inference time | Savings |
|-------------|---------------|---------|
| Baseline (fp32, 15 steps) | ~95ms | — |
| + TF32 matmul | ~54ms | 41ms |
| + bf16 autocast | ~25ms | 29ms |
| + Cross-attention KV cache | ~14ms (isolated) | 11ms |
| + Batched DINOv2 (4 cameras) | marginal | ~1ms |
| + torch.compile (denoise_step) | ~2ms saved | ~2ms |
| Real-world (10 steps, w/ S2) | 22-42ms | — |

## TODO

### High priority
- [ ] **S1 inference thread latency** — obs→infer gap is 8-51ms (typical ~14ms, spikes to 51ms) due to `threading.Event.wait()` OS scheduler jitter. Options: (1) busy-wait with CPU yield, (2) double-buffered obs with atomic swap, (3) redesign so inference thread owns obs capture.
- [ ] **S1 GPU priority over S2** — S1 inference spikes from 22ms to 59ms under S2 contention. Options: (1) CUDA MPS for proper time-slicing, (2) separate GPUs, (3) Triton kernel optimization.
- [ ] S2 LoRA finetuning — S2 cannot produce subtask transitions (stuck on "pick up the cylinder"). Plan: LoRA rank 32 on both Gemma 2B (q/k/v/o_proj, 18 layers) + SigLIP (QKV+out, 27 layers) ≈ 24M trainable params. Two losses: subtask cross-entropy (weight=10) + FAST token cross-entropy (weight=1). Needs per-frame `high_level_task` + `low_level_subtask` annotations.
- [ ] S2 latent normalization — z-score normalize S2 latents before feeding to S1 (44× cosine gap improvement validated). Requires S1 retraining.
- [ ] Generic SharedMemoryStore to replace per-type IPC classes
- [ ] Pre-decode and cache training images to avoid repeated video decode (main training bottleneck)

### Medium priority
- [ ] **Training DataLoader optimization** — GPU drops to 10% for ~2s every ~7s (video decode stall). Candidates: `persistent_workers=True` + `prefetch_factor=3` in DataLoader, pre-decode images to memmap. Needs benchmarking to measure actual improvement.
- [ ] `torch.compile` for S1 training (mode=default, skip DINOv2 — needs investigation)
- [ ] Separate backbone LR (lower for DINOv2 pretrained weights, matching ACT's approach)
- [ ] wandb integration for training monitoring
- [ ] Inference-time RTC guidance (LeRobot's `RTCProcessor`) on top of training-time RTC for extra smoothness
- [ ] Corrupt JPEG data — "premature end of data segment" from OpenCV cameras (not seen recently, may be resolved)

### Done
- [x] Intervention / inverse follow (`--teleop-config`, `--intervention-dataset`)
- [x] Inference episode recording to LeRobotDataset (`--record-dataset`)
- [x] Robot-agnostic control loop (joint names derived from robot, no hardcoded SO107)
- [x] Extracted helpers: `load_robot()`, `load_s1_policy()`, `create_recording_dataset()`
- [x] Bidirectional action attention (Pi0-style, `tgt_mask=None`) — v6
- [x] Cross-attention KV cache (pre-compute K,V from static context)
- [x] Batched DINOv2 (4 cameras in one forward pass)
- [x] bf16 autocast + TF32 matmul
- [x] torch.compile denoise_step
- [x] Image resize on CPU (cv2.resize, 0.7ms/4 images)
- [x] Persistent S2 process (s2_standalone)
- [x] Observation processor steps (DepthEdgeOverlayProcessorStep)
- [x] Grip drop diagnostics with inference-time obs saving
- [x] Soft landing with hold-position during torque ramp-down

### Future
- [ ] Co-training S1 + S2 (currently sequential: extract latents → train S1)
- [ ] Adaptive action horizon (short chunks for precision, long for transit)
- [ ] Switch S1 to using Pi0-style action expert architecture (proven at scale, matches Ψ₀)
- [ ] ONNX/TensorRT export for S1 (potential 2× inference speedup)
