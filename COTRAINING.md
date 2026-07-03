# Co-Training XLerobot with External Datasets

How to leverage external datasets (ALOHA, Open-X-Embodiment, other LeRobot-format
repos) to improve policies for the **xlerobot** embodiment (18-dim state/action,
cameras `head` / `left_wrist` / `right_wrist`, 30 fps, 360×640), and how to keep
improving them with human-in-the-loop (HIL) and RL data flywheels.

Companion files:
- `src/lerobot/data_processing/co_training_utils.py` — alignment + merge tool
  (also installed as `lerobot-cotrain-align`).
- `Guide.sh` — full copy-pasteable training commands (sections B-3, B-4, RA-BC).

---

## 1. The method in one picture

```
 external repos          your task repos              deployment
 (ALOHA / Open-X)        (trash_pickup, tool_pickup, …)
      │                        │
      ▼                        │
 [1] lerobot-cotrain-align     │        ← resample fps, remap cameras,
      │  (align + merge) ◄─────┘          remap joints by name, resize,
      ▼                                   re-encode to one schema
 merged co-train dataset
      │
      ▼
 [2] generalist pre-train (SmolVLA, ~60k steps)
      │
      ▼
 [3] per-task fine-tune (lower LR, ~20k steps)      ┌──────────────┐
      │           └── optional: RA-BC weighting     │  more data   │
      ▼                                             │  (HIL / RL)  │
 [4] deploy → run policy with VR intervention ──────►              │
      ▲                                             └──────┬───────┘
      └────────────── retrain on grown dataset ◄───────────┘
```

**Why a physical merge?** In this fork `MultiLeRobotDataset` is disabled
(`src/lerobot/datasets/factory.py` raises `NotImplementedError`), so passing a
JSON list of repo ids to `--dataset.repo_id` does **not** work. Instead,
external data is *aligned* to the exact xlerobot schema and *physically merged*
into one `LeRobotDataset` with `aggregate_datasets()`. This also guarantees a
single set of normalization statistics — critical when mixing embodiments.

What alignment does per source (see module docstring for details):
1. Temporal resample to a common fps (nearest-neighbour when downsampling).
2. State/action remapped **by joint name** against a reference xlerobot repo
   (`--match-features-from`), zero-padding missing joints and dropping extras.
3. Camera keys renamed (`top`/`cam_high` → `head`, wrist cams → `left_wrist` /
   `right_wrist`, …) and frames resized to one resolution.
4. Everything re-encoded so all shards have *identical* `features` dicts —
   `aggregate_datasets()` compares full feature dicts, not just shapes.

---

## 2. Step-by-step

### Step 0 — Inspect compatibility

```bash
lerobot-cotrain-align \
    --source-repos Odog16/trash_pickup Odog16/tool_pickup lerobot/aloha_mobile_cabinet \
    --target-repo-id dummy --inspect-only
```

Prints action/state dims, fps, and camera keys per repo so you can pick the
camera remap and confirm the reference schema.

### Step 1 — Align the external repos

```bash
lerobot-cotrain-align \
    --source-repos \
        lerobot/aloha_sim_insertion_human \
        lerobot/aloha_sim_transfer_cube_human \
        lerobot/aloha_mobile_cabinet \
        lerobot/aloha_static_battery \
    --target-repo-id Odog16/aloha_aligned_for_cotrain \
    --target-fps 30 \
    --target-image-size 360x640 \
    --target-state-dim 18 \
    --target-action-dim 18 \
    --match-features-from Odog16/tool_pickup \
    --camera-remap "top:head,cam_high:head,cam_left_wrist:left_wrist,cam_right_wrist:right_wrist" \
    --output-root ~/.cache/huggingface/lerobot/Odog16/aloha_aligned_for_cotrain \
    --push-to-hub false
```

Notes:
- **Always pass `--match-features-from <one of your xlerobot repos>`** so the
  state/action `names` lists match; the merge fails without it.
- Aligned shards are cached under `_align_tmp_*`; re-runs are incremental.
  `--force-rebuild` wipes stale caches, `--skip-missing-sources` tolerates 404s.

### Step 2 — Verify

```bash
cat ~/.cache/huggingface/lerobot/Odog16/aloha_aligned_for_cotrain/meta/info.json
# expect: fps 30, robot_type xlerobot, action/state shape [18],
#         video.height 360, video.width 640
```

Then re-run the Step 0 inspect with your task repos + the aligned repo — every
row should now show identical dims/fps/cameras.

### Step 3 — Merge with your task data (one co-train dataset)

Run the same tool once more with *all* sources — your xlerobot repos are
passed through in place (their metadata already matches), only the merge runs:

```bash
lerobot-cotrain-align \
    --source-repos \
        Odog16/trash_pickup Odog16/tool_pickup \
        Odog16/block_sorting_single Odog16/block_transfer \
        Odog16/aloha_aligned_for_cotrain \
    --target-repo-id Odog16/xlerobot_cotrain_v1 \
    --target-fps 30 --target-image-size 360x640 \
    --target-state-dim 18 --target-action-dim 18 \
    --match-features-from Odog16/tool_pickup \
    --output-root ~/.cache/huggingface/lerobot/Odog16/xlerobot_cotrain_v1 \
    --push-to-hub false
```

**Mixing ratio matters.** ALOHA repos contain far more episodes than your task
repos; if external data dominates, the policy drifts from your embodiment's
action distribution. Rules of thumb: keep external data ≤ 50–70 % of merged
frames for the generalist phase, and control the ratio by choosing *which*
external repos to include (there is no per-source weighting in the BC trainer).

### Step 4 — Phase 1: generalist pre-train

```bash
lerobot-train \
    --dataset.repo_id=Odog16/xlerobot_cotrain_v1 \
    --dataset.root=$HOME/.cache/huggingface/lerobot/Odog16/xlerobot_cotrain_v1 \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.train_state_proj=true \
    --policy.use_amp=true \
    --batch_size=16 --steps=60000 --save_freq=10000 \
    --output_dir=outputs/train/xlerobot_generalist_v1 \
    --job_name=xlerobot_generalist_v1 \
    --wandb.enable=true --wandb.project=lerobot
```

Language conditioning (SmolVLA) is what lets one policy absorb many tasks —
each merged episode keeps its own task string. The generalist checkpoint is
the reusable artifact: redo this step only when you add major new data.

### Step 5 — Phase 2: per-task specialist fine-tune

Fine-tune *from the generalist* on each single-task repo with lower LR and
fewer steps (full commands: `Guide.sh` step B-4):

```bash
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=smolvla \
    --policy.pretrained_path=outputs/train/xlerobot_generalist_v1/checkpoints/last/pretrained_model \
    --scheduler.type=cosine_decay_with_warmup --scheduler.peak_lr=5e-5 \
    --batch_size=16 --steps=20000 \
    --output_dir=outputs/train/trash_pickup_cotrain_v1
```

### Step 6 (optional but recommended) — RA-BC fine-tune

Reward-Aligned BC re-weights samples toward high-quality demonstrations
instead of treating all frames equally:

```bash
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=smolvla \
    --policy.pretrained_path=outputs/train/xlerobot_generalist_v1/checkpoints/last/pretrained_model \
    --use_rabc=true --rabc_head_mode=sparse --rabc_kappa=0.01 \
    --batch_size=16 --steps=20000 \
    --output_dir=outputs/train/trash_pickup_cotrain_rabc_v1
```

(`--use_rabc` & friends are legacy aliases auto-migrated to
`--sample_weighting.type=rabc`; see `src/lerobot/configs/train.py`.)
Progress/subtask annotations come from the SARM tooling in
`src/lerobot/data_processing/sarm_annotations/`.

---

## 3. Human-in-the-loop and RL: improving the policy *and* the data

Co-trained BC gets you a competent baseline. The gap to reliable performance is
closed by collecting the *right* new data — data from the states the policy
actually visits, labelled by what the human had to fix. This fork already has
the pieces:

### 3a. Intervention recording (DAgger-style flywheel)

The VR teleop exposes an intervention interface consumed by the HVLA runtime
(`s1_process.py`): **RIGHT A** toggles human takeover; while active the human
corrects the robot and frames are recorded with the intervention flag
(`hil_processor.AddTeleopEventsAsInfoStep`), then policy control resumes.

Loop:
1. Deploy the current specialist checkpoint.
2. Run episodes; intervene only when the policy goes wrong.
3. The recorded episodes are gold: they cover exactly the failure states of the
   current policy (on-policy states, expert corrections) — far more valuable
   per-frame than fresh full demonstrations.
4. Merge the new episodes into the task repo and re-run the Phase-2 (or RA-BC)
   fine-tune. 10–30 intervention episodes per iteration is typically enough to
   see a measurable success-rate jump.

### 3b. Reward-weighted data reuse (RA-BC + SARM)

Not all demos are equal. SARM subtask/progress annotation
(`data_processing/sarm_annotations/subtask_annotation.py`) scores episodes;
RA-BC (`--sample_weighting.type=rabc`) then up-weights smooth, successful
segments and down-weights hesitation and recovery noise. This *creates more
usable data* out of what you already have — old mediocre recordings still
contribute their good segments instead of being deleted.

### 3c. HIL-SERL: online RL with human oversight (`src/lerobot/rl/`)

For a single high-value task, run the actor/learner pair:

- `lerobot.rl.gym_manipulator` wraps the real robot as a gym env with
  intervention support, time limits, and ROI cropping (`crop_dataset_roi.py`).
- A **reward classifier** (configured via `processor.reward_classifier`,
  trained from success/failure frames you label) provides the sparse reward,
  so no hand-coded reward function is needed.
- `rl/actor.py` runs on the robot host collecting transitions (human can
  intervene at any time — interventions go into the replay buffer as
  high-quality actions); `rl/learner.py` trains SAC (`rl/algorithms/sac/`)
  off-policy on a GPU machine; `rl/data_sources/data_mixer.py` mixes offline
  demos with online transitions.
- Warm-start the buffer with your existing demonstrations so RL starts from
  the BC baseline instead of exploring from scratch.

HIL-SERL-style training typically reaches near-100 % success on a scoped task
within 1–3 hours of real-robot interaction, and every transition it collects is
also valid BC data for the next generalist merge.

### 3d. Which to use when

| Situation | Tool |
|---|---|
| New task, no data | Record 50+ demos → co-train (this doc §2) |
| Policy ~60–80 % success, fails in specific states | Intervention recording (§3a) → RA-BC fine-tune |
| Plenty of mixed-quality demos | SARM annotate + RA-BC (§3b) |
| One critical task needs near-perfect reliability | HIL-SERL online RL (§3c) |
| Many tasks, plateauing generalist | Add aligned external data, re-merge, redo Phase 1 |

The compounding loop: **deploy → intervene → annotate → re-weight → retrain →
deploy**. Each pass both improves the policy and grows a dataset that is
increasingly concentrated on hard states — which is exactly the data the next
generalist merge benefits from most.
