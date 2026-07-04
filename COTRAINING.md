# Co-Training XLerobot

How to train xlerobot policies on more than one dataset: merge your task
repos with converted external data, pre-train a generalist, fine-tune
specialists. Converting external data into the xlerobot schema →
[`DATA_CONVERSION.md`](./DATA_CONVERSION.md). Improving deployed policies
with human feedback → [`DAGGER_HIL.md`](./DAGGER_HIL.md). Full training
commands with all flags → `Guide.sh`.

```
 external data                  your task repos
 (ALOHA / Open-X / FastUMI)     (trash_pickup, tool_pickup, …)
      │                              │
      ▼  DATA_CONVERSION.md          │
 converted to xlerobot schema        │
      └──────────────┬───────────────┘
                     ▼
 [1] lerobot-cotrain-align  → one merged co-train dataset
                     ▼
 [2] generalist pre-train (SmolVLA, ~60k steps)
                     ▼
 [3] per-task specialist fine-tune (~20k steps, lower LR)
                     ▼            └── optional RA-BC weighting
 [4] deploy → DAgger corrections (DAGGER_HIL.md) → retrain
```

**Why a physical merge?** `MultiLeRobotDataset` is disabled in this fork
(`datasets/factory.py` raises `NotImplementedError`), so JSON-list
`--dataset.repo_id` does **not** work. `lerobot-cotrain-align` merges
schema-identical datasets with `aggregate_datasets()` and yields one set of
normalization statistics — critical when mixing embodiments.

---

## 1. Merge

After converting external sources (see DATA_CONVERSION.md), merge everything:

```bash
lerobot-cotrain-align \
    --source-repos \
        Odog16/trash_pickup Odog16/tool_pickup \
        Odog16/block_sorting_single Odog16/block_transfer \
        Odog16/aloha_aligned_for_cotrain \
        Odog16/umi_pretrain_clean_desktop \
    --target-repo-id Odog16/xlerobot_cotrain_v1 \
    --target-fps 30 --target-image-size 360x640 \
    --target-state-dim 18 --target-action-dim 18 \
    --match-features-from Odog16/tool_pickup \
    --push-to-hub false
```

Sources already matching the merge schema are passed through in place; only
the merge runs. **Mixing ratio matters**: keep external data ≤ 50–70 % of
merged frames so the policy doesn't drift from your embodiment's action
distribution — control the ratio by choosing which external repos (and
`--max-episodes` during conversion) to include.

## 2. Phase 1 — generalist pre-train

```bash
lerobot-train \
    --dataset.repo_id=Odog16/xlerobot_cotrain_v1 \
    --dataset.root=$HOME/.cache/huggingface/lerobot/Odog16/xlerobot_cotrain_v1 \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.train_state_proj=true --policy.use_amp=true \
    --batch_size=16 --steps=60000 --save_freq=10000 \
    --output_dir=outputs/train/xlerobot_generalist_v1 \
    --wandb.enable=true --wandb.project=lerobot
```

Language conditioning is what lets one policy absorb many tasks — each merged
episode keeps its own task string. The generalist checkpoint is the reusable
artifact; redo this phase only when you add major new data.

**UMI as a pretraining option.** With FastUMI retargeted at `--target-fps 30`
(DATA_CONVERSION.md §2), UMI tasks join this merge like any other source. If
you want a *UMI-heavy* pretrain instead (e.g. 10+ FastUMI tasks dwarfing your
own data), split Phase 1 in two: first train on the UMI-heavy merge, then
continue on an xlerobot-only merge before per-task fine-tuning — that keeps
the final action distribution native while still inheriting the manipulation
prior.

## 3. Phase 2 — per-task specialists (+ RA-BC)

Fine-tune from the generalist on each single-task repo with lower LR and
fewer steps (full commands: `Guide.sh` B-4 / RA-BC sections):

```bash
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=smolvla \
    --policy.pretrained_path=outputs/train/xlerobot_generalist_v1/checkpoints/last/pretrained_model \
    --scheduler.type=cosine_decay_with_warmup --scheduler.peak_lr=5e-5 \
    --batch_size=16 --steps=20000 \
    --output_dir=outputs/train/trash_pickup_cotrain_v1
```

Add `--use_rabc=true --rabc_head_mode=sparse --rabc_kappa=0.01` to weight
training toward high-quality demonstrations (see DAGGER_HIL.md §2).

## 4. Close the loop

Deploy the specialist and collect DAgger corrections with the VR teleop
(DAGGER_HIL.md §1); merge the correction episodes back into the task repo and
re-run Phase 2. This loop is where success rates actually climb.

---

## 5. Getting large amounts of valid xlerobot data

Ranked by **value per hour of your time** for this embodiment:

1. **DAgger corrections (highest value per frame).** Once any specialist
   runs, stop recording full demos — deploy and correct
   (`lerobot-rollout --strategy.type=dagger`). Corrections are on-policy
   states with expert actions, targeting exactly what the policy gets wrong.
   20 correction episodes routinely beat 100 fresh demos for improving an
   existing policy. Cost: ~30–60 min per iteration.

2. **VR teleop demonstrations (the gold standard for new tasks).** Native
   embodiment, native cameras, native action distribution — nothing converted
   data can match. Throughput is the limit (~40–60 episodes/hour with the
   Quest 3). Maximize *validity* while recording: vary object positions every
   episode, vary lighting/background every session, use RIGHT B to cut failed
   episodes early, and write distinct task strings — language variety is what
   the generalist feeds on. 50+ episodes/task is the floor; diminishing
   returns typically start ~150–200 without distribution shifts.

3. **Autonomous rollout harvesting (free data while you do something else).**
   Run specialists with `lerobot-rollout --strategy.type=sentry` (continuous
   recording with episode rotation) or `dagger --strategy.record_autonomous=true`.
   Raw autonomous data is mixed-quality by definition — make it valid by
   filtering: SARM-annotate and RA-BC-weight it, or keep only
   success-classified episodes. Scales with robot uptime, not human time.

4. **Mine the data you already have (SARM + RA-BC).** Old mixed-quality
   recordings, aborted sessions, superseded datasets: annotate and re-weight
   instead of deleting (DAGGER_HIL.md §2). Zero collection cost.

5. **Retargeted UMI data (scale, with an asterisk).** FastUMI-100K offers
   100k+ trajectories — 3 orders of magnitude above your per-task counts —
   via `lerobot-umi-retarget` (DATA_CONVERSION.md §2). The visual and
   workspace gaps make it a pretraining prior rather than in-domain data:
   expect it to improve the generalist's manipulation priors and language
   grounding, not to teach xlerobot-specific control. Start with the 5–10
   tasks closest to yours; keep it a minority of merged frames or use the
   two-stage pretrain (§2 above).

6. **Aligned ALOHA / Open-X (bimanual priors).** Real-robot joint-space data
   with a smaller domain gap than UMI on the action side (joints → joints),
   but a bigger camera/embodiment mismatch. Useful in the generalist merge at
   modest fractions; already wired via `lerobot-cotrain-align`.

7. **HIL-SERL transitions (small but perfect).** Online RL on a scoped
   single-arm task (DAGGER_HIL.md §3) produces on-policy, reward-labelled
   transitions that double as BC data. Use for the last mile on critical
   tasks, not for bulk collection.

The multiplier across all of these is the flywheel: every deployment session
should *record* (sentry/DAgger), every recording should get *scored*
(SARM/reward classifier), and every training round should *re-weight*
(RA-BC). Data that only sits on disk is the only invalid data.
