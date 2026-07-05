# Co-Training XLerobot

How to train xlerobot policies on more than one dataset: merge your task
repos with converted external data, pre-train a generalist, fine-tune
specialists. Converting external data into the xlerobot schema →
[`DATA_CONVERSION.md`](./DATA_CONVERSION.md). Improving deployed policies
with human feedback → [`DAGGER_HIL.md`](./DAGGER_HIL.md). Full training
commands with all flags → `Guide.sh`.

**Workflow UI:** the whole pipeline below (and variations of it) can be laid
out, script-generated, and run stage-by-stage from a local web UI:

```bash
uv run python scripts/workflow_ui/app.py        # → http://127.0.0.1:7799
```

Three tabs: **Workflow** (compose stages — merge / train / extract / SARM /
custom — with variables, generate standalone `.sh` scripts under
`workflows/<name>/scripts/`, run sequentially with live logs), **Compare
datasets** (preflight any external repo against your reference schema:
per-joint padding report, zero-fill severity in σ, camera/fps checks,
side-by-side [HF dataset visualizer](https://github.com/huggingface/lerobot-dataset-visualizer)
embeds, one-click apply of recommended mitigation flags), and **Guide**
(this document as color-coded phases). Two presets: **Home-tasks** (merge →
generalist → per-task fine-tunes) and **Master RGB-D / UMI** (§0/§2 below).
Generated stage scripts `conda activate lerobot` before running anything —
`lerobot-train` and friends need that env (GPU torch, matching Python), not
the repo's separate `uv`-managed `.venv` (see GUIDE.md §1). The commands
below assume `conda activate lerobot` is already active in your shell.

## 0. Record once: the master dataset

Record teleop sessions with **all 18 DOF and RGB-D active**
(`use_depth=True` on the head RealSense is now the xlerobot default —
recordings gain `observation.images.head_depth`). That master dataset is the
future-proof source of truth; derive per-run training views instead of
re-collecting:

```bash
lerobot-extract-subset \
    --source-repo Odog16/master_home_v1 \
    --target-repo-id Odog16/master_home_v1_bimanual12 \
    --profile bimanual12          # both 6-DOF arms, depth dropped
```

Profiles: `bimanual12` (12 arm dims — tensor shapes match open-source
bimanual datasets exactly, so those dims never need padding), `arms_head14`,
`full18`, or an explicit `--keep-names` list; optional `--target-fps` /
`--target-image-size` / `--keep-depth`. The subset keeps task strings and
drops straight into `lerobot-cotrain-align` as a source (or as the
`--match-features-from` reference for a 12-DOF merge).

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
    --pad-fill-mode ref-mean --override-padded-stats \
    --push-to-hub false
```

Sources already matching the merge schema are passed through in place; only
the merge runs. **Mixing ratio matters**: keep external data ≤ 50–70 % of
merged frames so the policy doesn't drift from your embodiment's action
distribution — control the ratio by choosing which external repos (and
`--max-episodes` during conversion) to include.

### Padding-bias mitigation

External sources cannot supply every xlerobot dim (a single-arm repo pads 11
of 18 joints). Padding those dims with **0** teaches the generalist a "frozen
joints" bias *and* pollutes the merged normalization stats. Three mitigations
are built in — the first two happen at merge time:

- `--pad-fill-mode ref-mean` — padded dims take the reference repo's per-joint
  **mean** instead of 0, so they sit at the centre of your real distribution.
  Velocity dims (`x.vel`, `y.vel`, `theta.vel`) stay 0 ("stopped" is correct).
  `state-copy` goes further: padded non-velocity *action* dims copy the
  same-name state value — an identity "hold position" action.
- `--override-padded-stats` — after merging, `meta/stats.json` entries for the
  padded dims are rewritten with the reference repo's mean/std/min/max, so
  your real base/lift/head motion isn't normalized into the tails.
- **Source down-weighting at train time** — the merge always writes
  `meta/cotrain_sources.json` (episode ranges + padded dims per source). Add
  `--sample_weighting.type=source` to `lerobot-train` and episodes from padded
  external sources get `--sample_weighting.external_weight` (default 0.3)
  instead of 1.0, shrinking the supervised-to-constant gradient without
  discarding their visual/language diversity. Requires the merged dataset to
  be local (`--dataset.root` or the HF cache).

Changing `--pad-fill-mode` invalidates the per-source `_align_tmp_*` caches
automatically (a `meta/cotrain_align_options.json` marker is compared).

## 2. Phase 1 — generalist pre-train

```bash
lerobot-train \
    --dataset.repo_id=Odog16/xlerobot_cotrain_v1 \
    --dataset.root=$HOME/.cache/huggingface/lerobot/Odog16/xlerobot_cotrain_v1 \
    --policy.type=smolvla --policy.push_to_hub=false \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.train_state_proj=true --policy.use_amp=true \
    --batch_size=16 --steps=60000 --save_freq=10000 \
    --sample_weighting.type=source --sample_weighting.external_weight=0.3 \
    --output_dir=outputs/train/xlerobot_generalist_v1 \
    --wandb.enable=true --wandb.project=lerobot
```

The `sample_weighting` lines down-weight padded external episodes (see
*Padding-bias mitigation* above); drop them for an all-native merge.
Per-repo overrides: `--sample_weighting.source_weights='{"lerobot/aloha_mobile_cabinet": 0.5}'`.

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

**UMI at scale (the Master RGB-D / UMI preset).** The two-stage split works
best in *12-DOF space*: stage 1 merges your `bimanual12` master subset (§0)
with UMI and bimanual sources — 12 names all match, so UMI can dominate the
mix without padding bias — and trains from `smolvla_base` with source
weighting; stage 2 continues that checkpoint on a native 18-DOF merge so the
final action distribution is yours. SmolVLA pads state/action internally, so
the 12→18 change is safe with `--policy.train_state_proj=true`. Start with
the 5–10 UMI tasks closest to your home tasks, and preflight every retargeted
repo in the UI's *Compare datasets* tab before spending encode time on it.

## 3. Phase 2 — per-task specialists (+ RA-BC)

Fine-tune from the generalist on each single-task repo with lower LR and
fewer steps (full commands: `Guide.sh` B-4 / RA-BC sections):

```bash
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=smolvla --policy.push_to_hub=false \
    --policy.pretrained_path=outputs/train/xlerobot_generalist_v1/checkpoints/last/pretrained_model \
    --scheduler.type=cosine_decay_with_warmup --scheduler.peak_lr=5e-5 \
    --batch_size=16 --steps=20000 \
    --output_dir=outputs/train/trash_pickup_cotrain_v1
```

Add `--use_rabc=true --rabc_head_mode=sparse --rabc_kappa=0.01` to weight
training toward high-quality demonstrations (see DAGGER_HIL.md §2). Generate
the required `sarm_progress.parquet` first — the workflow UI's **sarm** stage
runs it, or directly:

```bash
python -m lerobot.rewards.sarm.compute_rabc_weights \
    --dataset-repo-id Odog16/trash_pickup \
    --reward-model-path Odog16/sarm_xlerobot --head-mode sparse --device cuda
```

Annotate *after* any merge/extract (frame indices must match the dataset you
train on), and keep `--head-mode` consistent with `--rabc_head_mode`.

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
