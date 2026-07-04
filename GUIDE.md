# GUIDE.md — Running the Full XLerobot Co-Training Pipeline

The end-to-end recipe: record one master dataset, derive training views,
co-train with external data (ALOHA / Open-X / UMI) without inheriting a
padding bias, and fine-tune per-task specialists. Every step can be driven
from the **workflow UI** or from the raw CLI commands listed here — they are
the same commands; the UI just generates and runs them for you.

Deep dives: [`COTRAINING.md`](./COTRAINING.md) (concepts + flags),
[`DATA_CONVERSION.md`](./DATA_CONVERSION.md) (external → xlerobot schema),
[`DAGGER_HIL.md`](./DAGGER_HIL.md) (corrections loop),
[`RGBD_IMPLEMENTATION_GUIDE.md`](./RGBD_IMPLEMENTATION_GUIDE.md) (depth),
`Guide.sh` (hardware/session reference).

---

## 1. The workflow UI: access & execution model

```bash
uv run python scripts/workflow_ui/app.py        # → http://127.0.0.1:7799
```

- **Access.** The server binds to loopback only (no auth), so open the URL in
  a browser **on the same machine**. Working from another computer? Tunnel it:
  `ssh -L 7799:localhost:7799 <user>@<training-box>` then browse
  `http://localhost:7799` locally. Only use `--host 0.0.0.0` on a network you
  fully trust — anyone who can reach the port can run shell commands.
- **Execution.** *Run* / *Run all* execute stages **directly on the server's
  machine** — no copy-pasting. Each enabled stage is rendered to a standalone
  script under `workflows/<name>/scripts/NN_*.sh` (also via *Generate
  scripts*), so any stage can equally be run by hand, in tmux, or copied to
  another box. Scripts are self-contained: they `cd` to the repo root, set
  `set -euo pipefail`, and clear `PYTHONPATH` (Isaac Sim's bundled packages
  otherwise break torch imports).
- **Three tabs.** *Workflow* (compose/run stages), *Compare datasets*
  (preflight external repos — see step 4), *Guide* (this pipeline, color-coded).
  Every stage option has a **`?` key** next to its label — a one-glance
  explanation with a small depiction of what the option does; the colored
  stage-type badge does the same for the whole stage.
- **Fastest start:** click **+ Master RGB-D / UMI preset**, edit the variables
  at the top (repo names, step counts), and run stages top to bottom.

Logs land in `workflows/<name>/logs/`; run state survives page reloads; a
failing stage halts the sequence.

## 2. Phase 0 — record the master dataset (once)

Record with **everything on**: all 18 DOF (lift axis enabled) and RGB-D
(`use_depth=True` on the head RealSense is now the default in
`config_xlerobot.py`, adding `observation.images.head_depth`).

```bash
uv run lerobot-record \
    --robot.type=xlerobot \
    --robot.lift_axis.enabled=true \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/master_home_v1 \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up the trash and place it in the bin"
```

Recording discipline (it's what makes the data *valid*): vary object
positions every episode, vary lighting/background every session, cut failed
episodes early (RIGHT B on the Quest), write distinct task strings. 50+
episodes per task is the floor.

## 3. Derive training views (never re-record for a schema)

```bash
uv run lerobot-extract-subset \
    --source-repo Odog16/master_home_v1 \
    --target-repo-id Odog16/master_home_v1_bimanual12 \
    --profile bimanual12 --target-image-size 360x640
```

`bimanual12` keeps the 12 arm joints by name and drops depth — tensor shapes
then match open-source bimanual datasets exactly, so **those dims never need
padding**. Other options: `arms_head14`, `full18`, explicit `--keep-names`,
`--target-fps`, `--keep-depth`. UI: the teal **extract** stage.

## 4. Preflight every external dataset (before spending encode time)

UI → **Compare datasets**: reference = your task repo (e.g.
`Odog16/tool_pickup`), candidate = the external repo. You get: which joints
would be padded, how far a zero-fill lands from your real motion range (in
σ), camera/fps compatibility, episode-ratio warnings, and side-by-side
[HF dataset visualizer](https://github.com/huggingface/lerobot-dataset-visualizer)
embeds. **Apply to workflow** writes the recommended flags into your merge
stage. If *nothing* maps, the source needs conversion first:

```bash
# ALOHA / Open-X → xlerobot schema (see DATA_CONVERSION.md §1)
# UMI / FastUMI  → joint or EE space:
uv run lerobot-umi-retarget --source-repo ... --target-fps 30   # DATA_CONVERSION.md §2
```

## 5. Merge (with padding-bias mitigation on)

```bash
uv run lerobot-cotrain-align \
    --source-repos Odog16/master_home_v1_bimanual12 \
                   Odog16/umi_pretrain_clean_desktop \
                   Odog16/aloha_aligned_for_cotrain \
    --target-repo-id Odog16/cotrain_umi_bimanual_12dof \
    --target-fps 30 --target-image-size 360x640 \
    --target-state-dim 12 --target-action-dim 12 \
    --match-features-from Odog16/master_home_v1_bimanual12 \
    --pad-fill-mode ref-mean --override-padded-stats \
    --push-to-hub false
```

The three mitigations (all detailed in COTRAINING.md): `--pad-fill-mode
ref-mean` (padded dims sit at your mean pose, not 0), `--override-padded-stats`
(your stats stay authoritative), and the auto-written
`meta/cotrain_sources.json` manifest that enables train-time source
weighting. For the native merge, repeat with your 18-DOF task repos and
`--target-*-dim 18 --match-features-from Odog16/tool_pickup`.

## 6. Two-stage pre-train (UMI at scale)

**Stage 1 — 12-DOF, UMI/bimanual-heavy** (UMI may dominate; all names match):

```bash
uv run lerobot-train \
    --dataset.repo_id=Odog16/cotrain_umi_bimanual_12dof \
    --policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base \
    --policy.train_state_proj=true --policy.use_amp=true \
    --batch_size=16 --steps=40000 --save_freq=10000 \
    --sample_weighting.type=source --sample_weighting.external_weight=0.3 \
    --output_dir=outputs/train/stage1_umi_12dof \
    --wandb.enable=true --wandb.project=lerobot
```

**Stage 2 — native 18-DOF**, continuing from stage 1 (SmolVLA pads
state/action internally, so the 12→18 change is safe with
`train_state_proj=true`):

```bash
uv run lerobot-train \
    --dataset.repo_id=Odog16/cotrain_native_18dof \
    --policy.type=smolvla \
    --policy.pretrained_path=outputs/train/stage1_umi_12dof/checkpoints/last/pretrained_model \
    --policy.train_state_proj=true --policy.use_amp=true \
    --batch_size=16 --steps=30000 --save_freq=10000 \
    --output_dir=outputs/train/stage2_native_18dof \
    --wandb.enable=true --wandb.project=lerobot
```

Skipping UMI? Collapse this to the single generalist pre-train of the
**Home-tasks preset** (60k steps on one 18-DOF merge).

## 7. SARM annotation → RA-BC (optional but cheap quality win)

```bash
uv run python -m lerobot.rewards.sarm.compute_rabc_weights \
    --dataset-repo-id Odog16/trash_pickup \
    --reward-model-path Odog16/sarm_xlerobot \
    --head-mode sparse --device cuda
```

Writes `sarm_progress.parquet` next to the dataset. Annotate **after** any
merge/extract (frame indices must match), keep head-mode consistent with
training, and use it where data is mixed-quality (autonomous harvests, old
recordings). UI: the pink **sarm** stage.

## 8. Fine-tune per-task specialists

```bash
uv run lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=smolvla \
    --policy.pretrained_path=outputs/train/stage2_native_18dof/checkpoints/last/pretrained_model \
    --scheduler.type=cosine_decay_with_warmup --scheduler.peak_lr=5e-5 \
    --batch_size=16 --steps=20000 \
    --output_dir=outputs/train/trash_pickup_specialist
    # + --use_rabc=true --rabc_head_mode=sparse --rabc_kappa=0.01  (if annotated)
```

One fine-tune per home task, always from the same generalist checkpoint.
Evaluate **base/lift/head motion specifically** — not just gripper success —
to confirm the padding bias washed out.

## 9. Deploy & close the loop

```bash
uv run lerobot-rollout --strategy.type=dagger ...    # DAGGER_HIL.md §1
```

Deploy the specialist, collect VR corrections, merge them back into the task
repo, re-run step 8. 20 correction episodes routinely beat 100 fresh demos.
Between iterations, harvest autonomous data (`--strategy.type=sentry`) and
feed it through step 7.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `AttributeError: ... ParamSpec ... not writable` on any Python command | Isaac Sim's `PYTHONPATH` is shadowing the venv — run with `PYTHONPATH=` prefix (UI-generated scripts already do). |
| UI port busy (`Address already in use`) | `fuser -k 7799/tcp` or `--port 7800`. |
| A Hub repo 404s in merge/compare | Private or typo'd — `huggingface-cli login`, or `--skip-missing-sources`. |
| Merge fails on camera keys | Sources share no `observation.images.*` after remap — add `--camera-remap src:head,...` (Compare tab shows the mapping). |
| Stale aligned caches after changing flags | Usually auto-detected; force with `--force-rebuild`. |
| Compare tab visualizer iframes blank | The HF Space needs internet and public repos; the mapping report above them works regardless. |
