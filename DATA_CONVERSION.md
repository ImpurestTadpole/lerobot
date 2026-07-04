# Data Conversion for XLerobot

How to convert external datasets into the **xlerobot schema** (18-dim
state/action named `left_arm_*` / `right_arm_*` / `head_*` / `x.vel` … /
`gantry.height_mm`, cameras `head` / `left_wrist` / `right_wrist` at 360×640,
30 fps) so they can be merged and co-trained. What to do with the converted
data → [`COTRAINING.md`](./COTRAINING.md). Improving policies with human
feedback → [`DAGGER_HIL.md`](./DAGGER_HIL.md).

There are two converters, chosen by what the source data contains:

| Source data | Action space | Tool |
|---|---|---|
| Robot datasets with **joint** states (ALOHA, Open-X, other LeRobot repos) | joints | `lerobot-cotrain-align` |
| UMI-style datasets with **end-effector poses** (FastUMI-100K) | EE poses | `lerobot-umi-retarget` |

Both are registered CLIs (`pyproject.toml [project.scripts]`); run `--help`
for every flag.

---

## 1. Joint-space sources: `lerobot-cotrain-align`

Aligns each source (fps resample, joint remap **by name**, camera rename,
resize, re-encode) and physically merges everything into one dataset.
`MultiLeRobotDataset` is disabled in this fork, so a physical merge is the
only supported multi-dataset training path — it also gives one set of
normalization statistics, which matters when mixing embodiments.

```bash
# Inspect compatibility first (dims / fps / cameras per repo):
lerobot-cotrain-align \
    --source-repos Odog16/trash_pickup lerobot/aloha_mobile_cabinet \
    --target-repo-id dummy --inspect-only

# Align ALOHA repos to the xlerobot schema:
lerobot-cotrain-align \
    --source-repos \
        lerobot/aloha_sim_insertion_human \
        lerobot/aloha_mobile_cabinet \
        lerobot/aloha_static_battery \
    --target-repo-id Odog16/aloha_aligned_for_cotrain \
    --target-fps 30 \
    --target-image-size 360x640 \
    --target-state-dim 18 --target-action-dim 18 \
    --match-features-from Odog16/tool_pickup \
    --camera-remap "top:head,cam_high:head,cam_left_wrist:left_wrist,cam_right_wrist:right_wrist" \
    --push-to-hub false
```

Rules of thumb:

- **Always pass `--match-features-from <one of your xlerobot repos>`** —
  `aggregate_datasets()` compares full feature dicts including joint-name
  lists, not just shapes. Missing joints are zero-filled, extras dropped.
- Aligned shards cache under `_align_tmp_*`; re-runs are incremental.
  `--force-rebuild` wipes stale caches; `--skip-missing-sources` tolerates
  Hub 404s.
- Merging takes the **camera intersection** and the **minimum fps** across
  sources — convert everything to the same fps/cameras first if you don't
  want the merge to downgrade your task repos.

### Verify

```bash
cat ~/.cache/huggingface/lerobot/<repo>/meta/info.json
# expect: fps 30, robot_type xlerobot, state/action shape [18],
#         video.height 360, video.width 640
```

Then re-run `--inspect-only` with your task repos + the converted repo —
every row should show identical dims/fps/cameras.

---

## 2. UMI / end-effector sources: `lerobot-umi-retarget`

[FastUMI-100K](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot)
stores 100k+ handheld-gripper trajectories across 54 tasks — but as **relative
EE poses** (verified: `state[t]` = absolute pose relative to episode start,
`action[t] == state[t+1]`, meters/radians, gripper width 0–1) in LeRobot v2.1,
with wrist fisheye cameras only. Joint-name alignment can't ingest that;
retargeting converts the geometry.

### Recommended: batch mode (download → retarget → print the merge command)

```bash
lerobot-umi-retarget \
    --source-root ~/fastumi \
    --tasks dual_arm/Clean_Desktop dual_arm/Dispose_of_Desktop_Debris \
    --download-repo IPEC-COMMUNITY/FastUMI_100k_lerobot \
    --target-repo-id Odog16/umi_pretrain \
    --target-fps 30 \
    --match-features-from Odog16/tool_pickup \
    --max-episodes 200
```

This downloads just those tasks, writes one v3.0 dataset per task
(`Odog16/umi_pretrain_clean_desktop`, …) in the HF cache, and prints the
ready-to-run `lerobot-cotrain-align` command to merge them with your task
repos. Completed outputs are reused, so the command is resumable.

What happens inside (two stages):

1. **EE extraction** — integrates the episode-relative poses, maps axes
   (`--axis-map`), scales human reach into the SO-100 workspace (`--scale`,
   default 0.3), anchors episode starts mid-workspace. Defaults were tuned on
   real FastUMI data: the VR rest anchor left 46–87 % of frames unreachable;
   the mid-workspace anchor + scale 0.3 leaves 0 %.
2. **IK baking** (`--action-space joint`, default) — the same decomposed IK
   the VR teleop uses (`SO101Kinematics`) produces 18-dim joint vectors with
   names copied from `--match-features-from`. Episodes still exceeding
   `--max-clamp-frac` workspace violations are skipped and reported.

Key flags:

- `--target-fps 30` — linear EE interpolation + nearest video frame, so the
  output merges with your 30 fps repos without dragging them to 20 fps.
  Actions are re-derived as the one-step state shift, preserving
  `action[t] == state[t+1]` at any fps.
- `--action-space ee` — skips IK and writes robot-frame EE vectors with
  FastUMI names instead. Use this for a future EE-space policy (deployed with
  the stock `InverseKinematicsEEToJoints` processor — see the
  [action representations docs](https://huggingface.co/docs/lerobot/en/action_representations)).
  Both modes share stage 1, so joint- and EE-datasets stay consistent.
- `--head-fill black` (default) — UMI has no head camera; a black stream
  keeps the 3-camera schema so merges don't strip `head` from your task
  repos. `none` omits it (wrist-only pretraining; SmolVLA's `empty_cameras`
  covers the missing slot).
- `single_arm/*` tasks map to `--target-arm` (default `right`); the other
  wrist is black-filled.

### Expectations

Retargeted UMI data is a **pretraining prior, not in-domain data**: fisheye
optics and a handheld gripper in frame are a real visual gap, and workspace
scaling compresses spatial layout. Start with the 5–10 tasks closest to yours
and measure generalist transfer before scaling to all 100k trajectories.

---

## 3. Other format notes

- **LeRobot v2.1 → v3.0**: `uv run python src/lerobot/scripts/convert_dataset_v21_to_v30.py`
  converts standard v2.1 repos. (`lerobot-umi-retarget` reads FastUMI's v2.1
  layout directly — no pre-conversion needed.)
- **DAgger/rollout datasets** (`rollout_*` repos from `lerobot-rollout`) carry
  an extra boolean `intervention` feature. Training on them directly works
  (unused keys are ignored); `lerobot-cotrain-align` drops the extra feature
  when merging them into a task repo.
- **Depth streams** survive alignment only if *every* merged source has them
  (camera-intersection rule).
