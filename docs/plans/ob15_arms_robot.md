# Plan: `ob15_arms` robot type (arms-only control, station-held head/lift/base)

Branch: `ob15-arms-robot`. Execute on PC1. Do not change the existing `ob15` / `ob15_client` behavior.

## Goal

Add an OB15 robot type for task recording and policy runs:

- **Actions:** 12 values only (both arms + grippers).
- **Observation state:** arms + grippers + head pan/tilt + lift height. **No base velocity.**
- **Head + lift:** moved to a saved station pose at connect, then held. **Base:** always zero velocity.
- Any base/head/lift keys from VR teleop or a policy are dropped.
- Cameras and depth unchanged.

The full-body `ob15` type stays for VR setup, navigation and recovery.

## Facts from the current code (check before editing)

- `robots/ob15/ob15.py`: `OB15._state_ft` = 12 arm keys + `head_pan.pos`, `head_tilt.pos` + `x.vel`, `y.vel`, `theta.vel` + `{lift_axis.cfg.name}.height_mm` (default name `gantry`). `action_features = _state_ft`.
- `OB15.send_action` filters by key prefix. Missing base keys → zero velocity (EMA decays).
- `robots/ob15/ob15_client.py`: `OB15Client` has its own `_state_ft` (head only if `config.use_head`), sends the action dict as JSON to the host, returns values in `_state_order`.
- `robots/ob15/ob15_host.py`: runs the robot via `make_robot_from_config`. **Leave the host as full `ob15`.** Filtering and holding happen on the client.
- `robots/utils.py`: `make_robot_from_config` has `elif config.type == "ob15"` / `"ob15_client"` branches.
- `utils/feature_utils.py::build_dataset_frame` selects values by feature names, so extra teleop keys are ignored when recording.

## Tasks

> Commit `3a99cff` already implements tasks 1–9 (arms-only classes, stations, conversion script, tests, docs). **Remaining work = task 10 below.** Extend the existing code; do not rewrite it. Skills (`skills.yaml`) are **deferred** — see "Out of scope".

### 1–9. Done in `3a99cff`
- Arms-only robot types `ob15_arms` / `ob15_arms_client` (12 actions, no base velocity in state, head/lift held, base zero).
- `stations.py` (`~/.ob15/stations.json`), flat config fields `station`, `stations_file`, `head_pan`, `head_tilt`, `lift_height_mm`, `hold_on_connect`.
- Conversion script `data_processing/convert_to_arms_only.py`, tests `tests/robots/test_ob15_arms.py`, docs `docs/source/ob15_arms.mdx`.
- Current hold priority: explicit flag > station > current position at connect.

### 10. "The dataset remembers the hold pose"

> Detailed handoff for implementation and tests: `docs/plans/ob15_hold_from_task10.md`.

Goal: record the lift/head hold values with each dataset, and at inference move to those exact values automatically. No numbers to retype; no hard-coded heights in source.

**10a. Config fields** (add to both `OB15ArmsConfig` and `OB15ArmsClientConfig`, flat layout, all are CLI flags):
- `hold_from: str | None = None` — a local dataset root, a Hub dataset `repo_id`, a local policy checkpoint dir, or a Hub model `repo_id`.
- `rest_pose: dict[str, float] | None = None` — arm joints to move to before the lift/head move (None = skip).
- `settle_timeout_s: float = 15.0`, `lift_tol_mm: float = 5.0`, `head_tol: float = 2.0`.

**New priority** for each held value: explicit flag (`lift_height_mm` / `head_pan` / `head_tilt`) > `hold_from` > `station` > current position at connect.

**10b. Hold file format** — `ob15_hold.json`:
```json
{
  "version": 1,
  "robot_type": "ob15_arms_client",
  "hold": {"gantry.height_mm": 220.0, "head_pan.pos": 0.0, "head_tilt.pos": -35.0},
  "source": {"station": "kitchen_counter", "flags": {"lift_height_mm": 220.0}},
  "recorded_at": "<ISO timestamp>"
}
```
- Keys in `hold` match the robot feature names (lift key = `f"{lift_axis.name}.height_mm"`).

**10c. Helpers** (`robots/ob15_arms/arms_only.py` or new `robots/ob15_arms/arms_only.py`):
- `hold_metadata(robot) -> dict` — builds the JSON above from the resolved hold.
- `write_hold_file(dataset_root: Path, meta: dict)` → writes `<root>/meta/ob15_hold.json`.
- `read_hold_file(source: str) -> dict[str, float]`:
  1. Local dir: try `<source>/meta/ob15_hold.json` (dataset), then `<source>/ob15_hold.json` (checkpoint, also check `<source>/pretrained_model/ob15_hold.json`).
  2. Else Hub: `huggingface_hub.hf_hub_download(repo_id=source, filename="meta/ob15_hold.json", repo_type="dataset")`; on failure try `filename="ob15_hold.json", repo_type="model"`.
  3. Clear error naming every path tried.
- `resolve_hold_targets(config, current_obs, lift_name)` — update to the new priority.
- `wait_until_settled(read_obs_fn, targets, lift_tol_mm, head_tol, timeout_s) -> bool` — log a warning on timeout.

**10d. Robot classes** (`ob15_arms.py`, `ob15_arms_client.py`):
- `connect()`: super → read observation → move arms to `rest_pose` if set → resolve hold → send → `wait_until_settled` → log final hold values.
- Add `hold_metadata()` method (wraps 10c).

**10e. Recording hook** (`scripts/lerobot_record.py`):
- After the dataset is created/opened and the robot is connected: `if hasattr(robot, "hold_metadata"): write_hold_file(dataset.root, robot.hold_metadata())`.
- When resuming an existing dataset: if a hold file already exists and differs by more than the tolerances, **log a warning** (do not overwrite silently; add flag `--dataset.overwrite_hold=false` default).
- The file must be included in `push_to_hub` (it sits in `meta/`, check it uploads).

**10f. Training hook (small, optional but recommended)** (`scripts/lerobot_train.py`):
- When saving a checkpoint, if `<dataset.root>/meta/ob15_hold.json` exists, copy it to `<checkpoint>/pretrained_model/ob15_hold.json`. Then `--robot.hold_from=<policy path>` works and the policy carries its own pose.
- Guard so non-OB15 datasets are unaffected.

**10g. Tests** (extend `tests/robots/test_ob15_arms.py`, mocks only):
- Write → read round trip (local dataset dir and local checkpoint dir).
- Priority: flag > `hold_from` > station > current (one test per level).
- Missing file → clear error listing tried paths.
- Hub path: mock `hf_hub_download`.
- Record hook: hold file written once; mismatch on resume → warning, no overwrite.

**10h. Docs**: update `docs/source/ob15_arms.mdx` with the hold file, `hold_from`, and the commands below.

## Commands to verify on the robot

```bash
# Record (hold values saved to <dataset>/meta/ob15_hold.json)
lerobot-record --robot.type=ob15_arms_client \
  --robot.lift_height_mm=220 --robot.head_tilt=-35 \
  --dataset.repo_id=<user>/ob15_pick_cup --dataset.single_task="pick up the cup" ...

# Inference: pose read from the training dataset ...
lerobot-rollout --robot.type=ob15_arms_client \
  --robot.hold_from=<user>/ob15_pick_cup --policy.path=...

# ... or from the policy checkpoint (if 10f is done)
lerobot-rollout --robot.type=ob15_arms_client \
  --robot.hold_from=<path or repo of policy> --policy.path=<same>
```

Checks:
1. Connect → arms to rest pose (if set), then head and lift move to the hold pose, then stay still.
2. Move VR thumbsticks for base/head/lift during recording → nothing moves.
3. Dataset has `meta/ob15_hold.json` locally and on the Hub; `action` shape (12,); no base velocity in state.
4. Rollout with `hold_from` → lift within ±5 mm and head within ±2 of the saved values.
5. Explicit flag overrides `hold_from` (log shows which source won).

Rules: always home the lift at connect for task sessions (heights are relative to the homed zero). One hold pose per dataset.

## Done when
- `uv run pytest tests/robots/test_ob15_arms.py` passes.
- `pre-commit run --all-files` passes on changed files.
- Checks 1–5 pass on OB15.
- Existing `ob15` / `ob15_client` behavior and non-OB15 recording/training unchanged.

## Out of scope (later)
- **Skills registry** (`skills.yaml`: station + lift + head + prompt per skill, `--robot.skill=NAME`, `set_skill()` between episodes). Add when there are >~5 skills or when the ROS 2 behavior tree runs task sequences. It will read/write the same `hold` format.
- VR button to toggle full vs arms-only control in one session, and "Save station" wiring.
- ROS 2 behavior tree integration (`SetLift`, `LookAt`, `RunPolicy` nodes).
