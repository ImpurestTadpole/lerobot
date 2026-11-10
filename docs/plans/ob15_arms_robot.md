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

> Commit `3a99cff` already implements tasks 1 and 2–9 for stations. Remaining work: task 1b (skills) and the skill-related parts of 2, 3, 4, 5, 8 and 9. Extend the existing code; do not rewrite it.

### 1. Station file + loader
- New `robots/ob15/stations.py`:
  - `load_station(path: str, name: str) -> dict[str, float]` → keys `head_pan.pos`, `head_tilt.pos`, `<lift>.height_mm`.
  - `save_station(path, name, values)` (used later by a VR "Save station" button; not wired now).
- Default path `~/.ob15/stations.json`. Example:
  ```json
  {"kitchen_counter": {"head_pan.pos": 0.0, "head_tilt.pos": -35.0, "gantry.height_mm": 220}}
  ```
- Clear error if the file or station is missing.

### 1b. Skill file + loader (per-skill lift/head pose)
- A **skill** = one short, reusable action with a fixed setup (station, lift height, head angle) and one prompt, run by one policy. A **task** = a user goal made of several skills, sequenced by a script or the ROS 2 behavior tree.
- New `robots/ob15/skills.py`:
  - `@dataclass SkillPose`: `station: str | None`, `lift_height_mm`, `head_pan`, `head_tilt`, `task_prompt` (all optional).
  - `load_skill(path: str, name: str) -> SkillPose` (YAML, `yaml.safe_load`). Clear `KeyError` if missing.
- Default path `~/.ob15/skills.yaml`. Same file is read later by the ROS 2 behavior tree. Example:
  ```yaml
  skills:
    pick_cup_counter:
      station: kitchen_counter
      lift_height_mm: 220
      head_pan: 0.0
      head_tilt: -35.0
      task_prompt: "pick up the cup"
    load_dishwasher_low:
      station: kitchen_counter
      lift_height_mm: 120
      head_tilt: -50.0
      task_prompt: "put the plate in the dishwasher"
  ```
- Heights live in this file, **not** in code defaults.

### 2. Configs (`robots/ob15/config_ob15.py`)

**Status:** base version done in commit `3a99cff` (flat fields `station`, `stations_file`, `head_pan`, `head_tilt`, `lift_height_mm`, `hold_on_connect` on `OB15ArmsConfig` and `OB15ArmsClientConfig`). **Keep the flat layout** so existing flags keep working. Add these flat fields to both configs:

- `skill: str | None = None`, `skills_file: str = "~/.ob15/skills.yaml"`
- `rest_pose: dict[str, float] | None = None` (arm joints to move to before the lift/head move; None = skip)
- `settle_timeout_s: float = 15.0`, `lift_tol_mm: float = 5.0`, `head_tol: float = 2.0`

All fields are CLI flags via draccus, e.g. `--robot.skill=pick_cup_counter`, `--robot.lift_height_mm=230`.

**Priority** for each held value: explicit override (`head_pan` / `head_tilt` / `lift_height_mm`) > skill > skill's `station` (or `station` field) > current position read at connect.

### 3. Shared key helpers (`robots/ob15/arms_only.py`)
- `ARM_KEYS`: the 12 arm/gripper `.pos` keys, same order as `_state_ft`.
- `BASE_KEYS = ("x.vel", "y.vel", "theta.vel")`.
- `arms_action_features() -> dict[str, type]`.
- `arms_state_features(full_state_ft) -> dict` (drops `BASE_KEYS`, keeps order).
- `resolve_hold_targets(config, current_obs, lift_name) -> dict[str, float]` implementing the priority rule above. Returns `{"<lift>.height_mm", "head_pan.pos", "head_tilt.pos"}`.
- `wait_until_settled(read_obs_fn, targets, lift_tol_mm, head_tol, timeout_s) -> bool` (logs a warning on timeout).

### 4. `OB15Arms(OB15)` (`robots/ob15/ob15_arms.py`)
- `config_class = OB15ArmsConfig`, `name = "ob15_arms"`.
- Override `action_features` → `arms_action_features()`.
- Override `observation_features` → `arms_state_features(self._state_ft)` + `self._cameras_ft`.
- `connect()`: `super().connect()`, read observation, move arms to `rest_pose` if set, compute `self._hold` with `resolve_hold_targets`, send it, `wait_until_settled`, log the final hold values.
- `set_skill(name: str)`: same sequence (rest pose → resolve → move → settle). For use **between** episodes only (later: VR button / key). Never during an episode.
- `hold_metadata() -> dict`: skill name, station, resolved hold values. Write this into the dataset (`meta/info.json` extra field or a sidecar `meta/ob15_hold.json`) at record start.
- `get_observation()`: call super, pop `BASE_KEYS`.
- `send_action(action)`: keep only `ARM_KEYS`, merge `self._hold`, call `super().send_action(...)`, return only `ARM_KEYS` from the result.

### 5. `OB15ArmsClient(OB15Client)` (`robots/ob15/ob15_arms_client.py`)
- Same overrides as task 4, applied to the client features.
- `send_action`: keep `ARM_KEYS` + `self._hold` + explicit zeros for `BASE_KEYS`, send JSON to host, return only `ARM_KEYS`.
- Hold targets from the station file on PC1, or from the first received observation.

### 6. Registration
- `robots/ob15/__init__.py`: export the new classes/configs.
- `robots/utils.py`: add `"ob15_arms"` and `"ob15_arms_client"` branches.
- Check `scripts/lerobot_record.py`, `lerobot_teleoperate.py`, `lerobot_replay.py`, `lerobot_rollout.py`: they import `ob15` for registration; confirm the new configs register through that import (decorators live in `config_ob15.py`).

### 7. Dataset conversion script (`src/lerobot/robots/ob15/scripts/convert_to_arms_only.py`)
- Input: full-body OB15 dataset. Output: new dataset (new `repo_id`), original untouched.
- `action`: keep `ARM_KEYS` columns (by `names` in metadata).
- `observation.state`: drop `BASE_KEYS`.
- Copy cameras/depth/tasks/episodes unchanged. Recompute stats.
- `--report` flag: per episode, max range of head/lift/base dims (to find episodes where they moved a lot).

### 8. Tests (`tests/robots/test_ob15_arms.py`, mark `hardware` if needed, use mocks)
- `action_features` has exactly 12 keys in order.
- `observation_features` has no `BASE_KEYS`.
- `send_action` with extra base/head/lift keys → those keys never change from hold values; base keys sent as 0.
- Station loader: missing file, missing station.
- Skill loader: missing file, missing skill, partial entries.
- `resolve_hold_targets` priority: override > skill > station > current (one test per level).
- Conversion script on a tiny synthetic dataset: shapes and names correct.

### 9. Docs
- `docs/source/ob15_arms.mdx` (short): purpose, skill vs task, `skills.yaml` + `stations.json` formats, flags, commands below.

## Commands to verify on the robot

```bash
# Record one skill (VR teleop still works; base/head/lift inputs are ignored)
lerobot-record --robot.type=ob15_arms_client --robot.skill=pick_cup_counter \
  --dataset.single_task="pick up the cup" ...

# Quick override test
lerobot-record --robot.type=ob15_arms_client --robot.skill=pick_cup_counter \
  --robot.lift_height_mm=230 ...

# Run a policy with the same skill pose
lerobot-rollout --robot.type=ob15_arms_client --robot.skill=pick_cup_counter --policy.path=...
```

Checks:
1. Connect → arms to rest pose, then head and lift move to the skill pose, then stay still.
2. Move VR thumbsticks for base/head/lift during recording → nothing moves.
3. Recorded dataset: `action` shape (12,), state has no base velocity, hold metadata saved.
4. Rollout with the same skill → lift within ±5 mm and head within ±2 of the recorded hold values.
5. Old dataset converted → loads with `LeRobotDataset`, trains with `lerobot-train` for a few steps.

Rules: always home the lift at connect for task sessions (heights are relative to the homed zero). One dataset per skill. Same skill pose in recording and inference.

## Done when
- `uv run pytest tests/robots/test_ob15_arms.py` passes.
- `pre-commit run --all-files` passes on changed files.
- Checks 1–4 pass on OB15.
- Existing `ob15` / `ob15_client` behavior unchanged.

## Out of scope (later)
- VR button to toggle between full and arms-only control in one session, "Save station/skill" wiring, and calling `set_skill` from a VR button between episodes.
- Mismatch guard at inference (warn if skill hold values differ from the training dataset's saved hold metadata).
- ROS 2 behavior tree integration (`SetLift`, `LookAt`, `RunPolicy` nodes).
