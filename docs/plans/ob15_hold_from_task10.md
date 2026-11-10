# Task 10 handoff: "the dataset remembers the hold pose" (`hold_from`)

For: Claude Code session **PC2 training**. Implement, test and debug on PC2.
Parent plan: `docs/plans/ob15_arms_robot.md` (task 10). Tasks 1–9 are done in commit `3a99cff`.

## 0. Setup

```bash
git fetch origin
git checkout ob15-arms-robot && git pull
git checkout -b ob15-arms-hold-from
uv sync --locked --extra test --extra dev --extra feetech --extra dataset   # adjust extras if names differ
uv run pytest tests/robots/test_ob15_arms.py -q      # must pass BEFORE any change (baseline)
```

Git author: use the GitHub noreply address (`git config user.email "<id>+ImpurestTadpole@users.noreply.github.com"`), not a personal email.

Rules: extend the existing code, do not rewrite it. Keep the flat config layout. Do not change `ob15` / `ob15_client` behavior or non-OB15 record/train behavior.

## 1. Goal

- Recording with `ob15_arms` / `ob15_arms_client` writes the resolved head/lift hold pose to `<dataset_root>/meta/ob15_hold.json`.
- Training copies that file to `<checkpoint>/pretrained_model/ob15_hold.json`.
- At inference `--robot.hold_from=<dataset or policy, local path or Hub repo_id>` moves head/lift to the saved pose.
- Hold precedence: **explicit flag** (`head_pan` / `head_tilt` / `lift_height_mm`) **> `hold_from` > `station` > current position at connect**.

## 2. Existing code to build on (read first)

| File | What is there |
|---|---|
| `src/lerobot/robots/ob15/arms_only.py` | `ARM_KEYS`, `HEAD_KEYS`, `BASE_KEYS`, `lift_key()`, `hold_keys()`, `resolve_hold_targets(config, current_obs, lift_name)` (current < station < overrides), `hold_status()`, `hold_reached()`, `build_arms_command()`, `arms_only_result()`, constants `HEAD_TOL_DEG`, `LIFT_TOL_MM` |
| `src/lerobot/robots/ob15/ob15_arms.py` | `OB15Arms`: `connect()` → `_establish_hold()` (settle loop with module const `HOLD_SETTLE_TIMEOUT_S`), `check_hold()`, `_NoTargets` stand-in (used when `hold_on_connect=False`) |
| `src/lerobot/robots/ob15/ob15_arms_client.py` | `OB15ArmsClient`: same pattern over ZMQ (`_wait_first_observation`, `_push`, settle loop), its own `_NoTargets` |
| `src/lerobot/robots/ob15/config_ob15.py` | `OB15ArmsConfig`, `OB15ArmsClientConfig` with flat fields `station`, `stations_file`, `head_pan`, `head_tilt`, `lift_height_mm`, `hold_on_connect` |
| `src/lerobot/robots/ob15/stations.py` | JSON loader/saver pattern to copy (atomic write via `.tmp` + `os.replace`) |
| `src/lerobot/scripts/lerobot_record.py` | `record()`: dataset `resume` (~L601) / `create` (~L633); `robot.connect()` (~L658); `push_to_hub` (~L851) |
| `src/lerobot/scripts/lerobot_train.py` | `save_checkpoint(...)` call (~L1008), `checkpoint_dir = get_step_checkpoint_dir(...)`; `PRETRAINED_MODEL_DIR` from `lerobot.utils.constants` |
| `tests/robots/test_ob15_arms.py` | Fixtures to reuse: `_FakeHardware`, `arms_robot`, `_make_bus`, client ZMQ tests, conversion tests with `full_dataset` |

Line numbers are approximate; search by name.

## 3. Implementation

### 3a. Config (both `OB15ArmsConfig` and `OB15ArmsClientConfig`)
Add flat fields (each is a CLI flag via draccus):
```python
hold_from: str | None = None                 # dataset root / Hub dataset id / checkpoint dir / Hub model id
rest_pose: dict[str, float] | None = None    # arm/gripper targets sent before head/lift move
settle_timeout_s: float = 15.0
lift_tol_mm: float = 5.0
head_tol: float = 2.0
```
Update the docstring precedence line.

### 3b. New `src/lerobot/robots/ob15/hold_file.py`
- Constants: `HOLD_FILENAME = "ob15_hold.json"`, `DATASET_HOLD_PATH = "meta/ob15_hold.json"`, `HOLD_FILE_VERSION = 1`.
- File format:
  ```json
  {"version": 1, "robot_type": "ob15_arms_client",
   "hold": {"gantry.height_mm": 220.0, "head_pan.pos": 0.0, "head_tilt.pos": -35.0},
   "source": {"station": null, "hold_from": null, "overrides": {"head_pan": null, "head_tilt": -35.0, "lift_height_mm": 220.0}, "hold_on_connect": true},
   "recorded_at": "<ISO UTC>"}
  ```
- `build_hold_metadata(robot_type, hold, source) -> dict`
- `holds_match(a, b, head_tol, lift_tol) -> bool` (same keys, each within tolerance; `.height_mm` keys use `lift_tol`)
- `write_hold_file(dataset_root, meta, *, overwrite=False, head_tol, lift_tol) -> Path`
  - New file → atomic write.
  - Existing file + `overwrite=False` → keep it; if `holds_match` is False, `logger.warning` naming both poses and the flag `--dataset.overwrite_hold=true`.
- `read_hold_file(source: str) -> dict[str, float]`, search order:
  1. `source` is a file → parse it.
  2. `source` is a dir → `<dir>/meta/ob15_hold.json`, `<dir>/ob15_hold.json`, `<dir>/pretrained_model/ob15_hold.json`.
  3. Else Hub: `huggingface_hub.hf_hub_download(repo_id=source, filename="meta/ob15_hold.json", repo_type="dataset")`, then `filename="ob15_hold.json", repo_type="model"`.
  4. Nothing found → `FileNotFoundError` listing every path tried.
  - Validate: `hold` is a non-empty object of numbers (reject bools).
- `copy_hold_file_to_checkpoint(dataset_root, pretrained_dir) -> bool` (no-op, returns False, if source file absent).

### 3c. `arms_only.py`
- `resolve_hold_targets`: insert `hold_from` between station and overrides. Keys in the file that are not hold keys → ignored with a warning; hold keys missing from the file → fall back to lower-priority sources.
- `hold_source(config) -> dict` (station, hold_from, overrides, hold_on_connect) for the metadata.
- `rest_pose_command(config) -> dict | None`: returns arm-only targets; `ValueError` if `rest_pose` has any non-`ARM_KEYS` key.

### 3d. Robot classes (`ob15_arms.py`, `ob15_arms_client.py`)
- Replace module const `HOLD_SETTLE_TIMEOUT_S` and default tolerances with `config.settle_timeout_s`, `config.head_tol`, `config.lift_tol_mm` (in settle loop and `check_hold`).
- Connect sequence:
  1. read current pose → resolve hold (use `_NoTargets` when `hold_on_connect=False`; add `hold_from = None` to both `_NoTargets` classes).
  2. If `rest_pose`: send `build_arms_command(rest, freeze)` where `freeze` = current head/lift values (head/lift do not move yet); short wait (~2 s).
  3. Set `self._hold`, log it, run the settle loop sending `build_arms_command(rest or {}, self._hold)`.
- Add `hold_metadata() -> dict` → `build_hold_metadata(self.name, self._hold, hold_source(self.config))`; raise `DeviceNotConnectedError` if not connected.

### 3e. Record hook (`lerobot_record.py`)
- Add `overwrite_hold: bool = False` to the record dataset config dataclass (find where `push_to_hub`, `private`, `tags` are defined).
- Right after `robot.connect()` (dataset is already created or resumed):
  ```python
  if dataset is not None and hasattr(robot, "hold_metadata"):
      from lerobot.robots.ob15.hold_file import write_hold_file
      cfg_r = robot.config
      write_hold_file(dataset.root, robot.hold_metadata(), overwrite=cfg.dataset.overwrite_hold,
                      head_tol=cfg_r.head_tol, lift_tol=cfg_r.lift_tol_mm)
  ```
- Confirm `meta/ob15_hold.json` is uploaded by `dataset.push_to_hub(...)` (it uploads the dataset folder; check `allow_patterns`/`ignore_patterns` if any). If not uploaded, add an explicit `HfApi().upload_file(...)` after the push.

### 3f. Train hook (`lerobot_train.py`)
- After `save_checkpoint(...)`, on the main process only:
  ```python
  from lerobot.robots.ob15.hold_file import copy_hold_file_to_checkpoint
  copy_hold_file_to_checkpoint(dataset_root, checkpoint_dir / PRETRAINED_MODEL_DIR)
  ```
  `dataset_root` = the training dataset's local root (`dataset.root` or `dataset.meta.root`; check what exists; for multi-dataset training, only copy when exactly one source has a hold file, else log and skip).
- Also copy for the Hub push path if checkpoints are pushed (`save_checkpoint_to_hub`): the file is inside `pretrained_model/`, check it is included.
- Must be a no-op for non-OB15 datasets.

### 3g. Docs
Update `docs/source/ob15_arms.mdx`: hold file, `hold_from`, precedence, `rest_pose`, tolerances, the commands in section 5.

## 4. Tests (extend `tests/robots/test_ob15_arms.py`; mocks only, no hardware)

1. `write_hold_file` → `read_hold_file` round trip: local dataset dir (`meta/`), local checkpoint dir (`pretrained_model/`), direct file path.
2. `read_hold_file` errors: missing everywhere → `FileNotFoundError` listing tried paths; bad JSON; empty `hold`; non-number value.
3. Hub path: patch `huggingface_hub.hf_hub_download` → dataset hit; dataset miss + model hit; both miss.
4. Precedence (one test per level, using the `arms_robot`-style fixture):
   - current only; station > current; `hold_from` > station; flag > `hold_from`.
   - `hold_from` file missing a key → that key falls back to station/current.
5. `hold_on_connect=False` ignores `hold_from` (freezes current pose).
6. `rest_pose`: arms command sent first with head/lift frozen at current values; non-arm key → `ValueError`.
7. Settle uses config tolerances/timeout (set `settle_timeout_s` tiny + a fake that never reaches target → warning, no hang).
8. `hold_metadata()` contents: robot_type, hold, source fields.
9. Record hook (unit-level: call the helper the hook uses): new file written; resume with matching pose → unchanged; mismatching pose → warning + unchanged; `overwrite=True` → replaced.
10. `copy_hold_file_to_checkpoint`: copies when present, returns False when absent.
11. Client: same precedence via `OB15ArmsClient` with mocked ZMQ (reuse existing client test pattern).

Mark tests with the right pytest marks per `CLAUDE.md` (`hardware` / `dataset`) so they run in CI tiers.

## 5. Verify

```bash
uv run pytest tests/robots/test_ob15_arms.py -svv
uv run pytest tests/robots -q                       # nothing else broke
pre-commit run --files $(git diff --name-only origin/ob15-arms-robot)
```

On the robot (when available):
```bash
# Record a few episodes; check <root>/meta/ob15_hold.json exists and matches the flags
lerobot-record --robot.type=ob15_arms_client --robot.lift_height_mm=220 --robot.head_tilt=-35 \
  --dataset.repo_id=<user>/ob15_hold_test --dataset.single_task="test" --dataset.num_episodes=2 ...

# Restart with different manual lift height, then restore from the dataset
lerobot-rollout --robot.type=ob15_arms_client --robot.hold_from=<user>/ob15_hold_test --policy.path=...
```
Checks: lift within ±5 mm and head within ±2 of the file; log states which source won; flag beats `hold_from`.

## 6. Debug tips

- `hold_from` path vs Hub: a non-existent local path falls through to the Hub; log the final source used at INFO.
- Lift heights are relative to the homed zero: home the lift before checking values.
- If settle never completes on the client, print `last_remote_state` keys; the lift key is `f"{lift_axis.name}.height_mm"` (default `gantry`).
- If record hook fails on resume, check `dataset.root` exists before `robot.connect()` returns.

## 7. Done when

- All tests in section 4 pass; existing tests unchanged and passing; pre-commit clean on changed files.
- Commit(s) on `ob15-arms-hold-from`, pushed to `origin`. Open a PR into `ob15-arms-robot` (not `main`) with a summary and test output.
- Report back: what changed, test results, anything not finished or unverified on hardware.

## Prompt to paste into the PC2 session

> Read `docs/plans/ob15_hold_from_task10.md` on branch `ob15-arms-robot` of my fork. Follow it: create branch `ob15-arms-hold-from`, run the baseline tests, implement sections 3a–3g, write the tests in section 4, run section 5, debug until green, commit with my GitHub noreply email, push, and open a PR into `ob15-arms-robot`. Then report what changed and the test results.
