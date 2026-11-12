---
name: run-lerobot
description: Build, install, and run LeRobot's CLI (lerobot-train, lerobot-eval, lerobot-record, etc). Use when asked to run LeRobot, train or eval a policy, verify the installation works, smoke-test the CLI, or check that lerobot-train/lerobot-eval still work after a change.
---

LeRobot is a Python CLI suite (`lerobot-train`, `lerobot-eval`,
`lerobot-record`, `lerobot-teleoperate`, …) installed as console scripts
into the project venv — no GUI, no server, no port to manage. Drive it via
`.claude/skills/run-lerobot/smoke.sh`, which runs a real (tiny) end-to-end
train + eval cycle through the actual CLI entry points. All paths below are
relative to the repo root.

## Prerequisites

Base install needs nothing beyond what's in `pyproject.toml`. The smoke
test additionally needs the `pusht` simulator extra (not installed by
`uv sync` alone) and `accelerate` (declared under the `training` extra):

```bash
source .venv/bin/activate
python -m pip install "gym-pusht>=0.1.5,<0.2.0" "pymunk>=6.6.0,<7.0.0" accelerate
```

No `apt-get` packages were needed on this (Ubuntu/aarch64, Jetson) box —
the pusht simulator is pure Python + pygame/pymunk, no display required.

## Setup / Build

```bash
uv sync --locked          # or: pip install -e . if uv is unavailable
```

If `uv` isn't on `PATH` (it wasn't in this container), the project's own
`.venv` already has everything else needed to run the CLI; just `source
.venv/bin/activate` and use `python -m pip install <pkg>` (not bare `pip`
— see Gotchas) for anything extra.

## Run (agent path)

```bash
source .venv/bin/activate
bash .claude/skills/run-lerobot/smoke.sh            # writes to tests/outputs/act_smoke
```

What it does (verified this session, ~60s on CPU on a Jetson Orin):
1. `lerobot-train` — trains a tiny ACT policy (`dim_model=64`, 4 steps,
   batch_size=2) on the real `lerobot/pusht` dataset from the Hub,
   checkpointing at steps 2 and 4 and running an in-sim eval rollout at
   each checkpoint (writes an MP4 of the PushT rollout).
2. Asserts the checkpoint (`model.safetensors`) and eval video exist.
3. `lerobot-eval` — re-loads the step-4 checkpoint standalone and runs
   eval against it again, independent of the training loop.

Prints `==> SMOKE TEST PASSED` on success and exits non-zero on any
`test -f` assertion failure. Pass an alternate output dir as `$1`:

```bash
bash .claude/skills/run-lerobot/smoke.sh /tmp/my_smoke_run
```

To pull a viewable frame out of the produced rollout video (useful for
visually confirming the sim actually ran):

```bash
ffmpeg -y -i tests/outputs/act_smoke/eval/videos_step_000004/pusht_0/eval_episode_0.mp4 \
  -vf "select=eq(n\,2)" -vframes 1 /tmp/pusht_frame.png
```

This was done during development of this skill — the frame showed the
PushT T-block, pusher, and goal marker, confirming the sim renders
correctly headless.

## Run (human path)

Any of the `project.scripts` entries in `pyproject.toml` work directly
once the venv is active, e.g.:

```bash
lerobot-train --policy.type=act --env.type=pusht --dataset.repo_id=lerobot/pusht ...
lerobot-eval --policy.path=<checkpoint_dir> --env.type=pusht ...
```

`lerobot-record`/`lerobot-teleoperate`/`lerobot-calibrate` etc. need real
robot/camera hardware attached and were not exercised by this skill.

## Test

```bash
source .venv/bin/activate
PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  "tests/policies/test_policies.py::test_policy_defaults[act]" \
  "tests/policies/test_policies.py::test_act_backbone_lr" \
  "tests/policies/test_policies.py::test_act_temporal_ensembler" -q
# → 3 passed in ~8s
```

The `PYTHONPATH=` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` are required in
this container — see Gotchas.

## Gotchas

- **`pytest`/`pip` are not preinstalled in `.venv`.** This venv was
  created without `pip`/pytest bootstrapped. Fix: `python -m ensurepip
  --upgrade` once, then `python -m pip install pytest`. Always invoke
  install via `python -m pip install ...` with the venv's `python`
  explicit — a bare `pip install` on this box resolves to
  `/home/jetson/miniforge3/bin/pip` (conda base env) instead of the
  venv, silently installing into the wrong environment.
- **`PYTHONPATH` pollution from ROS.** This machine has ROS2 Humble on
  the environment's `PYTHONPATH`. `pytest`'s plugin autoloader picks up
  `launch_testing` from `/opt/ros/humble/...`, which then fails on a
  missing `lark` module and crashes pytest's own startup before any test
  runs. Fix: run pytest with `PYTHONPATH=` and
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (both shown above).
- **`torchcodec` can't load its native library on this Jetson image**
  (`OSError: libnvrtc.so.13: cannot open shared object file` — the
  installed CUDA toolkit only ships nvrtc 12.8, torchcodec's wheel wants
  CUDA 13's). This is **not fatal** — `lerobot-train`/`lerobot-eval`
  catch it and print `'torchcodec' is installed but cannot be loaded ...
  Falling back to 'pyav' as a default decoder.` and proceed correctly.
  `lerobot-info`, however, calls `importlib.import_module("torchcodec")`
  directly with no fallback and crashes outright — don't rely on
  `lerobot-info` to sanity-check the environment on this box.
- **The `pusht` extra isn't pulled in by a base `uv sync`.** `env.type=
  pusht` needs `gym-pusht`+`pymunk` installed separately (see
  Prerequisites) — without it `lerobot-train --env.type=pusht` fails at
  environment-creation time with an unhelpful gym registration error,
  not a clear "missing extra" message.
- **Training needs `accelerate` even for a single-process CPU run.**
  `lerobot-train` calls `require_package("accelerate", extra="training")`
  unconditionally early in `train()`; without it you get `ImportError:
  'accelerate' is required but not installed` regardless of policy type.
- **Prefer `env.type=pusht` over `env.type=aloha` for a fast smoke test.**
  The Makefile's own `test-act-ete-train` target uses `aloha`, which pulls
  in `gym-aloha` (MuJoCo) — much heavier to install than `gym-pusht`. ACT
  trains and evals identically against either; this skill's driver swaps
  to `pusht` for that reason.

## Troubleshooting

- **`ImportError: 'accelerate' is required but not installed`**: run
  `python -m pip install accelerate` in the venv (see Prerequisites).
- **`ImportError: 'diffusers' is required but not installed`**: only hit
  if you switch the driver to `--policy.type=diffusion`; install
  `lerobot[diffusion]`. The `act` policy used by this skill's driver
  doesn't need it.
- **`gym id 'gym_pusht/PushT-v0' not found`** (as a one-line warning,
  training/eval continues): harmless — LeRobot's env factory imports
  `gym_pusht` lazily right before creating the env; this line just logs
  that first import, not a failure.
- **`ModuleNotFoundError: No module named 'lerobot'` when running
  pytest**: you activated the venv but ran plain `pytest` — its shebang
  can point at a different interpreter. Use `python -m pytest` instead.
