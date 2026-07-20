This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).
> **XLerobot data/training pipeline → [`GUIDE.md`](./GUIDE.md)**, [`DATA_CONVERSION.md`](./DATA_CONVERSION.md), [`COTRAINING.md`](./COTRAINING.md), [`DAGGER_HIL.md`](./DAGGER_HIL.md), [`RGBD_IMPLEMENTATION_GUIDE.md`](./RGBD_IMPLEMENTATION_GUIDE.md).
> **AI usage expectations → [`AI_POLICY.md`](./AI_POLICY.md)** — disclose significant AI assistance in PRs and fully own/understand any AI-generated code before submitting it.

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing. This fork adds **XLerobot**, a mobile bimanual platform (see README's "XLerobot" section), plus its supporting data-conversion, co-training, and DAgger/human-in-the-loop tooling.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                        # All tests
uv run pytest tests/policies/test_act.py -svv                # One test file
uv run pytest tests/policies/test_act.py::test_act -svv      # One test case
DEVICE=cuda make test-end-to-end                             # All E2E train/eval smoke tests (see Makefile)
make annotation-e2e                                          # VLM annotation pipeline smoke test (stub backend, no GPU)
pre-commit run --all-files                                   # Lint + format (ruff, typos, prettier, gitleaks, zizmor, bandit, mypy)
uv run ruff check src tests --fix                            # Lint only
uv run mypy --config-file=pyproject.toml                     # Type check (gradual, see Notes)
```

Hardware- and environment-dependent tests self-skip via decorators in `tests/utils.py` (`require_cuda`, `require_cpu`, `require_x86_64_kernel`, `require_hf_token`, `require_env`, `skip_if_package_missing`) — no special flags needed to run the suite without a GPU/robot attached.

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-rollout`, `lerobot-annotate`, etc.), mapped in `pyproject.toml [project.scripts]`. That table is the authoritative list of every CLI command.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy (ACT, Diffusion, TDMPC, VQ-BeT, Pi0/Pi0Fast/Pi0.5, SmolVLA, GR00T, XVLA, EO-1, MolmoAct2, WALL-OSS, EVO1, VLA-JEPA, LingBot-VA, FastWAM, HVLA, …) lives in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`; policy components are resolved by naming convention (see `policies/common/`).
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps; shared builders live under `processor/` and are reused across policies (see `refactor(processors)` history).
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`. Sim benchmarks: LIBERO, MetaWorld, PushT, ALOHA sim.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers. `robots/xlerobot/` holds the mobile bimanual platform (dual Feetech buses, optional gantry lift axis, ZMQ host/client remote-inference split — see README's XLerobot section for bus/port layout).
- **`rl/`** — HIL-SERL style online RL: `actor.py`/`learner.py`/`trainer.py`, replay `buffer.py`, `gym_manipulator.py`, `train_rl.py`.
- **`rollout/`** — Policy rollout runtime used by `lerobot-rollout`: `strategies/`, `inference/`, `robot_wrapper.py`, `ring_buffer.py` — used for autonomous running + human takeover (DAgger workflow).
- **`rewards/`** — Reward/classifier models (SARM, TOPReward, Robometer) with their own `factory.py`/`pretrained.py`, mirroring the `policies/` pattern.
- **`annotations/`** — VLM-based subtask annotation pipeline (`steerable_pipeline/`), driven by `lerobot-annotate`.
- **`data_processing/`** — Cross-dataset tooling: `co_training_utils.py` (`lerobot-cotrain-align`), `extract_subset.py` (`lerobot-extract-subset`), `umi_retarget.py` (UMI end-effector → xlerobot schema), `sarm_annotations/`.
- **`jobs/`** — Dataset/Hub job helpers (`dataset.py`, `hf.py`).
- **`async_inference/`** — `policy_server.py` / `robot_client.py` split for running inference off-robot (gRPC-based; see `transport/`).
- **`model/`** — Shared model building blocks (attention, backbones) reused across policies.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module (mirrors `src/lerobot/`: `policies/`, `rl/`, `rollout` under `jobs`/`async_inference`, `annotations/`, etc.). Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`, recorded artifacts (small `.safetensors`, sample datasets) in `tests/artifacts/` via Git LFS. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `benchmark_tests.yml`, `release.yml` (PyPI publish on tags), `claude.yml` (Claude Code GitHub Action).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`scripts/`** (repo root, distinct from `src/lerobot/scripts/`) — dev/ops helpers: `workflow_ui/` (local web UI that generates and runs the co-training pipeline stage-by-stage, `uv run python scripts/workflow_ui/app.py` → `http://127.0.0.1:7799`), `ci/`, XLerobot deploy/sync shell scripts.
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root guide docs**: `AGENT_GUIDE.md` (user-facing agent companion to this file), `GUIDE.md` (full XLerobot co-training pipeline walkthrough), `COTRAINING.md`, `DATA_CONVERSION.md`, `DAGGER_HIL.md`, `RGBD_IMPLEMENTATION_GUIDE.md` (4-channel RGBD policy support), `Guide.sh` (hardware/session command reference), `AI_POLICY.md` (AI-assisted contribution rules), `CONTRIBUTING.md`.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config — including every extra and CLI script), `Makefile` (E2E test targets), `uv.lock`, `README.md` (general information).

## Notes

- **Mypy is gradual**: strict (`ignore_errors = false`) only for `lerobot.envs`, `lerobot.configs` (extra strictness: `disallow_untyped_defs`/`disallow_incomplete_defs`/`check_untyped_defs`), `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Everything else under `lerobot.*` has `ignore_errors = true`. Add type annotations when modifying these modules; other modules are best-effort.
- **Optional dependencies**: policies, envs, and robots are behind extras, organized as feature-scoped building blocks (`dataset`, `training`, `hardware`, `viz`) composed into user-facing extras (`core_scripts`, `evaluation`, `dataset_viz`) and per-integration extras (e.g. `lerobot[aloha]`, `lerobot[pi]`, `lerobot[feetech]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`, including inline notes on pinned versions (torchcodec platform matrix, placo/pin ABI pins) — don't loosen those pins without reading the adjacent comment.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
