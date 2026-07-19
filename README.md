<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml?query=branch%3Amain)
[![Tests](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry so that everyone can contribute to and benefit from shared datasets and pretrained models.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 State-of-the-art policies that have been shown to transfer to the real-world ready for training and deployment.

🤗 Comprehensive support for the open-source ecosystem to democratize physical AI.

## Quick Start

LeRobot can be installed directly from PyPI.

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> For detailed installation guide, please see the [Installation Documentation](https://huggingface.co/docs/lerobot/installation).

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices.

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**Supported Hardware:** SO100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones, OpenARM, Unitree G1, reBot B601.

While these devices are natively integrated into the LeRobot codebase, the library is designed to be extensible. You can easily implement the Robot interface to utilize LeRobot's data collection, training, and visualization tools for your own custom robot.

For detailed hardware setup guides, see the [Hardware Documentation](https://huggingface.co/docs/lerobot/integrate_hardware).

## XLerobot — Mobile Bimanual Robot

**XLerobot** is a fully mobile, bimanual manipulation platform built on top of LeRobot. It combines an omnidirectional wheeled base with two independent 6-DOF arms, a 2-DOF head, an optional vertical gantry lift axis, and a multi-camera vision system — all driven by Feetech STS3215 servo motors and integrated into the standard `Robot` interface.

### Hardware Overview

| Component | Details |
|---|---|
| **Mobile base** | 3-wheel omnidirectional (omni-wheels at 120°), velocity-controlled via Feetech STS3215 |
| **Left arm** | 6-DOF (shoulder pan/lift, elbow flex, wrist flex/roll, gripper) — STS3215 servos on Bus 1 |
| **Right arm** | 6-DOF (same joint layout as left) — STS3215 servos on Bus 2 |
| **Head** | 2-DOF pan + tilt — STS3215 servos on Bus 2 |
| **Gantry lift** | Optional Z-axis leadscrew (100 mm/rev, 575 mm stroke) — single STS3215 on Bus 2, motor ID 9 |
| **Head camera** | Intel RealSense D435i (640×360 RGB, 30 fps, serial `342222071125`) |
| **Wrist cameras** | 2× Innomaker USB cameras (640×360 MJPG, 30 fps) on `/dev/video6` and `/dev/video8` |
| **Compute** | NVIDIA Jetson (on-robot), with optional ZMQ bridge to a remote GPU workstation |

**Motor bus layout:**
- **Bus 1** (`/dev/ttyACM1`): Left arm (IDs 1–6) + omni-base wheels (IDs 7–9)
- **Bus 2** (`/dev/ttyACM0`): Right arm (IDs 1–6) + head pan/tilt (IDs 7–8) + gantry lift (ID 9, optional)

### Installation

Install LeRobot with the XLerobot dependencies (Feetech motors + RealSense camera):

```bash
pip install lerobot[feetech,realsense,viz]
```

### Configuration

XLerobot uses the `xlerobot` robot type. The default configuration is defined in `XLerobotConfig`:

```python
from lerobot.robots.xlerobot import XLerobot
from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig

config = XLerobotConfig(
    port1="/dev/ttyACM1",   # Bus 1: left arm + base
    port2="/dev/ttyACM0",   # Bus 2: right arm + head + lift
)
robot = XLerobot(config)
robot.connect()
```

**Enable the gantry lift axis:**

```python
from lerobot.robots.xlerobot.lift_axis import LiftAxisConfig

config = XLerobotConfig(
    lift_axis=LiftAxisConfig(
        enabled=True,
        soft_max_mm=575.0,      # travel range
        max_cmd_vel_frac=0.4,   # policy velocity cap (40% of v_max)
        home_on_connect=True,   # auto-home when robot.connect() is called
    )
)
```

**Remote mode (ZMQ bridge)** — run inference on a GPU workstation while the Jetson controls the hardware:

```python
# On the Jetson (robot side):
from lerobot.robots.xlerobot.xlerobot_host import XLerobotHost
host = XLerobotHost(robot)
host.run()

# On the GPU workstation (client side):
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
config = XLerobotClientConfig(remote_ip="JETSON_IP")
robot = XLerobot(config)  # streams observations + sends actions over ZMQ
```

### Calibration

Run the interactive calibration routine once per robot (saves to disk for future sessions):

```bash
lerobot-calibrate --robot.type=xlerobot --robot.port1=/dev/ttyACM1 --robot.port2=/dev/ttyACM0
```

The calibration wizard walks through each joint group in order:

1. **Left arm** — move to mid-range, then sweep full range of motion
2. **Right arm** — same procedure
3. **Head** — pan and tilt through full range
4. **Base wheels** — set to full-turn mode automatically (no manual sweep needed)
5. **Gantry lift** (if enabled) — drives downward until stall, then backs off to set home = 0 mm

Calibration is saved to `~/.cache/huggingface/lerobot/calibration/xlerobot.json` and auto-loaded on subsequent `robot.connect()` calls.

### Teleoperation

Keyboard teleop drives the base while a paired leader arm (e.g. SO-100) controls the robot arms:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.port1=/dev/ttyACM1 \
  --robot.port2=/dev/ttyACM0 \
  --teleop.type=so100_leader
```

**Base keyboard controls** (active during any teleoperation session):

| Key | Action |
|---|---|
| `i` / `k` | Forward / Backward |
| `j` / `l` | Strafe Left / Right |
| `u` / `o` | Rotate Left / Right |
| `n` / `m` | Speed Up / Speed Down (3 levels) |
| `b` | Quit |

### Data Collection

```bash
lerobot-record \
  --robot.type=xlerobot \
  --robot.port1=/dev/ttyACM1 \
  --robot.port2=/dev/ttyACM0 \
  --dataset.repo_id=YOUR_HF_USERNAME/xlerobot_task \
  --dataset.num_episodes=50
```

### Observation & Action Space

| Key | Type | Description |
|---|---|---|
| `left_arm_*.pos` | `float` ×6 | Left arm joint positions (normalized −100 to 100) |
| `right_arm_*.pos` | `float` ×6 | Right arm joint positions |
| `head_pan.pos`, `head_tilt.pos` | `float` ×2 | Head joint positions |
| `x.vel`, `y.vel`, `theta.vel` | `float` ×3 | Base body-frame velocities (m/s, deg/s) |
| `gantry.height_mm` | `float` | Lift axis height in mm (0 = home/bottom) |
| `head` | `uint8 (360, 640, 3)` | RealSense head camera (RGB) |
| `left_wrist` | `uint8 (360, 640, 3)` | Left wrist camera (RGB) |
| `right_wrist` | `uint8 (360, 640, 3)` | Right wrist camera (RGB) |

The camera names (`head`, `left_wrist`, `right_wrist`) are aligned with **SmolVLA**'s `OBS_IMAGE_1/2/3` convention, making the robot natively compatible with SmolVLA policies without any rename mapping.

### Visualization

Stream live robot data to a [Rerun](https://rerun.io) viewer during teleoperation or recording:

```bash
lerobot-teleoperate --robot.type=xlerobot ... --display_data=true
```

Set environment variables to tune streaming performance on Jetson:

```bash
export RERUN_FLUSH_TICK_SECS=0.008   # flush every 8ms for low latency
export RERUN_FLUSH_NUM_BYTES=16000   # ~16 KB per flush
export RERUN_LOG_FREQUENCY=2         # log every other frame to reduce CPU
export RERUN_DOWNSAMPLE_FACTOR=0.5   # resize images before sending
```

---

## LeRobot Dataset

To solve the data fragmentation problem in robotics, we utilize the **LeRobotDataset** format.

- **Structure:** Synchronized MP4 videos (or images) for vision and Parquet files for state/action data.
- **HF Hub Integration:** Explore thousands of robotics datasets on the [Hugging Face Hub](https://huggingface.co/lerobot).
- **Tools:** Seamlessly delete episodes, split by indices/fractions, add/remove features, and merge multiple datasets.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

Learn more about it in the [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## SoTA Models

LeRobot implements state-of-the-art policies in pure PyTorch, covering Imitation Learning, Reinforcement Learning, Vision-Language-Action (VLA) models, World Models, and Reward Models, with more coming soon. It also provides you with the tools to instrument and inspect your training process.

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

Training a policy is as simple as running a script configuration:

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| Category                   | Models                                                                                                                                                                                                                                                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Imitation Learning**     | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md), [Multitask DiT Policy](./docs/source/policy_multi_task_dit_README.md)                                                                                                                                                                    |
| **Reinforcement Learning** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL (coming soon)                                                                                                                                                                                                                                                                                |
| **VLAs Models**            | [Pi0](./docs/source/pi0.mdx), [Pi0Fast](./docs/source/pi0fast.mdx), [Pi0.5](./docs/source/pi05.mdx), [GR00T N1.7](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx), [EO-1](./docs/source/eo1.mdx), [MolmoAct2](./docs/source/molmoact2.mdx), [WALL-OSS](./docs/source/walloss.mdx), [EVO1](./docs/source/evo1.mdx) |
| **World Models**           | [VLA-JEPA](./docs/source/vla_jepa.mdx), [LingBot-VA](./docs/source/lingbot_va.mdx), [FastWAM](./docs/source/fastwam.mdx)                                                                                                                                                                                                                                                                   |
| **Reward Models**          | [SARM](./docs/source/sarm.mdx), [TOPReward](./docs/source/topreward.mdx), [Robometer](./docs/source/robometer.mdx)                                                                                                                                                                                                                                                                         |

Similarly to the hardware, you can easily implement your own policy & leverage LeRobot's data collection, training, and visualization tools, and share your model to the HF Hub

For detailed policy setup guides, see the [Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies). For GPU/RAM requirements and expected training time per policy, see the [Compute Hardware Guide](https://huggingface.co/docs/lerobot/hardware_guide).

## Inference & Evaluation

Evaluate your policies in simulation or on real hardware using the unified evaluation script. LeRobot supports standard benchmarks like **LIBERO**, **MetaWorld** and more to come.

```bash
# Evaluate a policy on the LIBERO benchmark
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

Learn how to implement your own simulation environment or benchmark and distribute it from the HF Hub by following the [EnvHub Documentation](https://huggingface.co/docs/lerobot/envhub)

## Resources

- **[Documentation](https://huggingface.co/docs/lerobot/index):** The complete guide to tutorials & API.
- **[Chinese Tutorials: LeRobot+SO-ARM101中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)** Detailed doc for assembling, teleoperate, dataset, train, deploy. Verified by Seed Studio and 5 global hackathon players.
- **[Discord](https://discord.gg/q8Dzzpym3f):** Join the `LeRobot` server to discuss with the community.
- **[X](https://x.com/LeRobotHF):** Follow us on X to stay up-to-date with the latest developments.
- **[Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial):** A free, hands-on course to learn robot learning using LeRobot.
- **[T-Shirt Folding Experiment](https://huggingface.co/spaces/lerobot/robot-folding):** An end-to-end demonstration of folding t-shirts with LeRobot.
- **[LeLab](https://github.com/huggingface/leLab):** A web interface for LeRobot — teleoperate, calibrate, record datasets, replay, and train your SO arm from the browser, no CLI required.

## Citation

If you use LeRobot in your project, please cite the GitHub repository to acknowledge the ongoing development and contributors:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Meftah, Khalil and Ellerbach, Maxime and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

If you are referencing our research or the academic paper, please also cite our ICLR publication:

<details>
<summary><b>ICLR 2026 Paper</b></summary>

```bibtex
@inproceedings{cadenelerobot,
  title={LeRobot: An Open-Source Library for End-to-End Robot Learning},
  author={Cadene, Remi and Alibert, Simon and Capuano, Francesco and Aractingi, Michel and Zouitine, Adil and Kooijmans, Pepijn and Choghari, Jade and Russi, Martino and Pascal, Caroline and Palma, Steven and Shukor, Mustafa and Moss, Jess and Soare, Alexander and Aubakirova, Dana and Lhoest, Quentin and Gallou\'edec, Quentin and Wolf, Thomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2602.22818}
}
```

</details>

## Contribute

We welcome contributions from everyone in the community! To get started, please read our [CONTRIBUTING.md](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md) guide. Whether you're adding a new feature, improving documentation, or fixing a bug, your help and feedback are invaluable. We're incredibly excited about the future of open-source robotics and can't wait to work with you on what's next—thank you for your support!

<p align="center">
  <img alt="SO101 Video" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>Built by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>
