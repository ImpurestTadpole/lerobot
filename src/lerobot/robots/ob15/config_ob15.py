# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
OB15 robot configuration.

OB15 is XLerobot with use_head=False — no head pan/tilt motors.
Everything else (ports, cameras, lift axis, teleop keys) is identical to
the operational XLerobot config in config_xlerobot.py.

Hardware summary:
  - Base:    3-wheel kiwi omnidirectional drive (STS3215 motors, IDs 7-9 on bus1)
  - Arms:    2× SO-101 (6 DoF each) on a gantry lift platform
  - Lift:    Linear gantry, ~575 mm travel (STS3215 motor ID 9 on bus2)
  - Head:    RealSense D435i on passive 2-axis RGBD gimbal — no pan/tilt motors
  - Wrists:  2× Innomaker USB cameras

Bus layout (from config_xlerobot.py):
  - port1 (ttyACM1): left arm 1-6 + base wheels 7-9
  - port2 (ttyACM0): right arm 1-6 + gantry 9  (no head motors 7-8)

Action / state space (15D):
  left_arm_{shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper}.pos  (6)
  right_arm_{shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper}.pos (6)
  x.vel  y.vel  theta.vel  (3, body-frame base velocities)

Gantry (lift) is recorded as gantry.height_mm in observations but is NOT
part of the 15D policy action space — it is positioned separately via
robot.lift_axis.home() or a dedicated set_height call before each episode.

Navigation / bus handoff note:
  bus1 (ttyACM1) is shared between:
    ob15 host  (lerobot conda env, Python 3.12) — arm + wheel control during policy
    sts3215_control (ROS2, Python 3.10)           — wheel driver used by Nav2
  They MUST NOT run simultaneously.
  See ~/ros2_ws/src/bob_1/launch/policy_mode.launch.py for the handoff procedure.
"""

from dataclasses import dataclass, field

from ..config import RobotConfig
from ..xlerobot.config_xlerobot import (
    XLerobotClientConfig,
    XLerobotConfig,
    XLerobotHostConfig,
    xlerobot_cameras_config,
)


@RobotConfig.register_subclass("ob15")
@dataclass
class OB15Config(XLerobotConfig):
    """
    OB15 robot — XLerobot without head pan/tilt motors.

    The only functional difference from XLerobotConfig is use_head=False,
    which removes head_pan / head_tilt from the motor bus and the
    state/action feature space (15D instead of 17D).

    All other settings — ports, cameras, lift axis, teleop keys — are
    inherited unchanged from XLerobotConfig so the robot behaves exactly
    like the operational xlerobot configuration.
    """

    use_head: bool = False


@dataclass
class OB15HostConfig(XLerobotHostConfig):
    """ZMQ host config for ob15 — identical to XLerobot host defaults."""
    pass


@RobotConfig.register_subclass("ob15_client")
@dataclass
class OB15ClientConfig(XLerobotClientConfig):
    """
    Remote OB15 over ZMQ (type: ob15_client).

    Inherits all defaults from XLerobotClientConfig (same ports, cameras,
    teleop keys). Use remote_ip to point at the Jetson.

    Example:
        lerobot-teleoperate \\
          --robot.type=ob15_client \\
          --robot.remote_ip=<jetson_ip>
    """
    remote_ip: str = "127.0.0.1"
    use_head: bool = False
