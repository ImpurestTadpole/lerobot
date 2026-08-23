#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for GemFollower (single GEM arm: mixed ODrive + Feetech). Both buses are mocked,
so these run without hardware or the `odrive` package installed."""

from unittest.mock import MagicMock, patch

from lerobot.robots.gem_follower.config_gem_follower import GemFollowerConfig
from lerobot.robots.gem_follower.gem_follower import GemFollower

_MODULE = "lerobot.robots.gem_follower.gem_follower"


def _make_bus(port, motors, calibration):
    bus = MagicMock(name=f"Bus({port})")
    bus.port = port
    bus.motors = motors
    bus.is_connected = False
    bus.is_calibrated = False
    return bus


def _build_robot(**config_kwargs) -> GemFollower:
    config = GemFollowerConfig(feetech_port="/dev/ttyUSB0", **config_kwargs)
    with (
        patch(f"{_MODULE}.FeetechMotorsBus", side_effect=_make_bus),
        patch(f"{_MODULE}.ODriveMotorsBus", side_effect=_make_bus),
    ):
        return GemFollower(config)


def test_registers_under_both_gem_and_gem_follower():
    from lerobot.robots import gem_follower  # noqa: F401
    from lerobot.robots.config import RobotConfig

    assert "gem" in RobotConfig.get_known_choices()
    assert "gem_follower" in RobotConfig.get_known_choices()


def test_construction_splits_motors_across_feetech_and_odrive_buses():
    robot = _build_robot()

    assert set(robot.feetech_bus.motors) == {
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
        "gripper",
    }
    assert set(robot.odrive_bus.motors) == {"joint_1"}


def test_action_and_observation_features_cover_all_8_joints():
    robot = _build_robot()
    expected = {f"{j}.pos" for j in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper")}
    assert set(robot.action_features) == expected
    assert set(robot.observation_features) >= expected


def test_is_connected_reflects_both_buses():
    robot = _build_robot()
    robot.feetech_bus.is_connected = True
    robot.odrive_bus.is_connected = True
    assert robot.is_connected

    robot.odrive_bus.is_connected = False
    assert not robot.is_connected


def test_send_action_routes_odrive_joint_to_pending_target_not_feetech_bus():
    robot = _build_robot()
    robot.feetech_bus.is_connected = True
    robot.odrive_bus.is_connected = True
    robot.send_action({"joint_1.pos": 45.0, "joint_2.pos": 10.0})

    assert robot._odrive_pending_target == 45.0
    robot.feetech_bus.sync_write.assert_called_once_with("Goal_Position", {"joint_2": 10.0})


def test_custom_feetech_motor_ids_are_respected():
    robot = _build_robot(feetech_motor_ids={"joint_2": 20, "joint_3": 21, "joint_4": 22, "joint_5": 23, "joint_6": 24, "joint_7": 25, "gripper": 26})
    assert robot.feetech_bus.motors["joint_2"].id == 20
    assert robot.feetech_bus.motors["gripper"].id == 26
