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

"""Tests for BiGemFollower (dual GEM arms: shared Feetech bus + 2 ODrive buses + optional
neck). Both bus types are mocked, so these run without hardware or the `odrive` package."""

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_gem_follower.config_bi_gem_follower import BiGemFollowerConfig
from lerobot.robots.bi_gem_follower.bi_gem_follower import BiGemFollower

_MODULE = "lerobot.robots.bi_gem_follower.bi_gem_follower"


def _make_bus(port, motors, calibration):
    bus = MagicMock(name=f"Bus({port})")
    bus.port = port
    bus.motors = motors
    bus.is_connected = False
    bus.is_calibrated = False
    return bus


def _build_robot(**config_kwargs) -> BiGemFollower:
    config = BiGemFollowerConfig(feetech_port="/dev/ttyUSB0", **config_kwargs)
    with (
        patch(f"{_MODULE}.FeetechMotorsBus", side_effect=_make_bus),
        patch(f"{_MODULE}.ODriveMotorsBus", side_effect=_make_bus),
    ):
        return BiGemFollower(config)


def test_registers_under_both_bi_gem_and_bi_gem_follower():
    from lerobot.robots import bi_gem_follower  # noqa: F401
    from lerobot.robots.config import RobotConfig

    assert "bi_gem" in RobotConfig.get_known_choices()
    assert "bi_gem_follower" in RobotConfig.get_known_choices()


def test_shares_one_feetech_bus_with_prefixed_left_right_motors():
    robot = _build_robot()

    expected_arm_motors = {
        f"{side}_{joint}" for side in ("left", "right") for joint in ("joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper")
    }
    expected_neck_motors = {"neck_pan", "neck_nod_left", "neck_nod_right"}
    assert set(robot.feetech_bus.motors) == expected_arm_motors | expected_neck_motors


def test_neck_disabled_removes_neck_motors():
    robot = _build_robot(neck=None)
    assert not any(m.startswith("neck_") for m in robot.feetech_bus.motors)
    assert "neck_yaw.pos" not in robot.action_features


def test_two_independent_odrive_buses():
    robot = _build_robot()
    assert set(robot.left_odrive_bus.motors) == {"left_joint_1"}
    assert set(robot.right_odrive_bus.motors) == {"right_joint_1"}
    assert robot.left_odrive_bus is not robot.right_odrive_bus


def test_duplicate_feetech_motor_ids_across_arms_raise():
    with pytest.raises(ValueError, match="unique"):
        _build_robot(right_arm_motor_ids={"joint_2": 2, "joint_3": 3, "joint_4": 4, "joint_5": 5, "joint_6": 6, "joint_7": 7, "gripper": 8})


def test_send_action_routes_each_side_odrive_joint_independently():
    # No polling thread running (configure() was never called), so send_action writes
    # each ODrive side's target directly rather than queuing it for the poll thread.
    robot = _build_robot()
    for bus in (robot.feetech_bus, robot.left_odrive_bus, robot.right_odrive_bus):
        bus.is_connected = True
    robot.send_action({"left_joint_1.pos": 10.0, "right_joint_1.pos": -10.0, "left_joint_2.pos": 5.0})

    robot.left_odrive_bus.sync_write.assert_called_once_with("Goal_Position", {"left_joint_1": 10.0})
    robot.right_odrive_bus.sync_write.assert_called_once_with("Goal_Position", {"right_joint_1": -10.0})
    # feetech_bus also gets a neck-pose sync_write (default neck config) -- check the arm
    # joint write specifically rather than asserting it's the bus's only call.
    robot.feetech_bus.sync_write.assert_any_call("Goal_Position", {"left_joint_2": 5.0})


def test_is_connected_reflects_all_three_buses():
    robot = _build_robot()
    for bus in (robot.feetech_bus, robot.left_odrive_bus, robot.right_odrive_bus):
        bus.is_connected = True
    assert robot.is_connected

    robot.right_odrive_bus.is_connected = False
    assert not robot.is_connected
