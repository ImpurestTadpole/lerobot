#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for OB15's per-limb port -> physical bus grouping. `FeetechMotorsBus` is mocked
throughout, so these run without hardware. Coverage is limited to the bus-construction
logic (which ports collapse onto which bus, dedup behavior) -- not actual motor
communication, which can't be verified without a real robot."""

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.ob15.config_ob15 import OB15Config
from lerobot.robots.ob15.ob15 import OB15

_MODULE = "lerobot.robots.ob15.ob15"


def _make_bus(port, motors, calibration):
    bus = MagicMock(name=f"FeetechMotorsBus({port})")
    bus.port = port
    bus.motors = motors
    bus.is_connected = False
    bus.is_calibrated = False
    return bus


def _build_robot(config: OB15Config) -> OB15:
    with patch(f"{_MODULE}.FeetechMotorsBus", side_effect=_make_bus):
        return OB15(config)


def test_default_topology_reproduces_original_2bus_wiring():
    config = OB15Config(cameras={})
    # lift_port must default onto bus2 (right_arm), not bus1 — ID 9 collides with the base.
    assert config.lift_port == config.right_arm_port == "/dev/ttyACM0"
    assert config.left_arm_port == config.base_port == "/dev/ttyACM1"

    robot = _build_robot(config)

    assert set(robot._buses.keys()) == {"/dev/ttyACM1", "/dev/ttyACM0"}
    assert len(robot._buses) == 2
    # left_arm + base share a bus; right_arm + head + lift share the other -- exactly the
    # original hardcoded bus1/bus2 grouping.
    assert robot._group_bus["left_arm"] is robot._group_bus["base"]
    assert robot._group_bus["right_arm"] is robot._group_bus["head"]
    assert robot._group_bus["right_arm"] is robot._group_bus["lift"]
    assert robot._group_bus["left_arm"] is not robot._group_bus["right_arm"]


def test_legacy_port1_port2_aliases_expand_onto_limb_ports():
    config = OB15Config(
        cameras={},
        port1="/dev/legacy_bus1",
        port2="/dev/legacy_bus2",
    )
    assert config.left_arm_port == config.base_port == "/dev/legacy_bus1"
    assert config.right_arm_port == config.head_port == config.lift_port == "/dev/legacy_bus2"

    robot = _build_robot(config)
    assert set(robot._buses.keys()) == {"/dev/legacy_bus1", "/dev/legacy_bus2"}
    assert robot._group_bus["left_arm"] is robot._group_bus["base"]
    assert robot._group_bus["right_arm"] is robot._group_bus["lift"]


def test_id_collision_on_shared_port_raises():
    """lift on bus1 with default motor_id=9 collides with base_right_wheel."""
    config = OB15Config(cameras={})
    config.lift_axis.bus = "lift"
    config.lift_port = config.base_port  # same physical bus as base wheels

    with pytest.raises(ValueError, match="Motor ID collision"):
        _build_robot(config)


def test_default_topology_bus_motor_sets_match_original():
    robot = _build_robot(OB15Config(cameras={}))

    left_bus_motors = set(robot._group_bus["left_arm"].motors)
    right_bus_motors = set(robot._group_bus["right_arm"].motors)
    assert left_bus_motors == {*robot.left_arm_motors, *robot.base_motors}
    assert right_bus_motors == {*robot.right_arm_motors, *robot.head_motors, "gantry"}


def test_fully_split_topology_gives_5_distinct_buses():
    config = OB15Config(
        cameras={},
        left_arm_port="/dev/ob15_left_arm",
        right_arm_port="/dev/ob15_right_arm",
        base_port="/dev/ob15_base",
        head_port="/dev/ob15_head",
    )
    config.lift_axis.bus = "lift"
    config.lift_port = "/dev/ob15_lift"

    robot = _build_robot(config)

    assert len(robot._buses) == 5
    buses = [robot._group_bus[g] for g in ("left_arm", "right_arm", "base", "head", "lift")]
    assert len({id(b) for b in buses}) == 5  # all distinct bus objects


def test_partial_split_only_base_separated():
    """Base gets its own port (e.g. to coexist with Nav2 without a full handoff); arms/head
    stay on the original grouping."""
    config = OB15Config(cameras={}, base_port="/dev/ob15_base_only")

    robot = _build_robot(config)

    assert len(robot._buses) == 3
    assert robot._group_bus["base"] is not robot._group_bus["left_arm"]
    # left_arm's bus no longer picks up the base motors it used to share a port with.
    assert set(robot._group_bus["left_arm"].motors) == set(robot.left_arm_motors)
    assert set(robot._group_bus["base"].motors) == set(robot.base_motors)
    assert robot._group_bus["right_arm"] is robot._group_bus["head"]


def test_lift_disabled_has_no_lift_group():
    config = OB15Config(cameras={})
    config.lift_axis.enabled = False

    robot = _build_robot(config)

    assert "lift" not in robot._group_bus
    assert not robot.lift_axis.enabled


def test_is_connected_reflects_all_buses():
    robot = _build_robot(OB15Config(cameras={}))
    for bus in robot._buses.values():
        bus.is_connected = True
    assert robot.is_connected

    next(iter(robot._buses.values())).is_connected = False
    assert not robot.is_connected


def test_is_calibrated_reflects_all_buses():
    robot = _build_robot(OB15Config(cameras={}))
    for bus in robot._buses.values():
        bus.is_calibrated = True
    assert robot.is_calibrated

    next(iter(robot._buses.values())).is_calibrated = False
    assert not robot.is_calibrated
