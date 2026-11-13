#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Bus-level checks for ``ob15_arms``.

Unlike ``test_ob15_arms.py`` (which replaces ``OB15.send_action`` / ``get_observation``), these run the
real ``OB15`` low-level code (base smoothing, arm/head writes, the lift P-controller) against fake motor
buses that remember what was written. They show what the motors would actually be told.
"""

import random
import re
import threading
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("zmq")

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.robots.ob15 import OB15, OB15Config
from lerobot.robots.ob15_arms import OB15Arms, OB15ArmsConfig
from lerobot.robots.ob15_arms.arms_only import (
    ARM_KEYS,
    BASE_KEYS,
    DEFAULT_HEAD_TILT_DEG,
    DEFAULT_LIFT_MM,
    hold_reached,
    read_hold_file,
    write_hold_file,
)
from lerobot.scripts.lerobot_record import RecordConfig, record
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

pytestmark = pytest.mark.hardware

LIFT = "gantry.height_mm"
WHEELS = ("base_left_wheel", "base_back_wheel", "base_right_wheel")
_FB = "lerobot.robots.ob15.ob15.FeetechMotorsBus"
TICKS_PER_WRITE = 0.1  # lift travel per Goal_Velocity write, in raw ticks per velocity unit


class FakeBus:
    """A motor bus with memory. Positions are instant, except the lift which integrates its velocity."""

    def __init__(self, port, motors, calibration=None):
        self.port, self.motors, self.calibration = port, motors, calibration or {}
        self.is_connected = False
        self.is_calibrated = True
        self.pos = {}
        self.vel = {}
        self.writes: list[tuple[str, dict]] = []  # (item, {motor: value}), in order

    def connect(self, *a, **k):
        self.is_connected = True

    def disconnect(self, disable_torque=True, *a, **k):
        self.is_connected = False
        self.torque_released = sorted(self.motors) if disable_torque else []

    def __getattr__(self, name):  # torque, configure_motors, write_calibration... are no-ops
        mock = MagicMock(name=name)
        setattr(self, name, mock)
        return mock

    def _pos(self, motor):
        return self.pos.setdefault(motor, 0.0)

    def read(self, item, motor, normalize=True, num_retry=0):
        if item == "Present_Temperature":
            return getattr(self, "temp_c", 35.0)
        return self._pos(motor) if item == "Present_Position" else self.vel.get(motor, 0.0)

    def sync_read(self, item, motors=None, normalize=True, num_retry=0):
        names = list(motors) if motors is not None else list(self.motors)
        if item == "Present_Position":
            return {m: self._pos(m) for m in names}
        return {m: self.vel.get(m, 0.0) for m in names}

    def write(self, item, motor, value, normalize=True, num_retry=0):
        self.writes.append((item, {motor: value}))
        if item == "Goal_Velocity":
            self.vel[motor] = value
            if motor == "gantry":  # lift: position integrates velocity
                self.pos[motor] = (self._pos(motor) + value * TICKS_PER_WRITE) % 4096

    def sync_write(self, item, values, normalize=True, num_retry=0):
        self.writes.append((item, dict(values)))
        for motor, value in values.items():
            if item == "Goal_Position":
                self.pos[motor] = value
            elif item == "Goal_Velocity":
                self.vel[motor] = value


@pytest.fixture
def make_robot(tmp_path, monkeypatch):
    """Factory: a connected OB15Arms on fake buses (head at 10/-5, lift homed at 100 mm)."""
    monkeypatch.setattr("lerobot.robots.ob15_arms.ob15_arms.HOLD_POLL_S", 0.0)
    robots = []

    def factory(**cfg):
        cfg.setdefault("calibration_dir", tmp_path / "calib")
        config = OB15ArmsConfig(cameras={}, **cfg)
        with patch(_FB, side_effect=FakeBus):
            robot = OB15Arms(config)
        for bus in robot._buses.values():
            bus.pos.update({"head_pan": 10.0, "head_tilt": -5.0})
        # Lift: already homed, currently 100 mm below the home position.
        robot.lift_axis._homed = True
        robot.lift_axis._z0_deg = 100.0 / robot.lift_axis._mm_per_deg
        with patch.object(OB15, "calibrate", lambda self: None):
            robot.connect()
        robots.append(robot)
        return robot

    yield factory
    for robot in robots:
        if robot.is_connected:
            robot.disconnect()


def _writes(robot, item, motors):
    """All values written for ``item`` to any of ``motors``, in order."""
    out = []
    for bus in robot._buses.values():
        for it, values in bus.writes:
            if it == item:
                out += [v for m, v in values.items() if m in motors]
    return out


def _arm_action(value=1.0):
    return dict.fromkeys(ARM_KEYS, value)


def test_connect_moves_head_and_lift_to_the_flags(make_robot):
    robot = make_robot(head_pan=0.0, head_tilt=-35.0, lift_height_mm=220.0)
    obs = robot.get_observation()
    assert obs["head_pan.pos"] == pytest.approx(0.0)
    assert obs["head_tilt.pos"] == pytest.approx(-35.0)
    assert obs[LIFT] == pytest.approx(220.0, abs=robot.config.lift_tol_mm)
    assert hold_reached(robot.check_hold())


def test_arm_commands_reach_the_motors_and_nothing_else_moves(make_robot):
    robot = make_robot(head_pan=0.0, head_tilt=-35.0, lift_height_mm=220.0)
    arm_motors = [*robot.left_arm_motors, *robot.right_arm_motors]
    for bus in robot._buses.values():
        bus.writes.clear()

    hostile = {**_arm_action(7.5), "head_pan.pos": 45.0, "head_tilt.pos": 30.0, LIFT: 5.0}
    hostile.update({"x.vel": 0.5, "y.vel": 0.5, "theta.vel": 90.0})  # base inputs from a VR teleop
    for _ in range(30):
        robot.send_action(hostile)

    # arms get the commanded value
    assert set(_writes(robot, "Goal_Position", arm_motors)) == {7.5}
    # head stays at the hold pose, whatever the action said
    assert set(_writes(robot, "Goal_Position", ["head_pan"])) == {0.0}
    assert set(_writes(robot, "Goal_Position", ["head_tilt"])) == {-35.0}
    # the wheels are not on the bus: they are never written at all, not even a zero
    assert not _writes(robot, "Goal_Velocity", WHEELS)
    assert not any(m in WHEELS for bus in robot._buses.values() for _, v in bus.writes for m in v)
    # the lift is still at the hold height
    assert robot.get_observation()[LIFT] == pytest.approx(220.0, abs=robot.config.lift_tol_mm)


def test_wheel_motors_are_not_on_any_bus(make_robot):
    robot = make_robot()
    assert robot.base_motors == []
    assert not any(m in WHEELS for bus in robot._buses.values() for m in bus.motors)
    # plain ob15 is unchanged: it still owns the wheels
    with patch(_FB, side_effect=FakeBus):
        plain = OB15(OB15Config(cameras={}))
    assert plain.base_motors == list(WHEELS)


def test_connect_succeeds_when_the_wheels_are_unplugged(tmp_path, monkeypatch):
    """The real left bus handshake fails on missing wheel IDs 7/8; ob15_arms must not ask for them."""
    monkeypatch.setattr("lerobot.robots.ob15_arms.ob15_arms.HOLD_POLL_S", 0.0)

    class NoWheelsBus(FakeBus):
        def connect(self, *a, **k):
            missing = [m for m in self.motors if m in WHEELS]
            if missing:
                raise RuntimeError(f"Missing motor IDs: {missing}")
            super().connect()

    config = OB15ArmsConfig(cameras={}, calibration_dir=tmp_path / "calib")
    with patch(_FB, side_effect=NoWheelsBus):
        robot = OB15Arms(config)
    robot.lift_axis._homed = True
    robot.lift_axis._z0_deg = 100.0 / robot.lift_axis._mm_per_deg
    with patch.object(OB15, "calibrate", lambda self: None):
        robot.connect()
    try:
        assert robot.is_connected
        assert set(robot.send_action(_arm_action())) == set(ARM_KEYS)
    finally:
        robot.disconnect()

    # plain ob15 asks for the wheels, so the same unplugged-wheel bus stops it
    with patch(_FB, side_effect=NoWheelsBus):
        plain = OB15(OB15Config(cameras={}, calibration_dir=tmp_path / "calib2"))
    with pytest.raises(RuntimeError, match="Missing motor IDs"):
        plain.connect()


def test_lift_drift_is_pulled_back_by_the_next_action(make_robot):
    robot = make_robot(head_pan=0.0, head_tilt=-35.0, lift_height_mm=220.0)
    gantry_bus = robot._group_bus["lift"]
    gantry_bus.pos["gantry"] = (gantry_bus.pos["gantry"] + 4096 * 0.3) % 4096  # ~30 mm of sag
    assert not hold_reached(robot.check_hold())
    for _ in range(300):
        robot.send_action(_arm_action())
    assert hold_reached(robot.check_hold())


def test_observation_and_action_spaces_have_no_base(make_robot):
    robot = make_robot()
    obs = robot.get_observation()
    assert not set(BASE_KEYS) & set(obs)
    assert {"head_pan.pos", "head_tilt.pos", LIFT} <= set(obs)
    # every declared feature is present (the real OB15 also returns an undeclared gantry.vel; recording ignores it)
    assert set(robot.observation_features) <= set(obs)
    assert tuple(robot.action_features) == ARM_KEYS
    assert set(robot.send_action(_arm_action())) == set(ARM_KEYS)


def test_no_flags_tilts_the_head_down_and_holds_pan_and_lift(make_robot):
    robot = make_robot()
    obs = robot.get_observation()
    assert obs["head_tilt.pos"] == pytest.approx(DEFAULT_HEAD_TILT_DEG) == pytest.approx(60.0)
    assert obs["head_pan.pos"] == pytest.approx(10.0)  # no default: the current pan is kept


def test_head_tilt_precedence_flag_over_hold_from_over_default(make_robot, tmp_path):
    recorded = make_robot(head_tilt=25.0)
    write_hold_file(tmp_path / "ds", recorded.hold_metadata(), head_tol=2.0, lift_tol=5.0)
    recorded.disconnect()
    assert make_robot(hold_from=str(tmp_path / "ds")).get_observation()["head_tilt.pos"] == pytest.approx(
        25.0
    )
    assert make_robot(head_tilt=10.0, hold_from=str(tmp_path / "ds")).get_observation()[
        "head_tilt.pos"
    ] == pytest.approx(10.0)


def test_hold_on_connect_false_freezes_the_current_tilt(make_robot):
    robot = make_robot(hold_on_connect=False)
    assert robot.get_observation()["head_tilt.pos"] == pytest.approx(-5.0)


def test_hold_from_a_recorded_dataset_restores_the_pose(make_robot, tmp_path):
    recorded = make_robot(head_pan=0.0, head_tilt=-35.0, lift_height_mm=220.0)
    write_hold_file(tmp_path / "ds", recorded.hold_metadata(), head_tol=2.0, lift_tol=5.0)
    recorded.disconnect()

    restored = make_robot(hold_from=str(tmp_path / "ds"))
    obs = restored.get_observation()
    assert obs["head_tilt.pos"] == pytest.approx(-35.0)
    assert obs[LIFT] == pytest.approx(220.0, abs=restored.config.lift_tol_mm)

    flagged = make_robot(hold_from=str(tmp_path / "ds"), lift_height_mm=150.0)
    assert flagged.get_observation()[LIFT] == pytest.approx(150.0, abs=flagged.config.lift_tol_mm)


def test_unhomed_lift_is_not_moved_and_connect_does_not_hang(tmp_path, monkeypatch, caplog):
    """OB15 refuses height commands until the lift is homed; the hold must warn, not hang or drive it."""
    monkeypatch.setattr("lerobot.robots.ob15_arms.ob15_arms.HOLD_POLL_S", 0.0)
    config = OB15ArmsConfig(cameras={}, calibration_dir=tmp_path, lift_height_mm=220.0, settle_timeout_s=0.2)
    config.lift_axis.home_on_connect = False
    with patch(_FB, side_effect=FakeBus):
        robot = OB15Arms(config)
    with patch.object(OB15, "calibrate", lambda self: None), caplog.at_level("WARNING"):
        robot.connect()
    try:
        assert "did not reach the hold pose" in caplog.text
        assert set(_writes(robot, "Goal_Velocity", ["gantry"])) == {0}  # never driven
    finally:
        robot.disconnect()


def test_default_calibration_folder_is_the_ob15_one():
    with patch(_FB, side_effect=FakeBus):
        robot = OB15Arms(OB15ArmsConfig(cameras={}))
    assert robot.calibration_dir.parts[-2:] == ("robots", "ob15")
    with patch(_FB, side_effect=FakeBus):
        custom = OB15Arms(OB15ArmsConfig(cameras={}, calibration_dir=Path("/tmp/somewhere")))
    assert custom.calibration_dir == Path("/tmp/somewhere")


# --- the real lerobot-record loop ---------------------------------------------------------------


@TeleoperatorConfig.register_subclass("ob15_arms_test_vr_like")
@dataclass
class _VRLikeTeleopConfig(TeleoperatorConfig):
    pass


class _VRLikeTeleop(Teleoperator):
    """Sends all 18 keys like the VR teleop: the arms vary, head/base/lift are hostile."""

    config_class = _VRLikeTeleopConfig
    name = "ob15_arms_test_vr_like"

    def __init__(self, config):
        super().__init__(config)
        self.config = config  # older record loops read teleop.config
        self._t = 0
        self._connected = False

    @cached_property
    def action_features(self):
        extra = ["head_pan.pos", "head_tilt.pos", *BASE_KEYS, LIFT]
        return dict.fromkeys([*ARM_KEYS, *extra], float)

    @cached_property
    def feedback_features(self):
        return {}

    is_connected = property(lambda self: self._connected)
    is_calibrated = property(lambda self: True)

    def connect(self, calibrate=True):
        self._connected = True

    def calibrate(self): ...

    def configure(self): ...

    def get_action(self):
        self._t += 1
        action = {k: float(np.sin(self._t / 5 + i)) for i, k in enumerate(ARM_KEYS)}
        action.update({"head_pan.pos": 40.0, "head_tilt.pos": 25.0, "x.vel": 0.4, "y.vel": 0.4})
        action.update({"theta.vel": 45.0, LIFT: 10.0})
        return action

    def send_feedback(self, feedback): ...

    def disconnect(self):
        self._connected = False


@pytest.mark.dataset
def test_lerobot_record_end_to_end(tmp_path, monkeypatch):
    """Two real episodes through ``record()``: 12-action dataset, hold file, wheels never driven."""
    monkeypatch.setattr("lerobot.robots.ob15_arms.ob15_arms.HOLD_POLL_S", 0.0)
    buses: list[FakeBus] = []

    def make_bus(**kwargs):
        bus = FakeBus(**kwargs)
        bus.pos.update({"head_pan": 10.0, "head_tilt": -5.0})
        buses.append(bus)
        return bus

    real_init = OB15Arms.__init__

    def init_homed(self, config):  # a lift that is already homed, 100 mm from the home position
        real_init(self, config)
        self.lift_axis._homed = True
        self.lift_axis._z0_deg = 100.0 / self.lift_axis._mm_per_deg

    root = tmp_path / "ds"
    cfg = RecordConfig(
        robot=OB15ArmsConfig(
            cameras={},
            calibration_dir=tmp_path / "cal",
            head_pan=0.0,
            head_tilt=-35.0,
            lift_height_mm=220.0,
        ),
        teleop=_VRLikeTeleopConfig(),
        dataset=DatasetRecordConfig(
            repo_id="local/ob15_arms_e2e",
            single_task="t",
            root=root,
            fps=10,
            episode_time_s=1,
            reset_time_s=0,
            num_episodes=2,
            push_to_hub=False,
            video=False,
            no_stamp=True,
        ),
        play_sounds=False,
    )
    with (
        patch(_FB, side_effect=make_bus),
        patch.object(OB15Arms, "__init__", init_homed),
        patch.object(OB15, "calibrate", lambda self: None),
        patch("lerobot.scripts.lerobot_record.make_teleoperator_from_config", _VRLikeTeleop),
    ):
        dataset = record(cfg)

    assert (dataset.num_episodes, dataset.num_frames) == (2, 20)
    features = dataset.meta.features
    assert tuple(features["action"]["names"]) == ARM_KEYS
    state_names = features["observation.state"]["names"]
    assert not set(BASE_KEYS) & set(state_names)
    assert {"head_pan.pos", "head_tilt.pos", LIFT} <= set(state_names)

    # the hold pose is saved with the dataset
    hold = read_hold_file(str(root))
    assert hold == {"head_pan.pos": 0.0, "head_tilt.pos": -35.0, LIFT: 220.0}

    # the wheels were never touched; the head never left the hold pose
    all_writes = [(item, m, v) for bus in buses for item, vals in bus.writes for m, v in vals.items()]
    assert not [1 for item, m, v in all_writes if m in WHEELS]
    assert {v for item, m, v in all_writes if item == "Goal_Position" and m == "head_tilt"} == {-35.0}

    # head and lift stay at the hold pose in every recorded frame
    states = np.stack(dataset.hf_dataset.to_pandas()["observation.state"].to_numpy())
    assert np.allclose(states[:, state_names.index("head_tilt.pos")], -35.0, atol=2.0)
    assert np.allclose(states[:, state_names.index(LIFT)], 220.0, atol=5.0)


def test_disconnect_skips_cameras_that_never_connected(make_robot):
    """connect() carries on when a camera fails to open; disconnect() must not then raise for it."""
    robot = make_robot()
    up, down = MagicMock(is_connected=True), MagicMock(is_connected=False)
    robot.cameras = {"up": up, "down": down}
    robot.disconnect()
    up.disconnect.assert_called_once()
    down.disconnect.assert_not_called()


class _HomingBus:
    """Lift motor that moves with its commanded velocity and stalls at a hard stop (the top limit)."""

    def __init__(self):
        self.pos, self.vel, self.writes = 2000.0, 0.0, []
        self.motors = {}

    def write(self, item, motor, value, **kw):
        self.writes.append((item, motor, value))
        if item == "Goal_Velocity":
            self.vel = float(value)

    def read(self, item, motor, normalize=True, **kw):
        if item == "Present_Position":
            self.pos = min(self.pos + self.vel * 0.03, 3000.0)  # hard stop at 3000
            return self.pos
        return 0.0  # Present_Current


@pytest.mark.parametrize(("release", "expect_torque_off"), [(True, True), (False, False)])
def test_home_releases_lift_torque_only_when_asked(monkeypatch, release, expect_torque_off):
    from lerobot.robots.ob15.lift_axis import LiftAxis, LiftAxisConfig

    monkeypatch.setattr("lerobot.robots.ob15.lift_axis.time.sleep", lambda s: None)
    bus = _HomingBus()
    lift = LiftAxis(
        LiftAxisConfig(enabled=True, home_on_connect=False, release_torque_after_home=release), bus
    )
    lift.configure()
    lift.home()
    assert lift._homed
    torque_off = [w for w in bus.writes if w[:2] == ("Torque_Enable", "gantry") and w[2] == 0]
    assert bool(torque_off) is expect_torque_off


def test_ob15_arms_keeps_the_lift_torqued_after_homing(tmp_path):
    with patch(_FB, side_effect=FakeBus):
        robot = OB15Arms(OB15ArmsConfig(cameras={}, calibration_dir=tmp_path / "c"))
        plain = OB15(OB15Config(cameras={}))
    assert robot.lift_axis.cfg.release_torque_after_home is False
    assert plain.lift_axis.cfg.release_torque_after_home is True  # ob15 keeps its original behaviour


def test_disconnect_releases_the_arms_but_keeps_the_lift_torqued(make_robot):
    robot = make_robot(disable_torque_on_disconnect=True)
    lift_bus = robot._group_bus["lift"]
    robot.send_action(_arm_action())
    robot.disconnect()
    assert "gantry" not in lift_bus.torque_released  # the lift is never released (it would fall)
    assert {"right_arm_gripper", "head_tilt"} <= set(lift_bus.torque_released)  # everything else is
    assert "gantry" in lift_bus.motors  # and the bus is left as it was
    stops = [v["gantry"] for item, v in lift_bus.writes if item == "Goal_Velocity" and "gantry" in v]
    assert stops and stops[-1] == 0  # stopped, not left moving


def test_disconnect_with_torque_kept_changes_nothing(make_robot):
    robot = make_robot(disable_torque_on_disconnect=False)
    lift_bus = robot._group_bus["lift"]
    robot.disconnect()
    assert lift_bus.torque_released == []


@pytest.mark.parametrize(("zero_at_limit", "expected_mm"), [(False, 0.0), (True, 10.0)])
def test_home_zero_is_at_the_hard_stop_only_when_asked(monkeypatch, zero_at_limit, expected_mm):
    """After backing off 10 mm the lift reads 0 mm (original ob15) or ~10 mm below the stop (zero_at_limit)."""
    from lerobot.robots.ob15.lift_axis import LiftAxis, LiftAxisConfig

    monkeypatch.setattr("lerobot.robots.ob15.lift_axis.time.sleep", lambda s: None)
    bus = _HomingBus()
    cfg = LiftAxisConfig(enabled=True, home_on_connect=False, zero_at_limit=zero_at_limit)
    lift = LiftAxis(cfg, bus)
    lift.configure()
    lift.home()
    assert lift.get_height_mm() == pytest.approx(expected_mm, abs=1.5)
    assert lift._cached_height_mm == pytest.approx(expected_mm, abs=1.5)


def test_ob15_arms_zeroes_the_lift_at_the_top_and_can_reach_zero(tmp_path):
    with patch(_FB, side_effect=FakeBus):
        robot = OB15Arms(OB15ArmsConfig(cameras={}, calibration_dir=tmp_path / "c"))
    assert robot.lift_axis.cfg.zero_at_limit is True
    assert robot.lift_axis.cfg.soft_min_mm == 1.0  # one millimetre of margin so it never presses the stop


def test_no_flags_holds_the_default_lift_height(make_robot):
    robot = make_robot()  # the fixture's lift starts 100 mm below the top
    assert DEFAULT_LIFT_MM == 25.0
    assert robot.get_observation()[LIFT] == pytest.approx(DEFAULT_LIFT_MM, abs=robot.config.lift_tol_mm)
    assert robot.hold_metadata()["hold"][LIFT] == DEFAULT_LIFT_MM
    # an explicit height still wins, and 0 is still the top
    assert make_robot(lift_height_mm=0.0).hold_metadata()["hold"][LIFT] == 0.0


def test_explicit_lift_height_and_hold_on_connect_false_still_win(make_robot):
    assert make_robot(lift_height_mm=40.0).get_observation()[LIFT] == pytest.approx(40.0, abs=5.0)
    frozen = make_robot(hold_on_connect=False)
    assert frozen.get_observation()[LIFT] == pytest.approx(100.0, abs=1.0)  # keeps where it was


def test_default_head_tilt_stays_inside_the_end_of_the_range(make_robot, monkeypatch):
    from lerobot.motors import MotorCalibration

    robot = make_robot()
    assert not robot.config.use_degrees  # head_tilt.pos is -100..100 of the calibrated range
    assert robot._default_head_tilt() == DEFAULT_HEAD_TILT_DEG
    monkeypatch.setattr("lerobot.robots.ob15_arms.ob15_arms.DEFAULT_HEAD_TILT_DEG", 120.0)
    assert robot._default_head_tilt() == 95.0  # never asks for the very end of the range

    # in degrees mode the margin is measured against this robot's calibrated tilt travel
    robot.config.use_degrees = True
    robot.calibration["head_tilt"] = MotorCalibration(
        id=8, drive_mode=0, homing_offset=0, range_min=1500, range_max=2181
    )  # +-29.9 deg
    assert robot._default_head_tilt() == pytest.approx(29.9 - 2.0, abs=0.2)


class _StuckBus:
    """A lift motor that never moves, whatever it is commanded (pushing against a stop / a jam)."""

    def __init__(self):
        self.writes, self.motors = [], {}

    def write(self, item, motor, value, **kw):
        self.writes.append((item, motor, value))

    def read(self, item, motor, normalize=True, **kw):
        return 3000.0 if item == "Present_Position" else 0.0


def _stuck_lift(stall_guard_s, monkeypatch):
    from lerobot.robots.ob15.lift_axis import LiftAxis, LiftAxisConfig

    clock = {"t": 1000.0}
    monkeypatch.setattr("lerobot.robots.ob15.lift_axis.time.monotonic", lambda: clock["t"])
    bus = _StuckBus()
    lift = LiftAxis(LiftAxisConfig(enabled=True, home_on_connect=False, stall_guard_s=stall_guard_s), bus)
    lift.configure()
    lift._homed = True
    return lift, bus, clock


def _drive(lift, bus, clock, seconds, target=100.0, step=0.1):
    out = []
    for _ in range(int(seconds / step)):
        clock["t"] += step
        n = len(bus.writes)
        lift.apply_action({"gantry.height_mm": target})
        out += [w[2] for w in bus.writes[n:] if w[0] == "Goal_Velocity"]
    return out


def test_stall_guard_stops_a_lift_that_is_driven_but_not_moving(monkeypatch):
    lift, bus, clock = _stuck_lift(3.0, monkeypatch)
    first = _drive(lift, bus, clock, 2.5)
    assert first and all(v != 0 for v in first)  # still trying for the first 3 s
    halted = _drive(lift, bus, clock, 5.0)  # guard fires at 3 s, then rests for stall_halt_s
    assert halted[-1] == 0 and set(halted[-20:]) == {0}
    assert any(v != 0 for v in halted[:6])  # (it was still driving until the guard fired)
    resumed = _drive(lift, bus, clock, 8.0)  # past the 10 s rest it tries again
    assert any(v != 0 for v in resumed)


def test_stall_guard_is_off_by_default(monkeypatch):
    lift, bus, clock = _stuck_lift(0.0, monkeypatch)
    driven = _drive(lift, bus, clock, 30.0)
    assert driven and all(v != 0 for v in driven)  # original ob15 behaviour: keeps driving


def test_arms_only_enables_the_stall_guard_plain_ob15_does_not(tmp_path):
    with patch(_FB, side_effect=FakeBus):
        arms = OB15Arms(OB15ArmsConfig(cameras={}, calibration_dir=tmp_path / "c"))
        plain = OB15(OB15Config(cameras={}))
    assert arms.lift_axis.cfg.stall_guard_s == 3.0 and arms.lift_axis.cfg.soft_min_mm == 1.0
    assert plain.lift_axis.cfg.stall_guard_s == 0.0


def _heat(robot, temp_c):
    robot._group_bus["lift"].temp_c = temp_c
    robot._last_health_check = -1e9  # due now


def test_a_warm_lift_motor_warns_but_keeps_holding(make_robot, caplog):
    robot = make_robot()
    _heat(robot, 58.0)
    with caplog.at_level("WARNING"):
        robot.send_action(_arm_action())
    assert any("lift motor is warm" in r.message for r in caplog.records)
    assert not robot._lift_released
    assert not [
        w for w in robot._group_bus["lift"].writes if w[0] == "Torque_Enable" and w[1] == {"gantry": 0}
    ]


def test_an_overheating_lift_motor_is_released_and_not_commanded_again(make_robot, caplog):
    robot = make_robot()
    lift_bus = robot._group_bus["lift"]
    _heat(robot, 66.0)
    with caplog.at_level("ERROR"):
        robot.send_action(_arm_action())
    assert robot._lift_released
    assert any("releasing its torque" in r.message for r in caplog.records)
    assert ("Torque_Enable", {"gantry": 0}) in lift_bus.writes
    n = len(lift_bus.writes)
    for _ in range(20):
        robot.send_action(_arm_action())
    assert not [
        w for w in lift_bus.writes[n:] if w[0] == "Goal_Velocity" and "gantry" in w[1]
    ]  # never driven again


def test_a_failing_temperature_read_does_not_break_recording(make_robot):
    robot = make_robot()
    robot._group_bus["lift"].read = MagicMock(side_effect=RuntimeError("bus error"))
    robot._last_health_check = -1e9
    robot.send_action(_arm_action())  # must not raise


def test_lift_exit_option_releases_it_when_asked(make_robot, caplog):
    kept = make_robot(disable_torque_on_disconnect=True)
    kept_bus = kept._group_bus["lift"]
    with caplog.at_level("WARNING"):
        kept.disconnect()
    assert "gantry" not in kept_bus.torque_released
    assert any("lift motor is left torqued" in r.message for r in caplog.records)

    released = make_robot(disable_torque_on_disconnect=True, release_lift_on_disconnect=True)
    released_bus = released._group_bus["lift"]
    released.disconnect()
    assert "gantry" in released_bus.torque_released


def test_record_resume_appends_a_second_task_to_the_same_dataset(tmp_path, monkeypatch):
    """One combined dataset: later sessions resume it and only change the task string (and arm/bracket)."""
    monkeypatch.setattr("lerobot.robots.ob15_arms.ob15_arms.HOLD_POLL_S", 0.0)
    real_init = OB15Arms.__init__

    def init_homed(self, config):
        real_init(self, config)
        self.lift_axis._homed = True
        self.lift_axis._z0_deg = 100.0 / self.lift_axis._mm_per_deg

    root = tmp_path / "combined"
    green = "cleaning up green brackets with the right arm"
    brown = "cleaning up brown brackets with the left arm"

    def session(task, resume, head_tilt):
        cfg = RecordConfig(
            robot=OB15ArmsConfig(
                cameras={}, calibration_dir=tmp_path / "cal", head_tilt=head_tilt, lift_height_mm=0.0
            ),
            teleop=_VRLikeTeleopConfig(),
            dataset=DatasetRecordConfig(
                repo_id="local/cleaning_up_brackets",
                single_task=task,
                root=root,
                fps=10,
                episode_time_s=1,
                reset_time_s=0,
                num_episodes=2,
                push_to_hub=False,
                video=False,
                no_stamp=True,  # the name stays exactly as given, so a later session finds the same dataset
            ),
            resume=resume,
            play_sounds=False,
        )
        with (
            patch(_FB, side_effect=FakeBus),
            patch.object(OB15Arms, "__init__", init_homed),
            patch.object(OB15, "calibrate", lambda self: None),
            patch("lerobot.scripts.lerobot_record.make_teleoperator_from_config", _VRLikeTeleop),
        ):
            return record(cfg)

    first = session(green, resume=False, head_tilt=65.0)
    assert (first.num_episodes, first.num_frames) == (2, 20)
    # a slightly different head pose than the first session, like the real green/brown sessions
    second = session(brown, resume=True, head_tilt=62.0)

    assert (second.num_episodes, second.num_frames) == (4, 40)  # appended, not a new dataset
    assert set(second.meta.tasks.index) == {green, brown}
    import glob

    import pandas as pd

    episodes = pd.concat(
        [
            pd.read_parquet(f)
            for f in sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
        ]
    )
    assert [list(t) for t in episodes.sort_values("episode_index")["tasks"]] == [
        [green],
        [green],
        [brown],
        [brown],
    ]
    frames = pd.concat(
        [
            pd.read_parquet(f)
            for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
        ]
    )
    task_index = frames.sort_values("index")["task_index"].to_numpy()
    assert set(task_index[:20]) != set(task_index[20:])  # frames carry their own task
    # the dataset keeps its original hold pose; the differing second session is reported, not silently mixed
    assert read_hold_file(str(root))["head_tilt.pos"] == 65.0


def test_rollout_state_includes_gantry_height(make_robot):
    """The rollout must hand the policy the same 15-value state the OB15 datasets were recorded with.

    ``gantry.height_mm`` has no ``.pos`` suffix; a ``.pos``/``.vel``-only filter used to drop it, so a
    15-wide checkpoint saw 14 values on the real robot.
    """
    from lerobot.rollout.context import _select_state_features

    robot = make_robot()
    state = [k for k, v in _select_state_features(robot.observation_features).items() if v is float]
    assert len(state) == 15
    assert state[:12] == list(ARM_KEYS)
    assert state[12:] == ["head_pan.pos", "head_tilt.pos", "gantry.height_mm"]
    # Base velocities stay out for the arms-only robot, and the action side is still the 12 arm keys.
    assert not any(k.startswith(("x.", "y.", "theta.")) for k in state)
    assert len(robot.action_features) == 12


def test_select_state_features_keeps_cameras_and_filters_other_scalars():
    from lerobot.rollout.context import _select_state_features

    features = {
        "a.pos": float,
        "b.vel": float,
        "gantry.height_mm": float,
        "battery_voltage": float,
        "cam": (480, 640, 3),
    }
    assert list(_select_state_features(features)) == ["a.pos", "b.vel", "gantry.height_mm", "cam"]
    assert list(_select_state_features(features, keep_all=True)) == list(features)


def test_rollout_position_hold_contract(make_robot):
    """The remote rollout refuses robots without a position-hold contract (it did on the first real run)."""
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    robot = make_robot()
    assert robot.supports_position_hold is True
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_hold()  # raised "ob15_arms has no supported local position-hold contract"
    assert wrapper.supports_hold

    applied = wrapper.send_action(_arm_action(5.0))
    assert set(applied) == set(ARM_KEYS) and len(applied) == 12
    wrapper.hold()  # re-sends the last applied targets
    for bus in robot._buses.values():
        assert all(isinstance(v, float) for v in bus.pos.values())


# --- lift failsafe: a velocity-mode motor keeps its last command if the host dies -------------------------


def _lift_bus(robot):
    return robot._group_bus["lift"]


def test_connect_zeroes_a_leftover_lift_velocity_before_anything_else(tmp_path, monkeypatch):
    """The lift ran down non-stop after a killed run: a new session must stop it before any other lift write."""
    monkeypatch.setattr("lerobot.robots.ob15_arms.ob15_arms.HOLD_POLL_S", 0.0)
    config = OB15ArmsConfig(cameras={}, calibration_dir=tmp_path / "calib")
    with patch(_FB, side_effect=FakeBus):
        robot = OB15Arms(config)
    bus = _lift_bus(robot)
    bus.vel["gantry"] = 400  # left running by the previous process
    robot.lift_axis._homed = True
    robot.lift_axis._z0_deg = 100.0 / robot.lift_axis._mm_per_deg
    with patch.object(OB15, "calibrate", lambda self: None):
        robot.connect()
    try:
        lift_writes = [(item, v["gantry"]) for item, v in bus.writes if "gantry" in v]
        assert lift_writes[0] == ("Goal_Velocity", 0), lift_writes[:3]
    finally:
        robot.disconnect()


def test_shutdown_signal_stops_the_lift_latches_and_chains(make_robot):
    import signal

    seen = []
    previous = signal.signal(signal.SIGHUP, lambda s, f: seen.append(s))  # what the rollout installs
    try:
        robot = make_robot()
        bus = _lift_bus(robot)
        bus.vel["gantry"] = 300
        signal.raise_signal(signal.SIGHUP)
        assert bus.vel["gantry"] == 0
        assert seen == [signal.SIGHUP]  # the previous handler still ran (graceful shutdown)
        # latched: a policy/teleop command can no longer start the lift again
        bus.writes.clear()
        robot.send_action({**_arm_action(1.0)})
        assert all(v["gantry"] == 0 for item, v in bus.writes if item == "Goal_Velocity" and "gantry" in v)
        assert bus.vel["gantry"] == 0
        robot.disconnect()
        assert signal.getsignal(signal.SIGHUP) is not robot._lift_signal_handler  # handler removed again
    finally:
        signal.signal(signal.SIGHUP, previous)


def test_failsafe_is_removed_on_disconnect_and_restores_the_previous_handler(make_robot):
    import signal

    marker = lambda s, f: None  # noqa: E731
    previous = signal.signal(signal.SIGTERM, marker)
    try:
        robot = make_robot()
        assert signal.getsignal(signal.SIGTERM) == robot._lift_signal_handler
        robot.disconnect()
        assert signal.getsignal(signal.SIGTERM) is marker
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_homing_error_zeroes_the_lift_velocity(make_robot):
    robot = make_robot()
    lift = robot.lift_axis
    bus = _lift_bus(robot)
    lift._homed = False

    def boom():
        raise RuntimeError("bus error mid-homing")

    lift._update_extended_ticks = boom
    with pytest.raises(RuntimeError):
        lift.home()
    assert bus.vel["gantry"] == 0


def test_lift_stop_does_not_configure_or_home(tmp_path):
    from lerobot.robots.ob15.lift_axis import LiftAxis, LiftAxisConfig

    bus = FakeBus("p", {})
    lift = LiftAxis(LiftAxisConfig(enabled=True, home_on_connect=True), bus)
    lift.attach()
    bus.vel[lift.cfg.name] = 250
    lift.stop()
    assert bus.vel[lift.cfg.name] == 0
    assert not any(item in ("Operating_Mode", "Torque_Enable") for item, _ in bus.writes)
    assert not lift._configured and not lift._homed


# --- camera read modes and the recording rate ------------------------------------------------------


class FakeCamera:
    """A camera with a read thread like the real ones: a NEW frame object every ``1/fps`` s, with the
    ``async_read`` (wait for the next new frame) and ``read_latest`` (take the buffer now) contract."""

    use_depth = False

    def __init__(self, fps=29.5, shape=(48, 64, 3)):
        self.period, self.shape = 1.0 / fps, shape
        self.is_connected = False
        self.latest_frame, self.latest_timestamp = None, None
        self._new = threading.Event()
        self._stop = threading.Event()
        self.frames_made = 0

    def connect(self, warmup=True):
        self.is_connected = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._new.wait(1.0)

    def _run(self):
        nxt = time.perf_counter()
        while not self._stop.is_set():
            frame = np.full(self.shape, self.frames_made % 250, dtype=np.uint8)
            self.frames_made += 1
            self.latest_frame, self.latest_timestamp = frame, time.perf_counter()
            self._new.set()
            nxt += self.period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def async_read(self, timeout_ms=200):
        if not self._new.wait(timeout_ms / 1000):
            raise TimeoutError("no new frame")
        self._new.clear()
        return self.latest_frame

    def read_latest(self, max_age_ms=500):
        if self.latest_frame is None:
            raise RuntimeError("has not captured any frames yet")
        if (time.perf_counter() - self.latest_timestamp) * 1e3 > max_age_ms:
            raise TimeoutError("latest frame is too old")
        return self.latest_frame

    def disconnect(self):
        self._stop.set()
        self.is_connected = False


def _camera_robot(make_robot, mode, fps=29.5, n=2, **cfg):
    robot = make_robot(camera_read_mode=mode, **cfg)
    robot.cameras = {f"cam{i}": FakeCamera(fps) for i in range(n)}
    for cam in robot.cameras.values():
        cam.connect()
    return robot


def test_invalid_camera_read_mode_is_rejected():
    with pytest.raises(ValueError, match="camera_read_mode"):
        with patch(_FB, side_effect=FakeBus):
            OB15Arms(OB15ArmsConfig(cameras={}, camera_read_mode="newest"))


def test_latest_mode_never_waits_for_a_frame_and_counts_repeated_frames(make_robot):
    robot = _camera_robot(make_robot, "latest", fps=20.0)
    try:
        robot.get_observation()  # warm the thread pool
        t0 = time.perf_counter()
        n = 200
        for _ in range(n):  # ~1 ms apart, far faster than the 50 ms camera period
            robot.get_observation()
            time.sleep(0.001)
        elapsed = time.perf_counter() - t0
        assert elapsed < n * 0.020, "latest mode must not block on the camera"
        stats = robot._frame_tracker.snapshot()
        assert set(stats) == {"cam0", "cam1"}
        for s in stats.values():
            assert s.reads >= n
            assert s.repeats > 0.8 * s.reads  # most reads saw the previous frame again
            assert s.longest_repeat_run >= 10
        report = "\n".join(robot.camera_frame_report())
        assert "camera cam0" in report and "repeated frames" in report
    finally:
        for cam in robot.cameras.values():
            cam.disconnect()


def test_fresh_mode_returns_new_frames_only_and_reports_nothing(make_robot):
    robot = _camera_robot(make_robot, "fresh", fps=50.0, n=1)
    try:
        seen = []
        for _ in range(6):
            seen.append(robot.get_observation()["cam0"])
        assert all(a is not b for a, b in zip(seen, seen[1:]))
        assert robot.camera_frame_report() == []
        assert robot._frame_tracker.snapshot() == {}
    finally:
        for cam in robot.cameras.values():
            cam.disconnect()


def test_latest_mode_stalled_camera_is_an_error_not_a_stale_frame(make_robot):
    robot = _camera_robot(make_robot, "latest", fps=50.0, n=1, camera_max_age_ms=100.0)
    cam = robot.cameras["cam0"]
    try:
        assert "cam0" in robot.get_observation()
        cam._stop.set()  # the camera thread dies; its buffer ages
        time.sleep(0.25)
        obs = robot.get_observation()
        assert "cam0" not in obs  # not served from the old buffer
        assert robot._frame_tracker.snapshot()["cam0"].errors == 1
    finally:
        cam.disconnect()


def _record_rate(tmp_path, monkeypatch, caplog, mode, work_s, fps=30, episode_time_s=4, cam_fps=29.5):
    """Run the real ``record()`` with fake cameras and a teleop costing ``work_s`` per tick; return the rate."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    monkeypatch.setattr("lerobot.robots.ob15_arms.ob15_arms.HOLD_POLL_S", 0.0)
    real_init = OB15Arms.__init__

    def init_homed(self, config):
        real_init(self, config)
        self.lift_axis._homed = True
        self.lift_axis._z0_deg = 100.0 / self.lift_axis._mm_per_deg

    class WorkTeleop(_VRLikeTeleop):
        def get_action(self):
            # stands in for VR IK + frame building + writer contention; a (lo, hi) pair is uniform jitter
            time.sleep(random.uniform(*work_s) if isinstance(work_s, tuple) else work_s)
            return super().get_action()

    cams_cfg = {f"cam{i}": OpenCVCameraConfig(index_or_path=i, fps=30, width=64, height=48) for i in range(3)}
    cfg = RecordConfig(
        robot=OB15ArmsConfig(
            cameras=cams_cfg, calibration_dir=tmp_path / "cal", head_pan=0.0, head_tilt=-35.0,
            lift_height_mm=220.0, camera_read_mode=mode,
        ),
        teleop=_VRLikeTeleopConfig(),
        dataset=DatasetRecordConfig(
            repo_id=f"local/rate_{mode}", single_task="t", root=tmp_path / f"ds_{mode}", fps=fps,
            episode_time_s=episode_time_s, reset_time_s=0, num_episodes=1, push_to_hub=False,
            video=False, no_stamp=True, num_image_writer_threads_per_camera=1,
        ),
        play_sounds=False,
    )
    fakes = {}

    def make_fakes(configs):
        for k in configs:
            fakes[k] = FakeCamera(cam_fps)
        return fakes

    with (
        patch(_FB, side_effect=lambda **kw: FakeBus(**kw)),
        patch.object(OB15Arms, "__init__", init_homed),
        patch.object(OB15, "calibrate", lambda self: None),
        patch("lerobot.robots.ob15.ob15.make_cameras_from_configs", make_fakes),
        patch("lerobot.scripts.lerobot_record.make_teleoperator_from_config", WorkTeleop),
        patch("lerobot.scripts.lerobot_record.init_logging", lambda: None),  # keep pytest's caplog handler
        caplog.at_level("INFO"),
    ):
        dataset = record(cfg)
    text = caplog.text
    hz = [float(m) for m in re.findall(r"effective cadence: ([0-9.]+) Hz", text)]
    return dataset, hz[-1], text


@pytest.mark.dataset
def test_record_with_latest_cameras_holds_the_target_rate_and_reports_it(tmp_path, monkeypatch, caplog):
    dataset, hz, text = _record_rate(tmp_path, monkeypatch, caplog, "latest", work_s=0.020)
    assert 29.0 <= hz <= 30.3, text
    # the loop reports its own statistics: percentiles, slow ticks and the camera repeats
    assert "tick period (ms" in text and "p99" in text and "slow ticks" in text
    assert "Camera frame report" in text and "cam0" in text and "repeated frames" in text
    assert "Achieved" not in text  # no below-target warning at the target rate
    assert dataset.num_frames >= int(4 * 29)


@pytest.mark.dataset
def test_record_warns_loudly_when_the_rate_is_below_target(tmp_path, monkeypatch, caplog):
    # 60 ms of work per tick cannot fit a 33 ms budget, whatever the camera mode
    dataset, hz, text = _record_rate(tmp_path, monkeypatch, caplog, "latest", work_s=0.060, episode_time_s=3)
    assert hz < 20, text
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "Achieved" in r.getMessage()]
    assert warnings and "against a 30 Hz target" in warnings[0] and "too fast" in warnings[0]
