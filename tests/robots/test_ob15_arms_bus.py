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
    assert obs["head_tilt.pos"] == pytest.approx(DEFAULT_HEAD_TILT_DEG) == pytest.approx(40.0)
    assert obs["head_pan.pos"] == pytest.approx(10.0)  # no default: the current pan is kept
    assert obs[LIFT] == pytest.approx(100.0, abs=1.0)


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
