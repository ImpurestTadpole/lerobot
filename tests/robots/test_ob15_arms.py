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

"""Tests for the arms-only OB15 robot types. Hardware and ZMQ are mocked."""

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("zmq")

from lerobot.robots import make_robot_from_config
from lerobot.robots.ob15 import (
    OB15,
    OB15Arms,
    OB15ArmsClient,
    OB15ArmsClientConfig,
    OB15ArmsConfig,
    OB15Client,
    OB15ClientConfig,
)
from lerobot.robots.ob15.arms_only import (
    ARM_KEYS,
    BASE_KEYS,
    build_arms_command,
    hold_reached,
    hold_status,
    resolve_hold_targets,
)
from lerobot.robots.ob15.stations import load_station, save_station

LIFT = "gantry.height_mm"
_MODULE = "lerobot.robots.ob15.ob15"


def _make_bus(port, motors, calibration):
    bus = MagicMock(name=f"FeetechMotorsBus({port})")
    bus.port = port
    bus.motors = motors
    bus.is_connected = True
    return bus


def _fake_disconnect(self):
    """Mark the mocked buses disconnected so Robot.__del__ doesn't run the real disconnect."""
    for bus in self._buses.values():
        bus.is_connected = False


class _FakeHardware:
    """Stands in for OB15's motor I/O: instantly reaches any commanded head/lift target."""

    def __init__(self):
        self.state = dict.fromkeys(ARM_KEYS, 0.0)
        self.state.update({"head_pan.pos": 10.0, "head_tilt.pos": -5.0, LIFT: 100.0})
        self.state.update(dict.fromkeys(BASE_KEYS, 0.0))
        self.sent: list[dict] = []

    def get_observation(self, skip_cameras=False, skip_depth=False):
        return dict(self.state)

    def send_action(self, action):
        self.sent.append(dict(action))
        for key in ("head_pan.pos", "head_tilt.pos", LIFT):
            if key in action:
                self.state[key] = action[key]
        return dict(action)


@pytest.fixture
def arms_robot(tmp_path):
    """A connected OB15Arms on a fake hardware layer, with station 'counter' defined."""
    stations = tmp_path / "stations.json"
    save_station(stations, "counter", {"head_pan.pos": 0.0, "head_tilt.pos": -35.0, LIFT: 220.0})
    config = OB15ArmsConfig(cameras={}, station="counter", stations_file=str(stations))
    with patch(f"{_MODULE}.FeetechMotorsBus", side_effect=_make_bus):
        robot = OB15Arms(config)
    hw = _FakeHardware()
    with (
        patch.object(OB15, "connect", lambda self, calibrate=True: None),
        patch.object(OB15, "get_observation", lambda self, **kw: hw.get_observation(**kw)),
        patch.object(OB15, "send_action", lambda self, a: hw.send_action(a)),
        patch.object(OB15, "disconnect", _fake_disconnect),
    ):
        robot.connect()
        yield robot, hw
        robot.disconnect()


# --- features ---------------------------------------------------------------------------------


def test_action_features_are_exactly_the_12_arm_keys_in_order(arms_robot):
    robot, _ = arms_robot
    assert tuple(robot.action_features) == ARM_KEYS
    assert len(ARM_KEYS) == 12


def test_observation_has_head_and_lift_but_no_base(arms_robot):
    robot, _ = arms_robot
    state_keys = set(robot.observation_features)
    assert {"head_pan.pos", "head_tilt.pos", LIFT} <= state_keys
    assert not state_keys & set(BASE_KEYS)


def test_get_observation_drops_base_keys(arms_robot):
    robot, hw = arms_robot
    hw.state["x.vel"] = 0.3
    with patch.object(OB15, "get_observation", lambda self, **kw: hw.get_observation(**kw)):
        obs = robot.get_observation()
    assert not set(BASE_KEYS) & set(obs)
    assert LIFT in obs


# --- hold behaviour ---------------------------------------------------------------------------


def test_connect_moves_head_and_lift_to_the_station(arms_robot):
    _, hw = arms_robot
    assert hw.state["head_pan.pos"] == 0.0
    assert hw.state["head_tilt.pos"] == -35.0
    assert hw.state[LIFT] == 220.0


def test_send_action_ignores_head_lift_and_base_inputs(arms_robot):
    robot, hw = arms_robot
    action = dict.fromkeys(ARM_KEYS, 1.0)
    action.update({"head_pan.pos": 45.0, "head_tilt.pos": 30.0, LIFT: 5.0, "x.vel": 0.5, "theta.vel": 20.0})
    result = robot.send_action(action)

    sent = hw.sent[-1]
    assert sent["head_pan.pos"] == 0.0
    assert sent["head_tilt.pos"] == -35.0
    assert sent[LIFT] == 220.0
    assert all(sent[k] == 0.0 for k in BASE_KEYS)
    assert tuple(result) == ARM_KEYS  # only the 12 arm keys come back


def test_head_and_lift_never_move_across_many_actions(arms_robot):
    robot, hw = arms_robot
    for i in range(50):
        robot.send_action({**dict.fromkeys(ARM_KEYS, float(i)), "head_tilt.pos": float(i), LIFT: float(i)})
    assert (hw.state["head_pan.pos"], hw.state["head_tilt.pos"], hw.state[LIFT]) == (0.0, -35.0, 220.0)


def test_check_hold_reports_ok_then_detects_drift(arms_robot):
    robot, hw = arms_robot
    with patch.object(OB15, "get_observation", lambda self, **kw: hw.get_observation(**kw)):
        assert hold_reached(robot.check_hold())
        hw.state[LIFT] = 200.0  # lift sagged 20 mm
        status = robot.check_hold()
    assert not hold_reached(status)
    assert status[LIFT]["ok"] is False
    assert status[LIFT]["error"] == pytest.approx(-20.0)
    assert status["head_pan.pos"]["ok"] is True


def test_hold_on_connect_false_freezes_the_current_pose(tmp_path):
    stations = tmp_path / "s.json"
    save_station(stations, "counter", {"head_pan.pos": 0.0, "head_tilt.pos": -35.0, LIFT: 220.0})
    config = OB15ArmsConfig(cameras={}, station="counter", stations_file=str(stations), hold_on_connect=False)
    with patch(f"{_MODULE}.FeetechMotorsBus", side_effect=_make_bus):
        robot = OB15Arms(config)
    hw = _FakeHardware()
    with (
        patch.object(OB15, "connect", lambda self, calibrate=True: None),
        patch.object(OB15, "get_observation", lambda self, **kw: hw.get_observation(**kw)),
        patch.object(OB15, "send_action", lambda self, a: hw.send_action(a)),
        patch.object(OB15, "disconnect", _fake_disconnect),
    ):
        robot.connect()
        robot.disconnect()  # inside the patch: Robot.__del__ would otherwise run the real disconnect
    assert hw.state["head_pan.pos"] == 10.0  # not moved to the station
    assert hw.state[LIFT] == 100.0


# --- hold target resolution -------------------------------------------------------------------


def test_override_beats_station_beats_current(tmp_path):
    stations = tmp_path / "s.json"
    save_station(stations, "a", {"head_pan.pos": 1.0, "head_tilt.pos": 2.0, LIFT: 3.0})
    current = {"head_pan.pos": 9.0, "head_tilt.pos": 9.0, LIFT: 9.0}

    cfg = OB15ArmsConfig(cameras={}, station="a", stations_file=str(stations), head_tilt=7.0)
    got = resolve_hold_targets(cfg, current, "gantry")
    assert got == {"head_pan.pos": 1.0, "head_tilt.pos": 7.0, LIFT: 3.0}

    cfg = OB15ArmsConfig(cameras={})
    assert resolve_hold_targets(cfg, current, "gantry") == current


def test_lift_disabled_has_no_lift_target():
    cfg = OB15ArmsConfig(cameras={})
    got = resolve_hold_targets(cfg, {"head_pan.pos": 0.0, "head_tilt.pos": 0.0}, None)
    assert set(got) == {"head_pan.pos", "head_tilt.pos"}


# --- stations ---------------------------------------------------------------------------------


def test_station_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Station file not found"):
        load_station(tmp_path / "missing.json", "x")
    path = tmp_path / "s.json"
    save_station(path, "a", {"head_pan.pos": 1.0})
    with pytest.raises(KeyError, match="Available"):
        load_station(path, "nope")
    path.write_text("not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_station(path, "a")


def test_save_station_keeps_other_stations(tmp_path):
    path = tmp_path / "s.json"
    save_station(path, "a", {"head_pan.pos": 1.0})
    save_station(path, "b", {"head_pan.pos": 2.0})
    assert json.loads(path.read_text()) == {"a": {"head_pan.pos": 1.0}, "b": {"head_pan.pos": 2.0}}


def test_hold_status_flags_missing_observation_key():
    status = hold_status({"head_pan.pos": 0.0}, {"head_pan.pos": 0.0, LIFT: 100.0})
    assert status["head_pan.pos"]["ok"] is True
    assert status[LIFT]["ok"] is False and status[LIFT]["actual"] is None


def test_build_arms_command_drops_extras_and_zeroes_base():
    cmd = build_arms_command({ARM_KEYS[0]: 3.0, "x.vel": 9.0, "head_pan.pos": 9.0}, {"head_pan.pos": 0.0})
    assert cmd[ARM_KEYS[0]] == 3.0
    assert cmd["head_pan.pos"] == 0.0
    assert all(cmd[k] == 0.0 for k in BASE_KEYS)


# --- client -----------------------------------------------------------------------------------


def test_plain_client_can_be_constructed():
    """Regression: OB15ClientConfig lacked base_speed_xy/base_speed_theta_deg/use_head, so the
    client crashed with AttributeError on construction."""
    client = OB15Client(OB15ClientConfig(remote_ip="127.0.0.1", cameras={}))
    assert len(client.action_features) == 18
    assert client.speed_levels[0]["xy"] == pytest.approx(0.1)


def test_arms_client_features_and_registration():
    client = make_robot_from_config(OB15ArmsClientConfig(remote_ip="127.0.0.1", cameras={}))
    assert isinstance(client, OB15ArmsClient)
    assert tuple(client.action_features) == ARM_KEYS
    assert not set(BASE_KEYS) & set(client.observation_features)
    assert LIFT in client.observation_features
    assert "x.vel" not in client._state_order  # observation.state vector has no base velocity


def test_arms_client_sends_hold_and_zero_base_over_zmq():
    client = OB15ArmsClient(OB15ArmsClientConfig(remote_ip="127.0.0.1", cameras={}))
    client._is_connected = True
    client.zmq_cmd_socket = MagicMock()
    client._hold = {"head_pan.pos": 0.0, "head_tilt.pos": -35.0, LIFT: 220.0}

    result = client.send_action({**dict.fromkeys(ARM_KEYS, 2.0), "head_tilt.pos": 50.0, "x.vel": 1.0})

    payload = json.loads(client.zmq_cmd_socket.send_string.call_args[0][0])
    assert payload["head_tilt.pos"] == -35.0
    assert payload[LIFT] == 220.0
    assert all(payload[k] == 0.0 for k in BASE_KEYS)
    assert result["action"].shape == (12,)
    assert set(result) == {*ARM_KEYS, "action"}


def test_arms_client_connect_holds_station_then_reports_drift(tmp_path):
    stations = tmp_path / "s.json"
    save_station(stations, "counter", {"head_pan.pos": 0.0, "head_tilt.pos": -35.0, LIFT: 220.0})
    client = OB15ArmsClient(
        OB15ArmsClientConfig(
            remote_ip="127.0.0.1", cameras={}, station="counter", stations_file=str(stations)
        )
    )
    hw = _FakeHardware()

    def fake_connect(self):
        self._is_connected = True
        self.zmq_cmd_socket = MagicMock()

    def fake_get_observation(self):
        self.last_remote_state = dict(hw.state)
        return dict(hw.state)

    def push(command):  # the host would apply the hold command; emulate it
        hw.send_action(command)

    with (
        patch.object(OB15Client, "connect", fake_connect),
        patch.object(OB15ArmsClient, "get_observation", fake_get_observation),
        patch.object(OB15ArmsClient, "_push", lambda self, command: push(command)),
        patch.object(OB15Client, "disconnect", lambda self: None),
    ):
        client.connect()
        assert (hw.state["head_pan.pos"], hw.state["head_tilt.pos"], hw.state[LIFT]) == (0.0, -35.0, 220.0)
        assert hold_reached(client.check_hold())
        hw.state["head_tilt.pos"] = -20.0
        status = client.check_hold()
    assert status["head_tilt.pos"]["ok"] is False


# --- dataset conversion / head-lift report ----------------------------------------------------

FULL_NAMES = [*ARM_KEYS, "head_pan.pos", "head_tilt.pos", *BASE_KEYS, LIFT]


@pytest.fixture
def full_dataset(tmp_path):
    """Two tiny full-body episodes: ep 0 is still; ep 1 tilts the head 10 deg and drifts the lift."""
    import numpy as np

    from lerobot.datasets import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (len(FULL_NAMES),), "names": FULL_NAMES},
        "observation.state": {"dtype": "float32", "shape": (len(FULL_NAMES),), "names": FULL_NAMES},
    }
    ds = LeRobotDataset.create(
        "test/ob15_full", fps=15, features=features, root=tmp_path / "full", use_videos=False
    )
    head_i, lift_i = FULL_NAMES.index("head_tilt.pos"), FULL_NAMES.index(LIFT)
    for ep in range(2):
        for t in range(10):
            vec = np.arange(len(FULL_NAMES), dtype=np.float32)
            vec[[FULL_NAMES.index(k) for k in BASE_KEYS]] = 0.0  # base stationary
            if ep == 1:
                vec[head_i] = -35.0 + t * (10.0 / 9)
                vec[lift_i] = 220.0 - t * 2.0
            ds.add_frame({"action": vec, "observation.state": vec, "task": "t"})
        ds.save_episode()
    ds.finalize()
    return tmp_path / "full"


def test_report_flags_only_episodes_where_head_or_lift_moved(full_dataset):
    from lerobot.robots.ob15.scripts.convert_to_arms_only import report

    rows = report(full_dataset)
    assert [bool(r["flags"]) for r in rows] == [False, True]
    assert rows[1]["head_tilt.pos"]["drift"] == pytest.approx(10.0, abs=0.01)
    assert rows[1][LIFT]["end"] == pytest.approx(202.0)
    assert rows[0]["head_tilt.pos"]["range"] == 0.0


def test_convert_to_arms_only_drops_base_and_keeps_arms(full_dataset, tmp_path):
    import numpy as np

    from lerobot.datasets import LeRobotDataset
    from lerobot.robots.ob15.scripts.convert_to_arms_only import convert

    out = convert(full_dataset, tmp_path / "arms", "test/ob15_arms")
    ds = LeRobotDataset("test/ob15_arms", root=out)

    assert tuple(ds.meta.features["action"]["names"]) == ARM_KEYS
    assert not set(BASE_KEYS) & set(ds.meta.features["observation.state"]["names"])
    sample = ds[3]
    assert sample["action"].shape == (12,)
    assert sample["observation.state"].shape == (len(FULL_NAMES) - 3,)
    np.testing.assert_allclose(sample["action"].numpy(), np.arange(12, dtype=np.float32))
    assert ds.meta.stats["action"]["mean"].shape == (12,)
    # source untouched
    src = LeRobotDataset("test/ob15_full", root=full_dataset)
    assert src.meta.features["action"]["shape"] == (len(FULL_NAMES),)


def test_convert_refuses_to_overwrite(full_dataset, tmp_path):
    from lerobot.robots.ob15.scripts.convert_to_arms_only import convert

    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        convert(full_dataset, tmp_path / "exists", "test/x")
