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


# --- hold file / hold_from / rest_pose (task 10) ----------------------------------------------

import contextlib  # noqa: E402
import logging  # noqa: E402

from lerobot.robots.ob15 import arms_only as hf  # noqa: E402
from lerobot.robots.ob15.arms_only import hold_source, rest_pose_command  # noqa: E402

HOLD = {"head_pan.pos": 0.0, "head_tilt.pos": -35.0, LIFT: 220.0}
OBS = {"head_pan.pos": 10.0, "head_tilt.pos": -5.0, LIFT: 100.0}  # what _FakeHardware starts at


def _meta(hold=None, robot_type="ob15_arms_client"):
    return hf.build_hold_metadata(
        robot_type,
        hold or HOLD,
        {"station": None, "hold_from": None, "overrides": {}, "hold_on_connect": True},
    )


def _dataset_with_hold(root, hold=None):
    hf.write_hold_file(root, _meta(hold), head_tol=2.0, lift_tol=5.0)
    return root


@contextlib.contextmanager
def _connected(monkeypatch, hw=None, **cfg):
    """A connected OB15Arms on fake hardware; yields (robot, hw)."""
    monkeypatch.setattr("lerobot.robots.ob15.ob15_arms.REST_POSE_WAIT_S", 0.0)
    config = OB15ArmsConfig(cameras={}, **cfg)
    with patch(f"{_MODULE}.FeetechMotorsBus", side_effect=_make_bus):
        robot = OB15Arms(config)
    hw = hw or _FakeHardware()
    with (
        patch.object(OB15, "connect", lambda self, calibrate=True: None),
        patch.object(OB15, "get_observation", lambda self, **kw: hw.get_observation(**kw)),
        patch.object(OB15, "send_action", lambda self, a: hw.send_action(a)),
        patch.object(OB15, "disconnect", _fake_disconnect),
    ):
        robot.connect()
        try:
            yield robot, hw
        finally:
            robot.disconnect()


def _hw_state(hw):
    return {k: hw.state[k] for k in ("head_pan.pos", "head_tilt.pos", LIFT)}


# 1. round trip
def test_hold_file_round_trip_dataset_checkpoint_and_direct_path(tmp_path):
    ds = _dataset_with_hold(tmp_path / "ds")
    assert (ds / "meta" / "ob15_hold.json").is_file()
    assert hf.read_hold_file(str(ds)) == HOLD

    ckpt = tmp_path / "ckpt"
    assert hf.copy_hold_file_to_checkpoint(ds, ckpt / "pretrained_model")
    assert hf.read_hold_file(str(ckpt)) == HOLD  # <dir>/pretrained_model/ob15_hold.json
    assert hf.read_hold_file(str(ckpt / "pretrained_model")) == HOLD  # <dir>/ob15_hold.json
    assert hf.read_hold_file(str(ds / "meta" / "ob15_hold.json")) == HOLD  # direct file


def test_hold_file_document_fields(tmp_path):
    ds = _dataset_with_hold(tmp_path / "ds")
    doc = json.loads((ds / "meta" / "ob15_hold.json").read_text())
    assert doc["version"] == hf.HOLD_FILE_VERSION == 1
    assert doc["robot_type"] == "ob15_arms_client"
    assert doc["hold"] == HOLD
    assert "recorded_at" in doc and "source" in doc
    assert not list((ds / "meta").glob("*.tmp"))  # atomic write leaves no temp file


# 2. read errors
def test_read_hold_file_missing_lists_every_tried_path(tmp_path):
    (tmp_path / "empty").mkdir()
    with (
        patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")),
        pytest.raises(FileNotFoundError) as exc,
    ):
        hf.read_hold_file(str(tmp_path / "empty"))
    msg = str(exc.value)
    for expected in ("meta/ob15_hold.json", "pretrained_model/ob15_hold.json", "hub:dataset:", "hub:model:"):
        assert expected in msg


@pytest.mark.parametrize(
    "content, match",
    [
        ("not json", "not valid JSON"),
        ('{"hold": {}}', "non-empty"),
        ('{"version": 1}', "non-empty"),
        ('{"hold": {"head_pan.pos": "left"}}', "must be a number"),
        ('{"hold": {"head_pan.pos": true}}', "must be a number"),
        ("[1, 2]", "JSON object"),
    ],
)
def test_read_hold_file_rejects_bad_content(tmp_path, content, match):
    path = tmp_path / "ob15_hold.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        hf.read_hold_file(str(path))


# 3. Hub
def _fake_download(tmp_path, available):
    def download(repo_id, filename, repo_type):
        if (repo_type, filename) not in available:
            raise RuntimeError(f"404 {repo_type} {filename}")
        path = tmp_path / f"{repo_type}.json"
        path.write_text(json.dumps(_meta(available[(repo_type, filename)])))
        return str(path)

    return download


def test_read_hold_file_from_hub_dataset(tmp_path):
    dl = _fake_download(tmp_path, {("dataset", "meta/ob15_hold.json"): HOLD})
    with patch("huggingface_hub.hf_hub_download", side_effect=dl) as mock:
        assert hf.read_hold_file("user/ob15_ds") == HOLD
    assert mock.call_args.kwargs == {
        "repo_id": "user/ob15_ds",
        "filename": "meta/ob15_hold.json",
        "repo_type": "dataset",
    }


def test_read_hold_file_from_hub_model_when_dataset_misses(tmp_path):
    model_hold = {**HOLD, LIFT: 150.0}
    dl = _fake_download(tmp_path, {("model", "ob15_hold.json"): model_hold})
    with patch("huggingface_hub.hf_hub_download", side_effect=dl) as mock:
        assert hf.read_hold_file("user/ob15_policy") == model_hold
    assert [c.kwargs["repo_type"] for c in mock.call_args_list] == ["dataset", "model"]


def test_read_hold_file_hub_miss_raises(tmp_path):
    with (
        patch("huggingface_hub.hf_hub_download", side_effect=_fake_download(tmp_path, {})),
        pytest.raises(FileNotFoundError, match="user/nothing"),
    ):
        hf.read_hold_file("user/nothing")


# 4. precedence (resolve_hold_targets, one test per level)
def _cfg(tmp_path, **kw):
    stations = tmp_path / "s.json"
    save_station(stations, "a", {"head_pan.pos": 1.0, "head_tilt.pos": 2.0, LIFT: 3.0})
    return OB15ArmsConfig(cameras={}, stations_file=str(stations), **kw)


def test_precedence_current_only(tmp_path):
    assert resolve_hold_targets(_cfg(tmp_path), OBS, "gantry") == OBS


def test_precedence_station_over_current(tmp_path):
    got = resolve_hold_targets(_cfg(tmp_path, station="a"), OBS, "gantry")
    assert got == {"head_pan.pos": 1.0, "head_tilt.pos": 2.0, LIFT: 3.0}


def test_precedence_hold_from_over_station(tmp_path):
    ds = _dataset_with_hold(tmp_path / "ds")
    got = resolve_hold_targets(_cfg(tmp_path, station="a", hold_from=str(ds)), OBS, "gantry")
    assert got == HOLD


def test_precedence_flag_over_hold_from(tmp_path, caplog):
    ds = _dataset_with_hold(tmp_path / "ds")
    cfg = _cfg(tmp_path, station="a", hold_from=str(ds), lift_height_mm=180.0, head_pan=7.0)
    with caplog.at_level(logging.INFO):
        got = resolve_hold_targets(cfg, OBS, "gantry")
    assert got == {"head_pan.pos": 7.0, "head_tilt.pos": -35.0, LIFT: 180.0}
    assert "sources" in caplog.text and str(ds) in caplog.text  # the log says where values came from


def test_hold_from_missing_key_falls_back_to_station_then_current(tmp_path, caplog):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"hold": {"head_tilt.pos": -20.0}}))
    with caplog.at_level(logging.WARNING):
        got = resolve_hold_targets(_cfg(tmp_path, station="a", hold_from=str(path)), OBS, "gantry")
    assert got == {"head_pan.pos": 1.0, "head_tilt.pos": -20.0, LIFT: 3.0}  # station fills the gaps
    assert "no value for" in caplog.text
    got = resolve_hold_targets(_cfg(tmp_path, hold_from=str(path)), OBS, "gantry")
    assert got == {"head_pan.pos": 10.0, "head_tilt.pos": -20.0, LIFT: 100.0}  # current fills the gaps


def test_hold_from_ignores_unknown_keys_with_warning(tmp_path, caplog):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"hold": {**HOLD, "x.vel": 1.0, "left_arm_gripper.pos": 5.0}}))
    with caplog.at_level(logging.WARNING):
        got = resolve_hold_targets(_cfg(tmp_path, hold_from=str(path)), OBS, "gantry")
    assert got == HOLD
    assert "non-hold keys" in caplog.text


def test_hold_from_with_lift_disabled_ignores_lift_value(tmp_path):
    ds = _dataset_with_hold(tmp_path / "ds")
    got = resolve_hold_targets(
        _cfg(tmp_path, hold_from=str(ds)), {"head_pan.pos": 5.0, "head_tilt.pos": 5.0}, None
    )
    assert got == {"head_pan.pos": 0.0, "head_tilt.pos": -35.0}


@pytest.mark.hardware
def test_robot_moves_to_hold_from_and_flag_beats_it(tmp_path, monkeypatch):
    ds = _dataset_with_hold(tmp_path / "ds")
    with _connected(monkeypatch, hold_from=str(ds)) as (_, hw):
        assert _hw_state(hw) == HOLD
    with _connected(monkeypatch, hold_from=str(ds), lift_height_mm=180.0) as (_, hw):
        assert _hw_state(hw) == {**HOLD, LIFT: 180.0}


# 5. hold_on_connect=False
@pytest.mark.hardware
def test_hold_on_connect_false_ignores_hold_from(tmp_path, monkeypatch):
    ds = _dataset_with_hold(tmp_path / "ds")
    with _connected(monkeypatch, hold_from=str(ds), hold_on_connect=False) as (robot, hw):
        assert _hw_state(hw) == OBS
        assert robot._hold == OBS


# 6. rest_pose
@pytest.mark.hardware
def test_rest_pose_sent_first_with_head_and_lift_frozen(tmp_path, monkeypatch):
    ds = _dataset_with_hold(tmp_path / "ds")
    rest = {"left_arm_shoulder_lift.pos": -90.0, "right_arm_gripper.pos": 12.0}
    with _connected(monkeypatch, hold_from=str(ds), rest_pose=rest) as (_, hw):
        first, second = hw.sent[0], hw.sent[1]
        assert first["left_arm_shoulder_lift.pos"] == -90.0 and first["right_arm_gripper.pos"] == 12.0
        assert {k: first[k] for k in OBS} == OBS  # head/lift have not moved yet
        assert all(first[k] == 0.0 for k in BASE_KEYS)
        assert {k: second[k] for k in HOLD} == HOLD  # then the move
        assert second["left_arm_shoulder_lift.pos"] == -90.0  # rest pose kept while settling
        assert _hw_state(hw) == HOLD


def test_rest_pose_rejects_non_arm_keys():
    assert rest_pose_command(OB15ArmsConfig(cameras={})) is None
    assert rest_pose_command(OB15ArmsConfig(cameras={}, rest_pose={ARM_KEYS[0]: 1})) == {ARM_KEYS[0]: 1.0}
    for bad in ("head_tilt.pos", LIFT, "x.vel"):
        with pytest.raises(ValueError, match="rest_pose"):
            rest_pose_command(OB15ArmsConfig(cameras={}, rest_pose={bad: 0.0}))
    with pytest.raises(
        ValueError, match="rest_pose"
    ):  # robots fail at construction, before touching hardware
        OB15ArmsClient(
            OB15ArmsClientConfig(remote_ip="127.0.0.1", cameras={}, rest_pose={"head_pan.pos": 0.0})
        )


# 7. settle uses the config
class _StuckHardware(_FakeHardware):
    """Never moves head/lift, whatever is commanded."""

    def send_action(self, action):
        self.sent.append(dict(action))
        return dict(action)


@pytest.mark.hardware
def test_settle_timeout_warns_and_does_not_hang(tmp_path, monkeypatch, caplog):
    ds = _dataset_with_hold(tmp_path / "ds")
    stuck = _StuckHardware()
    with (
        caplog.at_level(logging.WARNING),
        _connected(monkeypatch, stuck, hold_from=str(ds), settle_timeout_s=0.05),
    ):
        pass
    assert "did not reach the hold pose within 0.05s" in caplog.text


@pytest.mark.hardware
def test_settle_and_check_hold_use_config_tolerances(tmp_path, monkeypatch, caplog):
    path = tmp_path / "h.json"
    path.write_text(json.dumps({"hold": {**OBS, LIFT: 120.0}}))  # stuck lift is 20 mm off
    stuck = _StuckHardware()
    with (
        caplog.at_level(logging.WARNING),
        _connected(monkeypatch, stuck, hold_from=str(path), settle_timeout_s=0.05, lift_tol_mm=25.0) as (
            robot,
            _,
        ),
    ):
        assert hold_reached(robot.check_hold())
    assert "did not reach" not in caplog.text
    with _connected(monkeypatch, _StuckHardware(), hold_from=str(path), settle_timeout_s=0.05) as (robot, _):
        assert not hold_reached(robot.check_hold())  # default 5 mm


# 8. hold_metadata
@pytest.mark.hardware
def test_hold_metadata_contents(tmp_path, monkeypatch):
    with _connected(monkeypatch, station=None, head_tilt=-35.0, lift_height_mm=220.0) as (robot, _):
        meta = robot.hold_metadata()
    assert meta["robot_type"] == "ob15_arms"
    assert meta["version"] == 1
    assert meta["hold"] == {"gantry.height_mm": 220.0, "head_pan.pos": 10.0, "head_tilt.pos": -35.0}
    assert meta["source"] == {
        "station": None,
        "hold_from": None,
        "overrides": {"head_pan": None, "head_tilt": -35.0, "lift_height_mm": 220.0},
        "hold_on_connect": True,
    }
    assert meta["recorded_at"]


@pytest.mark.hardware
def test_hold_metadata_requires_connection():
    with patch(f"{_MODULE}.FeetechMotorsBus", side_effect=_make_bus):
        robot = OB15Arms(OB15ArmsConfig(cameras={}))
    from lerobot.utils.errors import DeviceNotConnectedError

    with pytest.raises(DeviceNotConnectedError):
        robot.hold_metadata()
    client = OB15ArmsClient(OB15ArmsClientConfig(remote_ip="127.0.0.1", cameras={}))
    with pytest.raises(DeviceNotConnectedError):
        client.hold_metadata()


def test_hold_source_reads_config_fields(tmp_path):
    cfg = _cfg(tmp_path, station="a", hold_from="u/ds", head_pan=1.0, hold_on_connect=False)
    assert hold_source(cfg) == {
        "station": "a",
        "hold_from": "u/ds",
        "overrides": {"head_pan": 1.0, "head_tilt": None, "lift_height_mm": None},
        "hold_on_connect": False,
    }


# 9. record hook helper (write_hold_file)
def test_write_hold_file_new_resume_match_mismatch_overwrite(tmp_path, caplog):
    root = tmp_path / "ds"
    path = hf.write_hold_file(root, _meta(), head_tol=2.0, lift_tol=5.0)
    original = path.read_text()

    # resume with a pose inside tolerance: unchanged, no warning
    near = {**HOLD, LIFT: 223.0, "head_tilt.pos": -34.0}
    with caplog.at_level(logging.WARNING):
        hf.write_hold_file(root, _meta(near), head_tol=2.0, lift_tol=5.0)
    assert path.read_text() == original and not caplog.records

    # resume with a different pose: unchanged + warning naming both poses and the flag
    far = {**HOLD, LIFT: 150.0}
    with caplog.at_level(logging.WARNING):
        hf.write_hold_file(root, _meta(far), head_tol=2.0, lift_tol=5.0)
    assert path.read_text() == original
    assert (
        "--dataset.overwrite_hold=true" in caplog.text and "150.0" in caplog.text and "220.0" in caplog.text
    )

    # overwrite=True replaces
    hf.write_hold_file(root, _meta(far), overwrite=True, head_tol=2.0, lift_tol=5.0)
    assert hf.read_hold_file(str(root)) == far


def test_holds_match_uses_per_key_tolerance():
    assert hf.holds_match(HOLD, {**HOLD, LIFT: 224.0}, head_tol=2.0, lift_tol=5.0)
    assert not hf.holds_match(HOLD, {**HOLD, LIFT: 226.0}, head_tol=2.0, lift_tol=5.0)
    assert not hf.holds_match(HOLD, {**HOLD, "head_pan.pos": 3.0}, head_tol=2.0, lift_tol=5.0)
    assert not hf.holds_match(HOLD, {"head_pan.pos": 0.0}, head_tol=2.0, lift_tol=5.0)  # different keys


def test_overwrite_hold_flag_defaults_off():
    from lerobot.configs.dataset import DatasetRecordConfig

    assert DatasetRecordConfig().overwrite_hold is False


# 10. checkpoint copy
def test_copy_hold_file_to_checkpoint(tmp_path):
    ds = _dataset_with_hold(tmp_path / "ds")
    out = tmp_path / "ckpt" / "pretrained_model"
    assert hf.copy_hold_file_to_checkpoint(ds, out) is True
    assert (out / "ob15_hold.json").read_text() == (ds / "meta" / "ob15_hold.json").read_text()
    other = tmp_path / "plain_ds"
    (other / "meta").mkdir(parents=True)
    out2 = tmp_path / "ckpt2" / "pretrained_model"
    assert hf.copy_hold_file_to_checkpoint(other, out2) is False
    assert not out2.exists()


def test_train_hook_copies_only_when_exactly_one_dataset_has_a_hold(tmp_path):
    ds = _dataset_with_hold(tmp_path / "ds")
    plain = tmp_path / "plain"
    (plain / "meta").mkdir(parents=True)
    single = MagicMock(spec=["root"], root=ds)
    assert hf.copy_dataset_hold_to_checkpoint(single, tmp_path / "o1") is True
    assert hf.copy_dataset_hold_to_checkpoint(MagicMock(spec=["root"], root=plain), tmp_path / "o2") is False
    multi = MagicMock(
        spec=["datasets"], datasets=[MagicMock(spec=["root"], root=ds), MagicMock(spec=["root"], root=plain)]
    )
    assert hf.copy_dataset_hold_to_checkpoint(multi, tmp_path / "o3") is True
    both = MagicMock(spec=["datasets"], datasets=[MagicMock(spec=["root"], root=ds)] * 2)
    assert hf.copy_dataset_hold_to_checkpoint(both, tmp_path / "o4") is False


# 11. client
def _client_with_fake_host(tmp_path, hw=None, **cfg):
    client = OB15ArmsClient(OB15ArmsClientConfig(remote_ip="127.0.0.1", cameras={}, **cfg))
    hw = hw or _FakeHardware()

    def fake_connect(self):
        self._is_connected = True
        self.zmq_cmd_socket = MagicMock()

    def fake_get_observation(self):
        self.last_remote_state = dict(hw.state)
        return dict(hw.state)

    patches = (
        patch.object(OB15Client, "connect", fake_connect),
        patch.object(OB15ArmsClient, "get_observation", fake_get_observation),
        patch.object(OB15ArmsClient, "_push", lambda self, command: hw.send_action(command)),
        patch.object(OB15Client, "disconnect", lambda self: None),
        patch("lerobot.robots.ob15.ob15_arms_client.REST_POSE_WAIT_S", 0.0),
    )
    return client, hw, patches


def _run_client(client, patches):
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        client.connect()


def test_client_precedence_hold_from_then_flag(tmp_path):
    ds = _dataset_with_hold(tmp_path / "ds")
    client, hw, patches = _client_with_fake_host(tmp_path, hold_from=str(ds))
    _run_client(client, patches)
    assert _hw_state(hw) == HOLD

    client, hw, patches = _client_with_fake_host(tmp_path, hold_from=str(ds), head_tilt=-10.0)
    _run_client(client, patches)
    assert _hw_state(hw) == {**HOLD, "head_tilt.pos": -10.0}
    meta = client.hold_metadata()
    assert meta["robot_type"] == "ob15_arms_client"
    assert meta["source"]["hold_from"] == str(ds) and meta["source"]["overrides"]["head_tilt"] == -10.0


def test_client_hold_on_connect_false_and_rest_pose(tmp_path):
    ds = _dataset_with_hold(tmp_path / "ds")
    client, hw, patches = _client_with_fake_host(tmp_path, hold_from=str(ds), hold_on_connect=False)
    _run_client(client, patches)
    assert _hw_state(hw) == OBS

    rest = {"right_arm_shoulder_lift.pos": -80.0}
    client, hw, patches = _client_with_fake_host(tmp_path, hold_from=str(ds), rest_pose=rest)
    _run_client(client, patches)
    assert hw.sent[0]["right_arm_shoulder_lift.pos"] == -80.0
    assert {k: hw.sent[0][k] for k in OBS} == OBS  # frozen
    assert {k: hw.sent[1][k] for k in HOLD} == HOLD
    assert _hw_state(hw) == HOLD


def test_client_settle_timeout_uses_config(tmp_path, caplog):
    ds = _dataset_with_hold(tmp_path / "ds")
    client, hw, patches = _client_with_fake_host(
        tmp_path, _StuckHardware(), hold_from=str(ds), settle_timeout_s=0.05
    )
    with caplog.at_level(logging.WARNING):
        _run_client(client, patches)
    assert "did not reach the hold pose within 0.05s" in caplog.text


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
