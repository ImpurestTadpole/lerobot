import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.ob15 import OB15Config
from lerobot.robots.ob15.camera_tuning import tune_camera, uvc_exposure_value
from lerobot.robots.ob15_arms import OB15ArmsConfig

pytestmark = pytest.mark.hardware


def _controls(run):
    return [c.args[0][-1].removeprefix("--set-ctrl=") for c in run.call_args_list]


def test_uvc_wrist_gets_60hz_and_a_fixed_exposure_in_100us_units():
    cam = SimpleNamespace(index_or_path="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2.2:1.0-video-index0")
    with patch("lerobot.robots.ob15.camera_tuning.subprocess.run") as run:
        tune_camera(cam, 60, 8.33)
    assert _controls(run) == ["power_line_frequency=2", "auto_exposure=1", "exposure_time_absolute=83"]
    assert run.call_args_list[0].args[0][:3] == ["v4l2-ctl", "-d", cam.index_or_path]


def test_50hz_and_power_line_only():
    cam = SimpleNamespace(index_or_path="/dev/video6")
    with patch("lerobot.robots.ob15.camera_tuning.subprocess.run") as run:
        tune_camera(cam, 50, None)
    assert _controls(run) == ["power_line_frequency=1"]


def test_nothing_is_touched_when_both_settings_are_none():
    cam = SimpleNamespace(index_or_path="/dev/video6")
    with patch("lerobot.robots.ob15.camera_tuning.subprocess.run") as run:
        tune_camera(cam, None, None)
    run.assert_not_called()


def test_a_failing_control_does_not_block_the_others_or_raise():
    cam = SimpleNamespace(index_or_path="/dev/video6")
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[-1])
        if "power_line" in cmd[-1]:
            raise subprocess.CalledProcessError(1, cmd)

    with patch("lerobot.robots.ob15.camera_tuning.subprocess.run", side_effect=run):
        tune_camera(cam, 60, 8.33)  # must not raise
    assert len(calls) == 3  # the exposure controls were still attempted


def test_missing_v4l2_ctl_is_not_fatal():
    with patch("lerobot.robots.ob15.camera_tuning.subprocess.run", side_effect=FileNotFoundError("v4l2-ctl")):
        tune_camera(SimpleNamespace(index_or_path="/dev/video6"), 60, 8.33)


def test_realsense_gets_the_power_line_option_and_no_exposure_change():
    sensor = MagicMock()
    cam = SimpleNamespace(rs_profile=MagicMock())
    cam.rs_profile.get_device.return_value.first_color_sensor.return_value = sensor
    fake_rs = MagicMock()
    with patch.dict("sys.modules", {"pyrealsense2": fake_rs}):
        tune_camera(cam, 60, 8.33)
    sensor.set_option.assert_called_once_with(fake_rs.option.power_line_frequency, 2.0)


def test_invalid_frequency_is_rejected():
    with pytest.raises(ValueError):
        tune_camera(SimpleNamespace(index_or_path="/dev/video6"), 55, None)


def test_exposure_conversion_and_clamping():
    assert uvc_exposure_value(8.33) == 83
    assert uvc_exposure_value(16.67) == 167
    assert uvc_exposure_value(0.0) == 1
    assert uvc_exposure_value(10_000) == 5000


def test_defaults_arms_only_is_tuned_plain_ob15_is_unchanged():
    arms, plain = OB15ArmsConfig(cameras={}), OB15Config(cameras={})
    assert (arms.camera_power_line_hz, arms.wrist_exposure_ms) == (60, 8.33)
    assert (plain.camera_power_line_hz, plain.wrist_exposure_ms) == (None, None)
