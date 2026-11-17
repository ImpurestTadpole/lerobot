"""Best-effort anti-flicker settings for the OB15 cameras.

Mains-powered lights flicker at twice the mains frequency (120 Hz on 60 Hz mains) and a camera whose exposure is
not a multiple of that period records moving horizontal bands. UVC cameras default to 50 Hz compensation, and their
auto-exposure can hunt, which also makes brightness drift between frames. This sets the power-line frequency on
every camera that supports it and (optionally) locks the exposure of the OpenCV/UVC cameras to a fixed time.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# V4L2 menu values of the power_line_frequency control (RealSense uses the same 1 = 50 Hz, 2 = 60 Hz).
_POWER_LINE_VALUE = {50: 1, 60: 2}
_UVC_EXPOSURE_UNIT_MS = 0.1  # exposure_time_absolute is in units of 100 microseconds
_UVC_EXPOSURE_RANGE = (1, 5000)


def uvc_exposure_value(exposure_ms: float) -> int:
    lo, hi = _UVC_EXPOSURE_RANGE
    return max(lo, min(hi, round(exposure_ms / _UVC_EXPOSURE_UNIT_MS)))


def _set_v4l2_controls(device: str, controls: list[str]) -> list[str]:
    """Set each control separately (a camera that lacks one must not block the others). Returns the failures."""
    failed = []
    for control in controls:
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", device, f"--set-ctrl={control}"],
                check=True,
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as e:
            failed.append(control)
            logger.warning(f"camera tuning: could not set {control} on {device}: {e}")
    return failed


def tune_camera(cam: Any, power_line_hz: int | None, exposure_ms: float | None = None) -> None:
    """Apply the settings to one connected camera. Never raises: a camera that cannot be tuned still works."""
    if power_line_hz is not None and power_line_hz not in _POWER_LINE_VALUE:
        raise ValueError(f"power_line_hz must be 50, 60 or None, got {power_line_hz!r}")
    if power_line_hz is None and exposure_ms is None:
        return
    try:
        path = getattr(cam, "index_or_path", None)
        if path is not None and str(path).startswith("/dev/"):  # OpenCV / UVC camera
            controls = []
            if power_line_hz is not None:
                controls.append(f"power_line_frequency={_POWER_LINE_VALUE[power_line_hz]}")
            if exposure_ms is not None:
                controls += ["auto_exposure=1", f"exposure_time_absolute={uvc_exposure_value(exposure_ms)}"]
            _set_v4l2_controls(str(Path(str(path))), controls)
        elif getattr(cam, "rs_profile", None) is not None and power_line_hz is not None:  # RealSense
            import pyrealsense2 as rs

            sensor = cam.rs_profile.get_device().first_color_sensor()
            sensor.set_option(rs.option.power_line_frequency, float(_POWER_LINE_VALUE[power_line_hz]))
    except Exception as e:  # tuning is an optimisation, never a reason to fail connect
        logger.warning(f"camera tuning failed for {cam}: {e}")
