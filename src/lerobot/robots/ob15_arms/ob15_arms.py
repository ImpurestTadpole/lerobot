"""Arms-only OB15: 12 arm/gripper actions, head + lift held at a hold pose, no base commands."""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.utils.errors import DeviceNotConnectedError

from ..ob15.ob15 import OB15
from .arms_only import (
    BASE_KEYS,
    DEFAULT_HEAD_TILT_DEG,
    arms_action_features,
    arms_only_result,
    arms_state_features,
    build_arms_command,
    build_hold_metadata,
    hold_reached,
    hold_source,
    hold_status,
    resolve_hold_targets,
)
from .config_ob15_arms import OB15ArmsConfig

logger = logging.getLogger(__name__)

HOLD_POLL_S = 0.1


class OB15Arms(OB15):
    """OB15 whose action space is the 12 arm/gripper joints only.

    On connect the head and lift move to the hold pose (overrides, hold_from, or the current
    pose) and are re-commanded on every ``send_action`` so the lift P-controller keeps holding.
    Base, head and lift keys in incoming actions are dropped. The wheel motors are not on the bus at all
    (``uses_base = False``), so nothing reads or writes them and an unplugged wheel cannot block connect. Everything else is inherited from ``OB15``.
    """

    config_class = OB15ArmsConfig
    name = "ob15_arms"
    uses_base = False  # wheel motors are not on the bus: unplugged/unpowered wheels do not block connect

    def __init__(self, config: OB15ArmsConfig):
        # Same hardware as ``ob15``: use its calibration folder (robots/ob15/<id>.json) unless one is given,
        # otherwise connect() would not find the existing calibration and start a manual one.
        if not config.calibration_dir:
            config.calibration_dir = HF_LEROBOT_CALIBRATION / ROBOTS / OB15.name
        # Homing parks the lift at the top; do not release its torque afterwards (that lets it fall).
        config.lift_axis.release_torque_after_home = False
        super().__init__(config)
        self._hold: dict[str, float] | None = None

    @property
    def _lift_name(self) -> str | None:
        return self.lift_axis.cfg.name if self.lift_axis.enabled else None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**arms_state_features(self._state_ft), **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return arms_action_features()

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        self._establish_hold()

    def _read_pose(self) -> dict[str, Any]:
        return OB15.get_observation(self, skip_cameras=True, skip_depth=True)

    def _establish_hold(self) -> None:
        cfg = self.config
        current = self._read_pose()
        # hold_on_connect=False: do not move to hold_from, just freeze head/lift where they are.
        self._hold = None
        hold = resolve_hold_targets(
            cfg if cfg.hold_on_connect else _NoTargets(),
            current,
            self._lift_name,
            default_head_tilt=DEFAULT_HEAD_TILT_DEG if cfg.hold_on_connect else None,
        )

        self._hold = hold
        logger.info(f"{self}: holding head/lift at {self._hold}")

        deadline = time.monotonic() + cfg.settle_timeout_s
        while True:
            OB15.send_action(self, build_arms_command({}, self._hold))
            status = hold_status(self._read_pose(), self._hold, cfg.head_tol, cfg.lift_tol_mm)
            if hold_reached(status):
                return
            if time.monotonic() >= deadline:
                bad = {k: v for k, v in status.items() if not v["ok"]}
                logger.warning(
                    f"{self}: head/lift did not reach the hold pose within {cfg.settle_timeout_s}s: {bad}"
                )
                return
            time.sleep(HOLD_POLL_S)

    def hold_metadata(self) -> dict[str, Any]:
        """Document for ``meta/ob15_hold.json``: the resolved hold pose and how it was requested."""
        if self._hold is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return build_hold_metadata(self.name, self._hold, hold_source(self.config))

    def get_observation(self, skip_cameras: bool = False, skip_depth: bool = False) -> dict[str, Any]:
        obs = super().get_observation(skip_cameras=skip_cameras, skip_depth=skip_depth)
        for key in BASE_KEYS:
            obs.pop(key, None)
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self._hold is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        sent = super().send_action(build_arms_command(action, self._hold))
        return arms_only_result(sent)

    def check_hold(self) -> dict[str, dict[str, float | bool]]:
        """Head/lift drift vs the hold pose right now. Call at the end of a task or episode."""
        if self._hold is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        status = hold_status(self._read_pose(), self._hold, self.config.head_tol, self.config.lift_tol_mm)
        for key, entry in status.items():
            if not entry["ok"]:
                logger.warning(f"{self}: {key} drifted from hold pose: {entry}")
        return status

    def disconnect(self) -> None:
        if self._hold is not None and self.is_connected:
            try:
                self.check_hold()
            except Exception as e:  # never block a disconnect on the diagnostic
                logger.debug(f"hold check at disconnect failed: {e}")
        self._hold = None
        super().disconnect()


class _NoTargets:
    """Config stand-in with no overrides, so the current pose is held."""

    head_pan = None
    head_tilt = None
    lift_height_mm = None
    hold_from = None
