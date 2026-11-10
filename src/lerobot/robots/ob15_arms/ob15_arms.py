"""Arms-only OB15: 12 arm/gripper actions, head + lift held at a hold pose, no base commands."""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.utils.errors import DeviceNotConnectedError

from ..ob15.ob15 import OB15
from .arms_only import (
    BASE_KEYS,
    arms_action_features,
    arms_only_result,
    arms_state_features,
    build_arms_command,
    hold_keys,
    rest_pose_command,
)
from .config_ob15_arms import OB15ArmsConfig
from .hold import hold_reached, hold_source, hold_status, resolve_hold_targets
from .hold_file import build_hold_metadata

logger = logging.getLogger(__name__)

HOLD_POLL_S = 0.1
REST_POSE_WAIT_S = 2.0


class OB15Arms(OB15):
    """OB15 whose action space is the 12 arm/gripper joints only.

    On connect the head and lift move to the hold pose (station, overrides, or the current
    pose) and are re-commanded on every ``send_action`` so the lift P-controller keeps holding.
    Base, head and lift keys in incoming actions are dropped and no base command is sent (the low-level
    base path reads a missing command as zero velocity). Everything else is inherited from ``OB15``.
    """

    config_class = OB15ArmsConfig
    name = "ob15_arms"

    def __init__(self, config: OB15ArmsConfig):
        rest_pose_command(config)  # fail early on a bad rest_pose
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
        # hold_on_connect=False: do not move to hold_from/station, just freeze head/lift where they are.
        self._hold = None
        hold = resolve_hold_targets(cfg if cfg.hold_on_connect else _NoTargets(), current, self._lift_name)

        rest = rest_pose_command(cfg)
        if rest:
            # Arms go to the rest pose first; head/lift stay where they are until that is done.
            freeze = {k: float(current[k]) for k in hold_keys(self._lift_name) if k in current}
            OB15.send_action(self, build_arms_command(rest, freeze))
            time.sleep(REST_POSE_WAIT_S)

        self._hold = hold
        logger.info(f"{self}: holding head/lift at {self._hold}")

        deadline = time.monotonic() + cfg.settle_timeout_s
        while True:
            OB15.send_action(self, build_arms_command(rest or {}, self._hold))
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
    """Config stand-in with no station or overrides, so the current pose is held."""

    station = None
    head_pan = None
    head_tilt = None
    lift_height_mm = None
    hold_from = None
