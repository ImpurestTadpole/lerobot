"""Remote (ZMQ) arms-only OB15 client. The host keeps running the full-body ``ob15`` robot."""

from __future__ import annotations

import json
import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceNotConnectedError

from .arms_only import (
    arms_action_features,
    arms_only_result,
    arms_state_features,
    build_arms_command,
    build_hold_metadata,
    hold_keys,
    hold_reached,
    hold_source,
    hold_status,
    resolve_hold_targets,
    rest_pose_command,
)
from .config_ob15 import OB15ArmsClientConfig
from .ob15_client import OB15Client

logger = logging.getLogger(__name__)

FIRST_OBS_TIMEOUT_S = 5.0
HOLD_POLL_S = 0.1
REST_POSE_WAIT_S = 2.0


class OB15ArmsClient(OB15Client):
    config_class = OB15ArmsClientConfig
    name = "ob15_arms_client"

    def __init__(self, config: OB15ArmsClientConfig):
        rest_pose_command(config)  # fail early on a bad rest_pose
        super().__init__(config)
        self._hold: dict[str, float] | None = None

    @property
    def _lift_name(self) -> str | None:
        return self.config.lift_axis.name if self.config.lift_axis.enabled else None

    @property
    def _state_order(self) -> tuple[str, ...]:
        # Drives the `observation.state` vector too, so it excludes the base velocities.
        return tuple(arms_state_features(self._state_ft).keys())

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**arms_state_features(self._state_ft), **self._cameras_ft, **self._depth_cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return arms_action_features()

    def _wait_first_observation(self) -> dict[str, Any]:
        deadline = time.monotonic() + FIRST_OBS_TIMEOUT_S
        while time.monotonic() < deadline:
            self.get_observation()
            if self.last_remote_state:
                return self.last_remote_state
            time.sleep(HOLD_POLL_S)
        raise DeviceNotConnectedError(f"{self}: no observation received from the host after connect.")

    def connect(self) -> None:
        super().connect()
        cfg = self.config
        current = self._wait_first_observation()
        # hold_on_connect=False: do not move to hold_from/station, just freeze head/lift where they are.
        hold = resolve_hold_targets(cfg if cfg.hold_on_connect else _NoTargets(), current, self._lift_name)

        rest = rest_pose_command(cfg)
        if rest:
            # Arms go to the rest pose first; head/lift stay where they are until that is done.
            freeze = {k: float(current[k]) for k in hold_keys(self._lift_name) if k in current}
            self._push(build_arms_command(rest, freeze))
            time.sleep(REST_POSE_WAIT_S)

        self._hold = hold
        logger.info(f"{self}: holding head/lift at {self._hold}")

        deadline = time.monotonic() + cfg.settle_timeout_s
        while True:
            self._push(build_arms_command(rest or {}, self._hold))
            self.get_observation()
            status = hold_status(self.last_remote_state, self._hold, cfg.head_tol, cfg.lift_tol_mm)
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

    def _push(self, command: dict[str, float]) -> None:
        self.zmq_cmd_socket.send_string(json.dumps(command))

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected or self._hold is None:
            raise DeviceNotConnectedError(f"{self} is not connected. You need to run `robot.connect()`.")
        command = build_arms_command(action, self._hold)
        self._push(command)
        sent = {k: command[k] for k in arms_action_features() if k in command}
        vec = np.array([sent.get(k, 0.0) for k in arms_action_features()], dtype=np.float32)
        result = arms_only_result({k: float(v) for k, v in sent.items()})
        result["action"] = vec
        return result

    def check_hold(self) -> dict[str, dict[str, float | bool]]:
        """Head/lift drift vs the hold pose, from the latest observation."""
        if self._hold is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.get_observation()
        status = hold_status(
            self.last_remote_state, self._hold, self.config.head_tol, self.config.lift_tol_mm
        )
        for key, entry in status.items():
            if not entry["ok"]:
                logger.warning(f"{self}: {key} drifted from hold pose: {entry}")
        return status

    def disconnect(self) -> None:
        if self._hold is not None and self._is_connected:
            try:
                self.check_hold()
            except Exception as e:
                logger.debug(f"hold check at disconnect failed: {e}")
        self._hold = None
        super().disconnect()


class _NoTargets:
    station = None
    head_pan = None
    head_tilt = None
    lift_height_mm = None
    hold_from = None
