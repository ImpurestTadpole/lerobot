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
    hold_reached,
    hold_status,
    resolve_hold_targets,
)
from .config_ob15 import OB15ArmsClientConfig
from .ob15_client import OB15Client

logger = logging.getLogger(__name__)

HOLD_SETTLE_TIMEOUT_S = 15.0
FIRST_OBS_TIMEOUT_S = 5.0
HOLD_POLL_S = 0.1


class OB15ArmsClient(OB15Client):
    config_class = OB15ArmsClientConfig
    name = "ob15_arms_client"

    def __init__(self, config: OB15ArmsClientConfig):
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
        current = self._wait_first_observation()
        cfg = self.config if self.config.hold_on_connect else _NoTargets()
        self._hold = resolve_hold_targets(cfg, current, self._lift_name)
        logger.info(f"{self}: holding head/lift at {self._hold}")

        deadline = time.monotonic() + HOLD_SETTLE_TIMEOUT_S
        while True:
            self._push(build_arms_command({}, self._hold))
            self.get_observation()
            status = hold_status(self.last_remote_state, self._hold)
            if hold_reached(status):
                return
            if time.monotonic() >= deadline:
                bad = {k: v for k, v in status.items() if not v["ok"]}
                logger.warning(
                    f"{self}: head/lift did not reach the hold pose within {HOLD_SETTLE_TIMEOUT_S}s: {bad}"
                )
                return
            time.sleep(HOLD_POLL_S)

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
        status = hold_status(self.last_remote_state, self._hold)
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
