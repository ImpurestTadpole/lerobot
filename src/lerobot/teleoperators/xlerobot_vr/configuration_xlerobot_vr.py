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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig

# Physical controller button -> semantic session/DAgger action, keyed as
# "<left|right>.<button>" (button names match the raw VR packet's `buttons` dict:
# x, y, a, b, menu, thumbstick).
#
# This map only covers discrete, edge-triggered buttons used for recording/DAgger
# control flow. Continuous motion axes (index trigger -> gripper, thumbstick
# movement -> base/lift velocity, controller pose -> arm IK) are not remappable
# here and keep their fixed bindings.
#
# Recognised semantic actions:
#   rerecord_episode   - discard and re-record the current episode
#   exit_early          - end the current episode early (save & continue)
#   stop_session        - stop recording / end the rollout session
#   reset_position       - reset the robot to its rest pose (momentary, held)
#   toggle_intervention   - DAgger human/policy handover (RIGHT A today)
#   upload_dataset        - push the recorded dataset to the Hub on demand
DEFAULT_VR_BUTTON_MAP: dict[str, str] = {
    "left.x": "rerecord_episode",
    "left.y": "rerecord_episode",
    "left.menu": "stop_session",
    "left.thumbstick": "reset_position",
    "right.b": "exit_early",
    "right.a": "toggle_intervention",
    "right.thumbstick": "upload_dataset",
}


@TeleoperatorConfig.register_subclass("xlerobot_vr")
@dataclass
class XLerobotVRTeleopConfig(TeleoperatorConfig):
    # VR sysytem setting
    vr_enabled: bool = True
    vr_connection_timeout: float = 10.0
    vr_data_timeout: float = 5.0

    kp: float = 1.0  # Proportional gain for arm control

    # XLeVR checkout (serves the WebXR page + certs, provides the xlevr package).
    # None = auto-resolve: XLEVR_PATH env var, then ~/XLeRobot/XLeVR and other
    # well-known locations (see vr_monitor._resolve_xlevr_path).
    xlevr_path: str | None = None

    # Maps physical controller buttons to semantic session/DAgger actions.
    # See ``DEFAULT_VR_BUTTON_MAP`` for the key format and recognised actions.
    # Override to remap buttons (e.g. --teleop.button_map='{"right.thumbstick":
    # "stop_session"}') without touching motion bindings.
    button_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_VR_BUTTON_MAP))

    # Stream the robot's camera feeds to the VR headset for display while teleoperating.
    stream_cameras_to_vr: bool = True
    camera_stream_fps: float = 10.0
    camera_stream_jpeg_quality: int = 80
    # Downscale frames to at most this width before JPEG-encoding for the VR stream, independent
    # of the camera's captured/recorded resolution (set in the robot's camera config). Keeps the
    # per-frame encode+send cost fixed even if capture resolution is bumped up for dataset
    # quality. Aspect ratio is preserved; frames already at or below this width pass through
    # unscaled. Set to 0 to disable and stream at full captured resolution.
    camera_stream_max_width: int = 480
    # Depth values (mm) beyond this are clipped before normalizing to the 0-255 false-color depth
    # view sent to the headset (toggled via RIGHT thumbstick click). Tune to the working distance
    # of the task — e.g. lower this for close-up manipulation to get more color contrast in the
    # near field, or raise it if the depth view looks saturated/flat at typical distances.
    camera_stream_depth_max_mm: int = 3000
