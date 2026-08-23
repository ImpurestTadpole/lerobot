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

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from ..config import RobotConfig
from .lift_axis import LiftAxisConfig

# From `lerobot-find-cameras opencv` on this Jetson:
#   #0 /dev/video2  usb-0:1.1:1.0-video-index2  D435i IR (GREY)     — unused
#   #1 /dev/video4  usb-0:1.1:1.3-video-index0  D435i RGB (YUYV)    — HEAD_RGB_BY_PATH
#   #2 /dev/video6  usb-0:2.4.3:1.0-video-index0 right Innomaker    — RIGHT_WRIST_BY_PATH
#   #3 /dev/video9  usb-0:2.4.2:1.0-video-index0 left Innomaker     — LEFT_WRIST_BY_PATH
# Do not use /dev/videoN: those indices shift. Do not use camera #0 as head RGB.

# D435i USB serial from the RealSense SDK (`pyrealsense2`). The V4L2/udev by-id serial
# (254343060891) is a different identifier — do not pass it to RealSenseCameraConfig.
REALSENSE_HEAD_SERIAL = "342222071125"

# OpenCV RGB fallback for the D435i (interface 1.3). Depth stays on the RealSense SDK;
# the Z16 node is usb-0:1.1:1.0-video-index0 and must not be opened as OpenCV.
HEAD_RGB_BY_PATH = "/dev/v4l/by-path/platform-3610000.usb-usb-0:1.1:1.3-video-index0"

# Wrist cameras share USB serial "SN0001", so by-path (physical port) is the stable id.
LEFT_WRIST_BY_PATH = "/dev/video6"
RIGHT_WRIST_BY_PATH = "/dev/video8"


def xlerobot_cameras_config() -> dict[str, CameraConfig]:
    """
    Camera configuration using SmolVLA's standardized naming convention.

    Camera naming aligns with SmolVLA's expected format:
    - camera1 = top/overhead view (was "head") - matches SmolVLA's OBS_IMAGE_1
    - camera2 = wrist view (was "left_wrist") - matches SmolVLA's OBS_IMAGE_2
    - camera3 = additional view (was "right_wrist") - matches SmolVLA's OBS_IMAGE_3

    This naming makes the robot natively compatible with SmolVLA policies without
    needing rename_map during training or inference.

    Note: camera1 MUST be opened FIRST to avoid resource conflicts.
    Head camera MUST be opened FIRST to avoid resource conflicts.
    Opening it after wrist cameras causes it to fail.
    """
    return {
        # camera1: D435i RGB + depth via the RealSense SDK (serial above, not /dev/video4).
        # 640x480 — reverted after two failed experiments: 1280x720 head + 960x540 wrists
        # collapsed the record loop from 30 Hz to ~7 Hz, and even 1280x720 head alone (wrists
        # left at 640x360) only sustained ~9-10 Hz. The RealSense color+depth capture/align at
        # 720p is itself too expensive for this Jetson's CPU budget at 30 Hz, independent of the
        # wrist cameras or VR streaming (which is already decoupled from capture resolution via
        # camera_stream_max_width). If higher head resolution is wanted later, try lowering the
        # head camera's fps (e.g. 15) instead of its resolution, or profile
        # cameras/realsense/camera_realsense.py's read path before retrying 720p.
        "head": RealSenseCameraConfig(
            serial_number_or_name=REALSENSE_HEAD_SERIAL,
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True,
        ),

        # camera2: left Innomaker (find-cameras #3). MJPG at 640x360 — cameras default to
        # YUYV 640x480 which saturates USB; MJPG is required for a 30 Hz control rate.
        "left_wrist": OpenCVCameraConfig(
            index_or_path=LEFT_WRIST_BY_PATH,
            fps=30,
            width=640,
            height=360,
            fourcc="MJPG",
            rotation=Cv2Rotation.NO_ROTATION,
            # Port 2.4.2 has dropped mid-session (usb disconnect). Extra warmup helps
            # the read thread recover; by-path still tracks the port after re-enumeration.
            warmup_s=5,
        ),

        # camera3: right Innomaker (find-cameras #2). Same MJPG note as left_wrist.
        "right_wrist": OpenCVCameraConfig(
            index_or_path=RIGHT_WRIST_BY_PATH,
            fps=30,
            width=640,
            height=360,
            fourcc="MJPG",
            rotation=Cv2Rotation.NO_ROTATION,
            warmup_s=5,
        ),
    }


@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLerobotConfig(RobotConfig):
    
    # USB serial: `ttyACM*` minor numbers are not stable across reboots — use udev symlinks for production.
    # This Jetson: ttyACM1 → bus1 (port1) left arm + base; ttyACM0 → bus2 (port2) right arm + head + lift.
    port1: str = "/dev/ttyACM1"  # bus1: left arm 1-6 + base 7-9
    port2: str = "/dev/ttyACM0"  # bus2: right arm 1-6 + head 7-8 + optional lift (motor 9 on bus2)
    camera_start_order: tuple[str, ...] | None = ("head", "left_wrist", "right_wrist")
    camera_start_delay_s: float = 2.0  # Increased delay to allow cameras to initialize properly (especially right_wrist)
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # Set to `False` for ob15 (no head pan/tilt hardware). Removes head_pan.pos / head_tilt.pos
    # from the state/action space and omits those motors from bus2, giving a 15D action space.
    use_head: bool = True

    # Optional gantry / Z lift axis (motor_id 9 on bus2). Activate with: lift_axis.enabled=True
    # or --robot.lift_axis.enabled=true. Calibration = homing during calibrate(); control =
    # gantry.height_mm (target mm) or gantry.vel; recorded as observation + action.
    lift_axis: LiftAxisConfig = field(default_factory=LiftAxisConfig)

    # Keyboard / VR base teleop tiers (n/m to cycle). Three levels: slow, medium, fast.
    base_speed_xy: tuple[float, float, float] = (0.15, 0.35, 0.5)  # m/s linear (fast ≈ kiwi max)
    base_speed_theta_deg: tuple[float, float, float] = (40.0, 80.0, 120.0)  # deg/s rotation

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "i",
            "backward": "k",
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            # Speed control
            "speed_up": "n",
            "speed_down": "m",
            # quit teleop
            "quit": "b",
        }
    )

    # Switches the head camera from OpenCV/V4L2 (RGB-only) to the RealSense SDK with depth capture
    # enabled. On by default — the head camera is a D435i, so depth is recorded automatically by
    # every lerobot-record/lerobot-teleoperate run with --robot.type=xlerobot (and ob15). Requires
    # `pyrealsense2` to be importable (see cameras/realsense/camera_realsense.py for the Jetson
    # GLIBC-mismatch guard). Set `--robot.use_realsense_depth=false` to fall back to plain RGB
    # (e.g. on a unit without a physical D435i, or if pyrealsense2 isn't installed).
    use_realsense_depth: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.use_realsense_depth:
            self.cameras["head"] = RealSenseCameraConfig(
                serial_number_or_name=REALSENSE_HEAD_SERIAL,
                fps=30,
                width=640,
                height=480,
                rotation=Cv2Rotation.NO_ROTATION,
                use_depth=True,
            )
        else:
            self.cameras["head"] = OpenCVCameraConfig(
                index_or_path=HEAD_RGB_BY_PATH,
                fps=30,
                width=640,
                height=480,
                fourcc="YUYV",
                rotation=Cv2Rotation.NO_ROTATION,
                warmup_s=2,
            )


# ZMQ bridge: host on robot (Jetson), client on GPU PC.


@dataclass
class XLerobotHostConfig:
    """ZMQ ports bound on the robot machine. Client connects to these."""

    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    connection_time_s: int = 3600
    watchdog_timeout_ms: int = 500
    max_loop_freq_hz: int = 30
    jpeg_quality: int = 90


@RobotConfig.register_subclass("xlerobot_client")
@dataclass
class XLerobotClientConfig(RobotConfig):
    """Remote XLerobot over ZMQ (`type: xlerobot_client` in robot JSON)."""

    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            "forward": "i",
            "backward": "k",
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            "speed_up": "n",
            "speed_down": "m",
            "quit": "b",
        }
    )
    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)
    lift_axis: LiftAxisConfig = field(default_factory=LiftAxisConfig)
    base_speed_xy: tuple[float, float, float] = (0.15, 0.35, 0.5)
    base_speed_theta_deg: tuple[float, float, float] = (40.0, 80.0, 120.0)
    # Must match the host's XLerobotConfig.use_head (set False for ob15_client).
    use_head: bool = True
    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
