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

from lerobot.cameras.configs import CameraConfig, Cv2Rotation, ColorMode
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from ..config import RobotConfig


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
        # camera1: Top/overhead view (head) - Intel RealSense D435i
        # MUST be opened FIRST to avoid resource conflicts
        # NOTE: Replace "YOUR_D435i_SERIAL_NUMBER" with your actual D435i serial number
        # You can find it by running: rs-enumerate-devices
        "head": RealSenseCameraConfig(
            serial_number_or_name="342222071125",  # Replace with your D435i serial number
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,  # Request BGR output
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True,
        ),
        
        # Original RGB camera (commented out, can be re-enabled if needed)
        # "head": OpenCVCameraConfig(
        #     index_or_path="/dev/video0", 
        #     fps=30,
        #     width=640,
        #     height=480,
        #     fourcc="MJPG",
        #     rotation=Cv2Rotation.NO_ROTATION,
        # ),
        
        # camera2: Wrist view (was "left_wrist")
        "left_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video8",  # Innomaker camera 2 (swapped)
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
            rotation=Cv2Rotation.NO_ROTATION,
            warmup_s=3,  # Increased warmup time for Innomaker cameras
        ),     
        
        # camera3: Additional view (was "right_wrist")
        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video6",  # Innomaker camera 1 (swapped)
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
            rotation=Cv2Rotation.NO_ROTATION,
            warmup_s=3,  # Increased warmup time for Innomaker cameras
        ),
    }


@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLerobotConfig(RobotConfig):
    
    port1: str = "/dev/ttyACM0"  # port to connect to the bus (left arm motors 1-6 + base motors 7-9)
    port2: str = "/dev/ttyACM1"  # port to connect to the bus (right arm motors 1-6 + head motors 7-8)
    camera_start_order: tuple[str, ...] | None = ("head", "left_wrist", "right_wrist")
    camera_start_delay_s: float = 1.0  # Increased delay to allow cameras to initialize properly
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

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



# ============================================================================
# CLIENT/HOST CONFIGURATIONS DISABLED - DIRECT USB CONNECTION ONLY
# ============================================================================
# The following configurations are commented out to enforce direct USB connection
# to the operating PC. Uncomment these if you need remote operation via ZMQ.
# ============================================================================

# @dataclass
# class XLerobotHostConfig:
#     # Network Configuration
#     port_zmq_cmd: int = 5555
#     port_zmq_observations: int = 5556
#
#     # Duration of the application
#     connection_time_s: int = 3600
#
#     # Watchdog: stop the robot if no command is received for over 0.5 seconds.
#     watchdog_timeout_ms: int = 500
#
#     # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
#     max_loop_freq_hz: int = 30

# @RobotConfig.register_subclass("xlerobot_client")
# @dataclass
# class XLerobotClientConfig(RobotConfig):
#     # Network Configuration
#     remote_ip: str
#     port_zmq_cmd: int = 5555
#     port_zmq_observations: int = 5556
#
#     teleop_keys: dict[str, str] = field(
#         default_factory=lambda: {
#             # Movement
#             "forward": "i",
#             "backward": "k",
#             "left": "j",
#             "right": "l",
#             "rotate_left": "u",
#             "rotate_right": "o",
#             # Speed control
#             "speed_up": "n",
#             "speed_down": "m",
#             # quit teleop
#             "quit": "b",
#         }
#     )
#
#     cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)
#
#     polling_timeout_ms: int = 15
#     connect_timeout_s: int = 5
