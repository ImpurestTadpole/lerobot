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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class RealSenseCameraConfig(CameraConfig):
    """Configuration class for Intel RealSense cameras.

    This class provides specialized configuration options for Intel RealSense cameras,
    including support for depth sensing and device identification via serial number or name.

    Example configurations for Intel RealSense D405:
    ```python
    # Basic configurations
    RealSenseCameraConfig("0123456789", 30, 1280, 720)  # 1280x720 @ 30FPS
    RealSenseCameraConfig("0123456789", 60, 640, 480)  # 640x480 @ 60FPS

    # Advanced configurations
    RealSenseCameraConfig("0123456789", 30, 640, 480, use_depth=True)  # With depth sensing
    RealSenseCameraConfig("0123456789", 30, 640, 480, rotation=Cv2Rotation.ROTATE_90)  # With 90° rotation
    ```

    Attributes:
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        serial_number_or_name: Unique serial number or human-readable name to identify the camera.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        use_rgb: Whether to enable the color stream. Defaults to True.
        use_depth: Whether to enable depth stream. Defaults to False.
        depth_width: Requested frame width in pixels for the depth stream. Defaults to None,
            which uses `width` (same resolution as color). Lower depth resolution reduces the
            USB/hardware-sync cost of `try_wait_for_frames()` on every hardware frame — unlike
            `depth_frame_interval`, which only skips decoding an already-arrived frame.
        depth_height: Requested frame height in pixels for the depth stream. Defaults to None,
            which uses `height` (same resolution as color).
        depth_frame_interval: Decode a depth frame only every Nth hardware frame (color stays
            full-rate). Lower CPU cost, lower depth temporal resolution. Defaults to 1 (every frame).
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        exposure: Manual exposure value for the color sensor. When set, auto-exposure is
            disabled and this fixed value is used. Valid ranges are camera-model specific
            and reported if the value is rejected. Defaults to None (leave unchanged).
        gain: Manual gain value for the color sensor. When set, auto-exposure is disabled
            and this fixed gain is used, which also freezes exposure at its current value
            when no exposure is configured. Valid ranges are camera-model specific and
            reported if the value is rejected. Defaults to None (leave unchanged).
        white_balance: Manual white balance value for the color sensor. When set, auto
            white balance is disabled and this fixed value is used. Valid ranges are
            camera-model specific and reported if the value is rejected. Defaults to None
            (leave unchanged).

    Note:
        - Either name or serial_number must be specified.
        - At least one of `use_rgb` or `use_depth` must be enabled.
        - Depth stream configuration (if enabled) will use the same FPS as the color stream.
        - The actual resolution and FPS may be adjusted by the camera to the nearest supported mode.
        - For `fps`, `width` and `height`, either all of them need to be set, or none of them.
    """

    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    use_rgb: bool = True
    use_depth: bool = False
    depth_width: int | None = None
    depth_height: int | None = None
    # Decode/postprocess a depth frame only every Nth hardware frame the background read thread
    # sees (color stays at the full rate every frame). The RealSense SDK delivers a synced
    # color+depth frameset per hardware frame regardless of this setting, so it doesn't reduce
    # USB bandwidth — but decoding+postprocessing the depth frame (the actual CPU cost) is skipped
    # on the other frames, which is where the real per-frame cost of use_depth=True comes from.
    # 1 = every frame (highest depth temporal resolution, highest CPU cost).
    depth_frame_interval: int = 1
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    exposure: int | None = None
    gain: int | None = None
    white_balance: int | None = None

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of `use_rgb` or `use_depth` must be enabled.")

        if self.depth_frame_interval < 1:
            raise ValueError(f"`depth_frame_interval` must be >= 1, got {self.depth_frame_interval}.")

        depth_res_values = (self.depth_width, self.depth_height)
        if any(v is not None for v in depth_res_values) and any(v is None for v in depth_res_values):
            raise ValueError("`depth_width` and `depth_height` must either both be set, or both be None.")
        if self.depth_width is not None and not self.use_depth:
            raise ValueError("`depth_width`/`depth_height` require `use_depth=True`.")
        if self.depth_width is not None and self.width is None:
            raise ValueError("`depth_width`/`depth_height` require `width`/`height`/`fps` to also be set.")

        manual_color_options = {
            "exposure": self.exposure,
            "gain": self.gain,
            "white_balance": self.white_balance,
        }
        configured_color_options = [name for name, value in manual_color_options.items() if value is not None]
        if configured_color_options and not self.use_rgb:
            raise ValueError(
                "Manual color sensor options require `use_rgb=True`. "
                f"Configured options: {configured_color_options}."
            )

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
