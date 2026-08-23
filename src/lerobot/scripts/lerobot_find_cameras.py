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

"""
Helper to find the camera devices available in your system.

Example:

```shell
lerobot-find-cameras
```
"""

# NOTE(Steven): RealSense can also be identified/opened as OpenCV cameras. If you know the camera is a RealSense, use the `lerobot-find-cameras realsense` flag to avoid confusion.
# NOTE(Steven): macOS cameras sometimes report different FPS at init time, not an issue here as we don't specify FPS when opening the cameras, but the information displayed might not be truthful.

import argparse
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lerobot.cameras import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def _resolve_stable_video_path(dev_path: str) -> str | None:
    """Resolve a `/dev/videoN` path to its stable `/dev/v4l/by-path/...` symlink, if any.

    `/dev/videoN` indices can shift across reboots or when unrelated USB devices are
    plugged/unplugged; the by-path symlink instead tracks the physical USB port, so it's what
    should go into a robot's camera config (`index_or_path=...`) for a stable setup.
    """
    by_path_dir = Path("/dev/v4l/by-path")
    if not by_path_dir.is_dir():
        return None
    for symlink in by_path_dir.iterdir():
        try:
            if symlink.resolve() == Path(dev_path).resolve():
                return str(symlink)
        except OSError:
            continue
    return None


def _realsense_usb_prefixes() -> set[str]:
    """USB device prefixes (minus interface) that have a RealSense by-id symlink.

    The D435i RGB interface (`:1.3`) often has no `/dev/v4l/by-id/` entry — only the
    depth module (`:1.0`) does. Matching the shared USB prefix still tags RGB/IR/depth.
    Example prefix: `platform-3610000.usb-usb-0:1.1`
    """
    prefixes: set[str] = set()
    by_id_dir = Path("/dev/v4l/by-id")
    by_path_dir = Path("/dev/v4l/by-path")
    if not by_id_dir.is_dir() or not by_path_dir.is_dir():
        return prefixes
    realsense_devs: set[Path] = set()
    for symlink in by_id_dir.iterdir():
        if "realsense" not in symlink.name.lower():
            continue
        try:
            realsense_devs.add(symlink.resolve())
        except OSError:
            continue
    if not realsense_devs:
        return prefixes
    for symlink in by_path_dir.iterdir():
        try:
            if symlink.resolve() not in realsense_devs:
                continue
        except OSError:
            continue
        name = symlink.name
        if "-video-index" not in name:
            continue
        device_and_iface = name.split("-video-index", 1)[0]
        prefixes.add(device_and_iface.rsplit(":", 1)[0])
    return prefixes


def _annotate_linux_opencv_camera(cam_info: dict[str, Any]) -> None:
    """Tag RealSense V4L2 nodes so they aren't mistaken for wrist webcams.

    A D435i enumerates several `/dev/video*` devices (Z16 depth, GREY IR, YUYV RGB, metadata).
    OpenCV can open the RGB and IR nodes, but RGB+depth must go through `type: intelrealsense`
    with the SDK serial from `lerobot-find-cameras realsense`.
    """
    try:
        resolved = Path(str(cam_info["id"])).resolve()
    except OSError:
        return

    by_id_dir = Path("/dev/v4l/by-id")
    if by_id_dir.is_dir():
        for symlink in by_id_dir.iterdir():
            try:
                if symlink.resolve() == resolved:
                    cam_info["stable_by_id"] = str(symlink)
                    break
            except OSError:
                continue

    by_path = str(cam_info.get("stable_by_path_id") or "")
    by_path_name = Path(by_path).name if by_path else ""
    if not any(by_path_name.startswith(prefix) for prefix in _realsense_usb_prefixes()):
        return

    if "video-index0" in by_path and ":1.0" in by_path:
        cam_info["note"] = (
            "Intel RealSense depth (Z16). Do not use as an OpenCV wrist camera. "
            "For RGB+depth: `lerobot-find-cameras realsense` then type=intelrealsense."
        )
    elif "video-index2" in by_path and ":1.0" in by_path:
        cam_info["note"] = (
            "Intel RealSense IR (GREY). Do not use as an OpenCV wrist/head RGB camera."
        )
    elif ":1.3" in by_path and "video-index0" in by_path:
        cam_info["note"] = (
            "Intel RealSense RGB (OpenCV fallback only). Prefer type=intelrealsense "
            "with use_depth=true for RGB+depth."
        )
    else:
        cam_info["note"] = (
            "Intel RealSense V4L2 node (metadata/aux). Not a wrist camera; "
            "use `lerobot-find-cameras realsense` for the SDK serial."
        )


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    """
    Finds all available OpenCV cameras plugged into the system.

    Returns:
        A list of all available OpenCV cameras with their metadata.
    """
    all_opencv_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for OpenCV cameras...")
    try:
        opencv_cameras = OpenCVCamera.find_cameras()
        for cam_info in opencv_cameras:
            if platform.system() == "Linux":
                stable_path = _resolve_stable_video_path(str(cam_info["id"]))
                if stable_path:
                    cam_info["stable_by_path_id"] = stable_path
                _annotate_linux_opencv_camera(cam_info)
            all_opencv_cameras_info.append(cam_info)
        logger.info(f"Found {len(opencv_cameras)} OpenCV cameras.")
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")

    return all_opencv_cameras_info


def find_all_realsense_cameras() -> list[dict[str, Any]]:
    """
    Finds all available RealSense cameras plugged into the system.

    Returns:
        A list of all available RealSense cameras with their metadata.
    """
    all_realsense_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for RealSense cameras...")
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        for cam_info in realsense_cameras:
            all_realsense_cameras_info.append(cam_info)
        logger.info(f"Found {len(realsense_cameras)} RealSense cameras.")
    except ImportError:
        logger.warning("Skipping RealSense camera search: pyrealsense2 library not found or not importable.")
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")

    return all_realsense_cameras_info


def find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, Any]]:
    """
    Finds available cameras based on an optional filter and prints their information.

    Args:
        camera_type_filter: Optional string to filter cameras ("realsense" or "opencv").
                            If None, lists all cameras.

    Returns:
        A list of all available cameras matching the filter, with their metadata.
    """
    all_cameras_info: list[dict[str, Any]] = []

    if camera_type_filter:
        camera_type_filter = camera_type_filter.lower()

    if camera_type_filter is None or camera_type_filter == "opencv":
        all_cameras_info.extend(find_all_opencv_cameras())
    if camera_type_filter is None or camera_type_filter == "realsense":
        all_cameras_info.extend(find_all_realsense_cameras())

    if not all_cameras_info:
        if camera_type_filter:
            logger.warning(f"No {camera_type_filter} cameras were detected.")
        else:
            logger.warning("No cameras (OpenCV or RealSense) were detected.")
    else:
        print("\n--- Detected Cameras ---")
        for i, cam_info in enumerate(all_cameras_info):
            print(f"Camera #{i}:")
            for key, value in cam_info.items():
                if key == "default_stream_profile" and isinstance(value, dict):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                else:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    return all_cameras_info


def save_image(
    img_array: np.ndarray,
    camera_identifier: str | int,
    images_dir: Path,
    camera_type: str,
) -> None:
    """
    Saves a single image to disk using Pillow. Handles color conversion if necessary.
    """
    try:
        img = Image.fromarray(img_array, mode="RGB")

        safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")
        filename_prefix = f"{camera_type.lower()}_{safe_identifier}"
        filename = f"{filename_prefix}.png"

        path = images_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        logger.info(f"Saved image: {path}")
    except Exception as e:
        logger.error(f"Failed to save image for camera {camera_identifier} (type {camera_type}): {e}")


def create_camera_instance(cam_meta: dict[str, Any], *, warmup_s: int = 1) -> dict[str, Any] | None:
    """Create and connect to a camera instance based on metadata."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance = None

    logger.info(f"Preparing {cam_type} ID {cam_id} with default profile")

    try:
        if cam_type == "OpenCV":
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
                warmup_s=warmup_s,
            )
            instance = OpenCVCamera(cv_config)
        elif cam_type == "RealSense":
            rs_config = RealSenseCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.RGB,
                warmup_s=warmup_s,
            )
            instance = RealSenseCamera(rs_config)
        else:
            logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
            return None

        if instance:
            logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
            instance.connect(warmup=True)
            return {"instance": instance, "meta": cam_meta}
    except Exception as e:
        logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
        if instance and instance.is_connected:
            instance.disconnect()
        return None


def process_camera_image(cam_dict: dict[str, Any], output_dir: Path, current_time: float) -> None:
    """Capture and process an image from a single camera."""
    cam = cam_dict["instance"]
    meta = cam_dict["meta"]
    cam_type_str = str(meta.get("type", "unknown"))
    cam_id_str = str(meta.get("id", "unknown"))

    try:
        image_data = cam.read()

        save_image(
            image_data,
            cam_id_str,
            output_dir,
            cam_type_str,
        )
    except TimeoutError:
        logger.warning(
            f"Timeout reading from {cam_type_str} camera {cam_id_str} at time {current_time:.2f}s."
        )
    except Exception as e:
        logger.error(f"Error reading from {cam_type_str} camera {cam_id_str}: {e}")
    return None


def cleanup_camera(cam_dict: dict[str, Any]) -> None:
    """Disconnect all cameras."""
    logger.info(f"Disconnecting camera with ID {cam_dict['meta'].get('id')}...")
    try:
        if cam_dict["instance"] and cam_dict["instance"].is_connected:
            cam_dict["instance"].disconnect()
    except Exception as e:
        logger.error(f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}")


def save_images_from_all_cameras(
    output_dir: Path,
    record_time_s: float = 2.0,
    camera_type: str | None = None,
    warmup_s: int = 1,
):
    """
    Connects to detected cameras (optionally filtered by type) and saves images from each.
    Uses default stream profiles for width, height, and FPS.

    Args:
        output_dir: Directory to save images.
        record_time_s: Duration in seconds to record images.
        camera_type: Optional string to filter cameras ("realsense" or "opencv").
                            If None, uses all detected cameras.
        warmup_s: Duration in seconds to warmup camera before recording images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving images to {output_dir}")
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)

    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot save images.")
        return

    logger.info(
        f"Starting image capture for {record_time_s} seconds from {len(all_camera_metadata)} cameras."
    )

    try:
        for cam_meta in all_camera_metadata:
            cam_dict = create_camera_instance(cam_meta, warmup_s=warmup_s)
            if cam_dict is None:
                continue
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < record_time_s:
                current_capture_time = time.perf_counter()
                process_camera_image(cam_dict, output_dir, current_capture_time)
            cleanup_camera(cam_dict)
    except KeyboardInterrupt:
        logger.info("Capture interrupted by user.")
    finally:
        print(f"Image capture finished. Images saved to {output_dir}")


def main():
    init_logging()

    parser = argparse.ArgumentParser(
        description="Unified camera utility script for listing cameras and capturing images."
    )
    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Specify camera type to capture from (e.g., 'realsense', 'opencv'). Captures from all if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="Directory to save images. Default: outputs/captured_images",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Time duration to attempt capturing frames. Default: 2 seconds.",
    )
    parser.add_argument(
        "--warmup-s",
        type=int,
        default=1,
        help="Time duration to warmup camera before attempting to capture frames. Default: 1 second.",
    )
    args = parser.parse_args()
    save_images_from_all_cameras(**vars(args))


if __name__ == "__main__":
    main()
