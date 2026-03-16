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

import contextlib
import numbers
import os
import queue
import sys
import threading

import cv2
import numpy as np
import rerun as rr

from lerobot.types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR


# ---------------------------------------------------------------------------
# Background visualization thread
# ---------------------------------------------------------------------------
# The main control loop drops data into this queue and immediately continues.
# A single daemon thread drains the queue and calls the actual rerun logging.
# Queue maxsize=3: buffers a few frames so the viewer looks smoother when the
# worker briefly falls behind (JPEG encoding 3 cams on Jetson can take >33ms).
# Frames are dropped when full — visualization never blocks the control loop.
#
# Why Rerun FPS can be slow:
# - This worker runs per frame: downsample, JPEG encode (3 cams), rr.log. On Jetson that can be >33ms.
# - Tune with: RERUN_DOWNSAMPLE_FACTOR=0.2, RERUN_JPEG_QUALITY=55, RERUN_LOG_FREQUENCY=2
# - RERUN_LOG_FREQUENCY=2 logs every other frame, halving viz CPU load while keeping it smooth.

_viz_queue: queue.Queue = queue.Queue(maxsize=3)
_viz_thread: threading.Thread | None = None


def _viz_worker() -> None:
    """Background thread that drains the visualization queue."""
    while True:
        item = _viz_queue.get()
        if item is None:  # sentinel → shut down
            break
        obs, action, compress = item
        try:
            _log_rerun_data_sync(obs, action, compress)
        except Exception:
            pass  # never crash the recording


def start_viz_thread() -> None:
    """Start the background visualization thread (idempotent)."""
    global _viz_thread
    if _viz_thread is not None and _viz_thread.is_alive():
        return
    _viz_thread = threading.Thread(target=_viz_worker, name="viz_worker", daemon=True)
    _viz_thread.start()


def stop_viz_thread() -> None:
    """Send the sentinel and wait for the worker to exit."""
    global _viz_thread
    if _viz_thread is not None and _viz_thread.is_alive():
        try:
            _viz_queue.put_nowait(None)
        except queue.Full:
            pass
        _viz_thread.join(timeout=2.0)
    _viz_thread = None


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def init_rerun(
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
    headless: bool = True,
    grpc_port: int = 9876,
    web_port: int = 9090,
    open_browser: bool = False,
    server_memory_limit: str = "200MB",
) -> None:
    """Initializes the Rerun SDK for visualizing the control loop.
    
    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server (upstream compatibility).
        port: Optional port for connecting to a Rerun server (upstream compatibility).
        headless: If True, run in headless mode with gRPC server (default).
                  If False, spawn a local GUI viewer.
                  Can be overridden by RERUN_HEADLESS env var ("true"/"false").
                  Ignored if ip and port are provided.
        grpc_port: Port for gRPC server (default 9876).
        web_port: Port for web viewer (default 9090) - DEPRECATED, not used anymore.
        open_browser: Whether to attempt opening browser (default False for headless).
        server_memory_limit: Server-side buffer for late viewers (default "25%").
    
    Notes:
        If ip and port are provided, uses upstream's connection logic.
        Otherwise, uses advanced headless mode: only the gRPC server is started on the Jetson.
        To view data, run the web viewer on your external computer (with GPU):
            rerun --serve-web --web-viewer-port 9090 --connect "rerun+http://JETSON_IP:9876/proxy"
        Then open http://localhost:9090 on your external computer's browser.

        Live / low-latency streaming:
        - RERUN_FLUSH_TICK_SECS (default 0.008): flush interval in seconds. 0.008 = 8ms (~1 frame at 120Hz).
          Set to 0.002 for minimal latency, or 0.033 for one frame at 30Hz.
        - RERUN_FLUSH_NUM_BYTES (default 64000): flush when this many bytes are buffered. Lower = more
          frequent flushes and lower latency, higher = fewer network round-trips.
        - RERUN_LOG_FREQUENCY=1 (default): log every frame; >1 skips frames and increases latency.
    """
    # Low-latency flush: send data to the viewer frequently so streaming feels live.
    # Must be set before rr.init(). Rerun's default is 200ms; we use 8ms for streaming.
    flush_tick = os.getenv("RERUN_FLUSH_TICK_SECS", "0.008")
    os.environ["RERUN_FLUSH_TICK_SECS"] = flush_tick

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "16000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

    rr.init(session_name)
    
    # Upstream compatibility: if ip and port are provided, use upstream's logic
    if ip and port:
        memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
        rr.spawn(memory_limit=memory_limit)
        return

    # User's advanced headless mode logic
    # Check if headless mode is overridden by environment variable
    headless_env = os.getenv("RERUN_HEADLESS")
    if headless_env is not None:
        headless = headless_env.lower() in ("true", "1", "yes")
    
    if headless:
        # Start ONLY gRPC server on Jetson (headless logging endpoint)
        # The web viewer should be run separately on external GPU-capable machine
        # Suppress output messages from Rerun server startup
        with suppress_output():
            server_uri = rr.serve_grpc(grpc_port=grpc_port, server_memory_limit=server_memory_limit)
        # Note: server_uri can be printed if debugging: print(f"gRPC server: {server_uri}")
    else:
        # Fallback to spawn a local viewer (for dev with GUI)
        memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
        rr.spawn(memory_limit=memory_limit)


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def _downsample_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Downsample an image for visualization bandwidth reduction.
    
    Args:
        image: Input image (HWC, CHW, or 2D format for depth images)
        scale_factor: Scaling factor (e.g., 0.5 for half size)
    
    Returns:
        Downsampled image in the same format as input
    """
    if scale_factor >= 1.0:
        return image
    
    # Check if CHW format (channels first)
    is_chw = image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4)
    
    if is_chw:
        # Convert CHW to HWC for cv2.resize
        image = np.transpose(image, (1, 2, 0))
    
    # Calculate new dimensions
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    
    # Downsample using cv2 (fast and high quality)
    downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    if is_chw:
        # Convert back to CHW
        downsampled = np.transpose(downsampled, (2, 0, 1))
    
    return downsampled


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """Non-blocking async rerun logging.

    Puts data into a queue consumed by the background viz thread so the main
    control loop is never blocked by image compression or rerun logging.
    Queue size of 5 provides ~150ms buffer. If the worker falls behind,
    new frames are silently dropped rather than blocking the control loop.
    
    Optimization: Check queue space before expensive array copying to avoid
    wasted work when the queue is full.
    """
    # Check if queue has space before doing expensive array copies
    if _viz_queue.full():
        return  # Skip this frame - worker is falling behind
    
    # Deep-copy numpy arrays so the main loop can reuse its buffers
    # Only do this if we know the queue has space
    obs_copy = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in (observation or {}).items()}
    act_copy = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in (action or {}).items()}
    
    try:
        _viz_queue.put_nowait((obs_copy, act_copy, compress_images))
    except queue.Full:
        pass  # drop frame – visualization is not worth blocking the robot


def _log_rerun_data_sync(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """Synchronous implementation (runs in the background viz thread)."""
    # Get configuration from environment
    # Default 0.5: readable for teleop, lightweight. Use downsample OR cv2 JPEG, not both heavy steps.
    downsample_factor = float(os.getenv("RERUN_DOWNSAMPLE_FACTOR", "0.3"))
    skip_depth = os.getenv("RERUN_SKIP_DEPTH", "true").lower() in ("true", "1", "yes")
    # int(float(...)) avoids ValueError if user sets e.g. RERUN_LOG_FREQUENCY=0.25. Default 1 = every frame.
    log_frequency = int(float(os.getenv("RERUN_LOG_FREQUENCY", "1")))

    # Frame counter for logging frequency
    if not hasattr(_log_rerun_data_sync, "_frame_counter"):
        _log_rerun_data_sync._frame_counter = 0
    _log_rerun_data_sync._frame_counter += 1

    # Skip this frame if logging frequency > 1
    if log_frequency > 1 and _log_rerun_data_sync._frame_counter % log_frequency != 0:
        return

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            
            # Skip depth in viz when RERUN_SKIP_DEPTH=true (default). No (skip_depth or True) so env is respected.
            if ("_depth" in str(k).lower() or k.endswith("_depth")) and skip_depth:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Check if this is an image (3D array or 2D array with reasonable dimensions)
                is_image = arr.ndim == 3 or (arr.ndim == 2 and arr.shape[0] > 10 and arr.shape[1] > 10)
                
                # Check if this is a depth image (uint16) - JPEG compression doesn't support uint16
                is_depth = arr.dtype == np.uint16 and ("depth" in str(k).lower() or k.endswith("_depth"))
                
                # Log depth image detection for debugging (only once per key)
                if is_depth and not hasattr(_log_rerun_data_sync, f"_depth_logged_{k}"):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"📊 Depth image detected: {k}, shape={arr.shape}, dtype={arr.dtype}")
                    setattr(_log_rerun_data_sync, f"_depth_logged_{k}", True)
                
                # Downsample only when factor < 1 (avoid double-encoding: downsample then one encode step)
                if is_image and downsample_factor < 1.0:
                    depth_factor = downsample_factor * 0.5 if is_depth else downsample_factor
                    arr = _downsample_image(arr, depth_factor)

                # Convert CHW -> HWC when needed (only for 3D arrays)
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))

                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                elif is_image and not is_depth:
                    # Always compress with cv2 JPEG — raw frames (~691KB each) saturate WiFi during
                    # fast motion. Unconditional so it works even without --display_ip/--display_port.
                    if arr.shape[-1] == 3:
                        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    else:
                        bgr = arr
                    quality = int(os.getenv("RERUN_JPEG_QUALITY", "60"))
                    success, encoded = cv2.imencode(
                        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
                    )
                    if success:
                        rr.log(
                            key,
                            rr.ImageEncoded(
                                contents=bytes(encoded),
                                format=rr.ImageFormat.JPEG,
                            ),
                            static=False,
                        )
                    else:
                        rr.log(key, rr.Image(arr), static=False)
                else:
                    rr.log(key, rr.Image(arr), static=False)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))