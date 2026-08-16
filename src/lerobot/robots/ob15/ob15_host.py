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

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import draccus
import numpy as np
import zmq

from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config

from .config_ob15 import OB15Config, OB15HostConfig
from .ob15 import OB15


def _observation_to_zmq_payload(
    obs: dict[str, Any],
    camera_keys: tuple[str, ...],
    depth_keys: tuple[str, ...],
    jpeg_quality: int,
) -> dict[str, Any]:
    """Scalars JSON-safe; RGB cameras → JPEG base64; depth cameras → 16-bit PNG base64."""
    out: dict[str, Any] = {}
    cam_set = set(camera_keys)
    depth_set = set(depth_keys)
    for k, v in obs.items():
        if k in cam_set:
            if not isinstance(v, np.ndarray) or v.size == 0:
                out[k] = ""
                continue
            ret, buffer = cv2.imencode(".jpg", v, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            out[k] = base64.b64encode(buffer).decode("utf-8") if ret else ""
            continue
        if k in depth_set:
            if not isinstance(v, np.ndarray) or v.size == 0:
                out[k] = ""
                continue
            # Depth is uint16 millimeters; PNG is the lossless codec that preserves that range.
            depth_2d = v[..., 0] if v.ndim == 3 else v
            ret, buffer = cv2.imencode(".png", depth_2d, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            out[k] = base64.b64encode(buffer).decode("utf-8") if ret else ""
            continue
        if isinstance(v, np.ndarray):
            if v.size == 1:
                out[k] = float(np.asarray(v).reshape(-1)[0])
            continue
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, (float, int, str, bool)) or v is None:
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


class OB15Host:
    def __init__(self, config: OB15HostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz
        self.jpeg_quality = config.jpeg_quality

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="OB15 ZMQ host (robot USB on this machine).")
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help="JSON profile (type=ob15). Default: OB15Config() defaults.",
    )
    parser.add_argument("--port-cmd", type=int, default=5555)
    parser.add_argument("--port-observations", type=int, default=5556)
    parser.add_argument("--connection-time-s", type=int, default=3600)
    parser.add_argument("--watchdog-timeout-ms", type=int, default=500)
    parser.add_argument("--max-loop-freq-hz", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    import lerobot.cameras.opencv.configuration_opencv  # noqa: F401
    import lerobot.cameras.realsense.configuration_realsense  # noqa: F401
    import lerobot.robots.ob15  # noqa: F401

    if args.robot_config:
        path = Path(args.robot_config).expanduser()
        with path.open() as f:
            profile = json.load(f)
        if "fields" in profile:
            config_dict = {"type": profile["type"]}
            for k, v in profile["fields"].items():
                config_dict[k] = v
            if "cameras" in profile:
                config_dict["cameras"] = profile["cameras"]
        else:
            config_dict = profile
        robot_cfg = draccus.decode(RobotConfig, config_dict)
        robot = make_robot_from_config(robot_cfg)
        if not isinstance(robot, OB15):
            raise ValueError("--robot-config must describe type=ob15 for this host.")
    else:
        robot = OB15(OB15Config(id="ob15_zmq_host"))

    host_config = OB15HostConfig(
        port_zmq_cmd=args.port_cmd,
        port_zmq_observations=args.port_observations,
        connection_time_s=args.connection_time_s,
        watchdog_timeout_ms=args.watchdog_timeout_ms,
        max_loop_freq_hz=args.max_loop_freq_hz,
        jpeg_quality=args.jpeg_quality,
    )
    host = OB15Host(host_config)
    cam_keys = tuple(robot.cameras.keys())
    depth_keys = tuple(
        f"{name}_depth" for name, cfg in robot.config.cameras.items() if getattr(cfg, "use_depth", False)
    )

    logging.info("Connecting OB15 (local USB)")
    robot.connect()

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info(
        "ZMQ host: cmd PULL *:%d, obs PUSH *:%d — point ob15_client.remote_ip here.",
        host_config.port_zmq_cmd,
        host_config.port_zmq_observations,
    )
    try:
        start = time.perf_counter()
        duration = 0.0
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                pass
            except Exception as e:
                logging.error("Command handling failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    "No command for > %d ms — stopping base (watchdog).",
                    host.watchdog_timeout_ms,
                )
                watchdog_active = True
                robot.stop_base()

            try:
                raw_obs = robot.get_observation()
                payload = _observation_to_zmq_payload(raw_obs, cam_keys, depth_keys, host.jpeg_quality)
                host.zmq_observation_socket.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.debug("Dropping observation (no client)")
            except Exception as e:
                logging.error("Observation send failed: %s", e)

            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt — exiting")
    finally:
        robot.disconnect()
        host.disconnect()
        logging.info("OB15 host shut down")


if __name__ == "__main__":
    main()
