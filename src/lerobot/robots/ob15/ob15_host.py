#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
OB15 host daemon — runs on the Jetson.

Wraps XLerobotHost with ob15-specific defaults (no head, 15D action space).

Usage:
    conda activate lerobot
    python -m lerobot.robots.ob15.ob15_host

    # With a custom robot profile JSON:
    python -m lerobot.robots.ob15.ob15_host --robot-config /path/to/ob15.json

IMPORTANT — bus conflict with ROS2:
    bus1 (ttyACM1) is shared with sts3215_control (ROS2 Nav2 wheel driver).
    Stop the ROS2 navigation stack before starting this daemon:

        ros2 launch bob_1 bringup.launch.py use_safety:=false  # then Ctrl-C
        # or specifically stop sts3215_node:
        ros2 lifecycle set /sts3215_node shutdown

    Then start this daemon. When done, restart navigation:
        ros2 launch bob_1 bringup.launch.py
"""

from __future__ import annotations

import argparse
import logging

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.xlerobot.xlerobot import XLerobot
from lerobot.robots.xlerobot.xlerobot_host import XLerobotHost, _observation_to_zmq_payload

from .config_ob15 import OB15Config, OB15HostConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="OB15 ZMQ host daemon (Jetson side)")
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help="Path to robot config JSON. Default: OB15Config() built-in defaults.",
    )
    parser.add_argument(
        "--zmq-cmd-port", type=int, default=5555, help="ZMQ port for receiving commands"
    )
    parser.add_argument(
        "--zmq-obs-port", type=int, default=5556, help="ZMQ port for sending observations"
    )
    parser.add_argument(
        "--watchdog-ms", type=int, default=500,
        help="Stop base if no command received for this many ms"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Control loop frequency (Hz)"
    )
    parser.add_argument("--id", type=str, default="ob15_host", help="Robot id string")
    args = parser.parse_args()

    if args.robot_config:
        import json
        import draccus
        from lerobot.robots.config import RobotConfig
        with open(args.robot_config) as f:
            config_dict = json.load(f)
        robot_cfg = draccus.decode(RobotConfig, config_dict)
        robot = make_robot_from_config(robot_cfg)
    else:
        robot = XLerobot(OB15Config(id=args.id))

    host_config = OB15HostConfig(
        port_zmq_cmd=args.zmq_cmd_port,
        port_zmq_observations=args.zmq_obs_port,
        watchdog_timeout_ms=args.watchdog_ms,
        max_loop_freq_hz=args.fps,
    )

    host = XLerobotHost(host_config)

    import time
    import zmq

    robot.connect()
    cam_keys = tuple(robot.config.cameras.keys())
    logging.info(
        "OB15 host ready. action_dim=%d  cameras=%s  ZMQ cmd=%d obs=%d",
        len(robot.action_features),
        cam_keys,
        host_config.port_zmq_cmd,
        host_config.port_zmq_observations,
    )
    logging.info("Waiting for client on %s ...", robot.config.port1)

    last_cmd_time = time.time()
    watchdog_active = False

    try:
        while True:
            loop_start = time.perf_counter()

            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                import json as _json
                data = dict(_json.loads(msg))
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                pass
            except Exception as e:
                logging.error("Command handling failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host_config.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    "No command for > %d ms — stopping base (watchdog).",
                    host_config.watchdog_timeout_ms,
                )
                watchdog_active = True
                robot.stop_base()

            try:
                raw_obs = robot.get_observation()
                payload = _observation_to_zmq_payload(
                    raw_obs, cam_keys, host_config.jpeg_quality
                )
                host.zmq_observation_socket.send_string(
                    __import__("json").dumps(payload), flags=zmq.NOBLOCK
                )
            except zmq.Again:
                logging.debug("Dropping observation (no client)")
            except Exception as e:
                logging.error("Observation send failed: %s", e)

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(1 / host_config.max_loop_freq_hz - elapsed, 0))

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt — exiting")
    finally:
        robot.disconnect()
        host.disconnect()
        logging.info("OB15 host shut down")


if __name__ == "__main__":
    main()
