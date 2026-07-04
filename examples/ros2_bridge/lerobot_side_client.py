#!/usr/bin/env python3
"""
ROS 2 ↔ LeRobot ZMQ bridge — the LeRobot side. Runs in the Python 3.12 venv.

Counterpart of ros_side_bridge.py (which owns all rclpy code on the system
Python 3.10). This side never imports ROS: it pulls JSON observations over
ZMQ and pushes JSON commands back, so it works from any interpreter.

IMPORTANT: run this from a shell that has NOT sourced /opt/ros/*/setup.bash —
sourcing exports a Python-3.10 PYTHONPATH that shadows the venv's packages
(see ROS2_INTEGRATION.md).

Demo loop (prints observation rate, sends a gentle sine twist):

    uv run python examples/ros2_bridge/lerobot_side_client.py --demo-twist

Or import the client from your own policy loop:

    from examples.ros2_bridge.lerobot_side_client import ROS2BridgeClient
    client = ROS2BridgeClient("localhost")
    obs = client.get_observation()          # joints + {name: HxWx3 RGB uint8}
    client.send_action(joint_positions=[...], twist=[vx, vy, wz])
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import time

import numpy as np


def decode_observation(raw: str) -> dict:
    """Bridge JSON → observation dict with numpy image arrays (RGB, HWC uint8)."""
    import cv2

    payload = json.loads(raw)
    images = {}
    for name, b64 in (payload.get("images") or {}).items():
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is not None:
            images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return {
        "stamp": payload.get("stamp"),
        "joint_names": payload.get("joint_names") or [],
        "joint_positions": np.asarray(payload.get("joint_positions") or [], dtype=np.float32),
        "joint_velocities": np.asarray(payload.get("joint_velocities") or [], dtype=np.float32),
        "images": images,
    }


class ROS2BridgeClient:
    """Latest-value observation/command link to ros_side_bridge.py."""

    def __init__(self, host: str = "localhost", obs_port: int = 5566, cmd_port: int = 5565):
        import zmq

        self._zmq = zmq
        ctx = zmq.Context()
        self.obs_sock = ctx.socket(zmq.PULL)
        self.obs_sock.setsockopt(zmq.CONFLATE, 1)
        self.obs_sock.connect(f"tcp://{host}:{obs_port}")
        self.cmd_sock = ctx.socket(zmq.PUSH)
        self.cmd_sock.setsockopt(zmq.CONFLATE, 1)
        self.cmd_sock.connect(f"tcp://{host}:{cmd_port}")

    def get_observation(self, timeout_s: float = 1.0) -> dict | None:
        """Newest observation, or None if the bridge sent nothing in time."""
        if self.obs_sock.poll(timeout=int(timeout_s * 1000)):
            return decode_observation(self.obs_sock.recv_string())
        return None

    def send_action(self, joint_positions=None, twist=None) -> None:
        cmd: dict = {}
        if joint_positions is not None:
            cmd["joint_positions"] = [float(v) for v in joint_positions]
        if twist is not None:
            cmd["twist"] = [float(v) for v in twist]
        if cmd:
            try:
                self.cmd_sock.send_string(json.dumps(cmd), flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                pass  # bridge not up yet; CONFLATE keeps only the newest anyway


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--obs-port", type=int, default=5566)
    parser.add_argument("--cmd-port", type=int, default=5565)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Seconds to run (0 = forever).")
    parser.add_argument("--demo-twist", action="store_true",
                        help="Send a small sine wave on twist.x to prove the "
                             "command path (watch: ros2 topic echo /cmd_vel).")
    args = parser.parse_args()

    client = ROS2BridgeClient(args.host, args.obs_port, args.cmd_port)
    print(f"listening tcp://{args.host}:{args.obs_port} … (Ctrl+C to stop)")
    n, t0, start = 0, time.monotonic(), time.monotonic()
    try:
        while True:
            obs = client.get_observation(timeout_s=1.0)
            if obs is not None:
                n += 1
                if args.demo_twist:
                    client.send_action(twist=[0.05 * math.sin(time.monotonic()), 0.0, 0.0])
            now = time.monotonic()
            if now - t0 >= 2.0:
                if obs is None:
                    print("no observations — is ros_side_bridge.py running?")
                else:
                    cams = {k: v.shape for k, v in obs["images"].items()}
                    print(f"{n / (now - t0):5.1f} Hz | joints={len(obs['joint_positions'])} "
                          f"{list(obs['joint_names'][:3])}… | images={cams or '{}'}")
                n, t0 = 0, now
            if args.duration and now - start >= args.duration:
                break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
