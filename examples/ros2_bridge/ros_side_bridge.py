#!/usr/bin/env python3
"""
ROS 2 ↔ LeRobot ZMQ bridge — the ROS side. RUNS ON THE SYSTEM PYTHON 3.10.

ROS 2 Humble ships rclpy compiled for Ubuntu 22.04's Python 3.10; LeRobot's
venv is Python 3.12, so `import rclpy` there is impossible. This node is the
seam between the two worlds: it lives entirely in the 3.10 interpreter,
subscribes to ROS topics, and forwards them to the LeRobot process over ZMQ
(and commands back), exactly like the xlerobot host/client split in
`src/lerobot/robots/xlerobot/`.

Run (terminal A — the ONLY place ROS is sourced):

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 examples/ros2_bridge/ros_side_bridge.py \
        --joint-states-topic /joint_states \
        --image head=/camera/color/image_raw \
        --cmd-vel-topic /cmd_vel

Counterpart (terminal B, NO ROS sourcing): lerobot_side_client.py.

Wire protocol (single-part JSON so ZMQ CONFLATE keeps only the newest):
    observations  PUSH :5566 → {"stamp", "joint_names", "joint_positions",
                                "joint_velocities", "images": {name: b64 jpeg}}
    commands      PULL :5565 ← {"joint_positions": [...]} and/or
                               {"twist": [vx, vy, wz]}

Dependencies on the SYSTEM python: rclpy (from Humble), pyzmq
(`/usr/bin/python3 -m pip install --user pyzmq`), python3-opencv (apt).
"""

import argparse
import base64
import json
import time


def image_msg_to_bgr(msg):
    """sensor_msgs/Image → BGR numpy, without cv_bridge (also 3.10-pinned)."""
    import numpy as np

    channels = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(msg.encoding)
    if channels is None:
        return None  # depth / exotic encodings: extend here if needed
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    arr = arr.reshape(msg.height, msg.step // channels // 1, channels)[
        :, : msg.width, :
    ] if channels > 1 else arr.reshape(msg.height, msg.step)[:, : msg.width]
    if msg.encoding == "rgb8":
        import cv2

        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-port", type=int, default=5566)
    parser.add_argument("--cmd-port", type=int, default=5565)
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--image", action="append", default=[],
                        metavar="NAME=TOPIC",
                        help="Camera to forward, e.g. head=/camera/color/image_raw "
                             "(repeatable).")
    parser.add_argument("--cmd-joints-topic", default="",
                        help="Publish received joint_positions here as "
                             "std_msgs/Float64MultiArray (empty = disabled).")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel",
                        help="Publish received twist here as geometry_msgs/Twist "
                             "(empty = disabled).")
    parser.add_argument("--rate", type=float, default=30.0,
                        help="Observation publish rate toward LeRobot (Hz).")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()

    # Imports live here so the module can be read/compiled anywhere; running
    # requires the sourced Humble environment.
    import cv2
    import rclpy
    import zmq
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64MultiArray

    class LeRobotBridge(Node):
        def __init__(self):
            super().__init__("lerobot_zmq_bridge")
            ctx = zmq.Context()
            # Same socket pattern as xlerobot_host: latest-value semantics.
            self.obs_sock = ctx.socket(zmq.PUSH)
            self.obs_sock.setsockopt(zmq.CONFLATE, 1)
            self.obs_sock.bind(f"tcp://*:{args.obs_port}")
            self.cmd_sock = ctx.socket(zmq.PULL)
            self.cmd_sock.setsockopt(zmq.CONFLATE, 1)
            self.cmd_sock.bind(f"tcp://*:{args.cmd_port}")

            self._joint_state = None
            self._images = {}
            self.create_subscription(
                JointState, args.joint_states_topic, self._on_joints, 10
            )
            for spec in args.image:
                name, _, topic = spec.partition("=")
                if not topic:
                    raise SystemExit(f"--image expects NAME=TOPIC, got {spec!r}")
                self.create_subscription(
                    Image, topic,
                    lambda msg, n=name: self._images.__setitem__(n, msg),
                    qos_profile_sensor_data,  # sensor topics are BEST_EFFORT
                )
            self.joint_pub = (
                self.create_publisher(Float64MultiArray, args.cmd_joints_topic, 10)
                if args.cmd_joints_topic else None
            )
            self.twist_pub = (
                self.create_publisher(Twist, args.cmd_vel_topic, 10)
                if args.cmd_vel_topic else None
            )
            self.create_timer(1.0 / args.rate, self._tick)
            self._sent = 0
            self._last_report = time.monotonic()
            self.get_logger().info(
                f"obs → tcp:{args.obs_port}  cmd ← tcp:{args.cmd_port}  "
                f"cams={[s.split('=')[0] for s in args.image]}"
            )

        def _on_joints(self, msg):
            self._joint_state = msg

        def _tick(self):
            # 1) forward the freshest observation
            payload = {"stamp": time.time(), "images": {}}
            if self._joint_state is not None:
                js = self._joint_state
                payload["joint_names"] = list(js.name)
                payload["joint_positions"] = list(js.position)
                payload["joint_velocities"] = list(js.velocity)
            for name, msg in self._images.items():
                bgr = image_msg_to_bgr(msg)
                if bgr is None:
                    continue
                ok, jpg = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                )
                if ok:
                    payload["images"][name] = base64.b64encode(jpg).decode("ascii")
            try:
                self.obs_sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
                self._sent += 1
            except zmq.Again:
                pass  # nobody connected yet

            # 2) apply the freshest command, if any
            try:
                cmd = json.loads(self.cmd_sock.recv_string(flags=zmq.NOBLOCK))
            except zmq.Again:
                cmd = None
            if cmd:
                if self.joint_pub and cmd.get("joint_positions") is not None:
                    m = Float64MultiArray()
                    m.data = [float(v) for v in cmd["joint_positions"]]
                    self.joint_pub.publish(m)
                if self.twist_pub and cmd.get("twist") is not None:
                    vx, vy, wz = (list(cmd["twist"]) + [0.0, 0.0, 0.0])[:3]
                    t = Twist()
                    t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
                    self.twist_pub.publish(t)

            now = time.monotonic()
            if now - self._last_report >= 5.0:
                self.get_logger().info(f"obs sent: {self._sent} (last 5 s)")
                self._sent, self._last_report = 0, now

    rclpy.init()
    node = LeRobotBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
