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

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_ob15 import OB15Config
from .lift_axis import LiftAxis

logger = logging.getLogger(__name__)


class OB15(Robot):
    """
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = OB15Config
    name = "ob15"

    def __init__(self, config: OB15Config):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys
        # EMA on base body-frame velocities (teleop + policy); shared motion language for train/infer.
        # alpha ~0.3 ≈ few frames at 30 Hz; lower = smoother, higher = snappier.
        self._base_vel_alpha = 0.35
        self._base_vel_smooth = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        # Define three speed levels and a current index
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.6, "theta": 180},  # fast
        ]
        self.speed_index = 0  # Start at slow
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # Motor definitions per logical limb group. IDs are bus-local (each physical bus is
        # independently addressed), so the same ID range (e.g. base's 7-9) can safely appear
        # on multiple groups as long as those groups never end up sharing a bus with a
        # colliding ID -- true for every grouping this class supports.
        group_motors: dict[str, dict[str, Motor]] = {
            "left_arm": {
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            "right_arm": {
                "right_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "right_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "right_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "right_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "right_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "right_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            "base": {
                "base_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            "head": {
                "head_pan": Motor(7, "sts3215", norm_mode_body),
                "head_tilt": Motor(8, "sts3215", norm_mode_body),
            },
        }
        self._group_port: dict[str, str] = {
            "left_arm": self.config.left_arm_port,
            "right_arm": self.config.right_arm_port,
            "base": self.config.base_port,
            "head": self.config.head_port,
        }

        # Optional gantry / Z lift axis: attaches to whichever group's bus `lift_axis.bus`
        # names, or its own dedicated port when `lift_axis.bus == "lift"`.
        if self.config.lift_axis.enabled:
            lift_bus_name = self.config.lift_axis.bus
            lift_port = (
                self.config.lift_port if lift_bus_name == "lift" else self._group_port[lift_bus_name]
            )
            self._group_port["lift"] = lift_port
            # RANGE_M100_100 for velocity control (lift axis drives it in velocity mode);
            # normalizes velocity to [-100, 100] automatically at the bus level.
            group_motors["lift"] = {
                self.config.lift_axis.name: Motor(
                    self.config.lift_axis.motor_id,
                    self.config.lift_axis.motor_model,
                    MotorNormMode.RANGE_M100_100,
                )
            }

        # Build one FeetechMotorsBus per unique port, merging motors (and calibration) from
        # every group that resolves to that port. This is what lets any subset of limbs move
        # to its own dedicated serial adapter -- or stay combined -- purely via config.
        self._buses: dict[str, FeetechMotorsBus] = {}
        self._group_bus: dict[str, FeetechMotorsBus] = {}
        for group, port in self._group_port.items():
            if port not in self._buses:
                merged_motors: dict[str, Motor] = {}
                merged_calibration: dict[str, MotorCalibration] = {}
                id_owners: dict[int, str] = {}
                for other_group, other_port in self._group_port.items():
                    if other_port != port:
                        continue
                    for motor_name, motor in group_motors[other_group].items():
                        if motor.id in id_owners:
                            raise ValueError(
                                f"Motor ID collision on {port}: '{motor_name}' and "
                                f"'{id_owners[motor.id]}' both use id={motor.id}. "
                                f"Groups sharing this port: "
                                f"{[g for g, p in self._group_port.items() if p == port]}. "
                                "Give colliding limbs distinct *_port values "
                                "(or attach the lift via lift_axis.bus / lift_port)."
                            )
                        id_owners[motor.id] = motor_name
                        merged_motors[motor_name] = motor
                        if motor_name in self.calibration:
                            merged_calibration[motor_name] = self.calibration[motor_name]
                self._buses[port] = FeetechMotorsBus(
                    port=port, motors=merged_motors, calibration=merged_calibration
                )
            self._group_bus[group] = self._buses[port]

        # Optional gantry / Z lift axis (already on its bus if enabled; attach() is a no-op
        # if the motor is already present).
        self.lift_axis = LiftAxis(self.config.lift_axis, self._group_bus.get("lift"))
        self.lift_axis.attach()

        self.left_arm_motors = list(group_motors["left_arm"])
        self.right_arm_motors = list(group_motors["right_arm"])
        self.head_motors = list(group_motors["head"])
        self.base_motors = list(group_motors["base"])
        self.cameras = make_cameras_from_configs(config.cameras)

        # Create persistent thread pool for parallel bus reads (avoid overhead of creating/destroying)
        self._executor = ThreadPoolExecutor(
            max_workers=max(len(self._buses), 1), thread_name_prefix="bus_reader"
        )

    @property
    def _state_ft(self) -> dict[str, type]:
        keys: tuple[str, ...] = (
                "left_arm_shoulder_pan.pos",
                "left_arm_shoulder_lift.pos",
                "left_arm_elbow_flex.pos",
                "left_arm_wrist_flex.pos",
                "left_arm_wrist_roll.pos",
                "left_arm_gripper.pos",
                "right_arm_shoulder_pan.pos",
                "right_arm_shoulder_lift.pos",
                "right_arm_elbow_flex.pos",
                "right_arm_wrist_flex.pos",
                "right_arm_wrist_roll.pos",
                "right_arm_gripper.pos",
                "head_pan.pos",
                "head_tilt.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
        )
        if self.lift_axis.enabled:
            # Only expose height_mm — not raw velocity — matching every other joint's .pos convention.
            # The P-controller in lift_axis.apply_action() converts height_mm targets to motor
            # velocity internally, so the policy never needs to reason in velocity units.
            keys = (
                *keys,
                f"{self.lift_axis.cfg.name}.height_mm",
            )
        return dict.fromkeys(keys, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features including depth images if available."""
        features = {}
        for cam_name, cam_config in self.config.cameras.items():
            # Add color camera
            features[cam_name] = (cam_config.height, cam_config.width, 3)
            # Add depth camera if enabled (RealSense cameras). Depth may be captured at a
            # different resolution than color (see RealSenseCameraConfig.depth_width/depth_height)
            # to cut the USB/hardware-sync cost of enabling depth; fall back to the color
            # dimensions when unset, matching the camera's own default.
            if hasattr(cam_config, 'use_depth') and cam_config.use_depth:
                depth_height = getattr(cam_config, 'depth_height', None) or cam_config.height
                depth_width = getattr(cam_config, 'depth_width', None) or cam_config.width
                features[f"{cam_name}_depth"] = (depth_height, depth_width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        # Robot is connected if all physical buses are connected (one per unique port).
        # Cameras are optional - robot can function without all cameras
        return all(bus.is_connected for bus in self._buses.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for bus in self._buses.values():
            bus.connect()

        # Check if calibration file exists and ask user if they want to restore it
        if self.calibration_fpath.is_file():
            logger.info(f"Calibration file found at {self.calibration_fpath}")
            # Non-interactive runs (robot_client under bridge/systemd) must not call input() — EOFError or hang.
            # isatty() can be true under some PTY wrappers; EOFError still happens on read.
            noninteractive = (
                sys.stdin is None
                or not sys.stdin.isatty()
                or os.environ.get("LEROBOT_NONINTERACTIVE", "").lower() in ("1", "true", "yes")
            )
            if noninteractive:
                user_input = ""
                logger.info("Non-interactive mode — auto-restoring calibration from file.")
            else:
                try:
                    user_input = input(
                        "Press ENTER to restore calibration from file, or type 'c' and press ENTER to run manual calibration: "
                    )
                except EOFError:
                    user_input = ""
                    logger.info("EOF on calibration prompt — auto-restoring calibration from file.")
            if user_input.strip().lower() != "c":
                logger.info("Attempting to restore calibration from file...")
                try:
                    # Load calibration data into bus memory, then write it to the motors.
                    for bus in self._buses.values():
                        bus_calibration = {k: v for k, v in self.calibration.items() if k in bus.motors}
                        bus.calibration = bus_calibration
                        bus.write_calibration(bus_calibration)
                    logger.info("Calibration data loaded into bus memory successfully!")
                    logger.info("Calibration restored successfully from file!")

                except Exception as e:
                    logger.warning(f"Failed to restore calibration from file: {e}")
                    if calibrate:
                        logger.info("Proceeding with manual calibration...")
                        self.calibrate()
            else:
                logger.info("User chose manual calibration...")
                if calibrate:
                    self.calibrate()
        elif calibrate:
            logger.info("No calibration file found, proceeding with manual calibration...")
            self.calibrate()

        # Connect cameras with staged startup (head first) to avoid USB bandwidth spikes
        requested_order = list(self.config.camera_start_order or ())
        seen: set[str] = set()
        ordered_camera_names: list[str] = []
        for name in requested_order:
            if name in self.cameras and name not in seen:
                ordered_camera_names.append(name)
                seen.add(name)
        for name in self.cameras.keys():
            if name not in seen:
                ordered_camera_names.append(name)
                seen.add(name)

        for idx, cam_name in enumerate(ordered_camera_names):
            cam = self.cameras[cam_name]
            try:
                cam.connect()
                logger.info(f"✅ Camera '{cam_name}' connected successfully")
            except Exception as e:
                logger.warning(f"⚠️  Camera '{cam_name}' failed to connect: {e}")
                logger.warning(f"   Continuing without camera '{cam_name}' - robot will still function")
            else:
                if self.config.camera_start_delay_s > 0 and idx < len(ordered_camera_names) - 1:
                    time.sleep(self.config.camera_start_delay_s)

        self.configure()
        self._base_vel_smooth = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return all(bus.is_calibrated for bus in self._buses.values())

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        ## calib left motors
        left_motors = self.left_arm_motors
        left_bus = self._group_bus["left_arm"]
        left_bus.disable_torque(left_motors)
        for name in left_motors:
            left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
        input(
            "Move left arm motors to the middle of their range of motion and press ENTER...."
        )
        homing_offsets_left = left_bus.set_half_turn_homings(left_motors)

        print(
            "Move all left arm joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins_left, range_maxes_left = left_bus.record_ranges_of_motion(left_motors)

        calibration_left = {}
        for name in left_motors:
            motor = left_bus.motors[name]
            calibration_left[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets_left[name],
                range_min=range_mins_left[name],
                range_max=range_maxes_left[name],
            )

        # calib right arm motors
        right_bus = self._group_bus["right_arm"]
        right_bus.disable_torque(self.right_arm_motors)
        for name in self.right_arm_motors:
            right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input(
            "Move right arm motors to the middle of their range of motion and press ENTER...."
        )

        homing_offsets_right = right_bus.set_half_turn_homings(self.right_arm_motors)

        print(
            "Move all right arm joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins_right, range_maxes_right = right_bus.record_ranges_of_motion(self.right_arm_motors)

        calibration_right = {}
        for name in self.right_arm_motors:
            motor = right_bus.motors[name]
            calibration_right[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets_right[name],
                range_min=range_mins_right[name],
                range_max=range_maxes_right[name],
            )

        # calib head motors
        head_bus = self._group_bus["head"]
        head_bus.disable_torque(self.head_motors)
        for name in self.head_motors:
            head_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input(
            "Move head motors to the middle of their range of motion and press ENTER...."
        )

        homing_offsets_head = head_bus.set_half_turn_homings(self.head_motors)

        print(
            "Move head pan and tilt through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins_head, range_maxes_head = head_bus.record_ranges_of_motion(self.head_motors)

        calibration_head = {}
        for name in self.head_motors:
            motor = head_bus.motors[name]
            calibration_head[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets_head[name],
                range_min=range_mins_head[name],
                range_max=range_maxes_head[name],
            )

        # calib base motors
        print("Base wheels use full turn mode, setting range to 0-4095...")
        range_mins_base = {}
        range_maxes_base = {}
        for name in self.base_motors:
            range_mins_base[name] = 0
            range_maxes_base[name] = 4095

        homing_offsets_base = dict.fromkeys(self.base_motors, 0)

        base_bus = self._group_bus["base"]
        calibration_base = {}
        for name, motor in base_bus.motors.items():
            if name.startswith("base"):
                calibration_base[name] = MotorCalibration(
                    id=motor.id,
                    drive_mode=0,
                    homing_offset=homing_offsets_base[name],
                    range_min=range_mins_base[name],
                    range_max=range_maxes_base[name],
                )

        # Calibrate lift axis (if enabled) - home to set zero position
        if self.lift_axis.enabled:
            print("\n" + "="*60)
            print("LIFT AXIS CALIBRATION")
            print("="*60)
            print("The lift axis will home by driving down until it stalls.")
            print("This sets the current position as 0mm.")
            user_input = input("Press ENTER to start lift axis homing, or 's' to skip: ")
            if user_input.strip().lower() != 's':
                try:
                    self.lift_axis.home(use_current=True)
                    print("✅ Lift axis homed successfully (zero position set)")
                except Exception as e:
                    logger.warning(f"⚠️  Lift axis homing failed: {e}")
                    print("⚠️  Lift axis homing failed - you may need to home it manually later")
            else:
                print("⏭️  Lift axis homing skipped")

        # Merge all group calibrations, then write each bus's slice to its motors (a shared
        # bus just gets the union of whichever groups resolved to it).
        self.calibration = {**calibration_left, **calibration_right, **calibration_head, **calibration_base}
        for bus in self._buses.values():
            bus_calibration = {name: cal for name, cal in self.calibration.items() if name in bus.motors}
            bus.write_calibration(bus_calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)


    def configure(self):
        # Configure every physical bus (one per unique port; a shared bus gets whichever
        # groups resolved to it). We assume that at connection time, arms are in rest
        # position, and torque can be safely disabled to run configuration.

        for bus in self._buses.values():
            bus.disable_torque()
            bus.configure_motors()

        left_bus = self._group_bus["left_arm"]
        right_bus = self._group_bus["right_arm"]
        head_bus = self._group_bus["head"]
        base_bus = self._group_bus["base"]

        # Configure left arm motors - position mode
        for name in self.left_arm_motors:
            left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            left_bus.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            left_bus.write("I_Coefficient", name, 0)
            left_bus.write("D_Coefficient", name, 43)

        # Configure right arm motors - position mode
        for name in self.right_arm_motors:
            right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            right_bus.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            right_bus.write("I_Coefficient", name, 0)
            right_bus.write("D_Coefficient", name, 43)

        # Configure head motors - position mode
        for name in self.head_motors:
            head_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            head_bus.write("P_Coefficient", name, 16)
            head_bus.write("I_Coefficient", name, 0)
            head_bus.write("D_Coefficient", name, 43)

        # Configure base motors - velocity mode
        for name in self.base_motors:
            base_bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        # Configure gantry / lift axis (velocity mode + wrap tracking)
        if self.lift_axis.enabled:
            self.lift_axis.configure()
            # Brief pause so any residual bus packets from homing (torque-disable writes,
            # velocity commands) are fully flushed before we broadcast enable_torque to all
            # motors.  Without this, a delayed response from the lift motor can collide with
            # the next write and produce "Incorrect status packet!" on an unrelated motor.
            time.sleep(0.35)

        # Enable torque on every bus (retries + spacing help after homing / heavy config traffic)
        buses = list(self._buses.values())
        for i, bus in enumerate(buses):
            bus.enable_torque(num_retry=4)
            if i < len(buses) - 1:
                time.sleep(0.05)


    def setup_motors(self) -> None:
        left_bus = self._group_bus["left_arm"]
        right_bus = self._group_bus["right_arm"]
        head_bus = self._group_bus["head"]
        base_bus = self._group_bus["base"]

        for motor in reversed(self.left_arm_motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            left_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {left_bus.motors[motor].id}")

        # Set up right arm motors
        for motor in reversed(self.right_arm_motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            right_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {right_bus.motors[motor].id}")

        # Set up base motors
        for motor in reversed(self.base_motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            base_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {base_bus.motors[motor].id}")

        # Set up head motors
        for motor in reversed(self.head_motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            head_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {head_bus.motors[motor].id}")


    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"base_left_wheel": value, "base_back_wheel": value, "base_right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using _degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x, y, theta_rad])

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Vector with raw wheel commands ("base_left_wheel", "base_back_wheel", "base_right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A dict (x.vel, y.vel, theta.vel) all in m/s
        """

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )

        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)
        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }  # m/s and deg/s

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        # Speed control
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        x_cmd = 0.0  # m/s forward/backward
        y_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed

        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    def _smooth_base_vel(self, cmd: dict[str, float]) -> dict[str, float]:
        """EMA on x/y/theta body velocities; damps one-frame spikes (stick release, noisy policy)."""
        for k in ("x.vel", "y.vel", "theta.vel"):
            raw = float(cmd.get(k, 0.0))
            self._base_vel_smooth[k] = (
                self._base_vel_alpha * raw + (1.0 - self._base_vel_alpha) * self._base_vel_smooth[k]
            )
        return dict(self._base_vel_smooth)

    def get_observation(self, skip_cameras: bool = False, skip_depth: bool = False) -> dict[str, Any]:
        """
        Get robot observation with parallel bus reads and detailed profiling.

        Optimization: reads every physical bus in parallel (one thread per unique port)
        using the persistent thread pool; groups sharing a bus are read sequentially
        within that bus's thread, since a single serial bus can't be read concurrently.

        Args:
            skip_cameras: If True, skip camera reads entirely (faster, ~10-20ms vs ~100ms)
            skip_depth: If True, skip depth image reads (saves ~20-30ms per RealSense camera).
                Defaults to False so that depth frames are recorded in datasets when available.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        total_start = time.perf_counter()

        # Parallel read across physical buses using persistent executor
        bus_start = time.perf_counter()

        groups_by_port: dict[str, list[str]] = {}
        for group, port in self._group_port.items():
            groups_by_port.setdefault(port, []).append(group)

        def read_bus(port: str, groups: list[str]) -> dict[str, Any]:
            """Read every logical group living on this physical bus (sequential -- a
            serial bus is exclusive -- but every bus runs in its own thread)."""
            t0 = time.perf_counter()
            bus = self._buses[port]
            result: dict[str, Any] = {}
            for group in groups:
                if group == "left_arm":
                    result["left_arm_pos"] = bus.sync_read("Present_Position", self.left_arm_motors)
                elif group == "right_arm":
                    result["right_arm_pos"] = bus.sync_read("Present_Position", self.right_arm_motors)
                elif group == "head":
                    result["head_pos"] = bus.sync_read("Present_Position", self.head_motors)
                elif group == "base":
                    result["base_vel"] = bus.sync_read("Present_Velocity", self.base_motors)
                elif group == "lift":
                    try:
                        result["lift_pos"] = bus.read(
                            "Present_Position", self.lift_axis.cfg.name, normalize=False
                        )
                        result["lift_vel"] = bus.read(
                            "Present_Velocity", self.lift_axis.cfg.name, normalize=False
                        )
                    except Exception as e:
                        logger.debug(f"⚠️  Failed to read lift axis on {port}: {e}")
            logger.debug(f"Bus {port} ({','.join(groups)}) read: {(time.perf_counter()-t0)*1e3:.1f}ms")
            return result

        futures = {
            port: self._executor.submit(read_bus, port, groups) for port, groups in groups_by_port.items()
        }
        results = {port: future.result() for port, future in futures.items()}

        left_arm_pos: dict[str, float] = {}
        right_arm_pos: dict[str, float] = {}
        head_pos: dict[str, float] = {}
        base_wheel_vel: dict[str, float] = {}
        lift_pos = None
        lift_vel = None
        for port_result in results.values():
            left_arm_pos = port_result.get("left_arm_pos", left_arm_pos)
            right_arm_pos = port_result.get("right_arm_pos", right_arm_pos)
            head_pos = port_result.get("head_pos", head_pos)
            base_wheel_vel = port_result.get("base_vel", base_wheel_vel)
            if "lift_pos" in port_result:
                lift_pos = port_result["lift_pos"]
                lift_vel = port_result["lift_vel"]

        bus_dt_ms = (time.perf_counter() - bus_start) * 1e3
        logger.debug(f"🔧 Parallel bus reads: {bus_dt_ms:.1f}ms")

        # Process base velocity
        proc_start = time.perf_counter()
        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )

        left_arm_state = {f"{k}.pos": v for k, v in left_arm_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_arm_pos.items()}
        head_state = {f"{k}.pos": v for k, v in head_pos.items()}
        obs_dict = {**left_arm_state, **right_arm_state, **head_state, **base_vel}

        # Add gantry/lift observation (height in mm + velocity feedback)
        if self.lift_axis.enabled:
            try:
                self.lift_axis.contribute_observation(obs_dict, pre_read_pos=lift_pos, pre_read_vel=lift_vel)
            except Exception as e:
                logger.debug(f"⚠️  Lift axis observation failed: {e}")

        proc_dt_ms = (time.perf_counter() - proc_start) * 1e3
        logger.debug(f"Processing: {proc_dt_ms:.1f}ms")

        # Capture images from cameras in parallel (skip if requested for performance)
        if not skip_cameras:
            cam_start = time.perf_counter()

            def read_camera(cam_key, cam):
                try:
                    if cam.is_connected:
                        color_frame = cam.async_read()
                        # For RealSense cameras with depth enabled, also read depth (unless skipped).
                        # Non-blocking peek (read_latest_depth), not async_read_depth: with
                        # depth_frame_interval > 1 the background thread only decodes a fresh
                        # depth frame every Nth hardware frame, so waiting for the "new frame"
                        # event (async_read_depth) would block this tick on the camera's slower
                        # depth cadence instead of returning immediately with the latest buffered
                        # frame (possibly a few ticks old — fine, same "last known-good" backfill
                        # already used for other camera hiccups, see last_camera_obs below).
                        depth_frame = None
                        if not skip_depth and hasattr(cam, 'use_depth') and cam.use_depth and hasattr(cam, 'read_latest_depth'):
                            try:
                                depth_frame = cam.read_latest_depth(max_age_ms=1000)
                            except Exception as e:
                                logger.debug(f"⚠️  Failed to read depth from camera '{cam_key}': {e}")
                        return cam_key, color_frame, depth_frame
                    else:
                        logger.debug(f"⚠️  Camera '{cam_key}' not connected, skipping")
                        return cam_key, None, None
                except Exception as e:
                    logger.warning(f"⚠️  Failed to read from camera '{cam_key}': {e}")
                    return cam_key, None, None

            # Submit all camera reads to thread pool
            camera_futures = [
                self._executor.submit(read_camera, cam_key, cam)
                for cam_key, cam in self.cameras.items()
            ]

            # Collect results
            for future in camera_futures:
                cam_key, color_frame, depth_frame = future.result()
                if color_frame is not None:
                    obs_dict[cam_key] = color_frame
                if depth_frame is not None:
                    # Ensure depth has a channel dimension to match dataset feature (H, W, 1)
                    try:
                        import numpy as np  # Local import to avoid top-level dependency if not needed
                        if depth_frame.ndim == 2:
                            depth_frame = np.expand_dims(depth_frame, axis=-1)
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to reshape depth frame for camera '{cam_key}': {e}")
                    # Add depth frame with "_depth" suffix
                    obs_dict[f"{cam_key}_depth"] = depth_frame

            cam_dt_ms = (time.perf_counter() - cam_start) * 1e3
            logger.debug(f"📷 Camera capture: {cam_dt_ms:.1f}ms")
        else:
            logger.debug("📷 Camera reads skipped for performance")

        total_dt_ms = (time.perf_counter() - total_start) * 1e3
        logger.debug(f"⏱️  TOTAL get_observation: {total_dt_ms:.1f}ms ({1000/total_dt_ms:.1f} Hz)")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command lekiwi to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_arm_pos = {k: v for k, v in action.items() if k.startswith("left_arm_") and k.endswith(".pos")}
        right_arm_pos = {k: v for k, v in action.items() if k.startswith("right_arm_") and k.endswith(".pos")}
        head_pos = {k: v for k, v in action.items() if k.startswith("head_") and k.endswith(".pos")}
        # Base: read raw body-frame cmds (0 if key absent) so EMA decays every tick.
        base_goal_vel_raw = {k: float(action.get(k, 0.0)) for k in ("x.vel", "y.vel", "theta.vel")}
        base_goal_vel_smooth = self._smooth_base_vel(base_goal_vel_raw)
        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel_smooth["x.vel"],
            base_goal_vel_smooth["y.vel"],
            base_goal_vel_smooth["theta.vel"],
        )


        if self.config.max_relative_target is not None:
            # Read present positions for left arm and right arm
            present_pos_left = self._group_bus["left_arm"].sync_read("Present_Position", self.left_arm_motors)
            present_pos_right = self._group_bus["right_arm"].sync_read("Present_Position", self.right_arm_motors)

            # Combine all present positions
            present_pos = {**present_pos_left, **present_pos_right}

            # Ensure safe goal position for each arm
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in chain(left_arm_pos.items(), right_arm_pos.items())
            }
            safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

            # Update the action with the safe goal positions
            left_arm_pos = {k: v for k, v in safe_goal_pos.items() if k in left_arm_pos}
            right_arm_pos = {k: v for k, v in safe_goal_pos.items() if k in right_arm_pos}

        left_arm_pos_raw = {k.replace(".pos", ""): v for k, v in left_arm_pos.items()}
        right_arm_pos_raw = {k.replace(".pos", ""): v for k, v in right_arm_pos.items()}
        head_pos_raw = {k.replace(".pos", ""): v for k, v in head_pos.items()}

        # Accumulate Goal_Position/Goal_Velocity writes per physical bus -- groups sharing a
        # bus get combined into a single sync_write per command type (exactly like the
        # un-split default), groups on their own dedicated port each get their own.
        pos_by_port: dict[str, dict[str, float]] = {}
        vel_by_port: dict[str, dict[str, float]] = {}
        for group, raw in (
            ("left_arm", left_arm_pos_raw),
            ("right_arm", right_arm_pos_raw),
            ("head", head_pos_raw),
        ):
            if raw:
                pos_by_port.setdefault(self._group_port[group], {}).update(raw)
        if base_wheel_goal_vel:
            vel_by_port.setdefault(self._group_port["base"], {}).update(base_wheel_goal_vel)

        for port, values in pos_by_port.items():
            self._buses[port].sync_write("Goal_Position", values)
        for port, values in vel_by_port.items():
            self._buses[port].sync_write("Goal_Velocity", values)

        # Apply gantry / lift action after bus writes (keeps base logic unchanged).
        normalized_lift_action = {}
        if self.lift_axis.enabled:
            try:
                lift_action = {k: v for k, v in action.items() if k.startswith(f"{self.lift_axis.cfg.name}.")}
                vel_key = f"{self.lift_axis.cfg.name}.vel"
                if vel_key in lift_action:
                    # Normalize for apply_action: lift_axis expects [-100, 100]
                    raw_v = lift_action[vel_key]
                    if abs(raw_v) > 100.0:
                        lift_action[vel_key] = (float(raw_v) / self.lift_axis.cfg.v_max) * 100.0
                        lift_action[vel_key] = max(-100.0, min(100.0, lift_action[vel_key]))
                self.lift_axis.apply_action(lift_action)
                # Normalized copy for logging/recording (single source of truth in lift_axis)
                normalized_lift_action = self.lift_axis.action_for_logging(lift_action)
            except Exception as e:
                logger.debug(f"⚠️  Lift axis action failed: {e}")

        # Record only height_mm as the lift action — matches the position-control convention
        # used for all arm joints (.pos) and the reference LeKiwi implementation.
        # The VR teleop may send gantry.vel to drive the motor, but what is saved in the
        # dataset is the resulting physical position so the policy can replay it via the
        # P-controller in lift_axis.apply_action().
        lift_keys: dict[str, Any] = {}
        if self.lift_axis.enabled:
            lift_keys[f"{self.lift_axis.cfg.name}.height_mm"] = self.lift_axis._cached_height_mm

        return {
            **left_arm_pos,
            **right_arm_pos,
            **head_pos,
            **base_goal_vel_smooth,
            **lift_keys,
        }

    def stop_base(self):
        try:
            base_bus = self._group_bus["base"]
            if base_bus.is_connected:
                base_bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
                self._base_vel_smooth = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                logger.info("Base motors stopped")
            else:
                logger.debug("Base bus not connected; skipping base stop.")
        except Exception as e:
            logger.warning(f"Failed to stop base cleanly: {e}")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Best-effort base stop; ignore errors on teardown
        try:
            self.stop_base()
        except Exception:
            pass

        # Best-effort lift stop; ignore errors on teardown
        try:
            if self.lift_axis.enabled:
                self.lift_axis.stop()
        except Exception:
            pass

        # Shutdown thread pool
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True, cancel_futures=True)

        for bus in self._buses.values():
            bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
