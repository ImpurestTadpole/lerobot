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
XLerobot VR Teleoperator
Refactored based on VR control logic from 8_xlerobot_VR_teleop.py, following teleop_keyboard format
"""
import math

import asyncio
import logging
import os
import sys
import threading
import time
import traceback
from queue import Queue
from typing import Any, Dict, Optional

import numpy as np

# from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.model.SO101Robot import SO101Kinematics

from ..teleoperator import Teleoperator
from .configuration_xlerobot_vr import XLerobotVRTeleopConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check VR Monitor availability
VR_AVAILABLE = True
try:
    # Dynamically import VR Monitor 
    from .vr_monitor import VRMonitor
except ImportError as e:
    VR_AVAILABLE = False
    VRMonitor = None
    logging.warning(f"VR Monitor not available: {e}")
except Exception as e:
    VR_AVAILABLE = False
    VRMonitor = None
    logging.warning(f"Could not import VR Monitor: {e}")


# Joint mapping configurations (copied from 8_xlerobot_VR_teleop.py)
LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}

RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

# Joint calibration coefficients (copied from 8_xlerobot_VR_teleop.py)
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      
    ['shoulder_lift', 2.0, 0.97],     
    ['elbow_flex', 0.0, 1.05],        
    ['wrist_flex', 0.0, 0.94],        
    ['wrist_roll', 0.0, 0.5],        
    ['gripper', 0.0, 1.0],           
]

class SimpleTeleopArm:
    """
    A class for controlling a robot arm using VR input with delta action control.
    
    This class provides inverse kinematics-based arm control with proportional control
    for smooth movement and gripper operations based on VR controller input.
    """
    
    def __init__(self, joint_map, initial_obs, kinematics, prefix="right", kp=1.5):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp  # Balanced for speed and smoothness (reduced from 2.5 to reduce jitter)
        self.kinematics = kinematics
        
        # Initial joint positions - adapted for XLerobot observation format
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Delta control state variables for VR input
        self.last_vr_time = 0.0
        self.vr_deadzone = 0.001  # Minimum movement threshold
        self.max_delta_per_frame = 0.005  # Maximum position change per frame
        
        # Set step size
        self.degree_step = 2
        self.xy_step = 0.005
        
        # Gripper smoothing state for variable precision control
        self.smoothed_gripper_value = self.joint_positions["gripper"]  # Initialize to current gripper position
        self.gripper_smoothing_alpha = 0.3  # Smoothing factor (0.0 = no smoothing, 1.0 = no smoothing)
        # Lower alpha = smoother but slower response, higher alpha = faster but less smooth
        
        # P control target positions, set to zero position
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        
        # Smoothed target positions for P-control (reduces jitter at 60Hz)
        # Initialize to current positions to avoid jumps
        self.smoothed_target_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        # Smoothing alpha for target positions (higher = more responsive, lower = smoother)
        # At 60Hz, alpha=0.3 means ~90% of target change happens in ~5 frames (~80ms)
        self.target_smoothing_alpha = 0.3
        
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        
        # Reset kinematics variables to initial state
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Reset delta control state
        self.last_vr_time = 0.0
        
        # Explicitly set wrist_flex
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        return action

    def handle_vr_input(self, vr_goal, gripper_state):
        """
        Handle VR input with delta action control - incremental position updates.
        
        Args:
            vr_goal: VR controller goal data containing target position and orientations
            gripper_state: Current gripper state (not used in current implementation)
        """
        if vr_goal is None:
            return
        
        # VR goal contains: target_position [x, y, z], wrist_roll_deg, wrist_flex_deg, gripper_closed
        if not hasattr(vr_goal, 'target_position') or vr_goal.target_position is None:
            return
            
        # Extract VR position data
        # Get current VR position
        current_vr_pos = vr_goal.target_position  # [x, y, z] in meters
        
        # Initialize previous VR position if not set
        if not hasattr(self, 'prev_vr_pos'):
            self.prev_vr_pos = current_vr_pos
            return  # Skip first frame to establish baseline
        
        # print(current_vr_pos)
        
        # Calculate relative change (delta) from previous frame
        raw_delta_x = (current_vr_pos[0] - self.prev_vr_pos[0]) * 170  # Scale for the shoulder
        raw_delta_y = (current_vr_pos[1] - self.prev_vr_pos[1]) * 80
        raw_delta_z = (current_vr_pos[2] - self.prev_vr_pos[2]) * 80

        # Update previous position for next frame
        self.prev_vr_pos = current_vr_pos
        
        # Delta control parameters - optimized for responsive, smooth control
        pos_scale = 0.015  # Reduced for more controlled, less sensitive movement
        angle_scale = 3.0  # Angle sensitivity scaling (for wrist flex/pitch)
        wrist_roll_scale = 1.0  # Separate, slower scaling for wrist roll
        delta_limit = 0.04  # Increased limit for faster movement
        angle_limit = 6.0  # Maximum angle delta per update (degrees)
        wrist_roll_limit = 3.0  # Maximum wrist roll delta per update (degrees)
        
        # Apply simple low-pass filter only to very high frequency noise
        # Use minimal smoothing - just enough to filter out sensor noise
        if not hasattr(self, 'filtered_delta_x'):
            self.filtered_delta_x = 0.0
            self.filtered_delta_y = 0.0
            self.filtered_delta_z = 0.0
        
        # Very light filtering (alpha=0.9 means 90% new, 10% old) - almost no lag
        filter_alpha = 0.9
        self.filtered_delta_x = filter_alpha * raw_delta_x + (1 - filter_alpha) * self.filtered_delta_x
        self.filtered_delta_y = filter_alpha * raw_delta_y + (1 - filter_alpha) * self.filtered_delta_y
        self.filtered_delta_z = filter_alpha * raw_delta_z + (1 - filter_alpha) * self.filtered_delta_z
        
        delta_x = self.filtered_delta_x * pos_scale
        delta_y = self.filtered_delta_y * pos_scale  
        delta_z = self.filtered_delta_z * pos_scale

        # Dead zone - very small, only filter out true noise
        threshold = 0.0005
        if delta_x < threshold and delta_x > -threshold:
            delta_x = 0.0
        if delta_y < threshold and delta_y > -threshold:
            delta_y = 0.0
        if delta_z < threshold and delta_z > -threshold:
            delta_z = 0.0

        # Limit delta values to prevent sudden movements
        delta_x = max(-delta_limit, min(delta_limit, delta_x))
        delta_y = max(-delta_limit, min(delta_limit, delta_y))
        delta_z = max(-delta_limit, min(delta_limit, delta_z))
        
        self.current_x += -delta_z  # VR Z maps to robot x, change the direction
        self.current_y += delta_y  # VR Y maps to robot y

        # Handle wrist angles with delta control - use relative changes
        if hasattr(vr_goal, 'wrist_flex_deg') and vr_goal.wrist_flex_deg is not None:
            # Initialize previous wrist_flex if not set
            if not hasattr(self, 'prev_wrist_flex'):
                self.prev_wrist_flex = vr_goal.wrist_flex_deg
                return
            
            # Calculate relative change from previous frame
            delta_pitch = (vr_goal.wrist_flex_deg - self.prev_wrist_flex) * angle_scale
            if delta_pitch < 1 and delta_pitch > -1:
                delta_pitch = 0.0
            delta_pitch = max(-angle_limit, min(angle_limit, delta_pitch))
            self.pitch += delta_pitch
            self.pitch = max(-90, min(90, self.pitch))  # Limit pitch range
            
            # Update previous value for next frame
            self.prev_wrist_flex = vr_goal.wrist_flex_deg
        
        if hasattr(vr_goal, 'wrist_roll_deg') and vr_goal.wrist_roll_deg is not None:
            # Initialize previous wrist_roll if not set
            if not hasattr(self, 'prev_wrist_roll'):
                self.prev_wrist_roll = vr_goal.wrist_roll_deg
                return
            
            # Use separate, slower scaling for wrist roll
            delta_roll = (vr_goal.wrist_roll_deg - self.prev_wrist_roll) * wrist_roll_scale
            delta_roll = max(-wrist_roll_limit, min(wrist_roll_limit, delta_roll))

            # Smaller dead zone for wrist roll to allow fine control
            if abs(delta_roll) < 0.5:
                delta_roll = 0.0
            
            current_roll = self.target_positions.get("wrist_roll", 0.0)
            new_roll = current_roll + delta_roll
            new_roll = max(-90, min(90, new_roll))  # Limit roll range
            self.target_positions["wrist_roll"] = new_roll
            
            # Update previous value for next frame
            self.prev_wrist_roll = vr_goal.wrist_roll_deg
        
        # VR Z axis controls shoulder_pan joint (delta control)
        if abs(delta_x) > 0.001:  # Only update if significant movement
            x_scale = 200.0  # Reduced scaling factor for delta control
            delta_pan = delta_x * x_scale
            delta_pan = max(-angle_limit, min(angle_limit, delta_pan))
            current_pan = self.target_positions.get("shoulder_pan", 0.0)
            new_pan = current_pan + delta_pan
            new_pan = max(-180, min(180, new_pan))  # Limit pan range
            self.target_positions["shoulder_pan"] = new_pan
        
        try:
            # Validate workspace before IK solving
            r = math.sqrt(self.current_x**2 + self.current_y**2)
            r_max = self.kinematics.l1 + self.kinematics.l2
            r_min = abs(self.kinematics.l1 - self.kinematics.l2)
            
            # Clamp to workspace if needed
            if r > r_max:
                scale = r_max / r
                self.current_x *= scale
                self.current_y *= scale
            elif r < r_min and r > 0:
                scale = r_min / r
                self.current_x *= scale
                self.current_y *= scale
            
            # Solve IK with improved precision
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            
            # Use lower alpha for smoother, more precise IK tracking
            # Lower values = more precise but slower response
            # Higher values = faster but less precise
            alpha = 0.15  # Reduced from 0.2 for better precision (was 0.1, then 0.2)
            
            # Apply exponential smoothing for precise tracking
            current_shoulder = self.target_positions.get("shoulder_lift", 0.0)
            current_elbow = self.target_positions.get("elbow_flex", 0.0)
            
            self.target_positions["shoulder_lift"] = (1-alpha) * current_shoulder + alpha * joint2_target
            self.target_positions["elbow_flex"] = (1-alpha) * current_elbow + alpha * joint3_target
            
        except Exception as e:
            print(f"[{self.prefix}] VR IK failed: {e}")
            # On IK failure, maintain current positions to prevent jumps
        
        # Calculate wrist_flex to maintain end-effector orientation
        self.target_positions["wrist_flex"] = (-self.target_positions["shoulder_lift"] - 
                                               self.target_positions["elbow_flex"] + self.pitch)
   
        # Handle gripper state with variable precision control
        # Extract trigger value as float (range: 0.0 to 1.0)
        # RANGE_0_100: 0 = fully open, 100 = fully closed
        trigger_value = float(vr_goal.metadata.get('trigger', 0.0))
        
        # Optional: Apply deadzone to prevent accidental small movements
        # Adjust deadzone threshold as needed (0.0 = no deadzone, higher = more deadzone)
        gripper_deadzone = 0.05  # 5% deadzone - adjust this value as needed
        if trigger_value < gripper_deadzone:
            trigger_value = 0.0
        else:
            # Normalize after deadzone: map [deadzone, 1.0] to [0.0, 1.0]
            trigger_value = (trigger_value - gripper_deadzone) / (1.0 - gripper_deadzone)
            trigger_value = max(0.0, min(1.0, trigger_value))  # Clamp to [0.0, 1.0]
        
        # Map trigger value [0.0, 1.0] to gripper range [0.0, 100.0]
        # 0.0 trigger = fully open (0.0), 1.0 trigger = fully closed (100.0)
        target_gripper_value = trigger_value * 100.0
        
        # Apply exponential smoothing to reduce jitter and provide smoother control
        # This helps prevent rapid oscillations when the trigger is held at a constant position
        self.smoothed_gripper_value = (
            (1.0 - self.gripper_smoothing_alpha) * self.smoothed_gripper_value +
            self.gripper_smoothing_alpha * target_gripper_value
        )
        
        # Clamp smoothed value to valid range
        self.smoothed_gripper_value = max(0.0, min(100.0, self.smoothed_gripper_value))
        self.target_positions["gripper"] = self.smoothed_gripper_value
        # Update smoothed gripper target (gripper already has its own smoothing, so just copy)
        self.smoothed_target_positions["gripper"] = self.smoothed_gripper_value
        
        # Smooth target positions before P-control to reduce jitter at 60Hz
        # This prevents motors from getting aggressive corrections every frame
        for joint in self.target_positions:
            if joint != "gripper":  # Gripper already smoothed separately above
                current_smoothed = self.smoothed_target_positions[joint]
                target = self.target_positions[joint]
                # Exponential smoothing: blend current smoothed value with new target
                self.smoothed_target_positions[joint] = (
                    (1.0 - self.target_smoothing_alpha) * current_smoothed +
                    self.target_smoothing_alpha * target
                )

    def p_control_action(self, robot_obs):
        """
        Generate proportional control action based on target positions.
        
        Args:
            robot: Robot instance to get current observations
            
        Returns:
            dict: Action dictionary with position commands for each joint
        """
        obs = robot_obs
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            # Use smoothed target instead of raw target to reduce jitter
            smoothed_target = self.smoothed_target_positions[j]
            error = smoothed_target - current[j]
            # Apply minimal deadzone only to eliminate true noise
            # Keep deadzone minimal for responsive control
            if abs(error) < 0.05:  # Small deadzone to prevent jitter from noise
                error = 0.0
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action


class SimpleHeadControl:
    """
    A class for controlling robot head motors using VR headset orientation.
    
    Maps VR headset yaw (pan) and pitch (tilt) to head motor commands with proportional control.
    """
    
    def __init__(self, initial_obs, kp=1.0, max_pan_deg=90, max_tilt_deg=45):
        """
        Initialize head control.
        
        Args:
            initial_obs: Initial robot observation to get current head positions
            kp: Proportional control gain
            max_pan_deg: Maximum pan angle in degrees (default ±90)
            max_tilt_deg: Maximum tilt angle in degrees (default ±45)
        """
        self.kp = kp
        self.max_pan_deg = max_pan_deg
        self.max_tilt_deg = max_tilt_deg
        
        # Conversion factors: head motors use RANGE_M100_100 normalization
        # Pan: ±90 degrees -> ±100 normalized (scale: 100/90)
        # Tilt: ±45 degrees -> ±100 normalized (scale: 100/45)
        self.pan_scale = 100.0 / max_pan_deg if max_pan_deg > 0 else 1.0
        self.tilt_scale = 100.0 / max_tilt_deg if max_tilt_deg > 0 else 1.0
        
        # Initialize head motor positions from observation (already normalized)
        self.target_positions = {
            "head_pan": initial_obs.get("head_pan.pos", 0.0),
            "head_tilt": initial_obs.get("head_tilt.pos", 0.0),
        }
        self.zero_pos = {"head_pan": 0.0, "head_tilt": 0.0}
        
    def handle_vr_input(self, headset_goal):
        """
        Handle VR headset orientation data.
        
        Args:
            headset_goal: ControlGoal with headset metadata containing head_pan and head_tilt in degrees
        """
        if headset_goal is None or not hasattr(headset_goal, 'metadata'):
            return
            
        # Extract head pan and tilt from headset goal metadata
        # head_pan: yaw angle (left/right), head_tilt: pitch angle (up/down)
        # Invert both pan and tilt to match robot movement direction
        head_pan_deg = -headset_goal.metadata.get('head_pan', 0.0)  # Inverted
        head_tilt_deg = -headset_goal.metadata.get('head_tilt', 0.0)  # Inverted
        
        # Clamp to safe ranges
        head_pan_deg = max(-self.max_pan_deg, min(self.max_pan_deg, head_pan_deg))
        head_tilt_deg = max(-self.max_tilt_deg, min(self.max_tilt_deg, head_tilt_deg))
        
        # Convert degrees to normalized positions (-100 to 100 range)
        # Head motors use RANGE_M100_100 normalization
        head_pan_norm = head_pan_deg * self.pan_scale
        head_tilt_norm = head_tilt_deg * self.tilt_scale
        
        # Clamp normalized values to safe range
        head_pan_norm = max(-100.0, min(100.0, head_pan_norm))
        head_tilt_norm = max(-100.0, min(100.0, head_tilt_norm))
        
        # Update target positions (in normalized units, matching robot observation format)
        self.target_positions["head_pan"] = head_pan_norm
        self.target_positions["head_tilt"] = head_tilt_norm
        
    def move_to_zero_position(self, robot):
        """Move head to zero position."""
        print(f"[HEAD] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)
        return action

    def p_control_action(self, robot_obs):
        """
        Generate proportional control action for head motors.
        
        Args:
            robot_obs: Robot observation dictionary
            
        Returns:
            dict: Action dictionary with position commands for head motors
        """
        action = {}
        for motor_name in self.target_positions:
            current = robot_obs.get(f"{motor_name}.pos", 0.0)
            error = self.target_positions[motor_name] - current
            control = self.kp * error
            action[f"{motor_name}.pos"] = current + control
        return action


def get_vr_base_action(left_goal, right_goal, robot):
    """
    Get base control commands from VR thumbstick input.
    
    Refactored to follow xlerobot's keyboard base control exactly.
    Control mapping (matching xlerobot keyboard layout):
    - Right thumbstick Y-axis: forward (positive) / backward (negative) [like 'i'/'k' keys]
    - Right thumbstick X-axis: left (negative) / right (positive) [like 'j'/'l' keys]  
    - Left thumbstick X-axis: rotate left (negative) / rotate right (positive) [like 'u'/'o' keys]
    
    Args:
        left_goal: Left VR controller goal data (for rotation control)
        right_goal: Right VR controller goal data (for translation control)
        robot: Robot instance with speed_levels attribute
        
    Returns:
        dict: Base velocity commands {"x.vel": float, "y.vel": float, "theta.vel": float}
    """
    # Get speed settings from robot (matching xlerobot's speed levels)
    if hasattr(robot, 'speed_levels') and hasattr(robot, 'speed_index'):
        speed_setting = robot.speed_levels[robot.speed_index]
        xy_speed = speed_setting["xy"]  # m/s (0.1, 0.2, or 0.3)
        theta_speed = speed_setting["theta"]  # deg/s (30, 60, or 90)
    else:
        # Fallback to medium speed
        xy_speed = 0.2  # m/s
        theta_speed = 60  # deg/s
    
    x_cmd = 0.0  # m/s forward/backward
    y_cmd = 0.0  # m/s lateral
    theta_cmd = 0.0  # deg/s rotation
    
    # Dead zone threshold (matching typical VR controller sensitivity)
    DEAD_ZONE = 0.15
    # Safety: if VR thumbstick metadata is stale, force base to stop.
    # This prevents the robot from continuing to spin if the recording loop drops updates.
    STALE_MS = 300
    
    # Right thumbstick controls XY translation (like keyboard i/k/j/l)
    if right_goal is not None and hasattr(right_goal, 'metadata'):
        metadata = right_goal.metadata
        if isinstance(metadata, dict):
            # If we got an explicit stop, obey it immediately.
            if metadata.get("base_stop", False):
                return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

            packet_ts_ms = metadata.get("packet_ts_ms")
            if isinstance(packet_ts_ms, (int, float)):
                now_ms = time.time() * 1000.0
                if (now_ms - float(packet_ts_ms)) > STALE_MS:
                    return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

            thumbstick = metadata.get('thumbstick', {})
            if isinstance(thumbstick, dict):
                thumb_x = thumbstick.get('x', 0)
                thumb_y = thumbstick.get('y', 0)
                
                # Log thumbstick values for debugging (only when above dead zone)
                if abs(thumb_x) > DEAD_ZONE or abs(thumb_y) > DEAD_ZONE:
                    logger.debug(f"🎮 RIGHT thumbstick: x={thumb_x:.2f}, y={thumb_y:.2f}")
                
                # Forward/backward (Y-axis)
                # Positive Y = forward (like 'i' key), negative Y = backward (like 'k' key)
                # Note: Inverted because VR thumbstick Y-axis is opposite to robot's forward direction
                if abs(thumb_y) > DEAD_ZONE:
                    x_cmd = -thumb_y * xy_speed  # Inverted to match robot's forward direction
                
                # Left/right lateral (X-axis)
                # Positive X = right (like 'l' key), negative X = left (like 'j' key)
                # Note: xlerobot uses opposite signs - positive for left, negative for right
                if abs(thumb_x) > DEAD_ZONE:
                    y_cmd = -thumb_x * xy_speed  # Inverted to match xlerobot keyboard
    
    # Left thumbstick controls rotation (like keyboard u/o)
    if left_goal is not None and hasattr(left_goal, 'metadata'):
        metadata = left_goal.metadata
        if isinstance(metadata, dict):
            # If we got an explicit stop, obey it immediately.
            if metadata.get("base_stop", False):
                return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

            packet_ts_ms = metadata.get("packet_ts_ms")
            if isinstance(packet_ts_ms, (int, float)):
                now_ms = time.time() * 1000.0
                if (now_ms - float(packet_ts_ms)) > STALE_MS:
                    return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

            thumbstick = metadata.get('thumbstick', {})
            if isinstance(thumbstick, dict):
                thumb_x = thumbstick.get('x', 0)
                
                # Log thumbstick values for debugging (only when above dead zone)
                if abs(thumb_x) > DEAD_ZONE:
                    logger.debug(f"🎮 LEFT thumbstick: x={thumb_x:.2f}")
                
                # Rotation (X-axis only)
                # Positive X = rotate right (like 'o' key), negative X = rotate left (like 'u' key)
                # Note: xlerobot uses opposite signs - positive for left, negative for right
                if abs(thumb_x) > DEAD_ZONE:
                    theta_cmd = -thumb_x * theta_speed  # Inverted to match xlerobot keyboard
    
    # Log base commands for debugging (only when non-zero)
    if x_cmd != 0 or y_cmd != 0 or theta_cmd != 0:
        logger.debug(f"🕹️  Base control: x={x_cmd:.3f} m/s, y={y_cmd:.3f} m/s, theta={theta_cmd:.1f} deg/s")
    
    return {
        "x.vel": x_cmd,
        "y.vel": y_cmd,
        "theta.vel": theta_cmd,
    }

class XLerobotVRTeleop(Teleoperator):
    """
    XLerobot VR Teleoperator class
    Following the format of teleop_keyboard, integrating VR control logic from 8_xlerobot_VR_teleop.py
    """

    config_class = XLerobotVRTeleopConfig
    name = "xlerobot_vr"

    def __init__(self, config: XLerobotVRTeleopConfig):
        super().__init__(config)
        self.config = config
        
        # VR system related
        self.vr_monitor = None
        self.vr_thread = None
        self.vr_data_queue = Queue()
        self.latest_vr_data = None
        
        # New: VR event handler
        self.vr_event_handler = None
                    
        # Kinematics instances
        self.kin_left = SO101Kinematics()
        self.kin_right = SO101Kinematics()
        
        # Arm controllers (initialized during calibrate, guarded elsewhere)
        self.left_arm = None
        self.right_arm = None
        
        # Head controller (initialized during calibrate)
        self.head_control = None
        
        # Base speed control
        self.current_base_speed = 0.0
        self.last_update_time = time.time()
        self.last_event_update_time = 0.0
        self.is_accelerating = False
        
        # Status flags
        self._connected = False
        self._calibrated = False
        
        # Store robot reference (set during connect)
        self.robot = None
        
        # Cache for observations to avoid double reads
        self._cached_obs = None
        self._obs_cache_time = 0.0
        self._obs_cache_duration = 0.01  # Cache for 10ms (faster than camera refresh)
        
        self.logs = {}

    @property
    def action_features(self) -> dict:
        """Define action feature structure"""
        # Define based on XLerobot's action space
        # Including dual arm joints, head motors, base movement
        features = {}
        
        # Left arm joints
        for joint_name in LEFT_JOINT_MAP.values():
            features[f"{joint_name}.pos"] = "float32"
        
        # Right arm joints
        for joint_name in RIGHT_JOINT_MAP.values():
            features[f"{joint_name}.pos"] = "float32"
            
        # Base control (according to XLerobot's base control method)
        features["base_action"] = "dict"
        
        return features

    @property
    def feedback_features(self) -> dict:
        """Define feedback feature structure"""
        return {}  # VR controllers usually don't need feedback

    @property
    def is_connected(self) -> bool:
        """Check connection status"""
        return (
            self._connected and 
            VR_AVAILABLE and 
            self.vr_monitor is not None and
            (self.vr_thread is not None and self.vr_thread.is_alive())
        )

    @property
    def is_calibrated(self) -> bool:
        """Check calibration status"""
        return self._calibrated

    def connect(self, calibrate: bool = True, robot=None) -> None:
        """Establish VR connection - optimized version"""
        if self.is_connected:
            raise RuntimeError(
                "XLerobot VR is already connected. Do not run `connect()` twice."
            )

        if not VR_AVAILABLE:
            raise RuntimeError(
                "VR Monitor is not available. Please check VR system installation."
            )

        try:
            logger.info("🔧 Initializing VR monitor...")
            self.vr_monitor = VRMonitor()
            
            # Use timeout mechanism to avoid infinite waiting
            init_success = False
            start_time = time.time()
            timeout = 10.0  # 10 second timeout
            
            while time.time() - start_time < timeout:
                if self.vr_monitor.initialize():
                    init_success = True
                    break
                time.sleep(0.1)
            
            if not init_success:
                raise Exception("VR monitor initialization timeout")
                
            logger.info("🚀 Starting VR monitoring...")
            self.vr_thread = threading.Thread(
                target=lambda: asyncio.run(self.vr_monitor.start_monitoring()), 
                daemon=True
            )
            self.vr_thread.start()
            
            # Wait for thread to start
            time.sleep(0.5)
            
            if not self.vr_thread.is_alive():
                raise Exception("VR monitoring thread failed to start")
                
            logger.info("✅ VR system ready")
            self._connected = True
            
            # Initialize VR event handler
            self.vr_event_handler = VREventHandler(self.vr_monitor)
            logger.info("🎮 VR event handler initialized")
            
            # Store robot reference for use in get_action
            self.robot = robot
            
            if calibrate and robot is not None:
                robot_obs = robot.get_observation()
                self.calibrate(robot_obs)
                
        except Exception as e:
            logger.error(f"[VR] Connection failed: {e}")
            self._connected = False
            raise RuntimeError(f"Failed to connect to VR: {e}")

    def calibrate(self, robot_obs: Optional[Dict] = None) -> None:
        """Calibrate VR controllers - optimized version"""
        if robot_obs is None:
            logger.warning("[VR] No robot observation provided for calibration")
            return
            
        try:
            # Initialize arm controllers
            self.left_arm = SimpleTeleopArm(
                LEFT_JOINT_MAP, robot_obs, self.kin_left, 
                prefix="left", kp=self.config.kp
            )
            self.right_arm = SimpleTeleopArm(
                RIGHT_JOINT_MAP, robot_obs, self.kin_right, 
                prefix="right", kp=self.config.kp
            )
            
            # Initialize head controller
            self.head_control = SimpleHeadControl(
                robot_obs, kp=self.config.kp
            )
            
            logger.info("[VR] Controllers initialized successfully (arms + head)")
            self._calibrated = True
            
        except Exception as e:
            logger.error(f"[VR] Calibration failed: {e}")
            self._calibrated = False
            raise


    def update_observation_cache(self, obs: dict[str, Any]) -> None:
        """
        Update the observation cache with externally-read observation.
        Called by lerobot_teleoperate.py to avoid double reads.
        """
        self._cached_obs = obs
        self._obs_cache_time = time.perf_counter()
    
    def _get_noop_action(self, robot_obs: dict[str, Any]) -> dict[str, Any]:
        """Generate a no-op action (current positions) with all required keys."""
        action = {}
        action.update(self._get_noop_left_arm_action(robot_obs))
        action.update(self._get_noop_right_arm_action(robot_obs))
        action.update(self._get_noop_head_action(robot_obs))
        # Base velocities default to zero
        action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        return action
    
    def _get_noop_left_arm_action(self, robot_obs: dict[str, Any]) -> dict[str, Any]:
        """Generate no-op action for left arm (current positions)."""
        action = {}
        for joint_name in LEFT_JOINT_MAP.values():
            key = f"{joint_name}.pos"
            action[key] = robot_obs.get(key, 0.0)
        return action
    
    def _get_noop_right_arm_action(self, robot_obs: dict[str, Any]) -> dict[str, Any]:
        """Generate no-op action for right arm (current positions)."""
        action = {}
        for joint_name in RIGHT_JOINT_MAP.values():
            key = f"{joint_name}.pos"
            action[key] = robot_obs.get(key, 0.0)
        return action
    
    def _get_noop_head_action(self, robot_obs: dict[str, Any]) -> dict[str, Any]:
        """Generate no-op action for head (current positions)."""
        action = {}
        for motor_name in ["head_pan", "head_tilt"]:
            key = f"{motor_name}.pos"
            action[key] = robot_obs.get(key, 0.0)
        return action
    
    def get_action(self) -> dict[str, Any]:
        """Get VR control action with detailed profiling"""
        total_start = time.perf_counter()
        
        action = {}
        
        # Quick check VR monitoring status and robot reference
        if not self.vr_monitor or self.robot is None:
            # Return no-op action (current positions) if VR/robot not ready
            try:
                robot_obs = self.robot.get_observation() if self.robot else {}
                action = self._get_noop_action(robot_obs)
            except:
                action = {}
            self.logs["read_pos_dt_s"] = time.perf_counter() - total_start
            return action
        
        # Get VR data once to avoid repeated calls
        vr_start = time.perf_counter()
        try:
            dual_goals = self.vr_monitor.get_latest_goal_nowait()
            if dual_goals is None:
                # Return no-op action (current positions) if no VR data yet
                try:
                    robot_obs = self.robot.get_observation()
                    action = self._get_noop_action(robot_obs)
                except:
                    action = {}
                self.logs["read_pos_dt_s"] = time.perf_counter() - total_start
                return action
                
            left_goal = dual_goals.get("left")
            right_goal = dual_goals.get("right")
            headset_goal = dual_goals.get("headset")
            
        except Exception as e:
            logger.warning(f"VR data acquisition failed: {e}")
            # Return no-op action (current positions) on VR data failure
            try:
                robot_obs = self.robot.get_observation()
                action = self._get_noop_action(robot_obs)
            except:
                action = {}
            self.logs["read_pos_dt_s"] = time.perf_counter() - total_start
            return action
        vr_dt_ms = (time.perf_counter() - vr_start) * 1e3
        logger.debug(f"🎮 VR data fetch: {vr_dt_ms:.1f}ms")
        
        # Get current robot observation with caching to avoid double reads
        # lerobot_teleoperate.py calls robot.get_observation() then teleop.get_action()
        # This cache prevents reading twice in the same loop iteration
        obs_start = time.perf_counter()
        current_time = time.perf_counter()
        
        # Use cached observation if recent (within 10ms)
        if (self._cached_obs is not None and 
            (current_time - self._obs_cache_time) < self._obs_cache_duration):
            robot_obs = self._cached_obs
            obs_dt_ms = (time.perf_counter() - obs_start) * 1e3
            logger.debug(f"🤖 Robot observation (cached): {obs_dt_ms:.2f}ms")
        else:
            # Read fresh observation and cache it
            try:
                robot_obs = self.robot.get_observation()
                self._cached_obs = robot_obs
                self._obs_cache_time = current_time
            except Exception as e:
                logger.warning(f"Failed to get robot observation: {e}")
                # Return no-op action (empty dict) if observation fails
                action = {}
                self.logs["read_pos_dt_s"] = time.perf_counter() - total_start
                return action
            obs_dt_ms = (time.perf_counter() - obs_start) * 1e3
            logger.debug(f"🤖 Robot observation (fresh): {obs_dt_ms:.1f}ms")
        
        # IK and control computation
        ik_start = time.perf_counter()
        try:
            current_time = time.perf_counter()
            
            # Robot control - high frequency execution
            if left_goal is not None and self.left_arm is not None:
                self.left_arm.handle_vr_input(left_goal, None)
                
            if right_goal is not None and self.right_arm is not None:
                self.right_arm.handle_vr_input(right_goal, None)
            
            # Head control - process headset orientation
            if headset_goal is not None and self.head_control is not None:
                self.head_control.handle_vr_input(headset_goal)
                # Log headset data occasionally for debugging
                if not hasattr(self, '_last_headset_log'):
                    self._last_headset_log = 0
                if (current_time - self._last_headset_log) >= 2.0:  # Log every 2 seconds
                    if hasattr(headset_goal, 'metadata'):
                        pan = headset_goal.metadata.get('head_pan', 0.0)
                        tilt = headset_goal.metadata.get('head_tilt', 0.0)
                        logger.debug(f"🎧 Headset: pan={pan:.1f}°, tilt={tilt:.1f}° -> targets: pan={self.head_control.target_positions['head_pan']:.1f}, tilt={self.head_control.target_positions['head_tilt']:.1f}")
                    self._last_headset_log = current_time
            elif headset_goal is None and self.head_control is not None:
                # Log occasionally if headset data is missing
                if not hasattr(self, '_last_headset_warning'):
                    self._last_headset_warning = 0
                if (current_time - self._last_headset_warning) >= 5.0:  # Warn every 5 seconds
                    logger.warning("⚠️  No headset data received - head control will maintain current position")
                    self._last_headset_warning = current_time
            
            # Event processing - optimized frequency (10Hz)
            if (current_time - self.last_event_update_time) >= 0.1:
                if left_goal is not None or right_goal is not None:
                    self._update_events_inline(left_goal, right_goal)
                self.last_event_update_time = current_time
            
            # Generate action dictionary - ensure all required keys are present
            left_action = self.left_arm.p_control_action(robot_obs) if self.left_arm is not None else self._get_noop_left_arm_action(robot_obs)
            right_action = self.right_arm.p_control_action(robot_obs) if self.right_arm is not None else self._get_noop_right_arm_action(robot_obs)
            head_action = self.head_control.p_control_action(robot_obs) if self.head_control is not None else self._get_noop_head_action(robot_obs)
            
            # Base control - ALWAYS try to get base action (even if goals are None)
            base_action = get_vr_base_action(left_goal, right_goal, self.robot)
            
            # Log base action for debugging (only if non-zero, and only occasionally to reduce overhead)
            if not hasattr(self, '_last_base_log_time'):
                self._last_base_log_time = 0
            current_time = time.perf_counter()
            if (current_time - self._last_base_log_time) >= 1.0:  # Log at most once per second
                if base_action.get("x.vel", 0) != 0 or base_action.get("y.vel", 0) != 0 or base_action.get("theta.vel", 0) != 0:
                    logger.debug(f"🕹️  VR Base action: x={base_action['x.vel']:.3f}, y={base_action['y.vel']:.3f}, theta={base_action['theta.vel']:.1f}")
            self._last_base_log_time = current_time
            
            # Merge actions
            action.update(left_action)
            action.update(right_action)
            action.update(head_action)
            action.update(base_action)
            
            # Verify base action was merged (debug)
            if "x.vel" in action or "y.vel" in action or "theta.vel" in action:
                logger.debug(f"✓ Base velocities in final action: x={action.get('x.vel', 0):.3f}, y={action.get('y.vel', 0):.3f}, theta={action.get('theta.vel', 0):.1f}")
            
        except Exception as e:
            logger.error(f"Action generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return no-op action (current positions) on exception
            try:
                action = self._get_noop_action(robot_obs)
            except:
                action = {}
        
        ik_dt_ms = (time.perf_counter() - ik_start) * 1e3
        logger.debug(f"🧮 IK + control: {ik_dt_ms:.1f}ms")
        
        total_dt_ms = (time.perf_counter() - total_start) * 1e3
        logger.debug(f"⏱️  TOTAL get_action: {total_dt_ms:.1f}ms")
        logger.debug(f"=" * 60)
        
        self.logs["read_pos_dt_s"] = time.perf_counter() - total_start
        return action
    
    def _update_events_inline(self, left_goal, right_goal=None):
        """
        Low frequency event update - 10Hz frequency, reuse already acquired VR goals.
        Only execute when event interval time is reached, greatly reducing processing overhead.
        """
        if not self.vr_event_handler:
            return

        # Directly use already acquired data, no need to call VR interface again
        try:
            if left_goal is not None and hasattr(left_goal, 'metadata'):
                self.vr_event_handler._process_left_controller(left_goal.metadata)
            if right_goal is not None and hasattr(right_goal, 'metadata'):
                self.vr_event_handler._process_right_controller(right_goal.metadata)
        except Exception as e:
            logger.debug(f"Low frequency event update failed: {e}")  # Downgrade to debug to avoid disrupting main flow

    def send_feedback(self) -> None:
        """Send feedback - optimized version, reduce blocking wait"""
        if not self.vr_monitor:
            logger.warning("VR monitor not available for feedback")
            return

        max_attempts = 200  # Maximum 200 attempts
        attempt = 0
        
        while attempt < max_attempts:
            try:
                dual_goals = self.vr_monitor.get_latest_goal_nowait()
                if dual_goals and sum(dual_goals.get('right').metadata['vr_position']):
                    logger.info("VR controller data received")
                    return
                    
            except Exception as e:
                logger.warning(f"Error getting VR data: {e}")
            
            attempt += 1
            logger.info(f'Waiting for VR controller data (attempt {attempt}/{max_attempts})')
            time.sleep(0.5)  # Reduce wait time from 8 seconds to 0.5 seconds
        
        logger.warning("Timeout waiting for VR controller data")

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        """Disconnect VR connection"""
        if not self.is_connected:
            raise RuntimeError(
                "XLerobot VR is not connected."
            )
        
        try:
            if self.vr_monitor:
                # VR Monitor usually runs in a thread, stop the thread
                pass
            
            self._connected = False
            self._calibrated = False
            print("[VR] Disconnected")
            
        except Exception as e:
            print(f"[VR] Error during disconnect: {e}")

    def move_to_zero_position(self, robot):
        """Move all controllers to zero position"""
        robot_obs = robot.get_observation()
        action = {}
        left_action = self.left_arm.move_to_zero_position(robot_obs)
        right_action = self.right_arm.move_to_zero_position(robot_obs)
        base_action = get_vr_base_action(None, None, robot)
        action.update(left_action)
        action.update(right_action)
        action.update(base_action)

        return action
    
    def get_vr_events(self):
        """Get VR event status.

        IMPORTANT: `lerobot_record.record_loop` checks `events["exit_early"]` *before* calling
        `teleop.get_action()` in each control-loop iteration.

        If we only update VR button events inside `get_action()` (even at 10Hz), there can be a
        1-iteration delay where a button press is detected late and can end the *next* episode
        right after it starts (appearing like episode overlap).

        To prevent cross-episode spillover, we refresh events here using the latest VR packet.
        """
        if self.vr_event_handler:
            # Refresh events from the latest VR goals so record_loop sees them immediately.
            try:
                if self.vr_monitor is not None:
                    dual_goals = self.vr_monitor.get_latest_goal_nowait()
                    if isinstance(dual_goals, dict):
                        self._update_events_inline(dual_goals.get("left"), dual_goals.get("right"))
            except Exception as e:
                logger.debug(f"VR event refresh failed (non-critical): {e}")

            # Get current event status (copy)
            events = self.vr_event_handler.get_events()
            
            # Automatically reset exit_early to prevent infinite loops
            # But keep rerecord_episode until it's processed in the recording loop
            if events.get("exit_early", False):
                self.vr_event_handler.events["exit_early"] = False
            
            return events
        else:
            # Return default event status
            return {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
                "reset_position": False,
                "back_position": False,
            }
    
    def reset_vr_events(self):
        """Reset VR event status"""
        if self.vr_event_handler:
            self.vr_event_handler.reset_events()
    
    def print_vr_control_guide(self):
        """Print VR control guide"""
        if self.vr_event_handler:
            self.vr_event_handler.print_control_guide()
        else:
            logger.info("VR event handler not initialized")


def init_vr_listener(teleop_vr):
    """
    Initialize VR listener, providing the same interface as init_keyboard_listener
    Used to replace keyboard event listening, used in record.py
    
    Args:
        teleop_vr: XLerobotVRTeleop instance
        
    Returns:
        tuple: (listener, events) - same return format as init_keyboard_listener
    """
    if not isinstance(teleop_vr, XLerobotVRTeleop):
        logger.error("teleop_vr must be an XLerobotVRTeleop instance")
        return None, {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "reset_position": False,
            "back_position": False,
        }
    
    # Print control guide
    teleop_vr.print_vr_control_guide()
    
    # Create virtual listener object (compatible with keyboard listener)
    class VRListener:
        def __init__(self, teleop_vr):
            self.teleop_vr = teleop_vr
            self.is_alive = True
            
        def stop(self):
            self.is_alive = False
            logger.info("VR listener stopped")
    
    vr_listener = VRListener(teleop_vr)
    
    # Get initial event status
    events = teleop_vr.get_vr_events()
    
    return vr_listener, events

class VREventHandler:
    """
    VR event handler, specifically handles recording control events
    Use left VR controller to replace keyboard control
    """
    
    def __init__(self, vr_monitor):
        self.vr_monitor = vr_monitor
        self.events = {
            "exit_early": False,      # Left controller right: Exit loop early (original right arrow key)
            "rerecord_episode": False, # Left controller left: Re-record episode (original left arrow key)
            "stop_recording": False,   # Left controller up: Stop recording (original ESC key)
            "reset_position": False,   # Left controller down: Reset robot (new feature)
            "back_position": False,    # In the bucket (new feature)
        }
        self.prev_states = {
            'thumbstick_x': 0,
            'thumbstick_y': 0,
            'trigger': False,
            # Left controller buttons
            'button_x': False,
            'button_y': False,
            'button_thumbstick': False,
            'button_menu': False,
            # Right controller buttons
            'right_button_b': False,
            # Snapshots for debug
            'buttons_snapshot': {},
            'right_buttons_snapshot': {},
        }
        self.threshold = 0.7  # Thumbstick trigger threshold
        
    def update_events(self):
        """Update VR event status"""
        if not self.vr_monitor:
            return self.events
            
        try:
            dual_goals = self.vr_monitor.get_latest_goal_nowait()
            if not dual_goals:
                return self.events
                
            left_goal = dual_goals.get("left")
            if not left_goal or not hasattr(left_goal, 'metadata'):
                return self.events
                
            self._process_left_controller(left_goal.metadata)
            
        except Exception as e:
            logger.error(f"VR事件更新失败: {e}")
            
        return self.events
    
    def _process_left_controller(self, metadata):
        """处理左手柄输入"""
        # Get button states from left controller
        # Web UI sends for left controller:
        #   x: physical X, y: physical Y
        buttons = metadata.get('buttons', {}) or {}
        button_x = bool(buttons.get('x', False))
        button_y = bool(buttons.get('y', False))
        button_thumbstick = bool(buttons.get('thumbstick', False))
        button_menu = bool(buttons.get('menu', False))

        # Log raw left button states whenever they change, for mapping debug
        prev_buttons_snapshot = self.prev_states.get('buttons_snapshot', {})
        if buttons != prev_buttons_snapshot:
            logger.debug(f"🎮 LEFT raw buttons: {buttons}")

        # Episode control mapping (requested):
        #   - LEFT X button -> Restart (re-record) current episode
        #   - LEFT Y button -> Restart (re-record) current episode
        if (button_x and not self.prev_states.get('button_x', False)) or (
            button_y and not self.prev_states.get('button_y', False)
        ):
            which = "X" if button_x else "Y"
            logger.info(f"🎮 VR LEFT {which} button pressed -> Re-record current episode")
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True

        # IMPORTANT: Do NOT map thumbstick *movement* to episode control events.
        # The left thumbstick X axis is used for base rotation, so mapping it to `exit_early`
        # causes accidental episode termination when rotating the robot.
        #
        # Explicit buttons only:
        # - LEFT Menu -> Stop recording (sticky)
        # - LEFT Thumbstick click -> Reset robot (instantaneous)
        if button_menu and not self.prev_states.get('button_menu', False):
            logger.info("🎮 VR LEFT menu button pressed -> Stop recording")
            self.events["stop_recording"] = True

        if button_thumbstick and not self.prev_states.get('button_thumbstick', False):
            logger.debug("🎮 VR LEFT thumbstick button pressed -> Reset robot")
            self.events["reset_position"] = True
        else:
            self.events["reset_position"] = False  # Reset event is instantaneous
            self.events["back_position"] = False
        
        # Detect trigger key events
        trigger = metadata.get('trigger', 0) > 0.5
        
        # Update status
        self.prev_states.update({
            'trigger': trigger,
            'button_x': button_x,
            'button_y': button_y,
            'button_thumbstick': button_thumbstick,
            'button_menu': button_menu,
            'buttons_snapshot': buttons,
        })

    def _process_right_controller(self, metadata):
        """Process right-hand controller input for episode control (B button = finish early)."""
        buttons = metadata.get('buttons', {}) or {}
        button_b = bool(buttons.get('b', False))  # Right B button

        prev_right_snapshot = self.prev_states.get('right_buttons_snapshot', {})
        if buttons != prev_right_snapshot:
            logger.debug(f"🎮 RIGHT raw buttons: {buttons}")

        # Requested mapping:
        #   - RIGHT B button -> finish episode early (save and continue to next episode)
        if button_b and not self.prev_states.get('right_button_b', False):
            logger.info("🎮 VR RIGHT B button pressed -> Finish episode early")
            self.events["exit_early"] = True

        self.prev_states['right_button_b'] = button_b
        self.prev_states['right_buttons_snapshot'] = buttons
    
    def reset_events(self):
        """Reset all event status"""
        for key in self.events:
            self.events[key] = False
        # Note: We don't reset prev_states here to maintain edge detection for buttons
    
    def get_events(self):
        """Get current event status"""
        return self.events.copy()
    
    def print_control_guide(self):
        """Print VR control guide"""
        guide = """
        ╔══════════════════════════════════════════════════════════════╗
        ║          🎮 XLerobot VR Control Guide 🎮                    ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  ARM CONTROL (Both controllers):                             ║
        ║  ├─ Grip button: Hold to control arm position                ║
        ║  ├─ Trigger: Gripper open (squeeze) / closed (release)       ║
        ║  └─ Move controller: Control arm position & wrist rotation   ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  BASE CONTROL (Thumbsticks - works anytime):                 ║
        ║  ├─ RIGHT thumbstick ↑↓: Forward / Backward                  ║
        ║  ├─ RIGHT thumbstick ←→: Left / Right lateral                ║
        ║  └─ LEFT thumbstick ←→: Rotate left / Rotate right           ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  EPISODE CONTROL (Buttons only):                              ║
        ║  ├─ LEFT X/Y button: Restart (re-record) current episode      ║
        ║  ├─ RIGHT B button: Finish episode early (save & next)        ║
        ║  ├─ LEFT Menu button: Stop recording                          ║
        ║  └─ LEFT Thumbstick click: Reset robot                        ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        logger.info(guide)