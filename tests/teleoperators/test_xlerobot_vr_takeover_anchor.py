# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""A VR takeover must start from the arm's current pose, not a fixed home.

`SimpleTeleopArm` is rebuilt on every DAgger intervention start (calibrate()). Its IK state used to
start at a fixed home (x=0.1629, y=0.1131, pitch=0), so the first VR delta pulled the arm toward that
home and the jump was recorded as part of the human correction.
"""

from types import SimpleNamespace

import pytest

from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.teleoperators.xlerobot_vr.xlerobot_vr import (
    RIGHT_JOINT_MAP,
    SimpleTeleopArm,
    _cartesian_from_joints,
)

POSES = [(-53.0, 33.0, 57.0), (-20.0, 60.0, 40.0), (10.0, 40.0, 70.0), (-80.0, 70.0, 20.0)]


def _obs(shoulder_lift, elbow_flex, wrist_flex):
    return {
        "right_arm_shoulder_pan.pos": -30.0,
        "right_arm_shoulder_lift.pos": shoulder_lift,
        "right_arm_elbow_flex.pos": elbow_flex,
        "right_arm_wrist_flex.pos": wrist_flex,
        "right_arm_wrist_roll.pos": 2.0,
        "right_arm_gripper.pos": 30.0,
    }


@pytest.mark.parametrize("shoulder_lift,elbow_flex,_wrist", POSES)
def test_cartesian_anchor_inverts_the_teleop_ik(shoulder_lift, elbow_flex, _wrist):
    kin = SO101Kinematics()
    x, y = _cartesian_from_joints(kin, shoulder_lift, elbow_flex)
    j2, j3 = kin.inverse_kinematics(x, y)
    assert j2 == pytest.approx(shoulder_lift, abs=0.05)
    assert j3 == pytest.approx(elbow_flex, abs=0.05)


@pytest.mark.parametrize("shoulder_lift,elbow_flex,wrist_flex", POSES)
def test_first_vr_frames_after_takeover_do_not_pull_the_arm_home(shoulder_lift, elbow_flex, wrist_flex):
    arm = SimpleTeleopArm(RIGHT_JOINT_MAP, _obs(shoulder_lift, elbow_flex, wrist_flex), SO101Kinematics())
    start = dict(arm.target_positions)
    # Controller held still: the first frame sets the baseline, the rest carry zero motion.
    goal = SimpleNamespace(
        target_position=[0.3, 1.2, -0.4], wrist_flex_deg=10.0, wrist_roll_deg=5.0, metadata={}
    )
    for _ in range(20):
        arm.handle_vr_input(goal, None)
    for joint in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        assert arm.target_positions[joint] == pytest.approx(start[joint], abs=0.5), joint
