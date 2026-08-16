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

import math

import numpy as np
import pytest

from lerobot.data_processing.joint_to_ee import (
    PROFILES,
    ArmGroup,
    ClassicSO101Solver,
    EeFormat,
    JointUnits,
    UrdfSolver,
    convert_joint_vector,
    infer_profile_from_names,
)
from tests.utils import skip_if_package_missing


def _left_arm_joints(**overrides: float) -> dict[str, float]:
    defaults = {
        "left_arm_shoulder_pan": 10.0,
        "left_arm_shoulder_lift": -45.0,
        "left_arm_elbow_flex": 60.0,
        "left_arm_wrist_flex": -15.0,
        "left_arm_wrist_roll": 5.0,
        "left_arm_gripper": 50.0,
    }
    defaults.update(overrides)
    return defaults

def test_infer_profile_xlerobot_full():
    src = [
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
        "gantry.height_mm",
    ]
    profile = infer_profile_from_names(src)
    assert profile is not None
    assert profile.pass_through_names == PROFILES["xlerobot_pass_through"].pass_through_names


def test_classic_fk_output_shape():
    solver = ClassicSO101Solver(ee_format=EeFormat.EULER)
    arm = PROFILES["bimanual12"].arms[0]
    joints = _left_arm_joints()
    ee = solver.arm_joints_to_ee(joints, arm)
    assert ee.shape == (7,)
    assert np.isfinite(ee).all()


def test_classic_fk_workspace_bounds():
    """Classic FK reach should stay within the SO-101 link-length workspace."""
    solver = ClassicSO101Solver(ee_format=EeFormat.EULER)
    arm = PROFILES["bimanual12"].arms[0]
    joints = _left_arm_joints(
        left_arm_shoulder_pan=5.0,
        left_arm_shoulder_lift=-50.0,
        left_arm_elbow_flex=70.0,
        left_arm_wrist_flex=-20.0,
        left_arm_wrist_roll=10.0,
        left_arm_gripper=80.0,
    )
    ee = solver.arm_joints_to_ee(joints, arm)
    kin = solver.kinematics
    reach = math.hypot(float(ee[0]), float(ee[1]))
    assert reach <= kin.l1 + kin.l2 + 1e-3
    assert reach >= abs(kin.l1 - kin.l2) - 1e-3
    assert ee[6] == pytest.approx(0.8)  # gripper normalized from 80


def test_convert_joint_vector_pass_through():
    src_names = [
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
        "gantry.height_mm",
    ]
    vec = np.arange(len(src_names), dtype=np.float32)
    profile = PROFILES["xlerobot_pass_through"]
    solver = ClassicSO101Solver(ee_format=EeFormat.ROTVEC)
    out = convert_joint_vector(
        vec,
        src_names,
        profile,
        solver,
        EeFormat.ROTVEC,
        JointUnits.DEGREES,
        norm_scale=1.8,
    )
    assert len(out) == len(profile.output_state_names(EeFormat.ROTVEC))
    # Pass-through tail should match source indices 12–17
    assert out[-6] == pytest.approx(12.0)
    assert out[-1] == pytest.approx(17.0)


@skip_if_package_missing("placo")
def test_urdf_fk_matches_classic_planar_reach():
    """URDF and classic solvers should agree on planar reach magnitude within tolerance."""
    from lerobot.data_processing.joint_to_ee import ensure_so101_urdf

    joints = _left_arm_joints(
        left_arm_shoulder_pan=0.0,
        left_arm_shoulder_lift=-50.0,
        left_arm_elbow_flex=70.0,
        left_arm_wrist_roll=0.0,
        left_arm_wrist_flex=-(-50.0 + 70.0),  # pitch ≈ 0
    )
    arm = ArmGroup(name="", joint_names=tuple(jn.replace("left_arm_", "") for jn in PROFILES["bimanual12"].arms[0].joint_names))
    joints_bare = {jn: joints[f"left_arm_{jn}"] for jn in arm.joint_names}

    classic = ClassicSO101Solver(ee_format=EeFormat.EULER)
    classic_ee = classic.arm_joints_to_ee(joints_bare, arm)

    urdf = UrdfSolver(ee_format=EeFormat.EULER, urdf_path=ensure_so101_urdf())
    urdf_ee = urdf.arm_joints_to_ee(joints_bare, arm)

    classic_r = math.hypot(float(classic_ee[0]), float(classic_ee[1]))
    urdf_r = math.hypot(float(urdf_ee[0]), float(urdf_ee[1]))
    assert abs(classic_r - urdf_r) < 0.05
