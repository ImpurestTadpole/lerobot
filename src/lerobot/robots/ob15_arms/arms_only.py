"""Keys and command building for the arms-only OB15 robot (``ob15_arms`` / ``ob15_arms_client``).

The robot exposes 12 actions (both arms + grippers). Head pan/tilt and the lift are moved to a hold pose
(see ``hold.py``) and then held, and no base command is ever sent, so those keys are never part of the
action space even though the head/lift stay in the observation.
"""

from __future__ import annotations

from typing import Any

ARM_KEYS: tuple[str, ...] = (
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
)
HEAD_KEYS: tuple[str, ...] = ("head_pan.pos", "head_tilt.pos")
BASE_KEYS: tuple[str, ...] = ("x.vel", "y.vel", "theta.vel")

# Defaults for "has the head/lift reached the hold pose".
HEAD_TOL_DEG = 2.0
LIFT_TOL_MM = 5.0


def lift_key(lift_name: str) -> str:
    return f"{lift_name}.height_mm"


def arms_action_features() -> dict[str, type]:
    return dict.fromkeys(ARM_KEYS, float)


def arms_state_features(full_state_ft: dict[str, Any]) -> dict[str, Any]:
    """Drop the base-velocity keys from a full-body state feature dict, keeping order."""
    return {k: v for k, v in full_state_ft.items() if k not in BASE_KEYS}


def hold_keys(lift_name: str | None) -> tuple[str, ...]:
    return (*HEAD_KEYS, lift_key(lift_name)) if lift_name else HEAD_KEYS


def rest_pose_command(config: Any) -> dict[str, float] | None:
    """Arm/gripper targets to send before the head/lift move, or None if no ``rest_pose``."""
    rest = getattr(config, "rest_pose", None)
    if not rest:
        return None
    bad = sorted(k for k in rest if k not in ARM_KEYS)
    if bad:
        raise ValueError(f"rest_pose may only contain arm/gripper keys {list(ARM_KEYS)}; got {bad}.")
    return {k: float(v) for k, v in rest.items()}


def build_arms_command(action: dict[str, Any], hold: dict[str, float]) -> dict[str, float]:
    """Arm keys from ``action`` + the hold targets. No base command is included.

    Any base/head/lift keys in ``action`` are ignored. ``OB15.send_action`` reads a missing base key as
    0.0 every tick (the smoothed base velocity decays to zero), so leaving the base out keeps the
    wheels at zero velocity without arms-only code driving the base.
    """
    command: dict[str, float] = {k: float(action[k]) for k in ARM_KEYS if k in action}
    command.update(hold)
    return command


def arms_only_result(sent: dict[str, Any]) -> dict[str, Any]:
    """Keep only the 12 arm keys from a send_action result."""
    return {k: sent[k] for k in ARM_KEYS if k in sent}
