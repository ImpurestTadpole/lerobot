#!/usr/bin/env python3
"""
Lift axis deadband calibration for OB15.

Finds the minimum motor velocity that reliably moves the lift in each
direction under actual load.  Run this once after assembly (and again
after any mechanical change — lubrication, arm payload change, etc.).

Usage
-----
    conda activate lerobot
    python deadband_calibration.py

The robot must already be homed.  Run with both arms attached at their
typical operating weight.

Output
------
Prints recommended values for:
    v_min_effective_up   (LiftAxisConfig field, raw ticks)
    v_min_effective_down (LiftAxisConfig field, raw ticks)

Copy the printed values into config_xlerobot.py (LiftAxisConfig defaults)
or pass them via your robot JSON config at ~/.config/lerobot/robots/ob15.json.
"""

from __future__ import annotations

import argparse
import time

from lerobot.robots.ob15.config_ob15 import OB15Config
from lerobot.robots.utils import make_robot_from_config


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mm_to_travel(robot) -> float:
    """Return current lift height in mm (from robot's LiftAxis)."""
    return robot.lift_axis.get_height_mm()


def test_velocity(robot, raw_velocity: int, duration_s: float = 0.5) -> float:
    """
    Command a raw velocity for `duration_s`, return mm of travel observed.
    Positive raw_velocity = physically ascending with home_at_top=True.
    """
    name = robot.lift_axis.cfg.name
    bus = robot.lift_axis._bus
    sign = -1 if robot.lift_axis.cfg.home_at_top else 1

    before_mm = _mm_to_travel(robot)
    bus.write("Goal_Velocity", name, int(sign * robot.lift_axis.cfg.dir_sign * raw_velocity))
    time.sleep(duration_s)
    bus.write("Goal_Velocity", name, 0)
    time.sleep(0.2)
    after_mm = _mm_to_travel(robot)
    return abs(after_mm - before_mm)


def find_v_min(robot, direction: int, low: int = 50, high: int = 400) -> int:
    """
    Binary search for the minimum raw velocity that moves the lift >= 1 mm
    in `direction` (+1 = increasing mm / physically descending, -1 = vice versa).

    Returns the smallest velocity at which the lift reliably moves.
    """
    print(f"\n  Binary search ({'+mm' if direction > 0 else '-mm'} direction) "
          f"in range [{low}, {high}] raw ticks ...")

    while high - low > 5:
        mid = (low + high) // 2
        motion = test_velocity(robot, direction * mid, duration_s=0.5)
        symbol = "OK" if motion >= 1.0 else "STALL"
        print(f"    vel={direction * mid:+5d}  motion={motion:.2f} mm  [{symbol}]")
        if motion < 1.0:
            low = mid       # stalled — need more velocity
        else:
            high = mid      # moved — try less

        # Guard: stay within soft limits
        cur = _mm_to_travel(robot)
        cfg = robot.lift_axis.cfg
        if cur <= cfg.soft_min_mm + 20 and direction < 0:
            print("  WARNING: approaching top limit — stopping search early")
            break
        if cur >= cfg.soft_max_mm - 20 and direction > 0:
            print("  WARNING: approaching bottom limit — stopping search early")
            break

    # Return to mid-range so subsequent search has room
    print("  Returning lift to mid-range ...")
    robot.lift_axis.apply_action({f"{robot.lift_axis.cfg.name}.height_mm": 280.0})
    _wait_for_settle(robot, 280.0, timeout_s=30.0)

    return high


def _wait_for_settle(robot, target_mm: float, timeout_s: float = 30.0) -> None:
    """Poll until lift is within 5 mm of target or timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cur = _mm_to_travel(robot)
        if abs(cur - target_mm) < 5.0:
            return
        robot.lift_axis.apply_action({f"{robot.lift_axis.cfg.name}.height_mm": target_mm})
        time.sleep(0.05)
    print(f"  WARNING: settle timeout (cur={_mm_to_travel(robot):.1f} mm)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="OB15 lift deadband calibration")
    parser.add_argument(
        "--low", type=int, default=50, help="Lower bound of binary search (raw ticks)"
    )
    parser.add_argument(
        "--high", type=int, default=400, help="Upper bound of binary search (raw ticks)"
    )
    parser.add_argument(
        "--duration", type=float, default=0.5, help="Test pulse duration in seconds"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OB15 Lift Deadband Calibration")
    print("=" * 60)
    print("Connecting to robot ...")

    cfg = OB15Config()
    cfg.lift_axis.home_on_connect = True
    robot = make_robot_from_config(cfg)
    robot.connect()

    if not robot.lift_axis.enabled:
        print("ERROR: lift axis is disabled in config. Aborting.")
        robot.disconnect()
        return

    print("\nRobot connected and homed.")
    print(f"Lift height: {_mm_to_travel(robot):.1f} mm")
    print(f"\nSearch range: [{args.low}, {args.high}] raw ticks")
    print(f"Test pulse: {args.duration} s per velocity step")
    print("\nIMPORTANT: ensure both arms are attached at typical operating weight.\n")

    input("Press ENTER to begin (Ctrl-C to abort) ... ")

    # Move to middle of range first so both searches have room
    print("\nMoving to mid-range (280 mm) ...")
    robot.lift_axis.apply_action({f"{robot.lift_axis.cfg.name}.height_mm": 280.0})
    _wait_for_settle(robot, 280.0, timeout_s=30.0)
    print(f"At {_mm_to_travel(robot):.1f} mm. Starting calibration.\n")

    # Search ascending (decreasing mm = against gravity with home_at_top=True)
    print("--- ASCENDING (decreasing mm, against gravity) ---")
    v_min_up = find_v_min(robot, direction=-1, low=args.low, high=args.high)

    # Search descending (increasing mm = gravity assists with home_at_top=True)
    print("\n--- DESCENDING (increasing mm, gravity assists) ---")
    v_min_down = find_v_min(robot, direction=+1, low=args.low, high=args.high)

    print("\n" + "=" * 60)
    print("Calibration complete.")
    print("=" * 60)
    print(f"\n  v_min_effective_up:   {v_min_up}")
    print(f"  v_min_effective_down: {v_min_down}")
    print("""
Copy these into LiftAxisConfig (lift_axis.py) or your robot JSON:

    "lift_axis": {{
        "v_min_effective_up":   {v_min_up},
        "v_min_effective_down": {v_min_down}
    }}

Re-run after any mechanical change (lubrication, arm payload, etc.).
""".format(v_min_up=v_min_up, v_min_down=v_min_down))

    robot.lift_axis.stop()
    robot.disconnect()


if __name__ == "__main__":
    main()
