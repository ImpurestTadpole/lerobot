from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import OperatingMode


class BusLike(Protocol):
    motors: Dict[str, object]

    def read(self, item: str, name: str, *, normalize: bool = True) -> float: ...

    def write(self, item: str, name: str, value: float) -> None: ...

    def sync_write(self, item: str, values: Dict[str, float]) -> None: ...


@dataclass
class LiftAxisConfig:
    """
    Optional gantry / Z lift axis driven by a Feetech servo in VELOCITY mode.

    The lift is controlled in *mm* via a small P controller that outputs a velocity command.
    Multi-turn motion is estimated by tracking Present_Position wrap-around (0..4095).
    """

    enabled: bool = True

    # How the axis is exposed to actions/observations.
    name: str = "gantry"

    # Where the motor is connected (bus2 = same port as right arm + head).
    bus: str = "bus2"  # "bus1" (left/base port) or "bus2" (right/head port)
    motor_id: int = 9
    motor_model: str = "sts3215"

    # Distance measurement: servo ticks (0..4095/rev) -> degrees -> mm.
    # Formula: height_mm = (extended_deg - z0_deg) * mm_per_deg, where
    #   mm_per_deg = (lead_mm_per_rev * output_gear_ratio) / 360.
    # Adjust lead_mm_per_rev to match your lead screw (mm of travel per motor revolution).
    #lead_mm_per_rev: float = 84.0  # mm of linear travel per motor rev (e.g. 8, 12, 20 for common leads)
    lead_mm_per_rev: float = 100.0 
    output_gear_ratio: float = 1.0  # If motor has reduction to screw: screw_rev/motor_rev (e.g. 0.5 for 2:1)

    # Height convention. Set home_at_top=True for: 0 = top (home), soft_max_mm = bottom.
    home_at_top: bool = True  # True: 0=top, bottom=e.g. 500mm; False: 0=bottom, top=600mm
    soft_min_mm: float = 0.0
    soft_max_mm: float = 525.0  # When home_at_top, set to 500 so bottom = 500mm

    # Homing: drive toward home (down if not home_at_top, up if home_at_top) until stall.
    home_down_speed: int = 1300
    home_stall_current_ma: int = 150
    home_backoff_deg: float = 5.0

    # Closed-loop on height -> velocity. Increase v_max and/or kp_vel for faster lift.
    kp_vel: float = 300.0
    v_max: int = 2000  # Max raw velocity to servo; raise (e.g. 1800) for higher speed
    on_target_mm: float = 1.0

    # Direction conventions
    dir_sign: int = 1  # +1 no inversion; -1 invert direction


class LiftAxis:
    """Gantry/Z-axis controller that plugs into an existing Feetech bus."""

    def __init__(self, cfg: LiftAxisConfig, bus1: Optional[BusLike], bus2: Optional[BusLike]):
        self.cfg = cfg
        self._bus = bus1 if cfg.bus == "bus1" else bus2
        self.enabled = bool(cfg.enabled and self._bus is not None)

        self._ticks_per_rev = 4096.0
        self._deg_per_tick = 360.0 / self._ticks_per_rev
        self._mm_per_deg = (cfg.lead_mm_per_rev * cfg.output_gear_ratio) / 360.0

        # Multi-turn tick tracking
        self._last_tick: float = 0.0
        self._extended_ticks: float = 0.0
        self._z0_deg: float = 0.0
        self._configured: bool = False
        
        # Cache current state to avoid redundant serial reads in apply_action
        self._cached_height_mm: float = 0.0
        self._cached_velocity: float = 0.0
        
        # Velocity filtering to reduce sensor noise and improve training data quality
        self._velocity_alpha: float = 0.3  # EMA smoothing factor (0=no filter, 1=no smoothing)
        self._filtered_velocity: float = 0.0

    def attach(self) -> None:
        """Registers the motor name on the selected bus (safe pre-connect)."""
        if not self.enabled:
            return
        if self.cfg.name not in self._bus.motors:
            self._bus.motors[self.cfg.name] = Motor(self.cfg.motor_id, self.cfg.motor_model, MotorNormMode.DEGREES)

    def configure(self) -> None:
        """Configures the motor in VELOCITY mode and resets wrap tracking."""
        if not self.enabled or self._configured:
            return
        self._bus.write("Operating_Mode", self.cfg.name, OperatingMode.VELOCITY.value)
        self._last_tick = float(self._bus.read("Present_Position", self.cfg.name, normalize=False))
        self._extended_ticks = 0.0
        self._configured = True

    def _update_extended_ticks(self) -> None:
        if not self.enabled:
            return
        cur = float(self._bus.read("Present_Position", self.cfg.name, normalize=False))  # 0..4095
        delta = cur - self._last_tick
        half = self._ticks_per_rev * 0.5
        if delta > +half:
            delta -= self._ticks_per_rev
        elif delta < -half:
            delta += self._ticks_per_rev
        self._extended_ticks += delta
        self._last_tick = cur

    def _extended_deg(self) -> float:
        return self.cfg.dir_sign * self._extended_ticks * self._deg_per_tick

    def get_height_mm(self) -> float:
        if not self.enabled:
            return 0.0
        self._update_extended_ticks()
        if self.cfg.home_at_top:
            # 0 = top (home), positive = down toward bottom
            return (self._z0_deg - self._extended_deg()) * self._mm_per_deg
        return (self._extended_deg() - self._z0_deg) * self._mm_per_deg

    def contribute_observation(self, obs: Dict[str, float], pre_read_pos: Optional[float] = None, pre_read_vel: Optional[float] = None) -> None:
        """Add lift axis height (mm) and velocity to observation dict.
        
        Args:
            obs: Observation dictionary to update
            pre_read_pos: Optional pre-read Present_Position value (avoids redundant serial read)
            pre_read_vel: Optional pre-read Present_Velocity value (avoids redundant serial read)
        """
        if not self.enabled:
            return
        
        # Update extended ticks tracking and get height
        if pre_read_pos is not None:
            # Mirror _update_extended_ticks() exactly using the pre-read position value
            # so we avoid a second serial read while still using the correct wrap-around
            # logic and height formula from get_height_mm() / _extended_deg().
            cur = float(pre_read_pos)
            delta = cur - self._last_tick
            half = self._ticks_per_rev * 0.5
            if delta > +half:
                delta -= self._ticks_per_rev
            elif delta < -half:
                delta += self._ticks_per_rev
            self._extended_ticks += delta
            self._last_tick = cur
            # Compute height with the same formula as get_height_mm()
            ext_deg = self._extended_deg()  # dir_sign * extended_ticks * deg_per_tick
            if self.cfg.home_at_top:
                self._cached_height_mm = float((self._z0_deg - ext_deg) * self._mm_per_deg)
            else:
                self._cached_height_mm = float((ext_deg - self._z0_deg) * self._mm_per_deg)
        else:
            # Fallback to synchronous read if no pre-read data
            self._cached_height_mm = float(self.get_height_mm())
        
        obs[f"{self.cfg.name}.height_mm"] = self._cached_height_mm
        
        # Get velocity
        if pre_read_vel is not None:
            # Use pre-read velocity data
            raw_velocity = float(pre_read_vel)
        else:
            # Fallback to synchronous read
            try:
                raw_velocity = float(self._bus.read("Present_Velocity", self.cfg.name, normalize=False))
            except Exception:
                raw_velocity = 0.0
        
        # Apply exponential moving average filter to reduce noise for training
        self._filtered_velocity = (self._velocity_alpha * raw_velocity + 
                                   (1 - self._velocity_alpha) * self._filtered_velocity)
        
        # Normalize velocity to [-100, 100] range to match RANGE_M100_100 motor norm mode
        # Raw motor units are ±v_max (typically ±2000), normalize to ±100 like other velocity motors
        normalized_velocity = (self._filtered_velocity / self.cfg.v_max) * 100.0
        self._cached_velocity = normalized_velocity
        
        obs[f"{self.cfg.name}.vel"] = self._cached_velocity

    def normalize_velocity_for_logging(self, vel: float) -> float:
        """
        Normalize velocity to [-100, 100] for logging and recorded data.
        Accepts raw motor units (e.g. ±v_max) or already-normalized values;
        ensures output is always in a consistent range so spikes do not distort data.
        """
        if not self.enabled or self.cfg.v_max <= 0:
            return 0.0
        # If already in normalized range, clamp only
        if abs(vel) <= 100.0:
            return max(-100.0, min(100.0, float(vel)))
        # Raw motor units: normalize by v_max
        normalized = (vel / float(self.cfg.v_max)) * 100.0
        return max(-100.0, min(100.0, normalized))

    def action_for_logging(self, action: Dict[str, float]) -> Dict[str, float]:
        """
        Return a copy of the lift-related action with velocity normalized to [-100, 100]
        for logging and recording. Use this when building the action dict that gets
        stored in the dataset so velocity values are consistent and spikes are removed.
        """
        if not self.enabled:
            return {}
        prefix = f"{self.cfg.name}."
        vel_key = f"{self.cfg.name}.vel"
        out: Dict[str, float] = {}
        for k, v in action.items():
            if not k.startswith(prefix):
                continue
            if k == vel_key:
                out[k] = self.normalize_velocity_for_logging(float(v))
            else:
                out[k] = float(v)
        return out

    def apply_action(self, action: Dict[str, float]) -> None:
        """
        Supported action keys:
        - f"{name}.height_mm": target height (mm)  (recommended)
        - f"{name}.vel"      : direct velocity command (advanced)
        """
        if not self.enabled:
            return
        self.configure()

        key_h = f"{self.cfg.name}.height_mm"
        key_v = f"{self.cfg.name}.vel"

        if key_h in action:
            target_mm = float(action[key_h])
            cur_mm = self._cached_height_mm  # Use cached value from observation
            err = target_mm - cur_mm
            if abs(err) <= self.cfg.on_target_mm:
                v_cmd = 0.0
            else:
                v_cmd = self.cfg.kp_vel * err
                v_cmd = max(-self.cfg.v_max, min(self.cfg.v_max, v_cmd))

            # Soft limits
            if (cur_mm >= self.cfg.soft_max_mm and v_cmd > 0) or (cur_mm <= self.cfg.soft_min_mm and v_cmd < 0):
                v_cmd = 0.0

            # When home_at_top, positive height = down, so motor direction is opposite
            sign = -1 if self.cfg.home_at_top else 1
            self._bus.write("Goal_Velocity", self.cfg.name, int(sign * self.cfg.dir_sign * v_cmd))

        if key_v in action:
            # Denormalize from [-100, 100] to raw motor units (RANGE_M100_100 normalization)
            # (send_action normalizes all inputs, both from VR and from policy)
            normalized_v = float(action[key_v])
            v = int((normalized_v / 100.0) * self.cfg.v_max)
            v = max(-self.cfg.v_max, min(self.cfg.v_max, v))
            try:
                cur_mm = self._cached_height_mm  # Use cached value from observation
                if (cur_mm >= self.cfg.soft_max_mm and v > 0) or (cur_mm <= self.cfg.soft_min_mm and v < 0):
                    v = 0
            except Exception:
                pass
            sign = -1 if self.cfg.home_at_top else 1
            self._bus.write("Goal_Velocity", self.cfg.name, v * sign * self.cfg.dir_sign)

    def home(self, use_current: bool = True) -> None:
        """
        Homes the axis by driving toward the home position until motion stalls, then sets 0mm there.
        - If home_at_top=False: drive down to bottom, 0mm = bottom.
        - If home_at_top=True: drive up to top, 0mm = top, bottom = soft_max_mm (e.g. 500mm).
        """
        if not self.enabled:
            return
        self.configure()
        name = self.cfg.name

        # Drive toward home: down (positive) or up (negative) depending on convention
        home_vel = self.cfg.home_down_speed if not self.cfg.home_at_top else -self.cfg.home_down_speed
        self._bus.write("Goal_Velocity", name, int(home_vel))
        stuck = 0
        last_tick = float(self._bus.read("Present_Position", name, normalize=False))

        for _ in range(600):  # ~30s @50ms
            time.sleep(0.05)
            self._update_extended_ticks()
            now_tick = self._last_tick
            moved = abs(now_tick - last_tick) > 10
            last_tick = now_tick

            cur_ma = 0.0
            if use_current:
                try:
                    raw_cur = float(self._bus.read("Present_Current", name, normalize=False))
                    # Empirical conversion used by AlohaMini fork (SDK units -> mA).
                    cur_ma = raw_cur * 6.5
                except Exception:
                    cur_ma = 0.0

            if (use_current and cur_ma >= self.cfg.home_stall_current_ma) or (not moved):
                stuck += 1
            else:
                stuck = 0
            if stuck >= 2:
                break

        # Stop and release briefly (lets gravity settle if needed)
        try:
            self._bus.write("Torque_Enable", name, 0)
        except Exception:
            self._bus.write("Goal_Velocity", name, 0)
        time.sleep(0.5)

        # Optional backoff (if you re-enable torque elsewhere)
        if self.cfg.home_backoff_deg:
            # We can't command degrees in velocity mode directly; just set zero here.
            pass

        self._update_extended_ticks()
        self._z0_deg = self._extended_deg()

    def stop(self) -> None:
        """Best-effort stop (sets Goal_Velocity=0)."""
        if not self.enabled:
            return
        try:
            self.configure()
            self._bus.write("Goal_Velocity", self.cfg.name, 0)
        except Exception:
            pass
