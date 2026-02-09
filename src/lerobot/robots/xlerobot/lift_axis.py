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

    # Mechanical conversion: 1 rev = 360° = 4096 ticks
    lead_mm_per_rev: float = 84.0
    output_gear_ratio: float = 1.0

    # Height convention: 0 = bottom (homed), 600 = top (mm from home).
    soft_min_mm: float = 0.0
    soft_max_mm: float = 600.0

    # Homing (drive down to hard stop, detect stall, backoff)
    home_down_speed: int = 1300
    home_stall_current_ma: int = 150
    home_backoff_deg: float = 5.0

    # Closed-loop on height -> velocity
    kp_vel: float = 300.0
    v_max: int = 1300
    on_target_mm: float = 1.0

    # Direction conventions
    dir_sign: int = -1  # +1 no inversion; -1 invert direction


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
        return (self._extended_deg() - self._z0_deg) * self._mm_per_deg

    def contribute_observation(self, obs: Dict[str, float]) -> None:
        if not self.enabled:
            return
        obs[f"{self.cfg.name}.height_mm"] = float(self.get_height_mm())
        try:
            obs[f"{self.cfg.name}.vel"] = float(self._bus.read("Present_Velocity", self.cfg.name, normalize=False))
        except Exception:
            pass

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
            cur_mm = self.get_height_mm()
            err = target_mm - cur_mm
            if abs(err) <= self.cfg.on_target_mm:
                v_cmd = 0.0
            else:
                v_cmd = self.cfg.kp_vel * err
                v_cmd = max(-self.cfg.v_max, min(self.cfg.v_max, v_cmd))

            # Soft limits
            if (cur_mm >= self.cfg.soft_max_mm and v_cmd > 0) or (cur_mm <= self.cfg.soft_min_mm and v_cmd < 0):
                v_cmd = 0.0

            self._bus.write("Goal_Velocity", self.cfg.name, int(self.cfg.dir_sign * v_cmd))

        if key_v in action:
            v = int(action[key_v])
            v = max(-self.cfg.v_max, min(self.cfg.v_max, v))
            try:
                cur_mm = self.get_height_mm()
                if (cur_mm >= self.cfg.soft_max_mm and v > 0) or (cur_mm <= self.cfg.soft_min_mm and v < 0):
                    v = 0
            except Exception:
                pass
            self._bus.write("Goal_Velocity", self.cfg.name, v * self.cfg.dir_sign)

    def home(self, use_current: bool = True) -> None:
        """
        Homes the axis by driving "down" until motion stalls (optionally using current),
        then sets the current position as 0mm.
        """
        if not self.enabled:
            return
        self.configure()
        name = self.cfg.name

        self._bus.write("Goal_Velocity", name, int(self.cfg.home_down_speed))
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
