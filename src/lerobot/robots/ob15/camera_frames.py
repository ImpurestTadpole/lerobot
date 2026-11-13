"""Bookkeeping for the cameras' non-blocking ``latest`` read mode.

``camera_read_mode="fresh"`` (the default) makes every control tick wait for the next *new* frame from every
camera, so the loop is slaved to the slowest camera: a 29.5 fps UVC camera caps a 30 Hz loop at 29.5 Hz, and a
tick that starts just after a frame was taken waits a whole frame period, so the rate can collapse to 15 Hz
for a tick whenever the work per tick gets close to the budget.

``"latest"`` instead takes whatever frame the camera thread has buffered, immediately. The loop's own clock then
sets the rate, and a camera that has not delivered a new frame since the previous tick yields the *same frame
object again*. That repeat is detected here (by object identity: the camera threads replace their buffer on every
new frame and never write into it), counted per camera and reported, so a repeated frame is never silently
recorded as new data.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Fraction of a window's reads that may be repeats before a WARNING is logged (one repeat in ~60 is the
#: expected price of a 29.5 fps camera under a 30 Hz loop, i.e. 1.7%).
REPEAT_WARN_FRACTION = 0.05
WARN_WINDOW_READS = 90
WARN_COOLDOWN_S = 10.0


@dataclass
class CameraFrameStats:
    reads: int = 0
    repeats: int = 0  # reads that returned the same frame as the previous read
    longest_repeat_run: int = 0
    errors: int = 0  # reads that raised (camera stalled, frame too old, thread dead)
    worst_age_ms: float = 0.0

    @property
    def repeat_fraction(self) -> float:
        return self.repeats / self.reads if self.reads else 0.0


class CameraFrameTracker:
    """Counts repeated (stale) frames and read errors per camera. Thread-safe: one reader thread per camera."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: dict[str, Any] = {}
        self._run: dict[str, int] = {}
        self._stats: dict[str, CameraFrameStats] = {}
        self._window: dict[str, list[int]] = {}  # key -> [reads, repeats] since the last warning check
        self._last_warn: dict[str, float] = {}

    def note_frame(self, key: str, frame: Any, age_ms: float | None = None) -> bool:
        """Record one successful read; return True when *frame* is the one returned by the previous read."""
        with self._lock:
            stats = self._stats.setdefault(key, CameraFrameStats())
            win = self._window.setdefault(key, [0, 0])
            repeated = self._last.get(key) is frame
            self._last[key] = frame
            stats.reads += 1
            win[0] += 1
            if age_ms is not None:
                stats.worst_age_ms = max(stats.worst_age_ms, age_ms)
            if repeated:
                stats.repeats += 1
                win[1] += 1
                self._run[key] = self._run.get(key, 0) + 1
                stats.longest_repeat_run = max(stats.longest_repeat_run, self._run[key])
            else:
                self._run[key] = 0
            if win[0] >= WARN_WINDOW_READS:
                frac = win[1] / win[0]
                now = time.monotonic()
                if frac > REPEAT_WARN_FRACTION and now - self._last_warn.get(key, -1e9) >= WARN_COOLDOWN_S:
                    self._last_warn[key] = now
                    logger.warning(
                        "Camera '%s' delivered no new frame on %.0f%% of the last %d ticks (the previous frame was "
                        "reused). The camera is slower than the control loop; the repeats are recorded as-is.",
                        key,
                        100 * frac,
                        win[0],
                    )
                win[0] = win[1] = 0
            return repeated

    def is_repeat(self, key: str, frame: Any) -> bool:
        """Would *frame* be a repeat of the previous read of *key*? (does not record anything)"""
        with self._lock:
            return self._last.get(key) is frame

    def note_error(self, key: str) -> None:
        with self._lock:
            self._stats.setdefault(key, CameraFrameStats()).errors += 1

    def snapshot(self) -> dict[str, CameraFrameStats]:
        with self._lock:
            return {k: CameraFrameStats(**vars(v)) for k, v in self._stats.items()}

    def report_lines(self) -> list[str]:
        lines = []
        for key, s in sorted(self.snapshot().items()):
            lines.append(
                f"  camera {key}: {s.reads} reads, {s.repeats} repeated frames ({100 * s.repeat_fraction:.1f}%), "
                f"longest repeat run {s.longest_repeat_run}, {s.errors} read errors, worst frame age {s.worst_age_ms:.0f} ms"
            )
        return lines


def read_color_latest(
    cam: Any,
    key: str,
    tracker: CameraFrameTracker,
    max_age_ms: float,
    fresh_wait_ms: float = 0.0,
) -> Any:
    """Newest buffered colour frame of *cam*, without waiting for the camera (see the module docstring).

    When the buffered frame is the one the previous read returned and *fresh_wait_ms* > 0, wait at most that long
    for a newer one (polling, so a stalled camera costs a bounded delay, never a hang). The wait is what turns the
    ~8% repeats a 29.5 fps camera produces under a 30 Hz loop (its frames arrive with jitter around the tick) into
    ~0, at a cost that is bounded and is only paid on ticks that would otherwise repeat.

    Raises whatever the camera raises when its buffer is older than *max_age_ms* (a stall), after counting it.
    """
    try:
        frame = cam.read_latest(max_age_ms=max_age_ms)
    except RuntimeError:
        # No frame captured yet (first tick after connect): block for one.
        frame = cam.async_read()
    except Exception:
        tracker.note_error(key)
        raise
    if fresh_wait_ms > 0 and tracker.is_repeat(key, frame):
        deadline = time.perf_counter() + fresh_wait_ms / 1e3
        while time.perf_counter() < deadline:
            time.sleep(0.0005)
            try:
                newer = cam.read_latest(max_age_ms=max_age_ms)
            except Exception:
                break
            if newer is not frame:
                frame = newer
                break
    tracker.note_frame(key, frame)
    return frame
