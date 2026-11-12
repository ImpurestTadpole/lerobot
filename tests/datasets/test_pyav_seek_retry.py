"""Regression test: pyav decoding retries with a larger seek margin when the decoder drops the
frame right after the landing keyframe (HEVC CRA + IDR_N_LP/RASL at some depth episode starts)."""

from fractions import Fraction

import numpy as np

from lerobot.datasets import video_utils


class _Frame:
    def __init__(self, pts):
        self.pts = pts

    def to_ndarray(self, format):  # noqa: A002
        return np.full((2, 2), self.pts % 4096, dtype=np.uint16)


class _Stream:
    time_base = Fraction(1, 15360)
    average_rate = Fraction(30)


class _Container:
    """Fake container: frames every 512 ticks (30 fps); a seek landing within 1 frame of the
    target loses the frame after the landing keyframe, like the real broken streams."""

    streams = type("S", (), {"video": [_Stream()]})()

    def __init__(self):
        self.start = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def seek(self, pts, backward=True, any_frame=False, stream=None):
        self.start = (pts // 512) * 512 if pts % 512 else pts - 512
        self.target = pts

    def decode(self, stream):
        land = self.start
        near = (self.target - land) <= 1024  # landed on the keyframe just before the target
        for i in range(40):
            pts = land + i * 512
            if near and i == 1:
                continue  # dropped by the decoder
            yield _Frame(pts)


def test_pyav_retries_when_frame_after_keyframe_is_dropped(monkeypatch):
    monkeypatch.setattr(video_utils.av, "open", lambda *a, **k: _Container())
    ts = 10 * 512 / 15360  # frame 10
    out = video_utils.decode_video_frames_pyav("x.mp4", [ts], 1e-4, is_depth=True)
    assert out.shape[0] == 1
    assert int(out[0, 0, 0, 0]) == (10 * 512) % 4096
