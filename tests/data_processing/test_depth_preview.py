import numpy as np

from lerobot.data_processing.depth_preview import INVALID_BELOW_MM, colorize_depth


def test_colorize_depth_shape_and_dtype():
    out = colorize_depth(np.full((1, 24, 42), 500.0, dtype=np.float32))
    assert out.shape == (24, 42, 3) and out.dtype == np.uint8


def test_invalid_pixels_are_black_and_valid_ones_are_not():
    d = np.full((8, 8), 800.0, dtype=np.float32)
    d[:, :4] = INVALID_BELOW_MM - 5  # the codec decodes "no reading" to ~10 mm
    out = colorize_depth(d)
    assert (out[:, :4] == 0).all()
    assert out[:, 4:].any(axis=-1).all()


def test_near_and_far_get_different_colours_and_clamp():
    near, far = colorize_depth(np.full((4, 4), 200.0)), colorize_depth(np.full((4, 4), 1500.0))
    assert not np.array_equal(near[0, 0], far[0, 0])
    assert np.array_equal(colorize_depth(np.full((4, 4), 50_000.0)), far)  # beyond far clamps to far
    assert np.array_equal(colorize_depth(np.full((4, 4), 100.0)), near)  # nearer than near clamps to near
    assert near[0, 0, 0] > near[0, 0, 2]  # near is warm (red > blue)
    assert far[0, 0, 2] > far[0, 0, 0]  # far is cool
