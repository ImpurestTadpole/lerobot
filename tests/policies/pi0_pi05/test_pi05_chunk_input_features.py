#!/usr/bin/env python
"""PI05 resizes images and zero-pads the state, so the policy server must accept real camera and state shapes."""

from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402
from lerobot.utils.feature_spec import FeatureSpec  # noqa: E402

_CAMS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _policy():
    config = PI05Config(
        input_features={
            **{f"observation.images.{c}": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for c in _CAMS},
            OBS_STATE: PolicyFeature(FeatureType.STATE, (32,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (32,))},
    )
    return SimpleNamespace(config=config)


def _features(image_hw=(540, 960), state_dim=15, cams=_CAMS):
    h, w = image_hw
    imgs = [FeatureSpec(f"observation.images.{c}", (h, w, 3), "uint8", kind="rgb", semantics="test-v1") for c in cams]
    return (*imgs, FeatureSpec(OBS_STATE, (state_dim,), "float32", semantics="test-v1"))


def _validate(features, policy=None):
    PI05Policy.validate_chunk_input_features(policy or _policy(), features)


def test_accepts_real_camera_resolution_and_narrow_state():
    _validate(_features())
    _validate(_features(image_hw=(360, 640), state_dim=15))
    _validate(_features(image_hw=(224, 224), state_dim=32))


def test_rejects_wrong_names():
    with pytest.raises(ValueError, match="must exactly match"):
        _validate(_features(cams=_CAMS[:2]))


def test_rejects_state_wider_than_padding_or_not_1d():
    with pytest.raises(ValueError, match="max_state_dim"):
        _validate(_features(state_dim=33))
    bad = (*_features()[:3], FeatureSpec(OBS_STATE, (3, 5), "float32", semantics="test-v1"))
    with pytest.raises(ValueError, match="1-D state"):
        _validate(bad)


def test_rejects_camera_that_is_not_rgb():
    feats = list(_features())
    feats[0] = FeatureSpec(feats[0].name, (540, 960), "float32", kind="tensor", semantics="test-v1")
    with pytest.raises(ValueError, match="RGB image"):
        _validate(tuple(feats))
