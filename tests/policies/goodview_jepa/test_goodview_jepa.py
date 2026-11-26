#!/usr/bin/env python

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
"""goodview_jepa: cross-view world-model target, frame stride, camera blanking, registration."""

from __future__ import annotations

import json

import pytest
import torch
import vla_jepa_test_helpers as h
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.goodview_jepa.configuration_goodview_jepa import GoodViewJEPAConfig
from lerobot.policies.goodview_jepa.convert_from_vla_jepa import convert
from lerobot.policies.goodview_jepa.modeling_goodview_jepa import (
    GoodViewJEPAModel,
    GoodViewJEPAPolicy,
    select_view,
)
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAModel, VLAJEPAPolicy
from lerobot.utils.constants import OBS_IMAGES

NUM_FRAMES = 4  # tubelet 1 in the test config -> 4 encoded steps, 3 context + 3 target


def _tagged_videos(b: int, v: int, t: int) -> torch.Tensor:
    """Constant clip per (sample, view, frame): value == 100 * sample + 10 * view + frame.

    The fake encoder returns each frame's pixel mean, so features carry these tags.
    """
    videos = torch.zeros(b, v, t, 3, 16, 16)
    for s in range(b):
        for view in range(v):
            for f in range(t):
                videos[s, view, f] = 100 * s + 10 * view + f
    return videos


def _run_world_model_loss(monkeypatch, mode: str, src: int = 0, dst: int = 1, real_views: int = 2):
    config = h.make_goodview_config(
        NUM_FRAMES, cross_view_mode=mode, cross_view_source=src, cross_view_dest=dst
    )
    model = GoodViewJEPAModel(config)
    seen: dict[str, torch.Tensor] = {}

    class _SpyPredictor(nn.Module):
        def forward(self, frame_tokens, action_tokens):
            seen["input"] = frame_tokens.detach().clone()
            return torch.zeros_like(frame_tokens)

    model.video_predictor = _SpyPredictor()
    real_l1 = torch.nn.functional.l1_loss

    def _spy_l1(pred, target, **kwargs):
        seen["target"] = target.detach().clone()
        return real_l1(pred, target, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "l1_loss", _spy_l1)
    b = 2
    videos = _tagged_videos(b, real_views, NUM_FRAMES)
    action_tokens = torch.zeros(b, (NUM_FRAMES - 1) * config.num_action_tokens_per_timestep, 8)
    model._world_model_loss(videos, action_tokens)
    return seen, model.video_encoder.config.hidden_size


def _tags(features: torch.Tensor, hidden: int, v: int = 2) -> list[list[list[int]]]:
    """[B, N, V*H] -> tag per (sample, step, view), read from the first feature of each view block."""
    return [
        [[round(features[s, n, view * hidden].item()) for view in range(v)] for n in range(features.shape[1])]
        for s in range(features.shape[0])
    ]


# ---------------------------------------------------------------------------
# Registration, factory, checkpoint conversion
# ---------------------------------------------------------------------------


def test_registered_as_separate_policy() -> None:
    assert "goodview_jepa" in PreTrainedConfig.get_known_choices()
    assert "vla_jepa" in PreTrainedConfig.get_known_choices()
    assert get_policy_class("goodview_jepa") is GoodViewJEPAPolicy
    assert get_policy_class("vla_jepa") is VLAJEPAPolicy
    config = make_policy_config("goodview_jepa", device="cpu")
    assert isinstance(config, GoodViewJEPAConfig)
    assert config.type == "goodview_jepa"
    assert config.cross_view_mode == "future"


def test_vla_jepa_config_has_no_goodview_fields() -> None:
    config = make_policy_config("vla_jepa", device="cpu")
    for name in ("cross_view_mode", "video_frame_stride", "camera_blank_prob"):
        assert not hasattr(config, name)


def test_factory_builds_processors(patch_vla_jepa_external_models) -> None:
    config = h.make_goodview_config()
    pre, post = make_pre_post_processors(config)
    assert pre is not None and post is not None


def test_convert_vla_jepa_checkpoint(patch_vla_jepa_external_models, tmp_path) -> None:
    h.set_seed_all(0)
    vla_config = h.make_config(num_video_frames=NUM_FRAMES)
    vla_config.world_model_num_views = 2  # cross-view needs two views (the published checkpoints have 2)
    vla = VLAJEPAPolicy(vla_config)
    src = tmp_path / "vla"
    vla.save_pretrained(src)
    dst = convert(str(src), tmp_path / "goodview", cross_view_mode="same_time", video_frame_stride=2)
    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["type"] == "goodview_jepa"
    assert cfg["cross_view_mode"] == "same_time"
    assert cfg["video_frame_stride"] == 2

    loaded = GoodViewJEPAPolicy.from_pretrained(dst)
    assert isinstance(loaded.config, GoodViewJEPAConfig)
    assert loaded.config.cross_view_mode == "same_time"
    src_state, dst_state = vla.state_dict(), loaded.state_dict()
    assert src_state.keys() == dst_state.keys()
    for key in src_state:
        assert torch.equal(src_state[key], dst_state[key]), key


def test_convert_rejects_other_policy_types(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"type": "act"}))
    with pytest.raises(ValueError, match="not vla_jepa"):
        convert(str(tmp_path), tmp_path / "out")


# ---------------------------------------------------------------------------
# World-model target
# ---------------------------------------------------------------------------


def test_select_view_repeats_one_view_per_sample() -> None:
    b, n, v, hd = 2, 3, 2, 4
    merged = torch.arange(b * n * v * hd, dtype=torch.float32).reshape(b, n, v * hd)
    out = select_view(merged, b, v, view=1)
    assert out.shape == merged.shape
    per_view = merged.reshape(b, n, v, hd)
    for slot in range(v):
        assert torch.equal(out.reshape(b, n, v, hd)[:, :, slot], per_view[:, :, 1])


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_mode_none_matches_vla_jepa_exactly(
    patch_vla_jepa_external_models, causal: bool, reduction: str
) -> None:
    """Drift guard: the copied `_world_model_loss` must equal VLA-JEPA's when cross-view is off."""
    h.set_seed_all(0)
    gv_config = h.make_goodview_config(NUM_FRAMES, cross_view_mode="none", causal_world_model_context=causal)
    vla_config = h.make_config(num_video_frames=NUM_FRAMES)
    vla_config.world_model_num_views = 2
    vla_config.causal_world_model_context = causal
    gv, vla = GoodViewJEPAModel(gv_config), VLAJEPAModel(vla_config)
    vla.load_state_dict(gv.state_dict())
    videos = torch.rand(2, 2, NUM_FRAMES, 3, 16, 16)
    action_tokens = torch.randn(
        2, (NUM_FRAMES - 1) * gv_config.num_action_tokens_per_timestep, h.QWEN_HIDDEN_SIZE
    )
    a = gv._world_model_loss(videos, action_tokens, reduction=reduction)
    b = vla._world_model_loss(videos, action_tokens, reduction=reduction)
    assert torch.equal(a, b)


def test_mode_none_keeps_original_target(patch_vla_jepa_external_models, monkeypatch) -> None:
    seen, hidden = _run_world_model_loss(monkeypatch, "none")
    for s in range(2):
        assert _tags(seen["input"], hidden)[s] == [
            [100 * s + 10 * view + f for view in range(2)] for f in range(3)
        ]
        assert _tags(seen["target"], hidden)[s] == [
            [100 * s + 10 * view + f for view in range(2)] for f in range(1, 4)
        ]


@pytest.mark.parametrize(("mode", "frame_offset"), [("same_time", 0), ("future", 1)])
@pytest.mark.parametrize(("src", "dst"), [(0, 1), (1, 0)])
def test_cross_view_input_and_target(
    patch_vla_jepa_external_models, monkeypatch, mode: str, frame_offset: int, src: int, dst: int
) -> None:
    seen, hidden = _run_world_model_loss(monkeypatch, mode, src, dst)
    assert seen["input"].shape == seen["target"].shape
    for s in range(2):
        # Predictor sees only camera src (in every slot), steps 0..T-2.
        assert _tags(seen["input"], hidden)[s] == [[100 * s + 10 * src + f] * 2 for f in range(3)]
        # Target is camera dst, same steps (B) or one step ahead (C).
        assert _tags(seen["target"], hidden)[s] == [
            [100 * s + 10 * dst + f + frame_offset] * 2 for f in range(3)
        ]


def test_cross_view_rejects_padded_view(patch_vla_jepa_external_models, monkeypatch) -> None:
    with pytest.raises(ValueError, match="real camera views"):
        _run_world_model_loss(monkeypatch, "future", real_views=1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"cross_view_mode": "sideways"}, "cross_view_mode"),
        ({"cross_view_dest": 0}, "must differ"),
        ({"cross_view_dest": 2}, "cross_view_dest"),
        ({"video_frame_stride": 0}, "video_frame_stride"),
        ({"camera_blank_prob": 1.5}, "camera_blank_prob"),
        ({"freeze_qwen": True}, "needs the world model"),
    ],
)
def test_config_validation(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        h.make_goodview_config(NUM_FRAMES, **overrides)


def test_freeze_qwen_allowed_without_cross_view() -> None:
    config = h.make_goodview_config(NUM_FRAMES, cross_view_mode="none", freeze_qwen=True)
    assert not config.enable_world_model


def test_video_frame_stride_sets_observation_indices() -> None:
    assert h.make_goodview_config(NUM_FRAMES).observation_delta_indices == [0, 1, 2, 3]
    assert h.make_goodview_config(NUM_FRAMES, video_frame_stride=3).observation_delta_indices == [0, 3, 6, 9]


# ---------------------------------------------------------------------------
# Training forward and camera blanking
# ---------------------------------------------------------------------------


def _two_camera_policy(**overrides) -> GoodViewJEPAPolicy:
    config = h.make_goodview_config(**overrides)
    config.input_features[f"{OBS_IMAGES}.wrist"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, h.IMAGE_SIZE, h.IMAGE_SIZE)
    )
    return GoodViewJEPAPolicy(config)


def _two_camera_batch() -> dict:
    batch = h.make_train_batch()
    batch[f"{OBS_IMAGES}.laptop"] = torch.ones_like(batch[f"{OBS_IMAGES}.laptop"])
    batch[f"{OBS_IMAGES}.wrist"] = torch.ones_like(batch[f"{OBS_IMAGES}.laptop"])
    return batch


@pytest.mark.parametrize("mode", ["none", "same_time", "future"])
def test_training_forward_backward(patch_vla_jepa_external_models, mode: str) -> None:
    h.set_seed_all(0)
    policy = _two_camera_policy(cross_view_mode=mode)
    policy.train()
    loss, logs = policy.forward(_two_camera_batch())
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in policy.model.video_predictor.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert logs["wm_loss"] >= 0


def test_camera_blanking_zeros_exactly_one_view_in_training(patch_vla_jepa_external_models) -> None:
    policy = _two_camera_policy(camera_blank_prob=1.0)
    policy.train()
    inputs = policy._prepare_model_inputs(_two_camera_batch(), training=True)
    for views in inputs["images"]:
        assert sum(bool((img == 0).all()) for img in views) == 1
    assert bool((inputs["videos"] > 0).all())  # world-model clips keep every camera


def test_camera_blanking_picks_both_cameras(patch_vla_jepa_external_models) -> None:
    h.set_seed_all(0)
    policy = _two_camera_policy(camera_blank_prob=1.0)
    policy.train()
    batch = h.make_train_batch(batch_size=32)
    for key in (f"{OBS_IMAGES}.laptop", f"{OBS_IMAGES}.wrist"):
        batch[key] = torch.ones(32, h.NUM_VIDEO_FRAMES, 3, h.IMAGE_SIZE, h.IMAGE_SIZE)
    inputs = policy._prepare_model_inputs(batch, training=True)
    blanked = {next(i for i, img in enumerate(views) if bool((img == 0).all())) for views in inputs["images"]}
    assert blanked == {0, 1}


def test_camera_blanking_off_in_eval_and_inference(patch_vla_jepa_external_models) -> None:
    policy = _two_camera_policy(camera_blank_prob=1.0)
    policy.eval()
    for training in (True, False):
        inputs = policy._prepare_model_inputs(_two_camera_batch(), training=training)
        for views in inputs["images"]:
            assert not any(bool((img == 0).all()) for img in views)


# ---------------------------------------------------------------------------
# Remote-week options: random_pair / mixed targets, per-camera jitter, new-layer warm-up
# ---------------------------------------------------------------------------


def _pair_tags(tensor: torch.Tensor, hidden: int, slots: int = 2) -> list:
    """Per sample, the (slot-0) tag of each encoded step, plus whether every slot holds the same camera."""
    b, n, _ = tensor.shape
    per_slot = tensor.reshape(b, n, slots, hidden)[..., 0]
    assert torch.equal(per_slot, per_slot[:, :, :1].expand_as(per_slot))
    return per_slot[:, :, 0].round().tolist()


def test_random_pair_uses_every_camera_and_future_target(patch_vla_jepa_external_models, monkeypatch) -> None:
    h.set_seed_all(0)
    config = h.make_goodview_config(NUM_FRAMES, cross_view_mode="random_pair")
    model = GoodViewJEPAModel(config)
    seen: dict[str, torch.Tensor] = {}

    class _SpyPredictor(nn.Module):
        def forward(self, frame_tokens, action_tokens):
            seen["input"] = frame_tokens.detach().clone()
            return torch.zeros_like(frame_tokens)

    model.video_predictor = _SpyPredictor()
    real_l1 = torch.nn.functional.l1_loss
    monkeypatch.setattr(
        torch.nn.functional,
        "l1_loss",
        lambda p, t, **k: (seen.__setitem__("target", t.detach().clone()), real_l1(p, t, **k))[1],
    )
    b, views = 64, 3
    videos = _tagged_videos(b, views, NUM_FRAMES)
    action_tokens = torch.zeros(b, (NUM_FRAMES - 1) * config.num_action_tokens_per_timestep, 8)
    model._world_model_loss(videos, action_tokens)
    hidden = model.video_encoder.config.hidden_size
    src, dst = model.last_view_pairs
    assert bool((src != dst).all())
    assert set(src.tolist()) == {0, 1, 2} and set(dst.tolist()) == {0, 1, 2}
    inputs, targets = _pair_tags(seen["input"], hidden), _pair_tags(seen["target"], hidden)
    tokens_per_step = seen["input"].shape[1] // (NUM_FRAMES - 1)
    for s in range(b):
        steps_in = inputs[s][::tokens_per_step]
        steps_out = targets[s][::tokens_per_step]
        assert steps_in == [100 * s + 10 * int(src[s]) + f for f in range(NUM_FRAMES - 1)]
        assert steps_out == [100 * s + 10 * int(dst[s]) + f + 1 for f in range(NUM_FRAMES - 1)]


def test_mixed_switches_between_none_and_random_pair(patch_vla_jepa_external_models, monkeypatch) -> None:
    config = h.make_goodview_config(NUM_FRAMES, cross_view_mode="mixed", mixed_none_prob=0.5)
    model = GoodViewJEPAModel(config)
    used = []
    monkeypatch.setattr(model, "_random_pair_loss", lambda *a, **k: used.append("pair") or torch.zeros(()))
    original = VLAJEPAModel._merge_views

    def _spy_merge(embeddings, b, v):
        used.append(f"merge{v}")
        return original(embeddings, b, v)

    monkeypatch.setattr(model, "_merge_views", _spy_merge)
    action_tokens = torch.zeros(
        2, (NUM_FRAMES - 1) * config.num_action_tokens_per_timestep, h.QWEN_HIDDEN_SIZE
    )
    h.set_seed_all(0)
    for _ in range(40):
        model._world_model_loss(torch.rand(2, 3, NUM_FRAMES, 3, 16, 16), action_tokens)
    assert used.count("pair") > 5 and used.count("merge2") > 5  # both branches; "none" keeps 2 views


def test_per_camera_jitter_changes_inputs_independently_not_videos(patch_vla_jepa_external_models) -> None:
    h.set_seed_all(0)
    policy = _two_camera_policy(per_camera_jitter=True, per_camera_jitter_prob=1.0)
    policy.train()
    batch = _two_camera_batch()
    for key in (f"{OBS_IMAGES}.laptop", f"{OBS_IMAGES}.wrist"):
        batch[key] = torch.full_like(batch[key], 0.5)
    policy.config.per_camera_jitter = False
    reference = policy._prepare_model_inputs(batch, training=True)
    policy.config.per_camera_jitter = True
    jittered = policy._prepare_model_inputs(batch, training=True)
    for ref_views, views in zip(reference["images"], jittered["images"], strict=True):
        assert all(not torch.equal(a, b) for a, b in zip(ref_views, views, strict=True))
        assert not torch.equal(views[0], views[1])  # an independent draw per camera
    assert torch.equal(reference["videos"], jittered["videos"])  # world-model clips stay clean
    policy.eval()
    for training in (True, False):
        out = policy._prepare_model_inputs(batch, training=training)
        for ref_views, views in zip(reference["images"], out["images"], strict=True):
            assert all(torch.equal(a, b) for a, b in zip(ref_views, views, strict=True))


def test_warmup_trains_only_the_new_layers_first(patch_vla_jepa_external_models) -> None:
    from lerobot.policies.goodview_jepa.configuration_goodview_jepa import NEW_ACTION_LAYERS

    h.set_seed_all(0)
    policy = _two_camera_policy(warmup_new_layers_steps=1)
    policy.train()

    def grads():
        policy.zero_grad()
        loss, _ = policy.forward(_two_camera_batch())
        loss.backward()
        new, other = 0.0, 0.0
        for name, p in policy.named_parameters():
            if p.grad is None:
                continue
            if any(name.endswith(layer) for layer in NEW_ACTION_LAYERS):
                new += float(p.grad.abs().sum())
            else:
                other += float(p.grad.abs().sum())
        return new, other

    new, other = grads()
    assert new > 0 and other == 0
    new, other = grads()
    assert new > 0 and other > 0


def test_new_options_default_off() -> None:
    c = h.make_goodview_config(NUM_FRAMES)
    assert (
        not c.per_camera_jitter and c.warmup_new_layers_steps == 0 and c.cross_view_mode in ("future", "none")
    )
