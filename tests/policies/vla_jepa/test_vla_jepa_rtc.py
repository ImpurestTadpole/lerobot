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
"""Guided Real-Time Chunking for the VLA-JEPA flow-matching action head."""

import torch

import conftest as h  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy


def _policy(rtc: RTCConfig | None) -> VLAJEPAPolicy:
    config = h.make_config()
    config.rtc_config = rtc
    policy = VLAJEPAPolicy(config)
    policy.eval()
    return policy


def test_supports_rtc_and_builds_the_processor(patch_vla_jepa_external_models) -> None:
    assert _policy(None).supports_rtc() and _policy(None).rtc_processor is None
    assert _policy(RTCConfig(execution_horizon=3)).rtc_processor is not None


def test_without_a_prefix_rtc_changes_nothing(patch_vla_jepa_external_models) -> None:
    h.set_seed_all(0)
    batch = h.make_inference_batch()
    policy = _policy(RTCConfig(execution_horizon=3))
    noise = torch.randn(*h.EXPECTED_ACTION_CHUNK_SHAPE)
    plain = policy.predict_action_chunk(batch, noise=noise)
    guided = policy.predict_action_chunk(batch, noise=noise, inference_delay=0, prev_chunk_left_over=None)
    assert torch.equal(plain, guided)


def test_prefix_pulls_the_frozen_steps_toward_the_previous_chunk(patch_vla_jepa_external_models) -> None:
    h.set_seed_all(0)
    batch = h.make_inference_batch()
    policy = _policy(RTCConfig(execution_horizon=h.ACTION_HORIZON, max_guidance_weight=10.0))
    noise = torch.randn(*h.EXPECTED_ACTION_CHUNK_SHAPE)
    plain = policy.predict_action_chunk(batch, noise=noise)
    target = plain + 0.5  # a "previous chunk" the new one should continue
    delay = 2
    guided = policy.predict_action_chunk(
        batch,
        noise=noise,
        inference_delay=delay,
        prev_chunk_left_over=target,
        execution_horizon=h.ACTION_HORIZON,
    )
    assert guided.shape == plain.shape and torch.isfinite(guided).all()
    before = (plain[:, :delay] - target[:, :delay]).abs().mean()
    after = (guided[:, :delay] - target[:, :delay]).abs().mean()
    assert after < before * 0.8  # the frozen prefix moves clearly toward the previous chunk


def test_rtc_disabled_in_config_ignores_the_prefix(patch_vla_jepa_external_models) -> None:
    h.set_seed_all(0)
    batch = h.make_inference_batch()
    policy = _policy(RTCConfig(enabled=False, execution_horizon=3))
    noise = torch.randn(*h.EXPECTED_ACTION_CHUNK_SHAPE)
    plain = policy.predict_action_chunk(batch, noise=noise)
    guided = policy.predict_action_chunk(
        batch, noise=noise, inference_delay=2, prev_chunk_left_over=plain + 1
    )
    assert torch.equal(plain, guided)
