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
"""Padded action steps (repeats of the episode's last action) must not contribute to the X-VLA loss."""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.xvla.action_hub import build_action_space  # noqa: E402
from lerobot.policies.xvla.modeling_xvla import drop_padded_steps  # noqa: E402

SPACES = [("auto", {"real_dim": 12, "max_dim": 20}, 12), ("so101_bimanual", {}, 12), ("ee6d", {}, 20)]


def _loss(space, pred, target, pad):
    return sum(space.compute_loss(*drop_padded_steps(pred, target, pad)).values())


@pytest.mark.parametrize("mode,kwargs,dim", SPACES)
def test_padded_steps_do_not_affect_loss_or_gradient(mode, kwargs, dim):
    space = build_action_space(mode, **kwargs)
    generator = torch.Generator().manual_seed(0)
    pred = torch.randn(2, 6, 20, generator=generator, requires_grad=True)
    target = torch.rand(2, 6, 20, generator=generator)
    pad = torch.zeros(2, 6, dtype=torch.bool)
    pad[1, 3:] = True

    loss = _loss(space, pred, target, pad)
    corrupted = target.clone()
    corrupted[1, 3:] = 1 - corrupted[1, 3:]
    assert torch.equal(loss, _loss(space, pred, corrupted, pad))

    loss.backward()
    assert torch.count_nonzero(pred.grad[1, 3:]) == 0
    assert torch.count_nonzero(pred.grad[1, :3]) > 0


def test_without_padding_the_loss_is_unchanged():
    space = build_action_space("auto", real_dim=12, max_dim=20)
    pred, target = torch.randn(2, 6, 20), torch.randn(2, 6, 20)
    expected = sum(space.compute_loss(pred, target).values())
    assert torch.allclose(_loss(space, pred, target, torch.zeros(2, 6, dtype=torch.bool)), expected)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        drop_padded_steps(torch.zeros(2, 6, 20), torch.zeros(2, 6, 20), torch.zeros(2, 5, dtype=torch.bool))
