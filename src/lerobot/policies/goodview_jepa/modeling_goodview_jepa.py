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
"""Good-view JEPA policy: VLA-JEPA with a cross-view world-model target.

Only two things differ from VLA-JEPA:
- `GoodViewJEPAModel._world_model_loss` can swap the predictor's input and target views.
- `GoodViewJEPAPolicy._prepare_model_inputs` can black out one camera in the Qwen input in training.
Module names and shapes are unchanged, so VLA-JEPA checkpoints load as they are.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAModel, VLAJEPAPolicy

from .configuration_goodview_jepa import GoodViewJEPAConfig


def select_view(merged: Tensor, b: int, v: int, view: int) -> Tensor:
    """Keep one camera of merged [B, N, V*H] features and repeat it in every view slot.

    Matches the `VLAJEPAModel._merge_views` layout (feature dim is view-major) and keeps the shape,
    so the pretrained predictor still loads.
    """
    n, vh = merged.shape[1], merged.shape[2]
    h = vh // v
    one = merged.reshape(b, n, v, h)[:, :, view : view + 1]
    return one.expand(b, n, v, h).reshape(b, n, vh)


class GoodViewJEPAModel(VLAJEPAModel):
    config: GoodViewJEPAConfig

    def _world_model_loss(self, videos: Tensor, action_tokens: Tensor, reduction: str = "mean") -> Tensor:
        """VLA-JEPA's JEPA encode + predictor L1 loss, with an optional cross-view input/target.

        Copied from `VLAJEPAModel._world_model_loss` with the cross-view block added; with
        `cross_view_mode="none"` it must match the parent exactly (tested), so a change to the parent
        shows up as a test failure rather than a silent drift.
        """
        if self.video_predictor is None:
            raise RuntimeError("The world model loss requires `enable_world_model=True`.")
        num_views = self.config.num_world_model_views
        real_views = videos.shape[1]  # before padding: padded slots are copies of view 0
        if videos.shape[1] < num_views:
            missing = num_views - videos.shape[1]
            videos = torch.cat([videos, videos[:, :1].repeat(1, missing, 1, 1, 1, 1)], dim=1)
        elif videos.shape[1] > num_views:
            videos = videos[:, :num_views]

        b, v, t_frames, c, h_img, w_img = videos.shape
        flat = videos.reshape(b * v, t_frames, c, h_img, w_img)
        video_pixels = self.video_processor(
            videos=list(flat),
            return_tensors="pt",
            device=self.video_encoder.device,
            do_rescale=False,
        )["pixel_values_videos"]  # [B*V, T, C, H, W]

        tubelet_size = self.video_encoder.config.tubelet_size
        with torch.no_grad():
            video_embeddings = self.video_encoder.get_vision_features(pixel_values_videos=video_pixels)
            video_embeddings = self._merge_views(video_embeddings, b, v)

        t_enc_total = self.config.num_video_frames // tubelet_size
        if t_enc_total < 2:
            zero_shape = (video_embeddings.shape[0],) if reduction == "none" else ()
            return torch.zeros(zero_shape, device=video_embeddings.device)

        t_enc_ctx = t_enc_total - 1
        tokens_per_frame = video_embeddings.shape[1] // t_enc_total
        if self.config.causal_world_model_context:
            with torch.no_grad():
                input_states = self._causal_video_embeddings(video_pixels, tubelet_size, t_enc_ctx)
                input_states = self._merge_views(input_states, b, v)
        else:
            input_states = video_embeddings[:, : tokens_per_frame * t_enc_ctx, :]
        gt_states = video_embeddings[:, tokens_per_frame:, :]

        mode = self.config.cross_view_mode
        if mode != "none":
            src, dst = self.config.cross_view_source, self.config.cross_view_dest
            if max(src, dst) >= real_views:
                raise ValueError(
                    f"cross_view_mode={mode!r} needs real camera views {src} and {dst}, "
                    f"but the batch has {real_views} (padded views are copies of view 0)."
                )
            input_states = select_view(input_states, b, v, src)
            target_all = select_view(video_embeddings, b, v, dst)
            if mode == "future":  # variant C: camera dst, one encoded step ahead
                gt_states = target_all[:, tokens_per_frame:, :]
            else:  # variant B: camera dst, same step
                gt_states = target_all[:, : tokens_per_frame * t_enc_ctx, :]

        expected_actions = t_enc_ctx * self.config.num_action_tokens_per_timestep
        if action_tokens.shape[1] < expected_actions:
            pad = action_tokens[:, -1:].repeat(1, expected_actions - action_tokens.shape[1], 1)
            action_tokens = torch.cat([action_tokens, pad], dim=1)

        predicted_states = self.video_predictor(
            input_states.float(), action_tokens[:, :expected_actions].float()
        )
        if reduction == "none":
            elementwise = F.l1_loss(predicted_states, gt_states.float(), reduction="none")
            return elementwise.mean(dim=tuple(range(1, elementwise.ndim)))
        return F.l1_loss(predicted_states, gt_states.float(), reduction="mean")


class GoodViewJEPAPolicy(VLAJEPAPolicy):
    """VLA-JEPA policy with a cross-view world-model target and training-time camera blanking."""

    config_class = GoodViewJEPAConfig
    name = "goodview_jepa"

    def __init__(self, config: GoodViewJEPAConfig, **kwargs) -> None:
        # Skip VLAJEPAPolicy.__init__, which would build (and load weights for) a plain VLAJEPAModel.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.model = GoodViewJEPAModel(config)
        self.reset()

    def _prepare_model_inputs(self, batch: dict[str, Tensor], training=True) -> dict[str, Any]:
        inputs = super()._prepare_model_inputs(batch, training=training)
        prob = self.config.camera_blank_prob
        images = inputs["images"]
        if not (training and self.training and prob > 0 and len(images[0]) > 1):
            return inputs
        # Per sample, maybe black out one random camera in the Qwen input. Images are float in [0, 1]
        # (IDENTITY normalization), so zero is the same black frame a test-time blackout gives. The
        # world-model videos are built separately and stay intact.
        num_views = len(images[0])
        blank = torch.rand(len(images)) < prob
        choice = torch.randint(num_views, (len(images),))
        inputs["images"] = [
            [torch.zeros_like(img) if blank[s] and view == choice[s] else img for view, img in enumerate(views)]
            for s, views in enumerate(images)
        ]
        return inputs
