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
"""Good-view JEPA: VLA-JEPA with a cross-view world-model target.

Same architecture, weights and processors as VLA-JEPA; only the world-model target, the clip frame
spacing and optional training-time camera blanking differ. With the defaults below it trains exactly
like VLA-JEPA.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

CROSS_VIEW_MODES = ("none", "same_time", "future", "random_pair", "mixed")
# Shape-changed tensors a new robot re-initialises (see `warmup_new_layers_steps`).
NEW_ACTION_LAYERS = (
    "action_model.action_encoder.layer1.weight",
    "action_model.action_decoder.layer2.weight",
    "action_model.action_decoder.layer2.bias",
    "action_model.state_encoder.layer1.weight",
)


@PreTrainedConfig.register_subclass("goodview_jepa")
@dataclass
class GoodViewJEPAConfig(VLAJEPAConfig):
    # World-model target. "none" keeps VLA-JEPA's target (every view, one encoded step ahead).
    # "same_time" (variant B) predicts view `cross_view_dest` at the same step from view
    # `cross_view_source`; "future" (variant C) predicts it one encoded step ahead. View indices follow
    # the order of `image_features`. The V-JEPA encoder stays frozen, so this changes the predictor's
    # target and what the action tokens must carry, not the encoder's representation. "same_time"
    # leaves the action tokens nothing to predict, so expect it to weaken the latent-action signal.
    cross_view_mode: str = "future"
    cross_view_source: int = 0
    cross_view_dest: int = 1
    # Spacing of the world-model clip frames. None keeps VLA-JEPA's default (consecutive frames, or
    # spread across the action chunk when it is longer than the clip). An int k samples frames
    # 0, k, 2k, ..., so with a tubelet of 2 each prediction looks 2*k raw frames ahead. Vary this, not
    # `num_video_frames` (which lengthens the clip and changes the action-token count), to vary the
    # prediction horizon.
    video_frame_stride: int | None = None
    # Training-time camera blanking: per sample, with this probability, one random camera is zeroed in
    # the Qwen input (world-model targets are untouched). Trains for the test-time "camera blacked
    # out" condition.
    camera_blank_prob: float = 0.0
    # "random_pair": per sample, predict a random camera from another random camera, one encoded step
    # ahead, choosing among ALL real cameras (the plain modes only see the first two). "mixed": per
    # batch, `mixed_none_prob` of the time the "none" target, otherwise "random_pair". Both keep the
    # 2-slot predictor, so pretrained weights load unchanged.
    mixed_none_prob: float = 0.5
    # Independent random colour jitter per camera on the Qwen inputs only (world-model clips stay
    # clean), applied to each camera of a sample with probability `per_camera_jitter_prob`.
    per_camera_jitter: bool = False
    per_camera_jitter_prob: float = 0.8
    jitter_brightness: float = 0.5
    jitter_contrast: float = 0.4
    jitter_saturation: float = 0.4
    jitter_hue: float = 0.05
    # For the first N training forwards only the shape-changed tensors in NEW_ACTION_LAYERS receive
    # gradients; every other parameter's gradient is zeroed. 0 = off.
    warmup_new_layers_steps: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cross_view_mode not in CROSS_VIEW_MODES:
            raise ValueError(
                f"`cross_view_mode` must be one of {CROSS_VIEW_MODES}, got {self.cross_view_mode!r}."
            )
        if self.cross_view_mode in ("random_pair", "mixed") and self.causal_world_model_context:
            raise ValueError(
                f"cross_view_mode={self.cross_view_mode!r} does not support causal_world_model_context."
            )
        if not 0.0 <= self.mixed_none_prob <= 1.0:
            raise ValueError(f"`mixed_none_prob` must be in [0, 1], got {self.mixed_none_prob}.")
        if not 0.0 <= self.per_camera_jitter_prob <= 1.0:
            raise ValueError(
                f"`per_camera_jitter_prob` must be in [0, 1], got {self.per_camera_jitter_prob}."
            )
        if self.warmup_new_layers_steps < 0:
            raise ValueError(f"`warmup_new_layers_steps` must be >= 0, got {self.warmup_new_layers_steps}.")
        if self.cross_view_mode != "none":
            if not self.enable_world_model:
                # VLAJEPAConfig turns the world model off under freeze_qwen; the target would be ignored.
                raise ValueError(
                    f"cross_view_mode={self.cross_view_mode!r} needs the world model, but it is off "
                    f"(freeze_qwen={self.freeze_qwen}, enable_world_model=False). Unfreeze Qwen or use "
                    "the vla_jepa policy."
                )
            num_views = self.num_world_model_views
            # random_pair / mixed draw their pairs from every real camera; the fixed indices are unused.
            for name in (
                ()
                if self.cross_view_mode in ("random_pair", "mixed")
                else ("cross_view_source", "cross_view_dest")
            ):
                index = getattr(self, name)
                if not 0 <= index < num_views:
                    raise ValueError(f"`{name}`={index} must be in [0, {num_views}) (world-model views).")
            if (
                self.cross_view_mode not in ("random_pair", "mixed")
                and self.cross_view_source == self.cross_view_dest
            ):
                raise ValueError("`cross_view_source` and `cross_view_dest` must differ.")
        if self.video_frame_stride is not None and self.video_frame_stride < 1:
            raise ValueError(f"`video_frame_stride` must be >= 1, got {self.video_frame_stride}.")
        if not 0.0 <= self.camera_blank_prob <= 1.0:
            raise ValueError(f"`camera_blank_prob` must be in [0, 1], got {self.camera_blank_prob}.")

    @property
    def observation_delta_indices(self) -> list[int]:
        if self.enable_world_model and self.video_frame_stride is not None:
            return [i * self.video_frame_stride for i in range(self.num_video_frames)]
        return super().observation_delta_indices
