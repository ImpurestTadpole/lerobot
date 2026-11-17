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
"""Black out one camera at test time, to test robustness to a failed camera.

A zeroed frame is "camera blacked out", not "camera removed": the policy still gets an input slot,
just a black one. A policy that never saw black frames in training is being tested off-distribution,
so compare like with like:
- train every policy in the comparison with the same `--policy.camera_blank_prob` (e.g. 0.1), or
- report the condition as "camera blacked out, no blanking in training" for all of them.

Apply it to the raw observation, before the policy preprocessor (VLA-JEPA images are in [0, 1] and
use IDENTITY normalization, so 0 is black for both training-time blanking and this helper). Use the
same seed and trial count for every policy.

    action = policy.select_action(preprocessor(drop_camera(observation)))
"""


def drop_camera(obs: dict, key: str = "observation.images.top") -> dict:
    """Return a copy of `obs` with camera `key` blacked out."""
    if key not in obs:
        raise KeyError(f"{key!r} not in observation keys {list(obs)}")
    obs = dict(obs)
    obs[key] = obs[key] * 0  # works for torch tensors and numpy arrays
    return obs
