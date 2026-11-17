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
"""Reuse the VLA-JEPA test fakes (fake Qwen and V-JEPA) for goodview_jepa.

The VLA-JEPA conftest is loaded by path under a unique module name, `vla_jepa_test_helpers`, so it
can't clash with other `conftest` modules; tests import it by that name.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

_path = Path(__file__).parents[1] / "vla_jepa" / "conftest.py"
_spec = importlib.util.spec_from_file_location("vla_jepa_test_helpers", _path)
helpers = importlib.util.module_from_spec(_spec)
sys.modules["vla_jepa_test_helpers"] = helpers
_spec.loader.exec_module(helpers)

# Re-export everything from the VLA-JEPA conftest. pytest puts every conftest directory on sys.path,
# so the vla_jepa tests' bare `from conftest import ...` can resolve to this module when both folders
# run together; with the same names it works either way. This also exposes the fixtures.
globals().update({k: v for k, v in vars(helpers).items() if not k.startswith("__")})


def make_goodview_config(num_video_frames: int = helpers.NUM_VIDEO_FRAMES, **overrides):
    """The VLA-JEPA test config as a GoodViewJEPAConfig, with good-view fields overridden."""
    from lerobot.policies.goodview_jepa.configuration_goodview_jepa import GoodViewJEPAConfig
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

    base = helpers.make_config(num_video_frames=num_video_frames)
    kwargs = {f.name: getattr(base, f.name) for f in fields(VLAJEPAConfig) if f.init}
    kwargs["world_model_num_views"] = 2  # cross-view modes need two views; the base test config has 1
    kwargs.update(overrides)
    config = GoodViewJEPAConfig(**kwargs)
    config.validate_features()
    return config


sys.modules["vla_jepa_test_helpers"].make_goodview_config = make_goodview_config
