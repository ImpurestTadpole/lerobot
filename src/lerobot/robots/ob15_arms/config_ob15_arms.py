# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from ..config import RobotConfig
from ..ob15.config_ob15 import OB15ClientConfig, OB15Config


@dataclass
class ArmsHoldOptions:
    """Head/lift hold options shared by ``ob15_arms`` and ``ob15_arms_client``.

    Head and lift are moved to a hold pose at connect and then held. Precedence: explicit flags
    (``head_pan`` / ``head_tilt`` / ``lift_height_mm``) > ``hold_from`` (pose saved with a dataset or
    policy) > named ``station`` > current positions at connect.
    """

    station: str | None = None
    stations_file: str = "~/.ob15/stations.json"
    head_pan: float | None = None
    head_tilt: float | None = None
    lift_height_mm: float | None = None
    hold_on_connect: bool = True
    # Dataset root / Hub dataset id / checkpoint dir / Hub model id holding `ob15_hold.json`.
    hold_from: str | None = None
    # Arm/gripper targets (ARM_KEYS) sent before the head/lift move to the hold pose.
    rest_pose: dict[str, float] | None = None
    settle_timeout_s: float = 15.0
    lift_tol_mm: float = 5.0
    head_tol: float = 2.0


@RobotConfig.register_subclass("ob15_arms")
@dataclass
class OB15ArmsConfig(ArmsHoldOptions, OB15Config):
    """OB15 controlled through the 12 arm/gripper joints only.

    No base command is sent (the base driver reads that as zero velocity) and base/head/lift keys in an
    incoming action are dropped. See ``ArmsHoldOptions`` for the hold pose.
    """


@RobotConfig.register_subclass("ob15_arms_client")
@dataclass
class OB15ArmsClientConfig(ArmsHoldOptions, OB15ClientConfig):
    """Remote (ZMQ) arms-only OB15. The host keeps running the full-body ``ob15`` robot."""
