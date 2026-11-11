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

"""The episode start/stop events the record loop sends to the VR headset (it plays a different chime for each)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.teleoperators.xlerobot_vr.xlerobot_vr import XLerobotVRTeleop


def _teleop(stream_cameras_to_vr=True, monitor=True):
    t = XLerobotVRTeleop.__new__(XLerobotVRTeleop)
    t.config = SimpleNamespace(stream_cameras_to_vr=stream_cameras_to_vr)
    t.vr_monitor = MagicMock() if monitor else None
    return t


def test_start_and_stop_send_distinct_events_with_the_recording_flag():
    t = _teleop()
    t.send_episode_event("start")
    t.send_episode_event("stop")
    sent = [c.args[0] for c in t.vr_monitor.send_status.call_args_list]
    assert sent == [
        {"episode_event": "start", "recording_active": True},
        {"episode_event": "stop", "recording_active": False},
    ]


@pytest.mark.parametrize("kwargs", [{"stream_cameras_to_vr": False}, {"monitor": False}])
def test_no_event_is_sent_without_the_vr_stream(kwargs):
    t = _teleop(**kwargs)
    t.send_episode_event("start")  # must not raise
    if t.vr_monitor is not None:
        t.vr_monitor.send_status.assert_not_called()
