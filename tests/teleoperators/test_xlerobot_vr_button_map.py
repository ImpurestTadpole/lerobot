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

"""Tests for the XLerobot VR ``button_map`` -> semantic-action dispatch.

These exercise ``VREventHandler`` directly with synthetic VR packet metadata
(no real VR hardware/websocket required), covering: default bindings match
today's fixed layout, remapping changes behaviour, the right-hand
missing-field guard, and the edge-cooldown debounce.
"""

import time

import pytest

from lerobot.teleoperators.xlerobot_vr.configuration_xlerobot_vr import (
    DEFAULT_VR_BUTTON_MAP,
    XLerobotVRTeleopConfig,
)
from lerobot.teleoperators.xlerobot_vr.xlerobot_vr import VREventHandler


def _meta(buttons: dict) -> dict:
    """Build a minimal VR goal metadata dict with the given raw buttons."""
    return {"buttons": buttons, "trigger": 0.0, "thumbstick": {"x": 0.0, "y": 0.0}}


def test_default_config_button_map_matches_default_map():
    cfg = XLerobotVRTeleopConfig()
    assert cfg.button_map == DEFAULT_VR_BUTTON_MAP


def test_default_left_x_or_y_triggers_rerecord_and_exit_early():
    handler = VREventHandler(vr_monitor=None)

    handler._process_left_controller(_meta({"x": True}))
    assert handler.events["rerecord_episode"] is True
    assert handler.events["exit_early"] is True

    handler._process_left_controller(_meta({"x": False}))

    handler.events["rerecord_episode"] = False
    handler.events["exit_early"] = False
    handler._process_left_controller(_meta({"y": True}))
    assert handler.events["rerecord_episode"] is True
    assert handler.events["exit_early"] is True


def test_default_left_menu_stops_session():
    handler = VREventHandler(vr_monitor=None)
    # _process_left_menu dispatches on RELEASE (a tap), not on the press itself, so it can
    # distinguish a quick tap from a long-press-for-passthrough (see _LONG_PRESS_S). A press with
    # no matching release must NOT fire stop_session yet.
    handler._process_left_controller(_meta({"menu": True}))
    assert handler.events["stop_recording"] is False

    handler._process_left_controller(_meta({"menu": False}))
    assert handler.events["stop_recording"] is True


def test_default_left_thumbstick_reset_position_is_self_clearing():
    handler = VREventHandler(vr_monitor=None)

    handler._process_left_controller(_meta({"thumbstick": True}))
    assert handler.events["reset_position"] is True

    # Held (not a new edge) -> self-clears back to False.
    handler._process_left_controller(_meta({"thumbstick": True}))
    assert handler.events["reset_position"] is False

    handler._process_left_controller(_meta({"thumbstick": False}))
    assert handler.events["reset_position"] is False


def test_default_right_b_exits_episode_early():
    handler = VREventHandler(vr_monitor=None)
    handler._process_right_controller(_meta({"b": True}))
    assert handler.events["exit_early"] is True


def test_default_right_a_exits_episode_early():
    # RIGHT A and B are both exit_early by default -- a clearly separate pair on the right
    # controller from LEFT X/Y's rerecord_episode, with no button bound to toggle_intervention
    # out of the box (DAgger sessions opt in explicitly via --teleop.button_map).
    handler = VREventHandler(vr_monitor=None)
    handler._process_right_controller(_meta({"a": True}))
    assert handler.events["exit_early"] is True
    assert handler._intervention_active is False


def test_toggle_intervention_requires_explicit_remap():
    handler = VREventHandler(vr_monitor=None, button_map={"right.a": "toggle_intervention"})
    assert handler._intervention_active is False

    handler._process_right_controller(_meta({"a": True}))
    assert handler._intervention_active is True

    handler._process_right_controller(_meta({"a": False}))
    time.sleep(0.6)  # clear the debounce window
    handler._process_right_controller(_meta({"a": True}))
    assert handler._intervention_active is False


def test_default_right_thumbstick_requests_upload():
    handler = VREventHandler(vr_monitor=None)
    handler._process_right_controller(_meta({"thumbstick": True}))
    assert handler.events["upload_requested"] is True


def test_right_button_debounce_ignores_rapid_repress():
    handler = VREventHandler(vr_monitor=None)

    handler._process_right_controller(_meta({"a": True}))
    assert handler.events["exit_early"] is True
    handler.events["exit_early"] = False

    handler._process_right_controller(_meta({"a": False}))
    # Re-press immediately, well inside the cooldown window -> dispatch is skipped, so the event
    # we just cleared stays cleared instead of firing again.
    handler._process_right_controller(_meta({"a": True}))
    assert handler.events["exit_early"] is False


def test_right_missing_button_field_treated_as_unchanged():
    handler = VREventHandler(vr_monitor=None)

    handler._process_right_controller(_meta({"b": True}))
    assert handler.events["exit_early"] is True
    handler.events["exit_early"] = False

    # Packet omits 'b' entirely (e.g. dropped field) -> must not read as a
    # release-then-repress on the next real packet.
    handler._process_right_controller(_meta({}))
    handler._process_right_controller(_meta({"b": True}))
    assert handler.events["exit_early"] is False


def test_button_map_can_be_remapped():
    remapped = {"right.b": "toggle_intervention", "right.a": "exit_early"}
    handler = VREventHandler(vr_monitor=None, button_map=remapped)

    handler._process_right_controller(_meta({"b": True}))
    assert handler._intervention_active is True
    assert handler.events["exit_early"] is False

    handler._process_right_controller(_meta({"b": False}))
    handler._process_right_controller(_meta({"a": True}))
    assert handler.events["exit_early"] is True


def test_unmapped_button_has_no_effect():
    handler = VREventHandler(vr_monitor=None, button_map={})
    handler._process_right_controller(_meta({"a": True, "b": True}))
    assert handler._intervention_active is False
    assert handler.events["exit_early"] is False
    assert handler.events["upload_requested"] is False


@pytest.mark.parametrize("semantic", ["rerecord_episode", "exit_early", "stop_session", "upload_dataset"])
def test_dispatch_semantic_is_idempotent_when_unmapped_button_omitted(semantic):
    """Sanity check that every non-toggle semantic maps to a distinct sticky event key."""
    handler = VREventHandler(vr_monitor=None, button_map={"left.x": semantic})
    handler._process_left_controller(_meta({"x": True}))

    expected = {
        "rerecord_episode": ("rerecord_episode", True),
        "exit_early": ("exit_early", True),
        "stop_session": ("stop_recording", True),
        "upload_dataset": ("upload_requested", True),
    }[semantic]
    assert handler.events[expected[0]] is expected[1]


class _FakeVRMonitor:
    """Records every send_status() call so tests can assert on the VR headset HUD payload."""

    def __init__(self):
        self.status_calls: list[dict] = []

    def send_status(self, status: dict) -> None:
        self.status_calls.append(status)


def test_recording_gate_arm_clears_hud_recording_indicator():
    monitor = _FakeVRMonitor()
    handler = VREventHandler(vr_monitor=monitor)

    handler.reset_recording_gate()

    assert monitor.status_calls[-1] == {"recording_enabled": False}
    assert handler.awaiting_recording_start is True


def test_recording_gate_open_sets_hud_recording_indicator():
    monitor = _FakeVRMonitor()
    handler = VREventHandler(vr_monitor=monitor)
    handler.reset_recording_gate()
    monitor.status_calls.clear()

    handler._process_left_x(True)

    assert monitor.status_calls[-1] == {"recording_enabled": True}
    assert handler.events["recording_gate_open"] is True
    assert handler.awaiting_recording_start is False
