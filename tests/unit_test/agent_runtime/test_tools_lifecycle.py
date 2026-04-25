# Copyright 2025 ApeCloud, Inc.
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

"""Contract tests for D9 §6 tool lifecycle adapter (Phase 8 #75)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aperag.domains.agent_runtime.schemas import AgentTimelineEventEnvelope
from aperag.domains.agent_runtime.tools.consent import ConsentService
from aperag.domains.agent_runtime.tools.elicitation import ElicitationService
from aperag.domains.agent_runtime.tools.lifecycle import (
    EVENT_TOOL_CONSENT_DECIDED,
    EVENT_TOOL_CONSENT_REQUESTED,
    EVENT_TOOL_ELICITATION_REQUESTED,
    EVENT_TOOL_ELICITATION_RESOLVED,
    LIFECYCLE_EVENT_TYPES,
    LifecycleEmitter,
    translate_lifecycle_envelope,
)
from aperag.domains.agent_runtime.uimessage import ElicitationData, ToolConsentData
from aperag.domains.agent_runtime.wire.parts import (
    DataElicitationPart,
    DataToolConsentPart,
)


def _envelope(event_type: str, data: dict) -> AgentTimelineEventEnvelope:
    return AgentTimelineEventEnvelope(
        event_id="evt-1",
        turn_id="turn-1",
        sequence=1,
        timestamp=datetime.now(timezone.utc),
        type=event_type,
        actor="system",
        data=data,
    )


def test_lifecycle_constants_set_includes_all_four_event_types():
    assert LIFECYCLE_EVENT_TYPES == {
        EVENT_TOOL_CONSENT_REQUESTED,
        EVENT_TOOL_CONSENT_DECIDED,
        EVENT_TOOL_ELICITATION_REQUESTED,
        EVENT_TOOL_ELICITATION_RESOLVED,
    }


def test_translate_returns_empty_for_unrelated_envelope_type():
    envelope = _envelope("turn.started", {})
    assert translate_lifecycle_envelope(envelope) == []


def test_translate_consent_requested_emits_data_tool_consent_part():
    payload = ToolConsentData(
        tool_call_id="call-1",
        tool_name="aperag_fs_write_file",
        metadata={"mcpServer": "aperag-fs", "mcpToolName": "write_file"},
        args_preview='{"path":"/tmp/x"}',
        args_hash="a" * 64,
        risk="writes_user_data",
        requested_at="2026-04-25T20:30:00+00:00",
        state="pending",
    )
    envelope = _envelope(
        EVENT_TOOL_CONSENT_REQUESTED,
        {"data": payload.model_dump(mode="json", by_alias=True)},
    )
    parts = translate_lifecycle_envelope(envelope)
    assert len(parts) == 1
    part = parts[0]
    assert isinstance(part, DataToolConsentPart)
    assert part.data.state == "pending"
    assert part.data.tool_call_id == "call-1"


def test_translate_consent_decided_passes_through_state():
    payload = ToolConsentData(
        tool_call_id="call-1",
        tool_name="t",
        args_preview="{}",
        args_hash="x" * 64,
        risk="writes_user_data",
        requested_at="2026-04-25T20:30:00+00:00",
        state="approved",
    )
    envelope = _envelope(
        EVENT_TOOL_CONSENT_DECIDED,
        {"data": payload.model_dump(mode="json", by_alias=True)},
    )
    parts = translate_lifecycle_envelope(envelope)
    assert len(parts) == 1
    assert parts[0].data.state == "approved"


def test_translate_elicitation_requested_emits_data_elicitation_part():
    payload = ElicitationData(
        elicitation_id="e1",
        server_name="aperag-fs",
        prompt="Provide path",
        schema={"required": ["path"]},
        response=None,
        state="pending",
    )
    envelope = _envelope(
        EVENT_TOOL_ELICITATION_REQUESTED,
        {"data": payload.model_dump(mode="json", by_alias=True)},
    )
    parts = translate_lifecycle_envelope(envelope)
    assert len(parts) == 1
    assert isinstance(parts[0], DataElicitationPart)
    assert parts[0].data.elicitation_id == "e1"
    assert parts[0].data.state == "pending"


def test_translate_accepts_unwrapped_envelope_data_too():
    """Some legacy / hand-crafted envelopes pass the inner shape directly.

    The translator should accept either ``{"data": {...}}`` (canonical
    wrapped) or the inner dict directly.
    """

    payload = ElicitationData(
        elicitation_id="e1",
        server_name="aperag-fs",
        prompt="?",
        schema={},
        state="pending",
    )
    envelope = _envelope(
        EVENT_TOOL_ELICITATION_REQUESTED,
        payload.model_dump(mode="json", by_alias=True),
    )
    parts = translate_lifecycle_envelope(envelope)
    assert len(parts) == 1
    assert parts[0].data.elicitation_id == "e1"


@pytest.mark.asyncio
async def test_lifecycle_emitter_request_consent_returns_envelope_payload():
    consent = ConsentService()
    elicitation = ElicitationService()
    emitter = LifecycleEmitter(consent=consent, elicitation=elicitation)
    emission = await emitter.request_consent(
        consent_id="c1",
        turn_id="turn-1",
        user_id="user-1",
        tool_name="t",
        raw_args={"x": 1},
        risk="writes_user_data",
    )
    assert emission.event_type == EVENT_TOOL_CONSENT_REQUESTED
    assert emission.payload.state == "pending"
    assert "data" in emission.envelope_data


@pytest.mark.asyncio
async def test_lifecycle_emitter_resolved_envelope_after_decision():
    consent = ConsentService()
    elicitation = ElicitationService()
    emitter = LifecycleEmitter(consent=consent, elicitation=elicitation)
    await emitter.request_consent(
        consent_id="c1",
        turn_id="turn-1",
        user_id="user-1",
        tool_name="t",
        raw_args={"x": 1},
        risk="writes_user_data",
    )
    await consent.decide("c1", "approved", actor_user_id="user-1")
    emission = await emitter.consent_decision_envelope("c1")
    assert emission.event_type == EVENT_TOOL_CONSENT_DECIDED
    assert emission.payload.state == "approved"


@pytest.mark.asyncio
async def test_lifecycle_emitter_request_elicitation_returns_envelope_payload():
    consent = ConsentService()
    elicitation = ElicitationService()
    emitter = LifecycleEmitter(consent=consent, elicitation=elicitation)
    emission = await emitter.request_elicitation(
        elicitation_id="e1",
        turn_id="turn-1",
        user_id="user-1",
        server_name="aperag-fs",
        prompt="?",
        schema={"required": ["x"]},
    )
    assert emission.event_type == EVENT_TOOL_ELICITATION_REQUESTED
    assert emission.payload.state == "pending"
