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

"""Contract tests for Phase 8 D8.1 (#73) — backend AI SDK v5 wire emitter.

Coverage:
* ``translate_envelope`` — one test per envelope-type → stream-part(s)
  mapping declared in the D8.1 lane spec.
* ``StreamPart`` discriminated-union JSON round-trip.
* ``safe_tool_name_resolver`` hook (#75 chenyexuan plug-in seam).
* SSE route — header advertises ``x-vercel-ai-ui-message-stream: v1``
  and ``Last-Event-ID`` resume skips already-seen envelopes.

These tests are fully self-contained — no Redis / DB / HTTP server.
The runtime is monkey-patched on the route module, mirroring the
existing ``tests/unit_test/agent_runtime/test_agent_runtime_v3.py``
pattern so #73 stays consistent with the rest of the agent_runtime
suite.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional

import pytest

from aperag.domains.agent_runtime.api import routes as agent_runtime_view
from aperag.domains.agent_runtime.db.models import AgentTurnStatus
from aperag.domains.agent_runtime.schemas import (
    AgentTimelineEventEnvelope,
    UserActivityContext,
    UserActivityEnvelope,
    UserActivityIntent,
)
from aperag.domains.agent_runtime.wire import (
    StreamPartAdapter,
    TranslatorState,
    dump_part,
    parse_part,
    translate_envelope,
)
from aperag.domains.agent_runtime.wire.parts import (
    AbortPart,
    DataActivityPart,
    DataCitationPart,
    ErrorPart,
    FinishPart,
    FinishStepPart,
    SourceUrlPart,
    StartPart,
    StartStepPart,
    TextDeltaPart,
    TextEndPart,
    TextStartPart,
    ToolInputAvailablePart,
    ToolInputStartPart,
    ToolOutputAvailablePart,
    ToolOutputErrorPart,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _envelope(
    event_type: str,
    *,
    sequence: int = 1,
    event_id: Optional[str] = None,
    label: Optional[str] = None,
    status: Optional[str] = None,
    data: Optional[dict] = None,
    actor: str = "agent",
    user_activity: Optional[UserActivityEnvelope] = None,
) -> AgentTimelineEventEnvelope:
    return AgentTimelineEventEnvelope(
        event_id=event_id or f"event-{sequence}",
        turn_id="turn-1",
        sequence=sequence,
        timestamp=_now(),
        type=event_type,
        technical_type=event_type,
        label=label,
        status=status,
        actor=actor,
        data=data or {},
        user_activity=user_activity,
    )


# ---------------------------------------------------------------------------
# translator unit tests
# ---------------------------------------------------------------------------


def test_translate_turn_started():
    state = TranslatorState()
    parts = translate_envelope(_envelope("turn.started", status="running"), state)

    assert len(parts) == 2
    assert isinstance(parts[0], StartPart)
    assert parts[0].message_id == "turn-1"
    assert isinstance(parts[1], StartStepPart)


def test_translate_text_delta_lifecycle():
    state = TranslatorState()

    first = translate_envelope(
        _envelope("text.delta", sequence=2, data={"delta": "Hel"}),
        state,
    )
    second = translate_envelope(
        _envelope("text.delta", sequence=3, data={"delta": "lo"}),
        state,
    )
    answer_artifact = translate_envelope(
        _envelope(
            "artifact.created",
            sequence=4,
            label="answer",
            data={"artifact_id": "art-1", "artifact_type": "answer"},
        ),
        state,
    )

    # First delta opens the text block.
    assert len(first) == 2
    assert isinstance(first[0], TextStartPart)
    assert first[0].id == "turn-1"
    assert isinstance(first[1], TextDeltaPart)
    assert first[1].id == "turn-1"
    assert first[1].delta == "Hel"

    # Subsequent deltas reuse the open block.
    assert len(second) == 1
    assert isinstance(second[0], TextDeltaPart)
    assert second[0].id == "turn-1"
    assert second[0].delta == "lo"

    # ``answer`` artifact closes the text block.
    assert len(answer_artifact) == 1
    assert isinstance(answer_artifact[0], TextEndPart)
    assert answer_artifact[0].id == "turn-1"
    assert state.text_block_open is False


def test_translate_tool_started_finished():
    state = TranslatorState()
    started = translate_envelope(
        _envelope(
            "tool.started",
            sequence=2,
            event_id="event-tool",
            label="search_collection",
            status="started",
            actor="tool",
            data={
                "tool_name": "search_collection",
                "args": {"query": "rag pipelines"},
            },
        ),
        state,
    )
    finished = translate_envelope(
        _envelope(
            "tool.finished",
            sequence=3,
            event_id="event-tool",
            label="search_collection",
            status="success",
            actor="tool",
            data={
                "tool_name": "search_collection",
                "result": {"items": [{"id": "doc-1"}]},
            },
        ),
        state,
    )

    assert len(started) == 2
    assert isinstance(started[0], ToolInputStartPart)
    assert started[0].tool_call_id == "event-tool"
    assert started[0].tool_name == "search_collection"
    assert started[0].metadata == {}
    assert isinstance(started[1], ToolInputAvailablePart)
    assert started[1].tool_call_id == "event-tool"
    assert started[1].input == {"query": "rag pipelines"}

    assert len(finished) == 1
    assert isinstance(finished[0], ToolOutputAvailablePart)
    assert finished[0].tool_call_id == "event-tool"
    assert finished[0].output == {"items": [{"id": "doc-1"}]}
    # Strict AI SDK v5: success path has no error_text field at all.
    assert not hasattr(finished[0], "error_text")


def test_translate_tool_failure():
    state = TranslatorState()
    parts = translate_envelope(
        _envelope(
            "tool.finished",
            sequence=4,
            event_id="event-fail",
            label="web_search",
            status="error",
            actor="tool",
            data={"tool_name": "web_search", "error": "rate limited"},
        ),
        state,
    )

    assert len(parts) == 1
    out = parts[0]
    # Strict AI SDK v5: failures emit ``tool-output-error``, NOT
    # ``tool-output-available`` with errorText (D8.0c+ #89 fix-forward).
    assert isinstance(out, ToolOutputErrorPart)
    assert out.tool_call_id == "event-fail"
    assert out.error_text == "rate limited"


def test_translate_reference_bundle():
    state = TranslatorState()
    items = [
        {
            "source_id": "src-1",
            "title": "RAG Primer",
            "snippet": "Retrieval augmented generation...",
            "uri": "https://example.com/rag",
            "metadata": {"url": "https://example.com/rag"},
        },
        {
            "source_id": "src-2",
            "title": "Pipeline patterns",
            "snippet": "On chunking strategies...",
            "uri": "https://example.com/pipeline",
            "metadata": {"url": "https://example.com/pipeline", "page_number": 7},
        },
    ]
    parts = translate_envelope(
        _envelope(
            "artifact.created",
            sequence=5,
            label="reference_bundle",
            actor="system",
            data={
                "artifact_type": "reference_bundle",
                "items": items,
            },
        ),
        state,
    )

    # 2 items → 4 parts (source-url + data-citation per item).
    assert len(parts) == 4
    assert isinstance(parts[0], SourceUrlPart)
    assert parts[0].url == "https://example.com/rag"
    assert isinstance(parts[1], DataCitationPart)
    assert parts[1].data.location.type == "url_citation"

    assert isinstance(parts[2], SourceUrlPart)
    assert parts[2].url == "https://example.com/pipeline"
    assert isinstance(parts[3], DataCitationPart)
    # Page metadata triggers ``page_location`` over the default URL one.
    assert parts[3].data.location.type == "page_location"
    assert parts[3].data.location.page_number == 7


def test_translate_turn_failed():
    state = TranslatorState()
    parts = translate_envelope(
        _envelope(
            "turn.failed",
            sequence=10,
            label="Failed",
            status="failed",
            actor="system",
            data={"error": "internal error", "error_type": "RuntimeError"},
        ),
        state,
    )

    assert len(parts) == 2
    assert isinstance(parts[0], ErrorPart)
    assert parts[0].error_text == "internal error"
    assert isinstance(parts[1], FinishPart)


def test_translate_turn_cancelled():
    state = TranslatorState()
    parts = translate_envelope(
        _envelope(
            "turn.cancelled",
            sequence=11,
            label="Cancelled",
            status="cancelled",
            actor="system",
        ),
        state,
    )

    assert len(parts) == 2
    assert isinstance(parts[0], AbortPart)
    assert isinstance(parts[1], FinishPart)


def test_translate_data_activity():
    state = TranslatorState()
    activity = UserActivityEnvelope(
        intent=UserActivityIntent.SEARCHING_KNOWLEDGE,
        title_key="activity.searching_knowledge.title",
        subtitle_key="activity.searching_knowledge.subtitle",
        detail_key="activity.searching_knowledge.detail.keyword",
        context=UserActivityContext(keyword="rag pipelines"),
    )
    parts = translate_envelope(
        _envelope(
            "agent.state.changed",
            sequence=12,
            label="Searching",
            status="searching",
            user_activity=activity,
        ),
        state,
    )

    assert len(parts) == 1
    out = parts[0]
    assert isinstance(out, DataActivityPart)
    assert out.data.intent == UserActivityIntent.SEARCHING_KNOWLEDGE.value
    assert out.data.label == "Searching"
    assert out.transient is True


def test_translate_turn_completed():
    state = TranslatorState()
    parts = translate_envelope(
        _envelope("turn.completed", sequence=20, label="Completed", status="completed", actor="system"),
        state,
    )
    assert len(parts) == 2
    assert isinstance(parts[0], FinishStepPart)
    assert isinstance(parts[1], FinishPart)


def test_translator_safe_tool_name_hook():
    """#75 chenyexuan plugs in a SafeToolName resolver — the
    translator must forward the resolver result without further
    sanitisation, and must NOT mutate the shared ``TranslatorState``
    (the resolver override is per-call only)."""

    state = TranslatorState()

    def resolver(raw_name: str) -> tuple[str, dict]:
        return f"safe_{raw_name}", {"mcpServer": "aperag-knowledge-base", "mcpToolName": raw_name}

    parts = translate_envelope(
        _envelope(
            "tool.started",
            sequence=2,
            event_id="event-resolver",
            label="search_collection",
            data={"tool_name": "search_collection", "args": {}},
        ),
        state,
        safe_tool_name_resolver=resolver,
    )

    assert isinstance(parts[0], ToolInputStartPart)
    assert parts[0].tool_name == "safe_search_collection"
    assert parts[0].metadata == {
        "mcpServer": "aperag-knowledge-base",
        "mcpToolName": "search_collection",
    }
    # The override does not leak onto shared state.
    assert state.safe_tool_name_resolver is None


def test_part_json_round_trip():
    """Every declared part type round-trips through the discriminated
    union TypeAdapter without information loss."""

    samples = [
        StartPart(message_id="turn-1"),
        StartStepPart(),
        FinishStepPart(),
        FinishPart(),
        AbortPart(),
        ErrorPart(error_text="boom"),
        TextStartPart(id="turn-1"),
        TextDeltaPart(id="turn-1", delta="hi"),
        TextEndPart(id="turn-1"),
        ToolInputStartPart(
            tool_call_id="t-1",
            tool_name="search_collection",
            metadata={"mcpServer": "aperag", "mcpToolName": "search_collection"},
        ),
        ToolInputAvailablePart(tool_call_id="t-1", tool_name="search_collection", input={"q": "rag"}),
        ToolOutputAvailablePart(tool_call_id="t-1", output={"items": []}),
        ToolOutputErrorPart(tool_call_id="t-2", error_text="rate limited"),
        SourceUrlPart(source_id="src-1", url="https://example.com/rag", title="RAG Primer"),
        DataCitationPart(
            data={
                "cited_text": "RAG is...",
                "location": {"type": "url_citation", "url": "https://example.com/rag"},
            }
        ),
        DataActivityPart(data={"intent": "thinking", "label": "Thinking"}),
    ]

    for part in samples:
        wire_dict = dump_part(part)
        wire_json = json.dumps(wire_dict, ensure_ascii=False, separators=(",", ":"))
        reparsed = parse_part(json.loads(wire_json))
        # Round-trip via wire alias mapping should preserve the model.
        assert reparsed.type == part.type
        assert StreamPartAdapter.dump_python(reparsed, by_alias=True) == wire_dict


def test_part_wire_uses_camel_case_keys():
    """The wire spec uses camelCase. Smoke-test that aliased fields
    serialize correctly (the FE consumer keys on these names)."""

    tool_input = dump_part(
        ToolInputStartPart(
            tool_call_id="t-1",
            tool_name="search_collection",
            metadata={},
        )
    )
    assert "toolCallId" in tool_input
    assert "toolName" in tool_input
    assert "tool_call_id" not in tool_input

    error = dump_part(ErrorPart(error_text="boom"))
    assert "errorText" in error
    assert "error_text" not in error


def test_tool_output_strict_split_per_ai_sdk_v5():
    """Strict AI SDK v5: tool failure emits ``tool-output-error``,
    success emits ``tool-output-available``. Neither part type carries
    the other's payload (D8.0c+ #89 fix-forward)."""

    success_dict = dump_part(ToolOutputAvailablePart(tool_call_id="t-ok", output={"items": []}))
    assert success_dict["type"] == "tool-output-available"
    # Success path MUST NOT carry an errorText key on the wire.
    assert "errorText" not in success_dict
    assert "error_text" not in success_dict

    error_dict = dump_part(ToolOutputErrorPart(tool_call_id="t-fail", error_text="rate limited"))
    assert error_dict["type"] == "tool-output-error"
    assert error_dict["errorText"] == "rate limited"
    assert error_dict["toolCallId"] == "t-fail"
    # Failure path MUST NOT carry an output key on the wire.
    assert "output" not in error_dict

    # Both shapes are recognized by the discriminated union TypeAdapter.
    assert isinstance(parse_part(success_dict), ToolOutputAvailablePart)
    assert isinstance(parse_part(error_dict), ToolOutputErrorPart)


def test_tool_output_available_rejects_error_text_kwarg():
    """ToolOutputAvailablePart no longer accepts ``error_text`` —
    constructing it with one must surface as a Pydantic error rather
    than silently coexist (regression guard for #89 strict split)."""

    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ToolOutputAvailablePart.model_validate(
            {
                "type": "tool-output-available",
                "toolCallId": "t-1",
                "output": {"items": []},
                "errorText": "should not be accepted",
            }
        )


# ---------------------------------------------------------------------------
# SSE route contract tests
# ---------------------------------------------------------------------------


def _build_envelope_for_route(sequence: int, event_type: str, **overrides) -> AgentTimelineEventEnvelope:
    return _envelope(event_type, sequence=sequence, **overrides)


class _FakeRouteRequest:
    def __init__(self, *, headers: Optional[dict] = None):
        self.base_url = "http://testserver/"
        self.headers = headers or {}

    async def is_disconnected(self) -> bool:
        return False


class _FakeRouteRuntimeManager:
    def __init__(self, events: list[AgentTimelineEventEnvelope]):
        self._events = events
        self.turn_service = SimpleNamespace(
            get_turn_snapshot=self._get_turn_snapshot,
            db_ops=SimpleNamespace(query_agent_turn=self._query_agent_turn),
        )
        self.event_service = SimpleNamespace(get_events_after=self._get_events_after)

    async def _get_turn_snapshot(self, *_args, **_kwargs):
        return SimpleNamespace()

    async def _query_agent_turn(self, *_args, **_kwargs):
        return SimpleNamespace(
            status=AgentTurnStatus.COMPLETED,
            timeline_cursor=self._events[-1].sequence if self._events else 0,
        )

    async def _get_events_after(self, _turn_id, after_sequence: int = 0, limit: int = 500):
        return [event for event in self._events if event.sequence > after_sequence][:limit]


@pytest.mark.asyncio
async def test_sse_wire_header_includes_ai_sdk_v5_marker(monkeypatch):
    events = [
        _build_envelope_for_route(1, "turn.started", status="running"),
        _build_envelope_for_route(2, "turn.completed", status="completed", actor="system"),
    ]
    fake_manager = _FakeRouteRuntimeManager(events)
    monkeypatch.setattr(agent_runtime_view, "runtime_manager", fake_manager)

    response = await agent_runtime_view.stream_turn_events_view(
        _FakeRouteRequest(),
        "chat-1",
        "turn-1",
        after_sequence=0,
        user=SimpleNamespace(id="user-1"),
    )

    # AI SDK v5 protocol marker.
    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
    # Content-Type from FastAPI's StreamingResponse uses the explicit
    # media_type with the framework's default charset.
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Drain the stream and confirm wire shape.
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    joined = "".join(chunks)
    assert '"type":"start"' in joined
    assert '"type":"start-step"' in joined
    assert '"type":"finish"' in joined
    # No legacy ``event:`` SSE field on AI SDK v5.
    assert "event: turn.started" not in joined


@pytest.mark.asyncio
async def test_sse_resume_from_last_event_id(monkeypatch):
    events = [
        _build_envelope_for_route(1, "turn.started", status="running"),
        _build_envelope_for_route(
            2,
            "text.delta",
            data={"delta": "hi"},
        ),
        _build_envelope_for_route(
            5,
            "text.delta",
            data={"delta": " there"},
        ),
        _build_envelope_for_route(6, "turn.completed", status="completed", actor="system"),
    ]
    fake_manager = _FakeRouteRuntimeManager(events)
    monkeypatch.setattr(agent_runtime_view, "runtime_manager", fake_manager)

    # First connection — no Last-Event-ID, sees everything.
    full_response = await agent_runtime_view.stream_turn_events_view(
        _FakeRouteRequest(),
        "chat-1",
        "turn-1",
        after_sequence=0,
        user=SimpleNamespace(id="user-1"),
    )
    full_chunks = "".join([chunk async for chunk in full_response.body_iterator])
    assert "id: 1" in full_chunks
    assert "id: 2" in full_chunks
    assert "id: 5" in full_chunks
    assert "id: 6" in full_chunks

    # Second connection — Last-Event-ID: 5 must skip envelopes ≤ 5.
    resumed_response = await agent_runtime_view.stream_turn_events_view(
        _FakeRouteRequest(headers={"last-event-id": "5"}),
        "chat-1",
        "turn-1",
        after_sequence=0,
        user=SimpleNamespace(id="user-1"),
    )
    resumed_chunks = "".join([chunk async for chunk in resumed_response.body_iterator])
    assert "id: 1" not in resumed_chunks
    assert "id: 2" not in resumed_chunks
    assert "id: 5" not in resumed_chunks
    assert "id: 6" in resumed_chunks
