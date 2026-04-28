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

"""``AgentTimelineEventEnvelope`` → AI SDK v5 stream parts translator.

The translator is a pure function (modulo a small per-turn state
object) that maps each timeline envelope to one-or-more
``StreamPart`` instances. The SSE route in ``api/routes.py`` runs
each emitted envelope through ``translate_envelope`` and yields the
resulting parts as individual SSE frames.

## Sequence-id semantics on fan-out

Multiple parts can share the source envelope's sequence number — for
instance, ``turn.started`` fans out to ``[start, start-step]``, and
an ``artifact.created(reference_bundle)`` with N items fans out to
``2N`` parts. To keep ``Last-Event-ID`` resume coherent (the
client's cursor must always point at the **next** envelope, not at a
mid-fanout offset), the SSE route writes the ``id:`` SSE field only
on the **last** part of a fan-out group; intermediate parts are
emitted as continuation frames without an ``id:``. On reconnect the
client resumes from the next envelope, which the translator
processes from a fresh ``TranslatorState``-aware position (or, for
text blocks, the FE consumer reconciles based on ``text-start`` /
``text-end`` IDs the server sent on the first attempt).

## Hooks reserved for #75 chenyexuan

The translator exposes a ``safe_tool_name_resolver`` keyword on
``translate_envelope`` (and threaded through ``TranslatorState``)
typed as ``Callable[[str], tuple[str, dict[str, Any]]] | None``.
When provided, raw tool names from the envelope are run through the
resolver; the returned ``(safe_name, metadata)`` populates the
``tool-input-start`` part. Default behavior (resolver=None) forwards
the raw name with empty metadata — matching pre-D9 wire shape so #73
can land without #75 being merged first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from aperag.domains.agent_runtime.schemas import (
    AgentTimelineEventEnvelope,
    UserActivityEnvelope,
)
from aperag.domains.agent_runtime.wire.parts import (
    AbortPart,
    ActivityData,
    CitationData,
    CitationLocation,
    DataActivityPart,
    DataCitationPart,
    ErrorPart,
    FinishPart,
    FinishStepPart,
    ReasoningDeltaPart,
    ReasoningEndPart,
    ReasoningStartPart,
    SourceUrlPart,
    StartPart,
    StartStepPart,
    StreamPart,
    TextDeltaPart,
    TextEndPart,
    TextStartPart,
    ToolInputAvailablePart,
    ToolInputStartPart,
    ToolOutputAvailablePart,
    ToolOutputErrorPart,
)

SafeToolNameResolver = Callable[[str], tuple[str, dict[str, Any]]]


@dataclass
class TranslatorState:
    """Per-turn translator bookkeeping.

    A fresh instance is created for each SSE stream (see
    ``stream_turn_events_view`` in ``api/routes.py``). Cross-turn
    state would be a bug — text-block ID reuse across turns would
    confuse the FE consumer. The state object intentionally has no
    ``turn_id`` field so callers can't accidentally swap it between
    turns mid-stream.

    Fields:
        text_block_open: True once a ``text-start`` has been emitted
            and not yet matched by a ``text-end``.
        text_block_id: The wire-side ``id`` of the open text block
            (we use ``turn_id`` per the D8.1 mapping table).
        safe_tool_name_resolver: Optional hook owned by #75 — see
            module docstring.
    """

    text_block_open: bool = False
    text_block_id: Optional[str] = None
    # Reasoning block state — mirror of text-block bookkeeping. Open on
    # first ``reasoning.delta``, closed when a non-reasoning event
    # arrives (tool call / text delta / turn end). Lets the FE
    # accumulate one chunk per "thinking N" block without each delta
    # spawning a new one.
    reasoning_block_open: bool = False
    reasoning_block_id: Optional[str] = None
    safe_tool_name_resolver: Optional[SafeToolNameResolver] = None
    _user_activities: dict[int, UserActivityEnvelope] = field(default_factory=dict)


def _resolve_tool_name(raw_name: str, resolver: Optional[SafeToolNameResolver]) -> tuple[str, dict[str, Any]]:
    """Run the raw tool name through the optional safe-name resolver.

    Returns ``(wire_name, metadata)``. With no resolver, ``metadata``
    is an empty dict — #75 will swap in a real resolver that returns
    SafeToolName + ``{mcpServer, mcpToolName}`` per D9 §A1+A6.
    """

    if resolver is None:
        # TODO(#75 chenyexuan): plug SafeToolName resolver to populate
        # metadata={"mcpServer": ..., "mcpToolName": ...} per D9 §A6.
        return raw_name, {}
    return resolver(raw_name)


def _user_activity_part(envelope: AgentTimelineEventEnvelope) -> Optional[DataActivityPart]:
    """Project the envelope's ``user_activity`` field onto a transient
    ``data-activity`` part. Returns ``None`` if the envelope has no
    activity (e.g. raw cached events that never went through
    ``EventService.adapt_event_envelope``)."""

    activity = envelope.user_activity
    if activity is None:
        return None
    return DataActivityPart(data=ActivityData(intent=activity.intent.value, label=envelope.label))


def _coerce_jsonish(value: Any) -> Any:
    """Best-effort decode of a JSON-string payload back into a Python
    object so the wire frame carries structured ``input`` / ``output``
    rather than a doubly-encoded string. Non-JSON strings pass
    through unchanged. Mirrors the runtime's ``_normalize_jsonish``
    behaviour but is duplicated here so the wire layer doesn't reach
    into the runtime module."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _extract_tool_input(envelope: AgentTimelineEventEnvelope) -> Any:
    data = envelope.data or {}
    if "args" in data:
        return _coerce_jsonish(data.get("args"))
    return None


def _extract_tool_output(envelope: AgentTimelineEventEnvelope) -> Any:
    data = envelope.data or {}
    if "result" in data:
        return _coerce_jsonish(data.get("result"))
    return None


def _is_failure_status(status: Optional[str]) -> bool:
    if not status:
        return False
    return status.lower() in {"error", "failed", "failure"}


def _reference_items(envelope: AgentTimelineEventEnvelope) -> list[dict[str, Any]]:
    """Pull the materialised reference list out of an
    ``artifact.created(reference_bundle)`` envelope.

    The runtime emits the bundle artifact with a ``data`` payload
    containing only ``artifact_id`` + ``artifact_type`` (the actual
    items live in the artifact row). For the wire layer we pull the
    items off ``data["items"]`` if the runtime inlined them; if not
    we fall back to ``data["payload"]["items"]``. When neither is
    present we emit an empty list and the caller emits no parts —
    the artifact ID is still resolvable via the snapshot endpoint.
    """

    data = envelope.data or {}
    items = data.get("items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    payload = data.get("payload")
    if isinstance(payload, dict):
        nested = payload.get("items")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    return []


def _build_citation_location(item: dict[str, Any]) -> CitationLocation:
    """Pick the best-fit citation location for a reference item.

    Default is ``url_citation`` (matches the current RAG tools that
    return URLs); ``page_location`` / ``char_location`` /
    ``content_block_location`` fire when the item metadata advertises
    page / char-offset / block-index keys.
    """

    metadata = item.get("metadata") or {}
    url = item.get("uri") or metadata.get("url")
    title = item.get("title") or metadata.get("title")

    page_number = metadata.get("page_number") or metadata.get("page")
    if isinstance(page_number, int):
        return CitationLocation(
            type="page_location",
            page_number=page_number,
            url=url,
            title=title,
        )

    start = metadata.get("start_char_index")
    end = metadata.get("end_char_index")
    if isinstance(start, int) and isinstance(end, int):
        return CitationLocation(
            type="char_location",
            start_char_index=start,
            end_char_index=end,
            url=url,
            title=title,
        )

    block_index = metadata.get("content_block_index")
    if isinstance(block_index, int):
        return CitationLocation(
            type="content_block_location",
            content_block_index=block_index,
            url=url,
            title=title,
        )

    return CitationLocation(type="url_citation", url=url or "", title=title)


def _translate_reference_bundle(
    envelope: AgentTimelineEventEnvelope,
) -> list[StreamPart]:
    parts: list[StreamPart] = []
    items = _reference_items(envelope)
    for index, item in enumerate(items):
        source_id = item.get("source_id") or item.get("id") or f"{envelope.event_id}-ref-{index}"
        url = item.get("uri") or (item.get("metadata") or {}).get("url")
        title = item.get("title") or (item.get("metadata") or {}).get("title")
        if url:
            parts.append(SourceUrlPart(source_id=str(source_id), url=url, title=title))
        snippet = item.get("snippet") or item.get("content") or ""
        location = _build_citation_location(item)
        parts.append(DataCitationPart(data=CitationData(cited_text=str(snippet), location=location)))
    return parts


def _translate_artifact(envelope: AgentTimelineEventEnvelope, state: TranslatorState) -> list[StreamPart]:
    artifact_type = (envelope.data or {}).get("artifact_type") or envelope.label
    artifact_type = (artifact_type or "").lower()

    if artifact_type == "answer":
        # Close the streaming text block opened by the first text.delta.
        if state.text_block_open and state.text_block_id is not None:
            text_id = state.text_block_id
            state.text_block_open = False
            state.text_block_id = None
            return [TextEndPart(id=text_id)]
        return []

    if artifact_type == "reference_bundle":
        return _translate_reference_bundle(envelope)

    if artifact_type == "error_summary":
        message = (envelope.data or {}).get("message") or envelope.label or "agent error"
        return [ErrorPart(error_text=str(message))]

    return []


def _translate_tool_started(envelope: AgentTimelineEventEnvelope, state: TranslatorState) -> list[StreamPart]:
    data = envelope.data or {}
    raw_tool_name = data.get("tool_name") or envelope.label or "tool"
    tool_call_id = str(data.get("tool_call_id") or envelope.event_id)
    safe_name, metadata = _resolve_tool_name(str(raw_tool_name), state.safe_tool_name_resolver)
    # Forward resolved titles (document_title / collection_title) on
    # the wire so the FE renderer surfaces real names DURING streaming
    # too — not only after refresh from the at-rest ToolPart.metadata.
    # Mirrors the at-rest composition in ``_compose_assistant_parts``.
    resolved_titles = data.get("titles")
    if isinstance(resolved_titles, dict) and resolved_titles:
        metadata = {**metadata, **resolved_titles}
    tool_input = _extract_tool_input(envelope)
    return [
        ToolInputStartPart(
            tool_call_id=tool_call_id,
            tool_name=safe_name,
            metadata=metadata,
        ),
        ToolInputAvailablePart(
            tool_call_id=tool_call_id,
            tool_name=safe_name,
            input=tool_input,
        ),
    ]


def _translate_tool_finished(envelope: AgentTimelineEventEnvelope) -> list[StreamPart]:
    data = envelope.data or {}
    tool_call_id = str(data.get("tool_call_id") or envelope.event_id)
    if _is_failure_status(envelope.status):
        # Per AI SDK v5 strict spec the failure path emits a separate
        # ``tool-output-error`` event, not ``tool-output-available`` with
        # an embedded ``errorText``. Best-effort error text — fall back
        # to the envelope label so the FE always has something to
        # render even when the tool didn't surface an explicit error
        # message.
        error_text = (
            str(data.get("error"))
            if data.get("error")
            else f"tool {envelope.label or 'unknown'} failed with status={envelope.status}"
        )
        return [
            ToolOutputErrorPart(
                tool_call_id=tool_call_id,
                error_text=error_text,
            )
        ]
    return [
        ToolOutputAvailablePart(
            tool_call_id=tool_call_id,
            output=_extract_tool_output(envelope),
        )
    ]


def _translate_text_delta(envelope: AgentTimelineEventEnvelope, state: TranslatorState) -> list[StreamPart]:
    delta = (envelope.data or {}).get("delta") or ""
    parts: list[StreamPart] = []
    # Close any open reasoning block before opening / continuing the
    # text block. The runtime's chunk-on-tool-call boundary already
    # handles closing reasoning at tool starts; this catches the
    # transition from reasoning straight to the answer text (no tool
    # call between them).
    parts.extend(_close_reasoning_block_if_open(state))
    if not state.text_block_open:
        text_id = envelope.turn_id
        state.text_block_open = True
        state.text_block_id = text_id
        parts.append(TextStartPart(id=text_id))
    assert state.text_block_id is not None  # invariant: opened above
    parts.append(TextDeltaPart(id=state.text_block_id, delta=str(delta)))
    return parts


def _close_reasoning_block_if_open(state: TranslatorState) -> list[StreamPart]:
    """Emit a ``reasoning-end`` if a reasoning block is currently open
    and reset the state. Called by anything that interrupts reasoning
    (text delta, tool start, turn end, artifact, ...). Idempotent —
    no-op when no block is open."""
    if state.reasoning_block_open and state.reasoning_block_id is not None:
        block_id = state.reasoning_block_id
        state.reasoning_block_open = False
        state.reasoning_block_id = None
        return [ReasoningEndPart(id=block_id)]
    return []


def _translate_reasoning_delta(envelope: AgentTimelineEventEnvelope, state: TranslatorState) -> list[StreamPart]:
    """Mirror of :func:`_translate_text_delta` for reasoning streams.

    Wave 9 task #2 followup #2: the runtime now emits
    ``reasoning.delta`` events carrying the model's actual thinking
    text (not just status badges). The FE consumes these as
    AI SDK v5 ``reasoning-start`` / ``reasoning-delta`` / ``reasoning-end``
    parts so the activity timeline can render Claude/Cursor-style
    "思考N" blocks at the right chronological position.

    The block stays open across consecutive ``reasoning.delta`` events
    and closes when any other event type arrives (tool start / text
    delta / turn end), so each persisted ``ReasoningPart`` chunk maps
    cleanly to one wire reasoning block.
    """
    delta = (envelope.data or {}).get("delta") or ""
    parts: list[StreamPart] = []
    if not state.reasoning_block_open:
        # Use a per-block id derived from turn_id + sequence so
        # multiple reasoning blocks in one turn don't collide on the
        # FE side. ``envelope.event_id`` is unique per emit and stable
        # for the duration of one block's first delta — good enough.
        block_id = f"reasoning-{envelope.event_id}"
        state.reasoning_block_open = True
        state.reasoning_block_id = block_id
        parts.append(ReasoningStartPart(id=block_id))
    assert state.reasoning_block_id is not None  # invariant: opened above
    parts.append(ReasoningDeltaPart(id=state.reasoning_block_id, delta=str(delta)))
    return parts


def translate_envelope(
    envelope: AgentTimelineEventEnvelope,
    state: TranslatorState,
    *,
    safe_tool_name_resolver: Optional[SafeToolNameResolver] = None,
) -> list[StreamPart]:
    """Translate one timeline envelope into one-or-more stream parts.

    Args:
        envelope: The runtime-emitted ``AgentTimelineEventEnvelope``.
        state: Per-turn ``TranslatorState`` (shared across calls for
            the same turn — caller must keep one instance per stream).
        safe_tool_name_resolver: Optional override for the resolver
            stored on ``state``. When provided, it shadows
            ``state.safe_tool_name_resolver`` for this single call —
            useful for tests that want to assert the hook fires
            without mutating shared state.

    Returns:
        Ordered list of ``StreamPart`` instances. Empty list means
        the envelope produces no FE-visible part (e.g. a raw event
        type the wire layer doesn't surface). Callers should still
        forward the SSE ``id:`` cursor on no-op envelopes so resume
        keeps moving.
    """

    # Local override without mutating shared state — see docstring.
    effective_state = state
    if safe_tool_name_resolver is not None:
        effective_state = TranslatorState(
            text_block_open=state.text_block_open,
            text_block_id=state.text_block_id,
            safe_tool_name_resolver=safe_tool_name_resolver,
        )

    event_type = envelope.type

    if event_type == "turn.started":
        return [StartPart(message_id=envelope.turn_id), StartStepPart()]

    if event_type == "agent.state.changed":
        activity = _user_activity_part(envelope)
        return [activity] if activity else []

    if event_type == "text.delta":
        parts = _translate_text_delta(envelope, effective_state)
        # Mirror state mutation back onto the caller's instance when
        # we used a shadow copy for the resolver override.
        if effective_state is not state:
            state.text_block_open = effective_state.text_block_open
            state.text_block_id = effective_state.text_block_id
            state.reasoning_block_open = effective_state.reasoning_block_open
            state.reasoning_block_id = effective_state.reasoning_block_id
        return parts

    if event_type == "reasoning.delta":
        parts = _translate_reasoning_delta(envelope, effective_state)
        if effective_state is not state:
            state.reasoning_block_open = effective_state.reasoning_block_open
            state.reasoning_block_id = effective_state.reasoning_block_id
        return parts

    if event_type in {"tool.started", "external_action.started"}:
        # A tool call interrupts any open reasoning block — close it
        # so the FE sees one chunk per "thinking N" block, mirroring
        # the runtime's at-rest chunk-on-tool-call boundary.
        parts = _close_reasoning_block_if_open(effective_state)
        parts.extend(_translate_tool_started(envelope, effective_state))
        if effective_state is not state:
            state.reasoning_block_open = effective_state.reasoning_block_open
            state.reasoning_block_id = effective_state.reasoning_block_id
        return parts

    if event_type in {"tool.finished", "external_action.finished"}:
        return _translate_tool_finished(envelope)

    if event_type == "artifact.created":
        parts = _translate_artifact(envelope, effective_state)
        if effective_state is not state:
            state.text_block_open = effective_state.text_block_open
            state.text_block_id = effective_state.text_block_id
        return parts

    if event_type == "turn.completed":
        # Close any open reasoning block on turn end so the FE never
        # sees a dangling reasoning-start without a matching
        # reasoning-end (mirrors how text blocks close at turn end).
        parts = _close_reasoning_block_if_open(effective_state)
        if effective_state is not state:
            state.reasoning_block_open = effective_state.reasoning_block_open
            state.reasoning_block_id = effective_state.reasoning_block_id
        parts.extend([FinishStepPart(), FinishPart()])
        return parts

    if event_type == "turn.failed":
        message = (envelope.data or {}).get("error") or envelope.label or "turn failed"
        parts = _close_reasoning_block_if_open(effective_state)
        if effective_state is not state:
            state.reasoning_block_open = effective_state.reasoning_block_open
            state.reasoning_block_id = effective_state.reasoning_block_id
        parts.extend([ErrorPart(error_text=str(message)), FinishPart()])
        return parts

    if event_type == "turn.cancelled":
        parts = _close_reasoning_block_if_open(effective_state)
        if effective_state is not state:
            state.reasoning_block_open = effective_state.reasoning_block_open
            state.reasoning_block_id = effective_state.reasoning_block_id
        parts.extend([AbortPart(), FinishPart()])
        return parts

    return []


__all__ = [
    "SafeToolNameResolver",
    "TranslatorState",
    "translate_envelope",
]
