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

"""Agent-runtime UIMessage at-rest schema (Phase 8 D8.2).

Canonical source: ``docs/modularization/agent-message-protocol-design.md``
section 2 (UIMessage at-rest), with §A1-§A7 amendments from
``docs/modularization/agent-runtime-mcp-design.md``.

A ``UIMessage`` is the single durable form of one agent turn's
assistant output (and, as Phase 8 progresses, the user input that
triggered it). Its ``parts: list[UIMessagePart]`` carry every shape the
FE renderer needs — text, tool-call lifecycle, source citations,
consent prompts, elicitation requests — so the FE never has to
reconstruct messages from a parallel timeline-event stream.

Design decisions pinned in this module (locked in PR description for
#75 D8.3 to consume):

* **role**: ``user | assistant | system`` (ChatML aligned; legacy
  ``human``/``ai`` are not supported at-rest).
* **transient exclusion**: a part may carry ``transient=True`` to opt
  out of persistence — currently only ``data-activity`` (ephemeral
  thinking/searching/etc. UX state). ``persistable_parts`` strips
  these before write; the wire emitter (D8.1) still includes them in
  live SSE.
* **tool-call lifecycle**: ``tool-<safeToolName>`` parts carry
  ``state ∈ {input-streaming | input-available | output-available |
  output-error}`` per AI SDK v5; ``argsPreview`` and ``argsHash`` per
  D9 §A7 are wire+at-rest, raw ``input`` payload is **never**
  persisted — the runtime keeps it in short-TTL Redis until the
  consent decision (D8.3 territory).
* **schema_version**: tag every persisted UIMessage with the runtime
  contract version constant from ``schemas.py``
  (``agent-runtime-v3.1``) so the FE renderer can branch on
  forward-compat.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from aperag.domains.agent_runtime.schemas import (
    AGENT_RUNTIME_SCHEMA_VERSION,
    UserActivityEnvelope,
)

# ---------------------------------------------------------------------
# Citation location (Anthropic-shape, per D8 §2 design table)
# ---------------------------------------------------------------------


class CharLocation(BaseModel):
    type: Literal["char_location"] = "char_location"
    doc_index: int
    doc_title: Optional[str] = None
    start_char: int
    end_char: int


class PageLocation(BaseModel):
    type: Literal["page_location"] = "page_location"
    doc_index: int
    doc_title: Optional[str] = None
    page_index: int


class ContentBlockLocation(BaseModel):
    type: Literal["content_block_location"] = "content_block_location"
    doc_index: int
    doc_title: Optional[str] = None
    block_index: int


class UrlCitationLocation(BaseModel):
    type: Literal["url_citation"] = "url_citation"
    url: str
    title: Optional[str] = None


CitationLocation = Union[
    CharLocation,
    PageLocation,
    ContentBlockLocation,
    UrlCitationLocation,
]


# ---------------------------------------------------------------------
# Part variants
# ---------------------------------------------------------------------


class TextPart(BaseModel):
    """Plain assistant / user text content."""

    type: Literal["text"] = "text"
    text: str


_TOOL_TYPE_PATTERN = r"^tool-[A-Za-z0-9_-]+$"


class ToolPart(BaseModel):
    """AI SDK v5 consolidated tool-call part (D8 §2.4 / D9 §A1+§A6).

    The ``type`` discriminator carries the ``SafeToolName`` directly
    (e.g. ``tool-aperag_knowledge_base_search_collection``); MCP server /
    tool identity lives in ``metadata``. This matches the AI SDK v5
    *at-rest* shape (one consolidated tool part per call, lifecycle
    captured in ``state``) — not the *wire* shape, which is a sequence
    of ``tool-input-*`` / ``tool-output-*`` events emitted by #73 and
    collapsed by the FE consumer into this consolidated form.

    SafeToolName resolution (raw MCP name → ``tool-<safeName>``) is
    #75's responsibility (``aperag.domains.agent_runtime.tools``);
    storage only enforces that the caller has already produced a
    canonical ``type`` string.
    """

    type: str = Field(pattern=_TOOL_TYPE_PATTERN)
    tool_call_id: str = Field(alias="toolCallId")
    state: Literal[
        "input-streaming",
        "input-available",
        "output-available",
        "output-error",
    ]
    input: Optional[Any] = None
    output: Optional[Any] = None
    error_text: Optional[str] = Field(default=None, alias="errorText")
    # First-class user-facing one-line summary (e.g. "搜索:张飞牛肉",
    # "阅读:example.com/page"). Per-tool-call free-form ≤ 200 chars
    # extracted by the runtime from the raw input dict. Rendered by the
    # FE as the activity-step subtitle so users see "what was searched
    # / read" even after refresh — when raw ``input`` is intentionally
    # not persisted (D9 §A7). Architect ratify msg=2639aeea pinned this
    # as a first-class field rather than nested under ``metadata`` so
    # FE doesn't fish through a catch-all dict for user-facing copy.
    summary: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class SourceUrlPart(BaseModel):
    type: Literal["source-url"] = "source-url"
    source_id: str = Field(alias="sourceId")
    url: str
    title: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class SourceDocumentPart(BaseModel):
    type: Literal["source-document"] = "source-document"
    source_id: str = Field(alias="sourceId")
    media_type: str = Field(alias="mediaType")
    title: str

    model_config = ConfigDict(populate_by_name=True)


class CitationData(BaseModel):
    cited_text: str
    location: CitationLocation


class DataCitationPart(BaseModel):
    """Anthropic-shape typed citation (D8 §2 + §2.5).

    Wrapped ``data: CitationData`` shape per D8 §2 same-schema canonical
    (architect lock 2026-04-25, msg=...): wire and at-rest must round-trip
    byte-for-byte with no wire/at-rest converter layer.
    """

    type: Literal["data-citation"] = "data-citation"
    data: CitationData


class ActivityData(BaseModel):
    """Inner payload mirroring ``UserActivityEnvelope`` for wire/at-rest
    same-schema parity. ``DataActivityPart`` is transient — never persisted —
    but the wrapped shape is preserved so the live SSE stream stays
    structurally identical to persisted variants.
    """

    activity: UserActivityEnvelope


class DataActivityPart(BaseModel):
    """Ephemeral UX activity (thinking / searching / etc.).

    ``transient=True`` is hard-coded — these parts are only ever
    present in the live SSE stream; ``persistable_parts`` strips them
    before at-rest write.
    """

    type: Literal["data-activity"] = "data-activity"
    data: ActivityData
    transient: Literal[True] = True


class ToolConsentData(BaseModel):
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    metadata: dict[str, Any] = Field(default_factory=dict)
    args_preview: str = Field(alias="argsPreview")
    args_hash: str = Field(alias="argsHash")
    risk: Literal[
        "writes_user_data",
        "calls_external_api",
        "modifies_system",
        "admin_only",
    ]
    requested_at: str = Field(alias="requestedAt")
    state: Literal["pending", "approved", "denied", "expired"]

    model_config = ConfigDict(populate_by_name=True)


class DataToolConsentPart(BaseModel):
    """Tool consent request / decision (D9 §A7).

    Persisted as part of the audit trail. Raw tool args are stored
    out-of-band by the runtime; only ``args_preview`` (≤ 500 chars
    redacted preview) and ``args_hash`` (sha256 over canonical JSON)
    appear here.
    """

    type: Literal["data-tool-consent"] = "data-tool-consent"
    data: ToolConsentData


class ElicitationData(BaseModel):
    elicitation_id: str = Field(alias="elicitationId")
    server_name: str = Field(alias="serverName")
    prompt: str
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    response: Optional[dict[str, Any]] = None
    state: Literal["pending", "answered", "cancelled"]

    model_config = ConfigDict(populate_by_name=True)


class DataElicitationPart(BaseModel):
    """User input request (D9 §5).

    Persisted — the user's response becomes part of message history.
    ``schema`` is the JSON-Schema fragment the FE should render as a
    form; ``response`` is the validated submission.
    """

    type: Literal["data-elicitation"] = "data-elicitation"
    data: ElicitationData


UIMessagePart = Union[
    TextPart,
    ToolPart,
    SourceUrlPart,
    SourceDocumentPart,
    DataCitationPart,
    DataActivityPart,
    DataToolConsentPart,
    DataElicitationPart,
]


# ---------------------------------------------------------------------
# UIMessage envelope
# ---------------------------------------------------------------------


class UIMessage(BaseModel):
    """At-rest message envelope. One per assistant turn (D8.2 first cut).

    A ``UIMessage`` is the durable transcript form for a single
    role-tagged contribution to a chat. Its ``parts`` carry every
    renderable shape the FE supports; ``schema_version`` lets the FE
    branch on forward-compat without a separate version negotiation.
    """

    schema_version: str = AGENT_RUNTIME_SCHEMA_VERSION
    id: str
    role: Literal["user", "assistant", "system"]
    parts: list[UIMessagePart] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Snapshot endpoint envelope (Phase 8 D8.4d / #90)
# ---------------------------------------------------------------------


RuntimeKind = Literal["agent_runtime", "direct_chat", "rag_chat"]


class AgentTurnSnapshot(BaseModel):
    """Canonical UIMessage at-rest snapshot for an agent turn (Phase 8 D8.4d / #90, D8.5-BE / #92).

    Replaces the legacy ``{turn, timeline, artifacts}`` snapshot shape
    with the ``UIMessage``-aligned canonical that the FE renderer
    (#76 / #77 / #78) consumes from both the live SSE stream and the
    at-rest reload path. Per D8 §2 wire / at-rest are byte-equal, so
    no client-side converter is required: a snapshot reload
    deserializes through the same ``UIMessagePart`` discriminated
    union the wire emits.

    ``runtime_kind`` (D8.5-BE) tags the runtime that produced the
    turn — ``agent_runtime`` for the agent reasoning loop,
    ``direct_chat`` / ``rag_chat`` reserved for future paths. ``role``
    keeps speaker semantics independent of runtime kind.

    ``parts`` is the persistable subset (transient ``data-activity``
    is stripped before write per :func:`persistable_parts`);
    ``status`` mirrors the runtime ``AgentTurnStatus`` enum value;
    ``error_text`` surfaces an ``error_summary`` artifact's message
    for failed turns. Turn-shape metadata (timestamps, ids, cursor)
    is kept top-level so the FE can render without a separate
    envelope round-trip.

    Lives in this module rather than ``schemas`` to keep the
    ``UIMessage`` family (parts + envelope) in a single source-of-truth
    file; ``schemas`` re-exports the name for back-compat with
    existing import sites.
    """

    schema_version: str = AGENT_RUNTIME_SCHEMA_VERSION
    turn_id: str
    chat_id: str
    runtime_kind: RuntimeKind = "agent_runtime"
    input_text: Optional[str] = None
    role: Literal["assistant"] = "assistant"
    status: str
    parts: list[UIMessagePart] = Field(default_factory=list)
    error_text: Optional[str] = None
    timeline_cursor: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _is_transient(part: UIMessagePart) -> bool:
    """Return True if the part declares ``transient=True``.

    Currently only ``DataActivityPart`` has the transient flag, but the
    helper is structured to handle future transient variants without
    edits.
    """
    return getattr(part, "transient", False) is True


def persistable_parts(parts: list[UIMessagePart]) -> list[UIMessagePart]:
    """Filter ``parts`` down to the at-rest persistable subset.

    The wire emitter (D8.1) calls this before writing to Redis snapshot
    or DB; the live SSE stream is unaffected. Preserves order.
    """
    return [part for part in parts if not _is_transient(part)]


def args_preview(raw: Any, *, limit: int = 500) -> str:
    """Build a redacted args preview suitable for at-rest persistence.

    Per D9 §A7 the raw tool input must never reach the wire or the at-rest
    store; we keep a length-bounded JSON-stringified preview for audit
    correlation. Long previews are truncated with an explicit marker.
    """
    try:
        text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(raw)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def args_hash(raw: Any) -> str:
    """sha256 of the canonical JSON form of ``raw`` (D9 §A7).

    Stable hash so consent audit + runtime memory can correlate across
    PR/process/replay; always 64 lowercase hex chars.
    """
    try:
        canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        canonical = str(raw)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
