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


class ToolPart(BaseModel):
    """AI SDK v5 tool-call lifecycle part.

    The ``type`` literal is ``tool-<safeToolName>`` per D8 §2.4 (e.g.
    ``tool-aperag_knowledge_base_search_collection``). At-rest we store
    the prefix-less convention via ``tool_name`` and the FE/wire emit
    handles the prefix join — this avoids embedding the ``-<name>``
    fragment in pydantic discriminator literals while keeping the AI
    SDK part identity intact.
    """

    type: Literal["tool"] = "tool"
    tool_name: str
    tool_call_id: str
    state: Literal[
        "input-streaming",
        "input-available",
        "output-available",
        "output-error",
    ]
    args_preview: Optional[str] = None
    args_hash: Optional[str] = None
    output: Optional[Any] = None
    error_text: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceUrlPart(BaseModel):
    type: Literal["source-url"] = "source-url"
    source_id: str
    url: str
    title: Optional[str] = None


class SourceDocumentPart(BaseModel):
    type: Literal["source-document"] = "source-document"
    source_id: str
    media_type: str
    title: str


class DataCitationPart(BaseModel):
    """Anthropic-shape typed citation (D8 §2 + §2.5)."""

    type: Literal["data-citation"] = "data-citation"
    cited_text: str
    location: CitationLocation


class DataActivityPart(BaseModel):
    """Ephemeral UX activity (thinking / searching / etc.).

    ``transient=True`` is hard-coded — these parts are only ever
    present in the live SSE stream; ``persistable_parts`` strips them
    before at-rest write.
    """

    type: Literal["data-activity"] = "data-activity"
    activity: UserActivityEnvelope
    transient: Literal[True] = True


class DataToolConsentPart(BaseModel):
    """Tool consent request / decision (D9 §A7).

    Persisted as part of the audit trail. Raw tool args are stored
    out-of-band by the runtime; only ``args_preview`` (≤ 500 chars
    redacted preview) and ``args_hash`` (sha256 over canonical JSON)
    appear here.
    """

    type: Literal["data-tool-consent"] = "data-tool-consent"
    tool_call_id: str
    tool_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    args_preview: str
    args_hash: str
    risk: Literal[
        "writes_user_data",
        "calls_external_api",
        "modifies_system",
        "admin_only",
    ]
    requested_at: str
    state: Literal["pending", "approved", "denied", "expired"]


class DataElicitationPart(BaseModel):
    """User input request (D9 §5).

    Persisted — the user's response becomes part of message history.
    ``schema`` is the JSON-Schema fragment the FE should render as a
    form; ``response`` is the validated submission.
    """

    type: Literal["data-elicitation"] = "data-elicitation"
    elicitation_id: str
    prompt: str
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    response: Optional[dict[str, Any]] = None
    state: Literal["pending", "submitted", "cancelled"]

    model_config = ConfigDict(populate_by_name=True)


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
