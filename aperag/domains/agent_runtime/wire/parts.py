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

"""AI SDK v5 ``UI Message Stream Protocol`` part models.

This module declares the wire-level Pydantic models that the agent
runtime SSE emitter pushes down to the FE ``@ai-sdk/react`` consumer.
Each part has a literal ``type`` discriminator; the union is exposed
both as ``StreamPart`` (typing alias for static checkers) and as a
``TypeAdapter[StreamPart]`` (``StreamPartAdapter``) so callers can
serialize / deserialize without a long if-isinstance chain.

The set of part types tracked here is the subset the agent runtime
actually emits today plus the placeholders needed by adjacent lanes:

* lifecycle: ``start`` / ``start-step`` / ``finish-step`` / ``finish``
  / ``abort`` / ``error``
* text: ``text-start`` / ``text-delta`` / ``text-end``
* tool lifecycle: ``tool-input-start`` / ``tool-input-delta`` /
  ``tool-input-available`` / ``tool-output-available`` /
  ``tool-output-error``
* sources: ``source-url`` / ``source-document``
* ApeRAG custom: ``data-citation`` / ``data-activity``
* placeholders for #75 chenyexuan: ``data-tool-consent`` /
  ``data-elicitation`` (declared so the discriminated union accepts
  them; emission flow lives in #75)

Naming follows the AI SDK v5 spec: hyphenated literals on the wire,
``snake_case`` field names on the Python side, with explicit
``Field(..., alias=...)`` only where the wire name conflicts with a
Python keyword (none currently do — the ``type`` field is fine).

The ``transient`` flag on ``data-activity`` is part of the AI SDK v5
contract (transient parts are not persisted into the at-rest UIMessage
history); the FE consumer drops it on assembly. We hard-code it to
``True`` rather than expose it as a constructor knob, mirroring the
spec.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# ---------------------------------------------------------------------------
# Lifecycle parts
# ---------------------------------------------------------------------------


class StartPart(BaseModel):
    """First frame on every stream — opens the assistant message."""

    type: Literal["start"] = "start"
    message_id: Optional[str] = Field(default=None, alias="messageId")

    model_config = ConfigDict(populate_by_name=True)


class StartStepPart(BaseModel):
    """Marks the start of a single agent reasoning + tool use step."""

    type: Literal["start-step"] = "start-step"


class FinishStepPart(BaseModel):
    """Closes the current step (paired with ``start-step``)."""

    type: Literal["finish-step"] = "finish-step"


class FinishPart(BaseModel):
    """Final frame — assistant turn complete."""

    type: Literal["finish"] = "finish"


class AbortPart(BaseModel):
    """Emitted when the turn is cancelled by the client / runtime."""

    type: Literal["abort"] = "abort"


class ErrorPart(BaseModel):
    """Stream-level error frame; FE surfaces ``errorText`` in the UI."""

    type: Literal["error"] = "error"
    error_text: str = Field(alias="errorText")

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Text parts
# ---------------------------------------------------------------------------


class TextStartPart(BaseModel):
    """Opens a streaming text block; subsequent ``text-delta`` parts
    must reuse the same ``id`` until a ``text-end`` closes the block."""

    type: Literal["text-start"] = "text-start"
    id: str


class TextDeltaPart(BaseModel):
    """Incremental token chunk for an open text block."""

    type: Literal["text-delta"] = "text-delta"
    id: str
    delta: str


class TextEndPart(BaseModel):
    """Closes the text block opened by ``text-start``."""

    type: Literal["text-end"] = "text-end"
    id: str


# ---------------------------------------------------------------------------
# Tool lifecycle parts
# ---------------------------------------------------------------------------


class ToolInputStartPart(BaseModel):
    """Announces an upcoming tool call.

    ``tool_name`` carries the wire-side tool name. Per D9 §A1+A6 this
    will be a ``SafeToolName`` (provider-safe ``[a-zA-Z0-9_-]``); for
    now (#73 scope) the translator forwards the raw envelope tool name
    and #75 (chenyexuan) will swap in the SafeToolName resolver.
    """

    type: Literal["tool-input-start"] = "tool-input-start"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class ToolInputDeltaPart(BaseModel):
    """Streaming partial JSON for the tool input.

    The current envelope contract delivers tool args atomically (no
    partial streaming), so the runtime emits ``tool-input-available``
    directly without intermediate deltas. This model is declared so
    the union accepts deltas the moment streaming-args support lands.
    """

    type: Literal["tool-input-delta"] = "tool-input-delta"
    tool_call_id: str = Field(alias="toolCallId")
    input_text_delta: str = Field(alias="inputTextDelta")

    model_config = ConfigDict(populate_by_name=True)


class ToolInputAvailablePart(BaseModel):
    """Final tool input — validated args ready for execution."""

    type: Literal["tool-input-available"] = "tool-input-available"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    input: Any = None

    model_config = ConfigDict(populate_by_name=True)


class ToolOutputAvailablePart(BaseModel):
    """Tool finished successfully — carries the tool's ``output`` payload.

    Per AI SDK v5 strict spec, success and failure are split onto two
    distinct wire events: ``tool-output-available`` (this class) for the
    success path and :class:`ToolOutputErrorPart` for the failure path.
    A failed tool call must NOT be emitted as ``tool-output-available``
    with a populated ``errorText``; the FE reducer collapses both events
    into the consolidated at-rest :class:`ToolPart` state machine
    (``state ∈ output-available | output-error``).

    ``extra="forbid"`` actively rejects the legacy
    ``ToolOutputAvailablePart(errorText=...)`` shape so that residual
    callers from before the D8.0c+ #89 split surface as a clean
    ``ValidationError`` instead of silently masquerading as success.
    """

    type: Literal["tool-output-available"] = "tool-output-available"
    tool_call_id: str = Field(alias="toolCallId")
    output: Any = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ToolOutputErrorPart(BaseModel):
    """Tool finished with a failure — carries the ``error_text``.

    Per AI SDK v5 strict spec the error path is a separate wire event
    type; ``error_text`` is required (the FE always has something to
    render) and there is no ``output`` field — partial failure outputs
    are flattened into the textual error_text by the runtime emitter.

    ``extra="forbid"`` mirrors the success-path discipline so an
    accidental ``output=...`` kwarg here surfaces as a
    ``ValidationError`` rather than silently flowing through.
    """

    type: Literal["tool-output-error"] = "tool-output-error"
    tool_call_id: str = Field(alias="toolCallId")
    error_text: str = Field(alias="errorText")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# ---------------------------------------------------------------------------
# Source parts
# ---------------------------------------------------------------------------


class SourceUrlPart(BaseModel):
    """Web-style source reference (URL + optional title)."""

    type: Literal["source-url"] = "source-url"
    source_id: str = Field(alias="sourceId")
    url: str
    title: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class SourceDocumentPart(BaseModel):
    """Document-style source reference (typed media + title)."""

    type: Literal["source-document"] = "source-document"
    source_id: str = Field(alias="sourceId")
    media_type: str = Field(alias="mediaType")
    title: str

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# ApeRAG custom (``data-*``) parts
# ---------------------------------------------------------------------------

# Citation location follows the Anthropic Messages typed-citations
# shape; only ``type`` is required at the wire level — extra keys
# (``url``, ``title``, ``start_char_index``, ``end_char_index``,
# ``page_number``, ``content_block_index``, ...) flow through
# untouched so #75 / FE renderers can pivot on ``location.type``.
CitationLocationType = Literal[
    "char_location",
    "page_location",
    "content_block_location",
    "url_citation",
]


class CitationLocation(BaseModel):
    """Anthropic-shape citation location.

    Extra fields are allowed (``model_config.extra = "allow"``) so
    that location-type-specific keys (start/end char index, page
    number, URL, etc.) round-trip without being declared here. The
    schema is intentionally loose — the strong contract is the
    discriminator ``type`` literal; consumers branch on it.
    """

    type: CitationLocationType
    url: Optional[str] = None
    title: Optional[str] = None
    start_char_index: Optional[int] = None
    end_char_index: Optional[int] = None
    page_number: Optional[int] = None
    content_block_index: Optional[int] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class CitationData(BaseModel):
    cited_text: str
    location: CitationLocation


class DataCitationPart(BaseModel):
    """Anthropic-shape typed citation (D8 §1 — see protocol design)."""

    type: Literal["data-citation"] = "data-citation"
    data: CitationData


class ActivityData(BaseModel):
    intent: str
    label: Optional[str] = None


class DataActivityPart(BaseModel):
    """Transient ``UserActivityIntent`` UX surface — preserves the
    pre-D8 ``user_activity`` envelope semantics on the new wire.

    ``transient: True`` is hard-coded per D8 protocol design — these
    parts are NOT persisted into the at-rest ``UIMessage`` history
    (FE drops them on message assembly).
    """

    type: Literal["data-activity"] = "data-activity"
    data: ActivityData
    transient: Literal[True] = True


# ---------------------------------------------------------------------------
# Consent + elicitation parts (D9 §3 + §5; populated by D8.3 / #75)
# ---------------------------------------------------------------------------
#
# These were placeholders during #73 (cuiwenbo). #75 (chenyexuan)
# refines them to the canonical wrapped shape and drops the transient
# flag — D9 §3.1 / §5.1 explicitly classify both as persisted parts
# (consent decisions are audit-trail relevant; elicitation responses
# are part of message history). The wire shape is byte-for-byte
# equivalent to the at-rest UIMessage classes in
# ``aperag/domains/agent_runtime/uimessage.py`` so the FE can
# round-trip wire frames into stored UIMessage parts without a
# converter.
#
# We re-import ``ToolConsentData`` / ``ElicitationData`` from the
# at-rest module to enforce single-source-of-truth on the payload
# shape — any field-level evolution lands once and both layers see
# it.


from aperag.domains.agent_runtime.uimessage import (  # noqa: E402  (import at end so wire layer's own classes take precedence in module symbol order)
    ElicitationData,
    ToolConsentData,
)


class DataToolConsentPart(BaseModel):
    """Tool-call consent prompt (D9 §3 + §A7).

    Per D9 §A7: ``argsPreview`` is a redacted preview safe to surface
    in UI; ``argsHash`` is a stable hash of the full args; the raw
    args stay backend-private. #75 populates the inner
    :class:`ToolConsentData` payload.

    Persisted (NOT transient) per D9 §3.1 -- consent decisions are
    part of the audit trail and must round-trip into the at-rest
    UIMessage history.
    """

    type: Literal["data-tool-consent"] = "data-tool-consent"
    data: ToolConsentData


class DataElicitationPart(BaseModel):
    """Schema-validated user-input elicitation (D9 §5).

    Persisted (NOT transient) per D9 §5.1 -- the user's answer is
    part of message history.
    """

    type: Literal["data-elicitation"] = "data-elicitation"
    data: ElicitationData


# ---------------------------------------------------------------------------
# Discriminated union + helpers
# ---------------------------------------------------------------------------


StreamPart = Annotated[
    Union[
        StartPart,
        StartStepPart,
        FinishStepPart,
        FinishPart,
        AbortPart,
        ErrorPart,
        TextStartPart,
        TextDeltaPart,
        TextEndPart,
        ToolInputStartPart,
        ToolInputDeltaPart,
        ToolInputAvailablePart,
        ToolOutputAvailablePart,
        ToolOutputErrorPart,
        SourceUrlPart,
        SourceDocumentPart,
        DataCitationPart,
        DataActivityPart,
        DataToolConsentPart,
        DataElicitationPart,
    ],
    Field(discriminator="type"),
]


StreamPartAdapter: TypeAdapter[StreamPart] = TypeAdapter(StreamPart)


def dump_part(part: BaseModel) -> dict[str, Any]:
    """Serialize a stream part to a wire-shaped ``dict``.

    AI SDK v5 wants camelCase keys (``toolCallId``, ``errorText``,
    ...). We declare those via ``Field(alias=...)`` and dump with
    ``by_alias=True`` so the wire JSON matches the spec verbatim.
    Pydantic's ``exclude_none`` is OFF on purpose: the spec treats
    ``null`` as a meaningful absence marker for some optional fields
    (``messageId`` on ``start``, ``title`` on ``source-url``), and
    the FE consumer's ``Last-Event-ID`` resume logic key-counts on
    field presence.
    """

    return part.model_dump(by_alias=True, mode="json")


def parse_part(payload: dict[str, Any]) -> StreamPart:
    """Validate a wire-shaped ``dict`` back into the discriminated union.

    Used by tests + #76 FE-parity contract checks. The discriminator
    is ``type``; on an unknown literal Pydantic raises a clear
    ``ValidationError`` rather than silently bucketing into a fallback
    type — this is intentional, since "silent fallback" would mask
    forward-compat bugs when the FE adds a new part literal.
    """

    return StreamPartAdapter.validate_python(payload)


__all__ = [
    "AbortPart",
    "ActivityData",
    "CitationData",
    "CitationLocation",
    "CitationLocationType",
    "DataActivityPart",
    "DataCitationPart",
    "DataElicitationPart",
    "DataToolConsentPart",
    "ErrorPart",
    "FinishPart",
    "FinishStepPart",
    "SourceDocumentPart",
    "SourceUrlPart",
    "StartPart",
    "StartStepPart",
    "StreamPart",
    "StreamPartAdapter",
    "TextDeltaPart",
    "TextEndPart",
    "TextStartPart",
    "ToolInputAvailablePart",
    "ToolInputDeltaPart",
    "ToolInputStartPart",
    "ToolOutputAvailablePart",
    "ToolOutputErrorPart",
    "dump_part",
    "parse_part",
]
