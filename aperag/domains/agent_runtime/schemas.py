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

"""Agent-runtime-domain Pydantic request / response schemas.

Phase 5 step 5-S5a carved ``AgentMessage`` out of
``aperag.schema.view_models``. Phase 5 step 5-S5b folds in the
envelope / request / response classes that previously lived in
``aperag/agent_runtime/schemas.py``. Cross-domain types stay as
direct imports — ``Collection`` from knowledge_base, ``File`` from
conversation, ``ModelSpec`` from ``aperag.schema.common``; G1 allows
domain→domain.

Module-level ``AGENT_RUNTIME_SCHEMA_VERSION`` constant is kept so
``AgentTurnEnvelope`` / ``AgentTimelineEventEnvelope`` /
``AgentArtifactEnvelope`` can tag every serialised payload with the
runtime contract version (declared ``v3.1``; bumping triggers
FE-side versioned parse).

The ``_bind_view_models_reexports`` hook at the bottom mirrors the
Phase 3 step 4b / Phase 5 step 5-S3 pattern: if this module loads
before ``aperag.schema.view_models``, the hook writes
``AgentMessage`` onto the view_models namespace so pre-migration
``from aperag.schema.view_models import AgentMessage`` callers still
resolve the canonical object. Write-only per lesson 9a-quad.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from aperag.domains.conversation.schemas import File
from aperag.domains.knowledge_base.schemas import Collection
from aperag.schema.common import ModelSpec

AGENT_RUNTIME_SCHEMA_VERSION = "agent-runtime-v3.1"


class VisibleAgentState(str, Enum):
    THINKING = "Thinking"
    SEARCHING = "Searching"
    CALLING_TOOL = "Calling Tool"
    READING_RESULT = "Reading Result"
    STREAMING_ANSWER = "Streaming Answer"
    COMPLETED = "Completed"
    FAILED = "Failed"


class UserActivityIntent(str, Enum):
    THINKING = "thinking"
    SEARCHING_KNOWLEDGE = "searching_knowledge"
    READING_SOURCE = "reading_source"
    COMPARING_RESULTS = "comparing_results"
    WRITING_ANSWER = "writing_answer"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class UserActivityContext(BaseModel):
    source_name: Optional[str] = None
    keyword: Optional[str] = None
    count: Optional[int] = None
    target_type: Optional[Literal["knowledge_base", "document", "web"]] = None
    scope_label: Optional[str] = None


class UserActivityEnvelope(BaseModel):
    intent: UserActivityIntent
    title_key: str
    subtitle_key: str
    detail_key: Optional[str] = None
    context: Optional[UserActivityContext] = None


class AgentTurnEnvelope(BaseModel):
    schema_version: str = AGENT_RUNTIME_SCHEMA_VERSION
    turn_id: str
    chat_id: str
    user_id: str
    bot_id: str
    request_id: str
    client_idempotency_key: str
    status: str
    input_text: str
    model_profile: dict[str, Any] = Field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    answer_artifact_id: Optional[str] = None
    reference_bundle_artifact_id: Optional[str] = None
    timeline_cursor: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AgentTimelineEventEnvelope(BaseModel):
    schema_version: str = AGENT_RUNTIME_SCHEMA_VERSION
    event_id: str
    turn_id: str
    sequence: int
    timestamp: datetime
    type: str
    technical_type: Optional[str] = None
    label: Optional[str] = None
    status: Optional[str] = None
    actor: Literal["agent", "tool", "system"]
    data: dict[str, Any] = Field(default_factory=dict)
    user_activity: Optional[UserActivityEnvelope] = None


class AgentArtifactEnvelope(BaseModel):
    schema_version: str = AGENT_RUNTIME_SCHEMA_VERSION
    artifact_id: str
    turn_id: str
    artifact_type: str
    summary: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    storage_ref: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ReferenceBundleItem(BaseModel):
    source_type: str
    source_id: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateTurnRequest(BaseModel):
    query: str
    completion: Optional[ModelSpec] = None
    collections: list[Collection] = Field(default_factory=list)
    web_search_enabled: bool = False
    language: Optional[
        Literal[
            "en-US",
            "zh-CN",
            "zh-TW",
            "ja-JP",
            "ko-KR",
            "fr-FR",
            "de-DE",
            "es-ES",
            "it-IT",
            "pt-BR",
            "ru-RU",
        ]
    ] = "en-US"
    files: list[File] = Field(default_factory=list)
    client_idempotency_key: Optional[str] = None


class CreateTurnResponse(BaseModel):
    turn: AgentTurnEnvelope
    stream_url: str


# ``AgentTurnSnapshot`` is the canonical UIMessage at-rest envelope and
# lives in :mod:`aperag.domains.agent_runtime.uimessage` next to the
# other UIMessage classes. It is re-exported here so the existing
# ``from aperag.domains.agent_runtime.schemas import AgentTurnSnapshot``
# import sites continue to work without a domain-internal hop.
from aperag.domains.agent_runtime.uimessage import (  # noqa: E402, F401
    AgentTurnSnapshot,
)


class CancelTurnResponse(BaseModel):
    turn_id: str
    status: str


class AgentMessage(BaseModel):
    """
    Message format for agent-type bots with additional capabilities
    """

    query: str = Field(..., description="User query", examples=["Tell me about ApeRAG features"])
    collections: list[Collection] = Field(
        ...,
        description="List of collection objects to search in",
        examples=[
            [
                {"id": "col_123", "title": "Example Collection"},
                {"id": "col_456", "title": "Another Collection"},
            ]
        ],
    )
    completion: Optional[ModelSpec] = Field(
        None,
        description="Model specification for completion including provider and model details",
    )
    web_search_enabled: Optional[bool] = Field(False, description="Whether to enable web search", examples=[True])
    language: Optional[
        Literal[
            "en-US",
            "zh-CN",
            "zh-TW",
            "ja-JP",
            "ko-KR",
            "fr-FR",
            "de-DE",
            "es-ES",
            "it-IT",
            "pt-BR",
            "ru-RU",
        ]
    ] = Field("en-US", description="Language preference for the response", examples=["en-US"])
    files: Optional[list[File]] = None


__all__ = [
    "AGENT_RUNTIME_SCHEMA_VERSION",
    "AgentArtifactEnvelope",
    "AgentMessage",
    "AgentTimelineEventEnvelope",
    "AgentTurnEnvelope",
    "AgentTurnSnapshot",
    "CancelTurnResponse",
    "CreateTurnRequest",
    "CreateTurnResponse",
    "ReferenceBundleItem",
    "UserActivityContext",
    "UserActivityEnvelope",
    "UserActivityIntent",
    "VisibleAgentState",
]
