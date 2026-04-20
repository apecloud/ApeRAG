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

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from aperag.schema import view_models

AGENT_RUNTIME_SCHEMA_VERSION = "agent-runtime-v3.1"


class VisibleAgentState(str, Enum):
    THINKING = "Thinking"
    SEARCHING = "Searching"
    CALLING_TOOL = "Calling Tool"
    READING_RESULT = "Reading Result"
    STREAMING_ANSWER = "Streaming Answer"
    COMPLETED = "Completed"
    FAILED = "Failed"


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
    label: Optional[str] = None
    status: Optional[str] = None
    actor: Literal["agent", "tool", "system"]
    data: dict[str, Any] = Field(default_factory=dict)


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
    completion: Optional[view_models.ModelSpec] = None
    collections: list[view_models.Collection] = Field(default_factory=list)
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
    files: list[view_models.File] = Field(default_factory=list)
    client_idempotency_key: Optional[str] = None


class CreateTurnResponse(BaseModel):
    turn: AgentTurnEnvelope
    stream_url: str


class AgentTurnSnapshot(BaseModel):
    turn: AgentTurnEnvelope
    timeline: list[AgentTimelineEventEnvelope] = Field(default_factory=list)
    artifacts: list[AgentArtifactEnvelope] = Field(default_factory=list)


class CancelTurnResponse(BaseModel):
    turn_id: str
    status: str
