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

"""Conversation-domain Pydantic request / response schemas.

Carved out of ``aperag.schema.view_models`` in Phase 5 step 5-S3 using
the same dual-hook shim cuiwenbo landed for Phase 3 step 4b
(``msg=f894cca4`` / ``msg=bdfe4da4`` / lesson 9a-tredec):

* ``view_models.py`` now ``try:``-imports these classes from here, so
  callers who imported ``aperag.schema.view_models`` first resolve the
  canonical class objects via the re-export window.
* The ``_bind_view_models_reexports`` hook at the end of this module
  covers the reverse load order — when this module finishes ahead of
  ``view_models``, the hook writes the class objects onto the
  ``view_models`` namespace so FastAPI response_model binding,
  Pydantic forward-ref resolution, and pre-migration ``from
  aperag.schema.view_models import Bot`` callers keep working.

The dual-hook intentionally does NOT read back from
``aperag.schema.view_models``; per lesson 9a-quad it is a
write-only shim. Phase 6 cleanup deletes it once every caller has
migrated to the canonical per-domain path.

Cross-domain references kept narrow:

* ``Agent.collections`` uses ``KBCollectionSchema`` — a forward ref to
  the ``Collection`` Pydantic schema the knowledge_base domain owns.
  Importing it directly (``from aperag.domains.knowledge_base.schemas
  import Collection``) crosses domains, which G1 does not ban (G1 only
  bans the three legacy aggregate modules).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field, conint

if TYPE_CHECKING:
    # Type-only import to keep the OpenAPI schema for ``ChatDetails.history``
    # populated without forming a runtime cycle: ``agent_runtime.uimessage``
    # imports ``AGENT_RUNTIME_SCHEMA_VERSION`` / ``UserActivityEnvelope`` from
    # ``agent_runtime.schemas``, which in turn imports ``File`` from this
    # module. Pydantic resolves the forward reference via
    # :func:`ChatDetails.model_rebuild` at the bottom of this file.
    from aperag.domains.agent_runtime.uimessage import AgentTurnSnapshot

from aperag.domains.knowledge_base.schemas import Collection as KBCollectionSchema
from aperag.schema.common import ModelSpec, PageResult, PaginatedResponse


class Agent(BaseModel):
    completion: Optional[ModelSpec] = None
    system_prompt_template: Optional[str] = None
    query_prompt_template: Optional[str] = None
    collections: Optional[list[KBCollectionSchema]] = None


class BotConfig(BaseModel):
    agent: Optional[Agent] = None


class Bot(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[Literal["knowledge", "common", "agent"]] = Field(
        None, description="The type of bot", examples=["agent"]
    )
    config: Optional[BotConfig] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class BotList(BaseModel):
    """
    A list of bots
    """

    items: Optional[list[Bot]] = None
    pageResult: Optional[PageResult] = None


class BotCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[Literal["agent"]] = Field(None, description="The supported bot type", examples=["agent"])
    config: Optional[BotConfig] = None


class BotUpdate(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    config: Optional[BotConfig] = None


class BotUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    config: Optional[BotConfig] = None


class Chat(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    bot_id: Optional[str] = None
    peer_id: Optional[str] = None
    peer_type: Optional[Literal["system", "feishu", "weixin", "weixin_official", "web", "dingtalk", "evaluation"]] = (
        None
    )
    status: Optional[Literal["active", "archived"]] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class ChatList(PaginatedResponse):
    """
    A list of chats with pagination
    """

    items: Optional[list[Chat]] = None


class ChatCreate(BaseModel):
    title: Optional[str] = None


class Reference(BaseModel):
    score: Optional[float] = None
    text: Optional[str] = None
    image_uri: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class File(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None


class ChatMessage(BaseModel):
    id: Optional[str] = None
    part_id: Optional[str] = None
    type: Optional[
        Literal[
            "welcome",
            "message",
            "start",
            "stop",
            "error",
            "tool_call_result",
            "thinking",
            "references",
        ]
    ] = None
    timestamp: Optional[float] = None
    role: Optional[Literal["human", "ai"]] = None
    data: Optional[str] = None
    references: Optional[list[Reference]] = None
    urls: Optional[list[str]] = None
    files: Optional[list[File]] = None


class ChatDetails(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    bot_id: Optional[str] = None
    peer_id: Optional[str] = None
    peer_type: Optional[Literal["system", "feishu", "weixin", "weixin_official", "web", "dingtalk", "evaluation"]] = (
        None
    )
    history: Optional[list[AgentTurnSnapshot]] = Field(
        None,
        description=(
            "Phase 8 D8.5-BE (#92): historical conversation turns as canonical "
            "``AgentTurnSnapshot`` envelopes — each turn carries the same "
            "``UIMessagePart[]`` shape the FE consumes from the live SSE stream "
            "(D8 §2 wire/at-rest byte-equal). Replaces the legacy "
            "``list[list[ChatMessage]]`` shape; FE renders historical turns with "
            "the same renderer used for live turns."
        ),
    )
    status: Optional[Literal["active", "archived"]] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class ChatUpdate(BaseModel):
    title: Optional[str] = None


class TitleGenerateRequest(BaseModel):
    max_length: Optional[conint(ge=6, le=50)] = Field(20, description="Maximum length of the generated title")
    language: Optional[Literal["zh-CN", "en-US", "ja-JP", "ko-KR"]] = Field(
        "zh-CN", description="Language for the title generation (IETF BCP 47 tag)"
    )
    turns: Optional[conint(ge=1)] = Field(1, description="Number of most recent conversation turns to consider")


class TitleGenerateResponse(BaseModel):
    title: str = Field(..., description="Generated title string")


class TurnFeedback(BaseModel):
    turn_id: str
    type: Literal["good", "bad"]
    tag: Optional[Literal["Harmful", "Unsafe", "Fake", "Unhelpful", "Other"]] = None
    message: Optional[str] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class TurnFeedbackList(BaseModel):
    items: list[TurnFeedback]


class Feedback(BaseModel):
    type: Optional[Literal["good", "bad"]] = None
    tag: Optional[Literal["Harmful", "Unsafe", "Fake", "Unhelpful", "Other"]] = None
    message: Optional[str] = None


# Alias retained for back-compat — ``view_models`` previously exposed both
# names pointing at the same class; keep the alias here so the dual-hook
# shim still resolves ``TurnFeedbackWrite``.
TurnFeedbackWrite = Feedback


# Phase 8 D8.5-BE (#92): resolve the forward reference to
# ``AgentTurnSnapshot`` after ``conversation.schemas`` finishes loading.
# A direct top-level import would form a cycle through
# ``agent_runtime.uimessage`` → ``agent_runtime.schemas`` → this module.
# The TYPE_CHECKING block at the top declares the import for static
# checkers / OpenAPI; this rebuild wires it up at runtime.
def _rebuild_chat_details() -> None:
    from aperag.domains.agent_runtime.uimessage import AgentTurnSnapshot  # noqa: F401

    ChatDetails.model_rebuild()


_rebuild_chat_details()


__all__ = [
    "Agent",
    "Bot",
    "BotConfig",
    "BotCreate",
    "BotList",
    "BotUpdate",
    "BotUpdateRequest",
    "Chat",
    "ChatCreate",
    "ChatDetails",
    "ChatList",
    "ChatMessage",
    "ChatUpdate",
    "Feedback",
    "File",
    "Reference",
    "TitleGenerateRequest",
    "TitleGenerateResponse",
    "TurnFeedback",
    "TurnFeedbackList",
    "TurnFeedbackWrite",
]
