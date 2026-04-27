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

"""Conversation-domain FastAPI routes.

Phase 5 step 5-S4g merges the pre-Phase-5 ``aperag/views/chat.py`` +
``aperag/views/bots_v2.py`` modules into this single
``aperag.domains.conversation.api.routes`` module. Two shim files at
the legacy paths re-export the corresponding router so
``aperag/app.py`` include calls and any third-party
``from aperag.views.{chat,bots_v2} import router`` callers still
resolve. The URL / method / response_model / audit-decorator /
status-code surface is byte-stable against the pre-move shape —
verified via ``scripts/export_openapi.py --check``.

Two routers are kept because the pre-move paths split between
``/api/v1`` (legacy ``chat.py`` — chat feedback / search / documents)
and ``/api/v2`` (``bots_v2.py`` — bot + chat CRUD + title). Merging
into a single router mounted twice would duplicate every URL. Phase
4 model_platform uses the same split-router pattern for the same
reason.

Lesson 9a-ter / lesson 9a-quatuordec / lesson 9a-quad applied:

* Handler ``user`` parameters are typed against the per-domain
  ``AuthenticatedUser(Protocol)`` from
  ``aperag.domains.conversation.ports`` rather than the identity-owned
  ``User`` ORM class. ``required_user`` still returns a real User row
  at runtime — the Protocol is structural.
* No ``from __future__ import annotations`` — FastAPI's 204-handler
  check at route registration dereferences the ``Response`` return
  annotation by value, and PEP 563 stringification would break that
  path (see ``docs/modularization/breaking-changes/phase3-knowledge_base.md``
  for the full lesson 9a-quatuordec trail).
* All six consumer services (``bot_service`` / ``chat_service_global``
  / ``chat_title_service`` / ``chat_collection_service`` /
  ``chat_document_service`` / ``turn_feedback_service_global``) are
  imported from their canonical per-domain homes, not from the legacy
  ``aperag/service/*`` shims. ``collection_service`` + the
  ``SearchResult`` / ``SearchRequest`` schemas come from the Phase 3
  knowledge_base / retrieval domains via direct cross-domain import
  (G1 permits domain→domain).
"""

import logging
from typing import Protocol

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile

from aperag.domains.conversation.schemas import (
    Bot,
    BotCreate,
    BotList,
    BotUpdate,
    BotUpdateRequest,
    Chat,
    ChatDetails,
    ChatList,
    ChatUpdate,
    Feedback,
    TitleGenerateRequest,
    TitleGenerateResponse,
    TurnFeedback,
    TurnFeedbackList,
)
from aperag.domains.conversation.service.bot_service import bot_service
from aperag.domains.conversation.service.chat_collection_service import chat_collection_service
from aperag.domains.conversation.service.chat_document_service import chat_document_service
from aperag.domains.conversation.service.chat_service import chat_service_global
from aperag.domains.conversation.service.chat_title_service import chat_title_service
from aperag.domains.conversation.service.turn_feedback_service import turn_feedback_service_global
from aperag.domains.knowledge_base.schemas import Document
from aperag.domains.knowledge_base.service.collection_service import collection_service
from aperag.domains.retrieval.schemas import SearchRequest, SearchResult
from aperag.exceptions import BusinessException
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

logger = logging.getLogger(__name__)


class AuthenticatedUser(Protocol):
    """Per-domain auth context for the conversation routes.

    Matches ``aperag.domains.conversation.ports.AuthenticatedUser`` —
    declared locally as well so the FastAPI ``Depends(required_user)``
    annotation resolves without a cross-module forward reference.
    The concrete identity-owned ``User`` row satisfies this Protocol
    structurally at runtime.
    """

    id: object
    role: str


chat_router = APIRouter(tags=["chats"])
bots_router = APIRouter(tags=["bots-v2"])


# -- chat feedback / search / document routes (formerly `aperag/views/chat.py`).


@chat_router.get("/chats/{chat_id}/feedback")
async def list_turn_feedback_view(
    request: Request, chat_id: str, user: AuthenticatedUser = Depends(required_user)
) -> TurnFeedbackList:
    return await turn_feedback_service_global.list_turn_feedbacks(str(user.id), chat_id)


@chat_router.post("/chats/{chat_id}/turns/{turn_id}/feedback")
@audit(resource_type="turn_feedback", api_name="UpsertTurnFeedback")
async def upsert_turn_feedback_view(
    request: Request,
    chat_id: str,
    turn_id: str,
    feedback: Feedback,
    user: AuthenticatedUser = Depends(required_user),
) -> TurnFeedback:
    return await turn_feedback_service_global.upsert_turn_feedback(str(user.id), chat_id, turn_id, feedback)


@chat_router.delete("/chats/{chat_id}/turns/{turn_id}/feedback")
@audit(resource_type="turn_feedback", api_name="DeleteTurnFeedback")
async def delete_turn_feedback_view(
    request: Request, chat_id: str, turn_id: str, user: AuthenticatedUser = Depends(required_user)
):
    await turn_feedback_service_global.delete_turn_feedback(str(user.id), chat_id, turn_id)
    return Response(status_code=204)


@chat_router.post("/chats/{chat_id}/search")
@audit(resource_type="search", api_name="SearchChatFiles")
async def search_chat_files_view(
    request: Request,
    chat_id: str,
    data: SearchRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> SearchResult:
    """Search files within a specific chat using hybrid search capabilities"""
    try:
        chat_collection_id = await chat_collection_service.get_user_chat_collection_id(str(user.id))
        if not chat_collection_id:
            raise HTTPException(status_code=404, detail="Chat collection not found")

        if not chat_id:
            raise HTTPException(status_code=400, detail="Chat ID is required")

        items, _ = await collection_service.execute_search_flow(
            data=data,
            collection_id=chat_collection_id,
            search_user_id=str(user.id),
            chat_id=chat_id,
            flow_name="chat_search",
            flow_title="Chat Search",
        )

        return SearchResult(
            id=None,
            query=data.query,
            vector_search=data.vector_search,
            fulltext_search=data.fulltext_search,
            graph_search=data.graph_search,
            summary_search=data.summary_search,
            items=items,
            created=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search chat files: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@chat_router.post("/chats/{chat_id}/documents")
@audit(resource_type="document", api_name="UploadChatDocument")
async def upload_chat_document_view(
    request: Request,
    chat_id: str,
    file: UploadFile = File(...),
    user: AuthenticatedUser = Depends(required_user),
) -> Document:
    """Upload a document to a chat session"""
    return await chat_document_service.upload_chat_document(chat_id=chat_id, user_id=str(user.id), file=file)


@chat_router.get("/chats/{chat_id}/documents/{document_id}")
async def get_chat_document_view(
    request: Request,
    chat_id: str,
    document_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Document:
    """Get chat document details"""
    document = await chat_document_service.get_chat_document_by_id(
        chat_id=chat_id, document_id=document_id, user_id=str(user.id)
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


# -- bot / chat CRUD + title generation (formerly `aperag/views/bots_v2.py`).


@bots_router.post("/bots", response_model=Bot)
@audit(resource_type="bot", api_name="CreateBotV2")
async def create_bot_v2_view(
    request: Request,
    body: BotCreate,
    user: AuthenticatedUser = Depends(required_user),
) -> Bot:
    """Create an agent bot owned by the current user."""
    return await bot_service.create_bot(str(user.id), body)


@bots_router.get("/bots", response_model=BotList)
async def list_bots_v2_view(
    user: AuthenticatedUser = Depends(required_user),
) -> BotList:
    """List bots owned by the current user."""
    return await bot_service.list_bots(str(user.id))


@bots_router.get("/bots/{bot_id}", response_model=Bot)
async def get_bot_v2_view(
    bot_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Bot:
    """Return one bot owned by the current user."""
    return await bot_service.get_bot(str(user.id), bot_id)


@bots_router.put("/bots/{bot_id}", response_model=Bot)
@audit(resource_type="bot", api_name="UpdateBotV2")
async def update_bot_v2_view(
    request: Request,
    bot_id: str,
    body: BotUpdateRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> Bot:
    """Update one bot. The path bot_id is canonical."""
    bot_update = BotUpdate(
        title=body.title,
        description=body.description,
        config=body.config,
    )
    return await bot_service.update_bot(str(user.id), bot_id, bot_update)


@bots_router.delete("/bots/{bot_id}", status_code=204)
@audit(resource_type="bot", api_name="DeleteBotV2")
async def delete_bot_v2_view(
    request: Request,
    bot_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Response:
    """Delete one bot owned by the current user. Idempotent."""
    await bot_service.delete_bot(str(user.id), bot_id)
    return Response(status_code=204)


@bots_router.post("/bots/{bot_id}/chats", response_model=Chat)
@audit(resource_type="chat", api_name="CreateChatV2")
async def create_chat_v2_view(
    request: Request,
    bot_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Chat:
    """Create a chat under one agent bot."""
    return await chat_service_global.create_chat(str(user.id), bot_id)


@bots_router.get("/bots/{bot_id}/chats", response_model=ChatList)
async def list_chats_v2_view(
    bot_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    user: AuthenticatedUser = Depends(required_user),
) -> ChatList:
    """List chats under one agent bot."""
    return await chat_service_global.list_chats(str(user.id), bot_id, page, page_size)


@bots_router.get("/bots/{bot_id}/chats/{chat_id}", response_model=ChatDetails)
async def get_chat_v2_view(
    bot_id: str,
    chat_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> ChatDetails:
    """Return one chat with history."""
    return await chat_service_global.get_chat(str(user.id), bot_id, chat_id)


@bots_router.put("/bots/{bot_id}/chats/{chat_id}", response_model=Chat)
@audit(resource_type="chat", api_name="UpdateChatV2")
async def update_chat_v2_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    body: ChatUpdate,
    user: AuthenticatedUser = Depends(required_user),
) -> Chat:
    """Update one chat. Path bot_id and chat_id are canonical."""
    return await chat_service_global.update_chat(str(user.id), bot_id, chat_id, body)


@bots_router.delete("/bots/{bot_id}/chats/{chat_id}", status_code=204)
@audit(resource_type="chat", api_name="DeleteChatV2")
async def delete_chat_v2_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Response:
    """Delete one chat under one agent bot. Idempotent."""
    await chat_service_global.delete_chat(str(user.id), bot_id, chat_id)
    return Response(status_code=204)


@bots_router.post("/bots/{bot_id}/chats/{chat_id}/title", response_model=TitleGenerateResponse)
async def generate_chat_title_v2_view(
    bot_id: str,
    chat_id: str,
    body: TitleGenerateRequest = TitleGenerateRequest(),
    user: AuthenticatedUser = Depends(required_user),
) -> TitleGenerateResponse:
    """Generate a title for one chat."""
    try:
        title = await chat_title_service.generate_title(
            user_id=str(user.id),
            bot_id=bot_id,
            chat_id=chat_id,
            max_length=body.max_length,
            language=body.language,
            turns=body.turns,
        )
        return TitleGenerateResponse(title=title)
    except BusinessException as be:
        raise HTTPException(status_code=400, detail={"error_code": be.error_code.name, "message": str(be)})
