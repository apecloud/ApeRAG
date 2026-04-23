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

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile

from aperag.db.models import User
from aperag.schema import view_models
from aperag.service.chat_collection_service import chat_collection_service
from aperag.service.chat_document_service import chat_document_service
from aperag.service.collection_service import collection_service
from aperag.service.turn_feedback_service import turn_feedback_service_global
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chats"])


# v1 `/bots/{bot_id}/chats*` routes (create/list/get/update/delete + title)
# were replaced by `bots_v2_router` in aperag/views/bots_v2.py (#1577) and
# the FE adapter switched over in #1579 (#25). The v1 handlers were
# removed in the `#26` final sweep. Remaining v1 chat routes below are
# the ones without v2 replacements yet: `/chats/{chat_id}/feedback*`,
# `/chats/{chat_id}/search`, `/chats/{chat_id}/documents*`.


@router.get("/chats/{chat_id}/feedback")
async def list_turn_feedback_view(
    request: Request, chat_id: str, user: User = Depends(required_user)
) -> view_models.TurnFeedbackList:
    return await turn_feedback_service_global.list_turn_feedbacks(str(user.id), chat_id)


@router.post("/chats/{chat_id}/turns/{turn_id}/feedback")
@audit(resource_type="turn_feedback", api_name="UpsertTurnFeedback")
async def upsert_turn_feedback_view(
    request: Request,
    chat_id: str,
    turn_id: str,
    feedback: view_models.Feedback,
    user: User = Depends(required_user),
) -> view_models.TurnFeedback:
    return await turn_feedback_service_global.upsert_turn_feedback(str(user.id), chat_id, turn_id, feedback)


@router.delete("/chats/{chat_id}/turns/{turn_id}/feedback")
@audit(resource_type="turn_feedback", api_name="DeleteTurnFeedback")
async def delete_turn_feedback_view(request: Request, chat_id: str, turn_id: str, user: User = Depends(required_user)):
    await turn_feedback_service_global.delete_turn_feedback(str(user.id), chat_id, turn_id)
    return Response(status_code=204)


@router.post("/chats/{chat_id}/search")
@audit(resource_type="search", api_name="SearchChatFiles")
async def search_chat_files_view(
    request: Request,
    chat_id: str,
    data: view_models.SearchRequest,
    user: User = Depends(required_user),
) -> view_models.SearchResult:
    """Search files within a specific chat using hybrid search capabilities"""
    try:
        # Get user's chat collection
        chat_collection_id = await chat_collection_service.get_user_chat_collection_id(str(user.id))
        if not chat_collection_id:
            raise HTTPException(status_code=404, detail="Chat collection not found")

        if not chat_id:
            raise HTTPException(status_code=400, detail="Chat ID is required")

        # Execute search flow using the helper method from collection_service
        items, _ = await collection_service.execute_search_flow(
            data=data,
            collection_id=chat_collection_id,
            search_user_id=str(user.id),
            chat_id=chat_id,  # Add chat_id for filtering in chat searches
            flow_name="chat_search",
            flow_title="Chat Search",
        )

        # Return search result without saving to database for chat searches
        from aperag.schema.view_models import SearchResult

        return SearchResult(
            id=None,  # No ID since not saved
            query=data.query,
            vector_search=data.vector_search,
            fulltext_search=data.fulltext_search,
            graph_search=data.graph_search,
            summary_search=data.summary_search,
            items=items,
            created=None,  # No creation time since not saved
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search chat files: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/chats/{chat_id}/documents")
@audit(resource_type="document", api_name="UploadChatDocument")
async def upload_chat_document_view(
    request: Request,
    chat_id: str,
    file: UploadFile = File(...),
    user: User = Depends(required_user),
) -> view_models.Document:
    """Upload a document to a chat session"""
    return await chat_document_service.upload_chat_document(chat_id=chat_id, user_id=str(user.id), file=file)


@router.get("/chats/{chat_id}/documents/{document_id}")
async def get_chat_document_view(
    request: Request,
    chat_id: str,
    document_id: str,
    user: User = Depends(required_user),
) -> view_models.Document:
    """Get chat document details"""
    document = await chat_document_service.get_chat_document_by_id(
        chat_id=chat_id, document_id=document_id, user_id=str(user.id)
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document
