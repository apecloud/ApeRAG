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

import json
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from aperag.db.models import User
from aperag.schema import view_models
from aperag.service.chat_document_service import chat_document_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chats/{chat_id}/documents", tags=["chat-documents"])
@audit(resource_type="document", api_name="UploadChatDocument")
async def upload_chat_document_view(
    request: Request,
    chat_id: str,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
) -> view_models.Document:
    """Upload a document to a chat session"""
    return await chat_document_service.upload_chat_document(
        chat_id=chat_id,
        user_id=str(user.id),
        file=file
    )


@router.get("/chats/{chat_id}/documents/{document_id}", tags=["chat-documents"])
async def get_chat_document_view(
    request: Request,
    chat_id: str,
    document_id: str,
    user: User = Depends(current_user),
) -> view_models.Document:
    """Get chat document details"""
    document = await chat_document_service.get_chat_document_by_id(
        chat_id=chat_id,
        document_id=document_id,
        user_id=str(user.id)
    )
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document





