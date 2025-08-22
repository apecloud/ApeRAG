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
from typing import Dict, List, Optional, TypedDict

from fastapi import HTTPException, UploadFile

from aperag.db.models import Document, DocumentStatus
from aperag.db.ops import async_db_ops
from aperag.service.chat_collection_service import chat_collection_service
from aperag.service.document_service import document_service
from aperag.utils.utils import utc_now


# Temporary type definitions until OpenAPI models are generated
class ChatDocumentResponse(TypedDict):
    id: str
    name: str
    size: int
    status: str
    chat_id: str
    message_id: Optional[str]
    progress: Dict[str, any]
    created: str
    updated: str


class ChatDocumentList(TypedDict):
    items: List[ChatDocumentResponse]

logger = logging.getLogger(__name__)

# File upload limits for chat documents
CHAT_DOCUMENT_LIMITS = {
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "max_files_per_message": 5,
    "max_files_per_chat": 100,
    "allowed_extensions": {'.pdf', '.doc', '.docx', '.txt', '.md'},
    "max_filename_length": 255
}


class ChatDocumentService:
    """
    Chat document service for handling document uploads in chat sessions
    """

    def __init__(self):
        self.db_ops = async_db_ops

    async def upload_chat_document(
        self, chat_id: str, message_id: str, user_id: str, file: UploadFile
    ) -> ChatDocumentResponse:
        """Upload chat document to user's chat collection"""
        # Validate file
        self._validate_file(file)

        # Get user's chat collection (should exist from registration)
        collection = await chat_collection_service.get_user_chat_collection(user_id)
        if not collection:
            # Create if missing (fallback)
            collection = await chat_collection_service.create_user_chat_collection(user_id)

        # Prepare document metadata
        doc_metadata = {
            "chat_id": chat_id,
            "message_id": message_id,
            "file_type": "chat_upload",
            "original_filename": file.filename,
            "upload_timestamp": utc_now().isoformat()
        }

        # Use document service to create document
        documents = await document_service.create_documents(
            user_id, collection.id, [file], doc_metadata
        )
        
        if not documents.items:
            raise HTTPException(status_code=500, detail="Failed to upload document")

        document = documents.items[0]
        
        logger.info(f"Chat document {document.id} uploaded for chat {chat_id}")
        
        return self._build_chat_document_response(document, chat_id, message_id)

    async def get_chat_document_by_id(
        self, chat_id: str, document_id: str, user_id: str
    ) -> Optional[ChatDocumentResponse]:
        """Get chat document by ID with chat ownership validation"""
        # Get user's chat collection
        collection = await chat_collection_service.get_user_chat_collection(user_id)
        if not collection:
            return None

        # Get document
        document = await self.db_ops.query_document_by_id(document_id)
        if not document or document.collection_id != collection.id:
            return None

        # Validate it's a chat document for the specified chat
        if document.doc_metadata:
            try:
                metadata = json.loads(document.doc_metadata)
                if (metadata.get("file_type") == "chat_upload" and 
                    metadata.get("chat_id") == chat_id):
                    return self._build_chat_document_response(
                        document, chat_id, metadata.get("message_id")
                    )
            except json.JSONDecodeError:
                pass

        return None

    async def list_chat_documents(
        self, chat_id: str, user_id: str
    ) -> ChatDocumentList:
        """List all documents for a chat session"""
        # Get user's chat collection
        collection = await chat_collection_service.get_user_chat_collection(user_id)
        if not collection:
            return {"items": []}

        # Get all documents in the collection
        documents = await self.db_ops.query_documents_by_collection_id(
            collection.id, include_deleted=False
        )

        # Filter for this chat's documents
        chat_documents = []
        for doc in documents:
            if doc.doc_metadata:
                try:
                    metadata = json.loads(doc.doc_metadata)
                    if (metadata.get("file_type") == "chat_upload" and 
                        metadata.get("chat_id") == chat_id):
                        chat_documents.append(
                            self._build_chat_document_response(
                                doc, chat_id, metadata.get("message_id")
                            )
                        )
                except json.JSONDecodeError:
                    continue

        return {"items": chat_documents}

    def _validate_file(self, file: UploadFile):
        """Validate uploaded file against chat document limits"""
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Check filename length
        if len(file.filename) > CHAT_DOCUMENT_LIMITS["max_filename_length"]:
            raise HTTPException(status_code=400, detail="Filename too long")

        # Check file extension
        import os
        _, ext = os.path.splitext(file.filename.lower())
        if ext not in CHAT_DOCUMENT_LIMITS["allowed_extensions"]:
            allowed = ", ".join(CHAT_DOCUMENT_LIMITS["allowed_extensions"])
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {allowed}"
            )

        # Check file size
        if file.size and file.size > CHAT_DOCUMENT_LIMITS["max_file_size"]:
            max_size_mb = CHAT_DOCUMENT_LIMITS["max_file_size"] // (1024 * 1024)
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {max_size_mb}MB"
            )

    def _build_chat_document_response(
        self, document: Document, chat_id: str, message_id: str = None
    ) -> ChatDocumentResponse:
        """Build chat document response from Document model"""
        # Get processing progress
        progress = self._get_document_progress(document)
        
        return {
            "id": document.id,
            "name": document.name,
            "size": document.size,
            "status": document.status.value,
            "chat_id": chat_id,
            "message_id": message_id,
            "progress": progress,
            "created": document.gmt_created.isoformat(),
            "updated": document.gmt_updated.isoformat(),
        }

    def _get_document_progress(self, document: Document) -> dict:
        """Get document processing progress"""
        # This is a simplified progress calculation
        # In a real implementation, you'd get this from DocumentIndex table
        
        total_steps = 4  # Parse, Vector, Fulltext, Summary/Graph
        completed_steps = 0
        current_step = "Uploading"

        if document.status == DocumentStatus.PENDING:
            current_step = "Pending"
            completed_steps = 1
        elif document.status == DocumentStatus.RUNNING:
            current_step = "Processing"
            completed_steps = 2
        elif document.status == DocumentStatus.COMPLETE:
            current_step = "Complete"
            completed_steps = total_steps
        elif document.status == DocumentStatus.FAILED:
            current_step = "Failed"
            completed_steps = 0

        return {
            "current_step": current_step,
            "total_steps": total_steps,
            "completed_steps": completed_steps
        }


# Global service instance
chat_document_service = ChatDocumentService()
