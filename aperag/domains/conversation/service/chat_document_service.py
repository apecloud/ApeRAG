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

"""Chat-document service for the ``conversation`` domain.

Cross-service hookups are direct sibling / cross-domain imports:

* ``chat_collection_service`` is a sibling inside this domain
  (Phase 5 step 5-S4f moved it here) — direct import, no Protocol seam.
* ``aperag.domains.knowledge_base.service.document_service`` is a
  cross-domain direct import (legal under G1 — the strict ban only
  covers the three legacy aggregate modules).

``aperag.domains.knowledge_base.schemas.Document`` is the canonical
response type after Phase 3 step 5b3; keeping the annotation on
``upload_chat_document`` preserves the OpenAPI response_model.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile

from aperag.db.ops import async_db_ops
from aperag.domains.conversation.service.chat_collection_service import chat_collection_service
from aperag.domains.knowledge_base.schemas import Document
from aperag.domains.knowledge_base.service.document_service import document_service
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)


class ChatDocumentService:
    """
    Chat document service for handling document uploads in chat sessions
    """

    def __init__(self):
        self.db_ops = async_db_ops

    async def upload_chat_document(self, chat_id: str, user_id: str, file: UploadFile) -> Document:
        """Upload chat document to user's chat collection"""
        collection = await chat_collection_service.get_user_chat_collection(user_id)
        if not collection:
            # Create if missing (fallback)
            collection = await chat_collection_service.create_user_chat_collection(user_id)

        # Prepare document metadata (without message_id initially)
        doc_metadata = {
            "chat_id": chat_id,
            "file_type": "chat_upload",
        }

        documents = await document_service.create_documents(
            user_id, collection.id, [file], doc_metadata, ignore_duplicate=True
        )

        if not documents.items:
            raise HTTPException(status_code=500, detail="Failed to upload document")

        return documents.items[0]

    async def get_chat_document_by_id(self, chat_id: str, document_id: str, user_id: str) -> Optional[Document]:
        """Get chat document by ID with chat ownership validation"""
        collection = await chat_collection_service.get_user_chat_collection(user_id)
        if not collection:
            return None

        document = await self.db_ops.query_document_by_id(document_id)
        if not document or document.collection_id != collection.id:
            return None

        return document

    async def get_documents_metadata(self, chat_id: str, document_ids: List[str], user_id: str) -> List[Dict[str, Any]]:
        """Get metadata for documents to be stored in chat message"""
        if not document_ids:
            return []

        collection = await chat_collection_service.get_user_chat_collection(user_id)
        if not collection:
            return []

        documents_metadata = []
        for document_id in document_ids:
            document = await self.db_ops.query_document_by_id(document_id)
            if not document or document.collection_id != collection.id:
                continue

            if document.doc_metadata:
                try:
                    metadata = json.loads(document.doc_metadata)
                    if metadata.get("file_type") == "chat_upload" and metadata.get("chat_id") == chat_id:
                        file_info = {
                            "id": document.id,
                            "name": document.name,
                            "size": document.size,
                            "status": document.status.value,
                            "created": document.gmt_created.isoformat(),
                            "updated": document.gmt_updated.isoformat(),
                        }
                        documents_metadata.append(file_info)
                except json.JSONDecodeError:
                    continue

        return documents_metadata

    async def has_documents_in_chat(self, chat_id: str, user_id: str) -> bool:
        """Return whether the current chat already has searchable uploaded files."""
        collection = await chat_collection_service.get_user_chat_collection(user_id)
        if not collection:
            return False

        documents = await self.db_ops.query_documents([user_id], collection.id)
        for document in documents:
            if not document.doc_metadata:
                continue
            try:
                metadata = json.loads(document.doc_metadata)
            except json.JSONDecodeError:
                continue

            if metadata.get("file_type") == "chat_upload" and metadata.get("chat_id") == chat_id:
                return True

        return False

    async def associate_documents_with_message(
        self, chat_id: str, message_id: str, files: List[str], user: str
    ) -> List[Dict[str, Any]]:
        """Handle file metadata retrieval and document association for chat messages."""
        if not files:
            return []

        result = []
        try:
            result = await self.get_documents_metadata(chat_id=chat_id, document_ids=files, user_id=user)
            await self._associate_documents_with_message(
                chat_id=chat_id, message_id=message_id, document_ids=files, user_id=user
            )
        except Exception as e:
            logger.warning(f"Failed to associate documents with message {message_id}: {e}")

        return result

    async def _associate_documents_with_message(
        self, chat_id: str, message_id: str, document_ids: List[str], user_id: str
    ) -> None:
        """Associate uploaded documents with a message when user sends the message"""
        collection = await chat_collection_service.get_user_chat_collection(user_id)
        if not collection:
            return

        for document_id in document_ids:
            document = await self.db_ops.query_document_by_id(document_id)
            if not document or document.collection_id != collection.id:
                continue

            if document.doc_metadata:
                try:
                    metadata = json.loads(document.doc_metadata)
                    if metadata.get("file_type") == "chat_upload" and metadata.get("chat_id") == chat_id:
                        metadata["message_id"] = message_id
                        document.doc_metadata = json.dumps(metadata)
                        document.gmt_updated = utc_now()

                        await self.db_ops.update_document(document)
                        logger.info(f"Associated document {document_id} with message {message_id}")
                except json.JSONDecodeError:
                    continue


# Global service instance — wired by the legacy shim at
# ``aperag.service.chat_document_service``.
chat_document_service = ChatDocumentService()
