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
import os
from typing import List, Optional

from asgiref.sync import sync_to_async
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.config import get_async_session, settings
from aperag.db import models as db_models
from aperag.db.ops import (
    AsyncDatabaseOps,
    async_db_ops,
)
from aperag.docparser.doc_parser import DocParser
from aperag.index.manager import document_index_manager
from aperag.objectstore.base import get_object_store
from aperag.schema import view_models
from aperag.schema.view_models import Document, DocumentList
from aperag.utils.constant import QuotaType
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS
from aperag.exceptions import ResourceNotFoundException, CollectionInactiveException, QuotaExceededException, DocumentNotFoundException, invalid_param

logger = logging.getLogger(__name__)


def _trigger_index_reconciliation():
    """
    Trigger index reconciliation task asynchronously for better real-time responsiveness.

    This is called after document create/update/delete operations to immediately
    process index changes, improving responsiveness compared to relying only on
    periodic reconciliation. The periodic task interval can be increased since
    we have real-time triggering.
    """
    try:
        # Import here to avoid circular dependencies and handle missing celery gracefully
        from config.celery_tasks import reconcile_indexes_task

        # Trigger the reconciliation task asynchronously
        reconcile_indexes_task.delay()
        logger.debug("Index reconciliation task triggered for real-time processing")
    except ImportError:
        logger.warning("Celery not available, skipping index reconciliation trigger")
    except Exception as e:
        logger.warning(f"Failed to trigger index reconciliation task: {e}")


class DocumentService:
    """Document service that handles business logic for documents"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def build_document_response(
        self, document: db_models.Document, session: AsyncSession
    ) -> view_models.Document:
        """Build Document response object for API return."""
        # Get index status from new tables
        index_status_info = await document_index_manager.get_document_index_status(session, document.id)

        # Convert new format to old API format for backward compatibility
        indexes = index_status_info.get("indexes", {})

        # Map new states to old enum values for API compatibility
        def map_state_to_old_enum(actual_state: str):
            if actual_state == "absent":
                return "SKIPPED"
            elif actual_state == "creating":
                return "RUNNING"
            elif actual_state == "present":
                return "COMPLETE"
            elif actual_state == "failed":
                return "FAILED"
            else:
                return "PENDING"

        return Document(
            id=document.id,
            name=document.name,
            status=document.status,
            vector_index_status=map_state_to_old_enum(indexes.get("vector", {}).get("actual_state", "absent")),
            fulltext_index_status=map_state_to_old_enum(indexes.get("fulltext", {}).get("actual_state", "absent")),
            graph_index_status=map_state_to_old_enum(indexes.get("graph", {}).get("actual_state", "absent")),
            size=document.size,
            created=document.gmt_created,
            updated=document.gmt_updated,
        )

    async def create_documents(
        self, user: str, collection_id: str, files: List[UploadFile]
    ) -> view_models.DocumentList:
        if len(files) > 50:
            raise invalid_param("file_count", "documents are too many, add document failed")

        # Check collection exists and is active
        collection = await self.db_ops.query_collection(user, collection_id)
        if collection is None:
            raise ResourceNotFoundException("Collection", collection_id)
        if collection.status != db_models.CollectionStatus.ACTIVE:
            raise CollectionInactiveException(collection_id)

        if settings.max_document_count:
            document_limit = await self.db_ops.query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
            if document_limit is None:
                document_limit = settings.max_document_count
            if await self.db_ops.query_documents_count(user, collection_id) >= document_limit:
                raise QuotaExceededException("document", document_limit)

        supported_file_extensions = DocParser().supported_extensions()
        supported_file_extensions += SUPPORTED_COMPRESSED_EXTENSIONS

        response = []

        for item in files:
            file_suffix = os.path.splitext(item.filename)[1].lower()
            if file_suffix not in supported_file_extensions:
                raise invalid_param("file_type", f"unsupported file type {file_suffix}")
            if item.size > settings.max_document_size:
                raise invalid_param("file_size", "file size is too large")

            # Direct call to repository method, which handles its own transaction
            document_instance = await self.db_ops.create_document(
                user=user, collection_id=collection.id, name=item.filename, size=item.size
            )

            obj_store = get_object_store()
            upload_path = f"{document_instance.object_store_base_path()}/original{file_suffix}"

            # Read file content from UploadFile
            file_content = await item.read()
            # Reset file pointer for potential future use
            await item.seek(0)

            # Use sync_to_async to call the synchronous put method with file content
            await sync_to_async(obj_store.put)(upload_path, file_content)

            # Update document with object path
            metadata = json.dumps({"object_path": upload_path})
            updated_doc = await self.db_ops.update_document_by_id(
                user, collection_id, document_instance.id, metadata
            )

            # Get document response
            async def _get_doc_response(session):
                return await self.build_document_response(updated_doc, session)
            doc_response = await self.db_ops._execute_query(_get_doc_response)
            response.append(doc_response)

            # Create index specs for the new document
            index_types = [db_models.DocumentIndexType.VECTOR, db_models.DocumentIndexType.FULLTEXT]
            collection_config = json.loads(collection.config)
            if collection_config.get("enable_knowledge_graph", False):
                index_types.append(db_models.DocumentIndexType.GRAPH)

            async def _create_indexes(session):
                await document_index_manager.create_document_indexes(session, updated_doc.id, user, index_types)
            await self.db_ops.execute_with_transaction(_create_indexes)

        # Trigger index reconciliation after successful document creation
        _trigger_index_reconciliation()

        return DocumentList(items=response)

    async def list_documents(self, user: str, collection_id: str) -> view_models.DocumentList:
        documents = await self.db_ops.query_documents([user], collection_id)
        response = []
        async for session in get_async_session():
            for document in documents:
                response.append(await self.build_document_response(document, session))
        return DocumentList(items=response)

    async def get_document(self, user: str, collection_id: str, document_id: str) -> view_models.Document:
        document = await self.db_ops.query_document(user, collection_id, document_id)
        if document is None:
            raise DocumentNotFoundException(document_id)
        async for session in get_async_session():
            return await self.build_document_response(document, session)

    async def update_document(
        self, user: str, collection_id: str, document_id: str, document_in: view_models.DocumentUpdate
    ) -> view_models.Document:
        instance = await self.db_ops.query_document(user, collection_id, document_id)
        if instance is None:
            raise DocumentNotFoundException(document_id)

        if document_in.config:
            try:
                config = json.loads(document_in.config)
                metadata = json.loads(instance.metadata)
                metadata["labels"] = config["labels"]
                updated_metadata = json.dumps(metadata)

                # Direct call to repository method, which handles its own transaction
                updated_doc = await self.db_ops.update_document_by_id(
                    user, collection_id, document_id, updated_metadata
                )

                if not updated_doc:
                    raise DocumentNotFoundException(document_id)

                # Update index specs to trigger re-indexing
                async def _update_indexes(session):
                    await document_index_manager.update_document_indexes(session, updated_doc.id)
                await self.db_ops.execute_with_transaction(_update_indexes)

                # Get document response
                async def _get_doc_response(session):
                    return await self.build_document_response(updated_doc, session)
                result = await self.db_ops._execute_query(_get_doc_response)
            except json.JSONDecodeError:
                raise invalid_param("config", "invalid document config")
        else:
            async def _get_doc_response(session):
                return await self.build_document_response(instance, session)
            result = await self.db_ops._execute_query(_get_doc_response)

        # Trigger index reconciliation after successful document update
        _trigger_index_reconciliation()

        return result

    async def delete_document(self, user: str, collection_id: str, document_id: str) -> Optional[view_models.Document]:
        """Delete document by ID (idempotent operation)
        
        Returns the deleted document or None if already deleted/not found
        """
        document = await self.db_ops.query_document(user, collection_id, document_id)
        if document is None:
            # Document already deleted or never existed - idempotent operation
            return None

        # Delete object storage files
        obj_store = get_object_store()
        await sync_to_async(obj_store.delete_objects_by_prefix)(f"{document.object_store_base_path()}/")

        # Direct call to repository method, which handles its own transaction
        deleted_doc = await self.db_ops.delete_document_by_id(user, collection_id, document_id)

        if deleted_doc:
            # Mark index specs for deletion
            async def _delete_indexes(session):
                await document_index_manager.delete_document_indexes(session, document_id)
            await self.db_ops.execute_with_transaction(_delete_indexes)

            # Get document response
            async def _get_doc_response(session):
                return await self.build_document_response(deleted_doc, session)
            result = await self.db_ops._execute_query(_get_doc_response)

            # Trigger index reconciliation after successful document deletion
            _trigger_index_reconciliation()

            return result
        
        return None

    async def delete_documents(self, user: str, collection_id: str, document_ids: List[str]) -> dict:
        # Direct call to repository method, which handles its own transaction
        success_ids, failed_ids = await self.db_ops.delete_documents_by_ids(user, collection_id, document_ids)

        # Delete indexes for successful deletions
        for doc_id in success_ids:
            async def _delete_indexes(session):
                await document_index_manager.delete_document_indexes(session, doc_id)
            await self.db_ops.execute_with_transaction(_delete_indexes)

        result = {"success": success_ids, "failed": failed_ids}

        # Trigger index reconciliation after successful batch document deletion
        if result.get("success"):  # Only trigger if at least one document was deleted successfully
            _trigger_index_reconciliation()

        return result


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
document_service = DocumentService()
