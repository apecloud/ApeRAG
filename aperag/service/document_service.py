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
from aperag.exceptions import (
    CollectionInactiveException,
    DocumentNotFoundException,
    QuotaExceededException,
    ResourceNotFoundException,
    invalid_param,
)
from aperag.index.manager import document_index_manager
from aperag.objectstore.base import get_object_store
from aperag.schema import view_models
from aperag.schema.view_models import Document, DocumentList
from aperag.utils.constant import QuotaType
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS

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
        # Import here to avoid circular dependencies
        from aperag.tasks.scheduler import create_task_scheduler
        from aperag.config import settings

        # Use task scheduler to trigger reconciliation
        task_scheduler = create_task_scheduler(settings.task_scheduler_type)
        task_scheduler.schedule_reconcile_indexes()
        logger.debug("Index reconciliation task triggered for real-time processing")
    except ImportError:
        logger.warning("Task scheduler not available, skipping index reconciliation trigger")
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

        # Prepare file data and validate all files before starting any database operations
        file_data = []
        for item in files:
            file_suffix = os.path.splitext(item.filename)[1].lower()
            if file_suffix not in supported_file_extensions:
                raise invalid_param("file_type", f"unsupported file type {file_suffix}")
            if item.size > settings.max_document_size:
                raise invalid_param("file_size", "file size is too large")

            # Read file content from UploadFile
            file_content = await item.read()
            # Reset file pointer for potential future use
            await item.seek(0)

            file_data.append(
                {"filename": item.filename, "size": item.size, "suffix": file_suffix, "content": file_content}
            )

        # Process all files in a single transaction for atomicity
        async def _create_documents_atomically(session):
            from aperag.db.models import Document, DocumentStatus

            documents_created = []
            obj_store = get_object_store()
            uploaded_files = []  # Track uploaded files for cleanup

            try:
                for file_info in file_data:
                    # Create document in database directly using session
                    document_instance = Document(
                        user=user,
                        name=file_info["filename"],
                        status=DocumentStatus.PENDING,
                        size=file_info["size"],
                        collection_id=collection.id,
                    )
                    session.add(document_instance)
                    await session.flush()
                    await session.refresh(document_instance)

                    # Upload to object store
                    upload_path = f"{document_instance.object_store_base_path()}/original{file_info['suffix']}"
                    await sync_to_async(obj_store.put)(upload_path, file_info["content"])
                    uploaded_files.append(upload_path)

                    # Update document with object path
                    metadata = json.dumps({"object_path": upload_path})
                    document_instance.doc_metadata = metadata
                    session.add(document_instance)
                    await session.flush()
                    await session.refresh(document_instance)

                    # Create index specs for the new document
                    index_types = [db_models.DocumentIndexType.VECTOR, db_models.DocumentIndexType.FULLTEXT]
                    collection_config = json.loads(collection.config)
                    if collection_config.get("enable_knowledge_graph", False):
                        index_types.append(db_models.DocumentIndexType.GRAPH)

                    await document_index_manager.create_document_indexes(
                        session, document_instance.id, user, index_types
                    )

                    # Build response object
                    doc_response = await self.build_document_response(document_instance, session)
                    documents_created.append(doc_response)

                return documents_created

            except Exception as e:
                # Clean up uploaded files on database transaction failure
                for upload_path in uploaded_files:
                    try:
                        await sync_to_async(obj_store.delete_objects_by_prefix)(upload_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup uploaded file during rollback: {cleanup_error}")
                raise e

        response = await self.db_ops.execute_with_transaction(_create_documents_atomically)

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
                metadata = json.loads(instance.doc_metadata or "{}")
                metadata["labels"] = config["labels"]
                updated_metadata = json.dumps(metadata)

                # Update document and indexes atomically in a single transaction
                async def _update_document_atomically(session):
                    from sqlalchemy import select

                    from aperag.db.models import Document, DocumentStatus

                    # Update document metadata
                    stmt = select(Document).where(
                        Document.id == document_id,
                        Document.collection_id == collection_id,
                        Document.user == user,
                        Document.status != DocumentStatus.DELETED,
                    )
                    result = await session.execute(stmt)
                    document = result.scalars().first()

                    if not document:
                        raise DocumentNotFoundException(document_id)

                    document.doc_metadata = updated_metadata
                    session.add(document)
                    await session.flush()
                    await session.refresh(document)

                    # Update index specs to trigger re-indexing
                    await document_index_manager.update_document_indexes(session, document.id)

                    # Build response object
                    return await self.build_document_response(document, session)

                result = await self.db_ops.execute_with_transaction(_update_document_atomically)
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

        # Delete document and indexes atomically in a single transaction
        async def _delete_document_atomically(session):
            from sqlalchemy import select

            from aperag.db.models import Document, DocumentStatus, utc_now

            # Get and delete document
            stmt = select(Document).where(
                Document.id == document_id,
                Document.collection_id == collection_id,
                Document.user == user,
                Document.status != DocumentStatus.DELETED,
            )
            result = await session.execute(stmt)
            doc_to_delete = result.scalars().first()

            if not doc_to_delete:
                return None

            # Soft delete document
            doc_to_delete.status = DocumentStatus.DELETED
            doc_to_delete.gmt_deleted = utc_now()
            session.add(doc_to_delete)
            await session.flush()
            await session.refresh(doc_to_delete)

            # Mark index specs for deletion
            await document_index_manager.delete_document_indexes(session, document_id)

            # Build response object
            return await self.build_document_response(doc_to_delete, session)

        result = await self.db_ops.execute_with_transaction(_delete_document_atomically)

        if result:
            # Delete object storage files after successful database transaction
            obj_store = get_object_store()
            try:
                await sync_to_async(obj_store.delete_objects_by_prefix)(f"{document.object_store_base_path()}/")
            except Exception as e:
                logger.warning(f"Failed to delete object storage files for document {document_id}: {e}")

            # Trigger index reconciliation after successful document deletion
            _trigger_index_reconciliation()

            return result

        return None

    async def delete_documents(self, user: str, collection_id: str, document_ids: List[str]) -> dict:
        # Delete documents and indexes atomically in a single transaction
        async def _delete_documents_atomically(session):
            from sqlalchemy import select

            from aperag.db.models import Document, DocumentStatus, utc_now

            # Get documents to delete
            stmt = select(Document).where(
                Document.id.in_(document_ids),
                Document.collection_id == collection_id,
                Document.user == user,
                Document.status != DocumentStatus.DELETED,
            )
            result = await session.execute(stmt)
            documents_to_delete = result.scalars().all()

            if not documents_to_delete:
                return [], list(document_ids)

            # Soft delete documents
            success_ids = []
            for doc in documents_to_delete:
                doc.status = DocumentStatus.DELETED
                doc.gmt_deleted = utc_now()
                session.add(doc)
                success_ids.append(doc.id)

            await session.flush()

            # Delete indexes for all successful deletions
            for doc_id in success_ids:
                await document_index_manager.delete_document_indexes(session, doc_id)

            # Calculate failed IDs
            failed_ids = list(set(document_ids) - set(success_ids))
            return success_ids, failed_ids

        success_ids, failed_ids = await self.db_ops.execute_with_transaction(_delete_documents_atomically)

        result = {"success": success_ids, "failed": failed_ids}

        # Trigger index reconciliation after successful batch document deletion
        if result.get("success"):  # Only trigger if at least one document was deleted successfully
            _trigger_index_reconciliation()

        return result


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
document_service = DocumentService()
