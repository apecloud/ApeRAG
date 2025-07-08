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
import mimetypes
import os
from typing import List

from asgiref.sync import sync_to_async
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.config import get_async_session, settings
from aperag.db import models as db_models
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
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
from aperag.schema.view_models import Chunk, DocumentList, DocumentPreview
from aperag.utils.constant import QuotaType
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS
from aperag.utils.utils import generate_vector_db_collection_name, utc_now
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor
from config.settings import VECTOR_DB_CONTEXT, VECTOR_DB_TYPE

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
        """Build Document response object for API return using new status model."""
        from sqlalchemy import select

        from aperag.db.models import DocumentIndex

        # Get all document indexes for status calculation
        document_indexes = await session.execute(
            select(DocumentIndex).where(
                DocumentIndex.document_id == document.id,
                DocumentIndex.status != db_models.DocumentIndexStatus.DELETING,
                DocumentIndex.status != db_models.DocumentIndexStatus.DELETION_IN_PROGRESS,
            )
        )
        indexes = document_indexes.scalars().all()

        # Map index states to API response format
        index_status = {}
        index_updated = {}

        # Initialize all types as SKIPPED (when no record exists)
        all_types = [
            db_models.DocumentIndexType.VECTOR,
            db_models.DocumentIndexType.FULLTEXT,
            db_models.DocumentIndexType.GRAPH,
        ]
        for index_type in all_types:
            index_status[index_type] = "SKIPPED"

        # Update with actual states from database
        for index in indexes:
            index_status[index.index_type] = index.status
            index_updated[index.index_type] = index.gmt_updated

        return view_models.Document(
            id=document.id,
            name=document.name,
            status=document.status,
            vector_index_status=index_status.get(db_models.DocumentIndexType.VECTOR, "SKIPPED"),
            fulltext_index_status=index_status.get(db_models.DocumentIndexType.FULLTEXT, "SKIPPED"),
            graph_index_status=index_status.get(db_models.DocumentIndexType.GRAPH, "SKIPPED"),
            vector_index_updated=index_updated.get(db_models.DocumentIndexType.VECTOR, None),
            fulltext_index_updated=index_updated.get(db_models.DocumentIndexType.FULLTEXT, None),
            graph_index_updated=index_updated.get(db_models.DocumentIndexType.GRAPH, None),
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

                    # Use index manager to create indexes with new status model
                    await document_index_manager.create_or_update_document_indexes(
                        document_id=document_instance.id, index_types=index_types, session=session
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

    async def _delete_document(self, session: AsyncSession, user: str, collection_id: str, document_id: str):
        """
        Core logic to delete a single document and its associated resources.
        This method is designed to be called within a transaction.
        """
        # Validate document existence and ownership
        document = await self.db_ops.query_document(user, collection_id, document_id)
        if document is None:
            # Silently ignore if document not found, as it might have been deleted by another process
            logger.warning(f"Document {document_id} not found for deletion, skipping.")
            return

        # Use index manager to mark all related indexes for deletion
        await document_index_manager.delete_document_indexes(document_id=document.id, index_types=None, session=session)

        # Delete from object store
        obj_store = get_object_store()
        metadata = json.loads(document.doc_metadata) if document.doc_metadata else {}
        if metadata.get("object_path"):
            try:
                # Use delete_objects_by_prefix to remove all related files (original, chunks, etc.)
                await sync_to_async(obj_store.delete_objects_by_prefix)(document.object_store_base_path())
                logger.info(f"Deleted objects from object store with prefix: {document.object_store_base_path()}")
            except Exception as e:
                logger.warning(f"Failed to delete objects for document {document.id} from object store: {e}")

        # Mark document as deleted
        document.status = db_models.DocumentStatus.DELETED
        document.gmt_deleted = utc_now()
        session.add(document)
        await session.flush()
        logger.info(f"Successfully marked document {document.id} as deleted.")

        return document

    async def delete_document(self, user: str, collection_id: str, document_id: str) -> dict:
        """Delete a single document and trigger index reconciliation."""

        async def _delete_document_atomically(session: AsyncSession):
            return await self._delete_document(session, user, collection_id, document_id)

        result = await self.db_ops.execute_with_transaction(_delete_document_atomically)

        # Trigger reconciliation to process the deletion
        _trigger_index_reconciliation()
        return result

    async def delete_documents(self, user: str, collection_id: str, document_ids: List[str]) -> dict:
        """Delete multiple documents and trigger index reconciliation."""

        async def _delete_documents_atomically(session: AsyncSession):
            deleted_ids = []
            for doc_id in document_ids:
                await self._delete_document(session, user, collection_id, doc_id)
                deleted_ids.append(doc_id)
            return {"deleted_ids": deleted_ids, "status": "success"}

        result = await self.db_ops.execute_with_transaction(_delete_documents_atomically)

        # Trigger reconciliation to process deletions
        _trigger_index_reconciliation()
        return result

    async def rebuild_document_indexes(
        self, user_id: str, collection_id: str, document_id: str, index_types: List[str]
    ) -> dict:
        """
        Rebuild specified indexes for a document

        Args:
            user_id: User ID
            collection_id: Collection ID
            document_id: Document ID
            index_types: List of index types to rebuild ('VECTOR', 'FULLTEXT', 'GRAPH')

        Returns:
            dict: Success response
        """
        if len(set(index_types)) != len(index_types):
            raise invalid_param("index_types", "duplicate index types are not allowed")

        logger.info(f"Rebuilding indexes for document {document_id} with types: {index_types}")

        # Convert index types to enum values outside transaction
        from aperag.db.models import DocumentIndexType

        index_type_enums = []
        for index_type in index_types:
            if index_type == "VECTOR":
                index_type_enums.append(DocumentIndexType.VECTOR)
            elif index_type == "FULLTEXT":
                index_type_enums.append(DocumentIndexType.FULLTEXT)
            elif index_type == "GRAPH":
                index_type_enums.append(DocumentIndexType.GRAPH)
            else:
                raise invalid_param("index_type", f"Invalid index type: {index_type}")

        # Execute all operations atomically in a single transaction
        async def _rebuild_document_indexes_atomically(session):
            # Verify document exists and user has access
            document = await self.db_ops.query_document(user_id, collection_id, document_id)
            if not document:
                raise DocumentNotFoundException(f"Document {document_id} not found")

            if document.collection_id != collection_id:
                raise ResourceNotFoundException(f"Document {document_id} not found in collection {collection_id}")

            # Verify user has access to the collection
            collection = await self.db_ops.query_collection(user_id, collection_id)
            if not collection or collection.user != user_id:
                raise ResourceNotFoundException(f"Collection {collection_id} not found or access denied")
            collection_config = json.loads(collection.config)
            if not collection_config.get("enable_knowledge_graph", False):
                # Only remove GRAPH type if it's actually in the list to avoid ValueError
                if db_models.DocumentIndexType.GRAPH in index_type_enums:
                    index_type_enums.remove(db_models.DocumentIndexType.GRAPH)

            # Trigger index rebuild by incrementing version for selected index types
            await document_index_manager.create_or_update_document_indexes(session, document_id, index_type_enums)

            logger.info(f"Successfully triggered rebuild for document {document_id} indexes: {index_types}")

            return {"code": "200", "message": f"Index rebuild initiated for types: {', '.join(index_types)}"}

        result = await self.db_ops.execute_with_transaction(_rebuild_document_indexes_atomically)

        # Trigger index reconciliation after successful rebuild initiation
        _trigger_index_reconciliation()

        return result

    async def get_document_chunks(self, user_id: str, collection_id: str, document_id: str) -> List[Chunk]:
        """
        Get all chunks of a document.
        """
        async for session in get_async_session():
            # 1. Get the document to verify ownership and get collection_id
            stmt = select(db_models.Document).filter(
                db_models.Document.id == document_id,
                db_models.Document.collection_id == collection_id,
                db_models.Document.user == user_id,
            )
            result = await session.execute(stmt)
            document = result.scalars().first()
            if not document:
                raise DocumentNotFoundException(document_id)

            # 2. Get the chunk IDs (ctx_ids) from the document_index table
            stmt = select(db_models.DocumentIndex).filter(
                db_models.DocumentIndex.document_id == document_id,
                db_models.DocumentIndex.index_type == db_models.DocumentIndexType.VECTOR,
            )
            result = await session.execute(stmt)
            doc_index = result.scalars().first()

            if not doc_index or not doc_index.index_data:
                return []

            try:
                index_data = json.loads(doc_index.index_data)
                ctx_ids = index_data.get("context_ids", [])
            except (json.JSONDecodeError, AttributeError):
                return []

            if not ctx_ids:
                return []

            # 3. Retrieve chunks from Qdrant
            try:
                collection_name = generate_vector_db_collection_name(collection_id=document.collection_id)
                ctx = json.loads(VECTOR_DB_CONTEXT)
                ctx["collection"] = collection_name
                vector_store_adaptor = VectorStoreConnectorAdaptor(VECTOR_DB_TYPE, ctx=ctx)
                qdrant_client = vector_store_adaptor.connector.client

                points = qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=ctx_ids,
                    with_payload=True,
                )

                # 4. Format the response
                chunks = []
                for point in points:
                    if point.payload:
                        # In llama-index-0.10.13, the payload is stored in _node_content
                        node_content = point.payload.get("_node_content")
                        if node_content and isinstance(node_content, str):
                            try:
                                payload_data = json.loads(node_content)
                                chunks.append(
                                    Chunk(
                                        id=point.id,
                                        text=payload_data.get("text", ""),
                                        metadata=payload_data.get("metadata", {}),
                                    )
                                )
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse _node_content for point {point.id}")
                        else:
                            # Fallback for older or different data structures
                            chunks.append(
                                Chunk(
                                    id=point.id,
                                    text=point.payload.get("text", ""),
                                    metadata=point.payload.get("metadata", {}),
                                )
                            )

                return chunks
            except Exception as e:
                logger.error(
                    f"Failed to retrieve chunks from vector store for document {document_id}: {e}", exc_info=True
                )
                raise HTTPException(status_code=500, detail="Failed to retrieve chunks from vector store")

    async def get_document_preview(self, user_id: str, collection_id: str, document_id: str) -> DocumentPreview:
        """
        Get all preview-related information for a document.
        """
        async for session in get_async_session():
            # 1. Get document and vector index in one go
            doc_stmt = select(db_models.Document).filter(
                db_models.Document.id == document_id,
                db_models.Document.collection_id == collection_id,
                db_models.Document.user == user_id,
            )
            doc_result = await session.execute(doc_stmt)
            document = doc_result.scalars().first()
            if not document:
                raise DocumentNotFoundException(document_id)

            index_stmt = select(db_models.DocumentIndex).filter(
                db_models.DocumentIndex.document_id == document_id,
                db_models.DocumentIndex.index_type == db_models.DocumentIndexType.VECTOR,
            )
            index_result = await session.execute(index_stmt)
            doc_index = index_result.scalars().first()

            # 2. Get chunks
            chunks = await self.get_document_chunks(user_id, collection_id, document_id)

            # 3. Get markdown content
            obj_store = get_object_store()
            markdown_content = ""
            # The parsed markdown file is stored with the name "parsed.md"
            markdown_path = f"{document.object_store_base_path()}/parsed.md"
            try:
                md_stream = await sync_to_async(obj_store.get)(markdown_path)
                if md_stream:
                    markdown_content = md_stream.read().decode("utf-8")
            except Exception:
                logger.warning(f"Could not find or read markdown file at {markdown_path}")

            # 4. Determine paths
            doc_metadata = json.loads(document.doc_metadata) if document.doc_metadata else {}
            doc_object_path = doc_metadata.get("object_path")
            if doc_object_path:
                doc_object_path = os.path.basename(doc_object_path)

            converted_pdf_object_path = None
            index_data = json.loads(doc_index.index_data) if doc_index and doc_index.index_data else {}
            if index_data.get("has_pdf_source_map") and not document.name.lower().endswith(".pdf"):
                # If the parsing result contains pdf_source_map metadata,
                # it means it is a PDF or has been converted to a PDF.
                # But only converted documents have a converted.pdf file.
                pdf_path = f"{document.object_store_base_path()}/converted.pdf"
                exists = await sync_to_async(obj_store.obj_exists)(pdf_path)
                if exists:
                    converted_pdf_object_path = "converted.pdf"

            # 5. Construct and return response
            return DocumentPreview(
                doc_object_path=doc_object_path,
                doc_filename=document.name,
                converted_pdf_object_path=converted_pdf_object_path,
                markdown_content=markdown_content,
                chunks=chunks,
            )

    async def get_document_object(self, user_id: str, collection_id: str, document_id: str, path: str):
        """
        Get a file object associated with a document from the object store.
        """
        async for session in get_async_session():
            # 1. Verify user has access to the document
            stmt = select(db_models.Document).filter(
                db_models.Document.id == document_id,
                db_models.Document.collection_id == collection_id,
                db_models.Document.user == user_id,
            )
            result = await session.execute(stmt)
            document = result.scalars().first()
            if not document:
                raise DocumentNotFoundException(document_id)

            # Construct the full path and perform security check
            full_path = os.path.join(document.object_store_base_path(), path)
            if not full_path.startswith(document.object_store_base_path()):
                raise HTTPException(status_code=403, detail="Access denied to this object path")

            # 2. Get the object from object store
            try:
                obj_store = get_object_store()
                file_data_stream = await sync_to_async(obj_store.get)(full_path)

                if not file_data_stream:
                    raise HTTPException(status_code=404, detail="Object not found at specified path")

                def file_iterator():
                    with file_data_stream:
                        yield from file_data_stream

                # 3. Stream the response
                content_type, _ = mimetypes.guess_type(full_path)
                if content_type is None:
                    content_type = "application/octet-stream"

                return StreamingResponse(file_iterator(), media_type=content_type)
            except Exception as e:
                logger.error(f"Failed to get object for document {document_id} at path {full_path}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to get object from store")


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
document_service = DocumentService()
