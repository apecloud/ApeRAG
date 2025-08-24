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
import re
from typing import List

from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.config import settings
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
from aperag.objectstore.base import get_async_object_store
from aperag.schema import view_models
from aperag.schema.view_models import Chunk, DocumentList, DocumentPreview, VisionChunk
from aperag.utils.pagination import (
    ListParams,
    PaginatedResponse,
    PaginationHelper,
    PaginationParams,
    SearchParams,
    SortParams,
)
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS
from aperag.utils.utils import generate_vector_db_collection_name, utc_now
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor

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

    async def _validate_collection(self, user: str, collection_id: str) -> db_models.Collection:
        """
        Validate that collection exists and is active.
        Returns the collection if valid, raises exception otherwise.
        """
        collection = await self.db_ops.query_collection(user, collection_id)
        if collection is None:
            raise ResourceNotFoundException("Collection", collection_id)
        if collection.status != db_models.CollectionStatus.ACTIVE:
            raise CollectionInactiveException(collection_id)
        return collection

    def _validate_file(self, filename: str, size: int) -> str:
        """
        Validate file extension and size.
        Returns the file suffix if valid, raises exception otherwise.
        """
        supported_file_extensions = DocParser().supported_extensions()
        supported_file_extensions += SUPPORTED_COMPRESSED_EXTENSIONS

        file_suffix = os.path.splitext(filename)[1].lower()
        if file_suffix not in supported_file_extensions:
            raise invalid_param("file_type", f"unsupported file type {file_suffix}")
        if size > settings.max_document_size:
            raise invalid_param("file_size", "file size is too large")

        return file_suffix

    async def _check_document_quotas(self, session: AsyncSession, user: str, collection_id: str, count: int):
        """
        Check and consume document quotas.
        Raises QuotaExceededException if quota would be exceeded.
        """
        from sqlalchemy import func, select

        from aperag.service.quota_service import quota_service

        # Check and consume user quota
        await quota_service.check_and_consume_quota(user, "max_document_count", count, session)

        # Check per-collection quota
        stmt = (
            select(func.count())
            .select_from(db_models.Document)
            .where(
                db_models.Document.collection_id == collection_id,
                db_models.Document.status != db_models.DocumentStatus.DELETED,
                db_models.Document.status != db_models.DocumentStatus.UPLOADED,  # Don't count temporary uploads
            )
        )
        existing_doc_count = await session.scalar(stmt)

        # Get per-collection quota limit
        from aperag.db.models import UserQuota

        stmt = select(UserQuota).where(UserQuota.user == user, UserQuota.key == "max_document_count_per_collection")
        result = await session.execute(stmt)
        per_collection_quota = result.scalars().first()

        if per_collection_quota and (existing_doc_count + count) > per_collection_quota.quota_limit:
            raise QuotaExceededException(
                "max_document_count_per_collection", per_collection_quota.quota_limit, existing_doc_count
            )

    def _get_index_types_for_collection(self, collection_config: dict) -> list:
        """
        Get the list of index types to create based on collection configuration.
        """
        index_types = [
            db_models.DocumentIndexType.VECTOR,
            db_models.DocumentIndexType.FULLTEXT,
        ]

        if collection_config.get("enable_knowledge_graph", False):
            index_types.append(db_models.DocumentIndexType.GRAPH)
        if collection_config.get("enable_summary", False):
            index_types.append(db_models.DocumentIndexType.SUMMARY)
        if collection_config.get("enable_vision", False):
            index_types.append(db_models.DocumentIndexType.VISION)

        return index_types

    async def _create_document_record(
        self,
        session: AsyncSession,
        user: str,
        collection_id: str,
        filename: str,
        size: int,
        status: db_models.DocumentStatus,
        file_suffix: str,
        file_content: bytes,
    ) -> db_models.Document:
        """
        Create a document record in database and upload file to object store.
        Returns the created document instance.
        """
        # Create document in database
        document_instance = db_models.Document(
            user=user,
            name=filename,
            status=status,
            size=size,
            collection_id=collection_id,
        )
        session.add(document_instance)
        await session.flush()
        await session.refresh(document_instance)

        # Upload to object store
        async_obj_store = get_async_object_store()
        upload_path = f"{document_instance.object_store_base_path()}/original{file_suffix}"
        await async_obj_store.put(upload_path, file_content)

        # Update document with object path
        metadata = json.dumps({"object_path": upload_path})
        document_instance.doc_metadata = metadata
        session.add(document_instance)
        await session.flush()
        await session.refresh(document_instance)

        return document_instance

    async def _query_documents_with_indexes(
        self, user: str, collection_id: str, document_id: str = None
    ) -> List[db_models.Document]:
        """
        Common function to query documents with their indexes using JOIN.
        If document_id is provided, query single document, otherwise query all documents.
        """

        async def _execute_query(session):
            from sqlalchemy import and_, outerjoin, select

            # Create JOIN query between Document and DocumentIndex tables
            # Use outerjoin to get all documents even if they don't have indexes
            query = (
                select(
                    db_models.Document,
                    db_models.DocumentIndex.index_type,
                    db_models.DocumentIndex.index_data,
                    db_models.DocumentIndex.status.label("index_status"),
                    db_models.DocumentIndex.gmt_created.label("index_created_at"),
                    db_models.DocumentIndex.gmt_updated.label("index_updated_at"),
                    db_models.DocumentIndex.error_message.label("index_error_message"),
                )
                .select_from(
                    outerjoin(
                        db_models.Document,
                        db_models.DocumentIndex,
                        db_models.Document.id == db_models.DocumentIndex.document_id,
                    )
                )
                .where(
                    and_(
                        db_models.Document.user == user,
                        db_models.Document.collection_id == collection_id,
                        db_models.Document.status != db_models.DocumentStatus.DELETED,
                        db_models.Document.status
                        != db_models.DocumentStatus.UPLOADED,  # Filter out temporary uploaded documents
                        db_models.Document.status
                        != db_models.DocumentStatus.EXPIRED,  # Filter out temporary uploaded documents
                    )
                )
                .order_by(db_models.Document.gmt_created.desc())
            )

            # Add document_id filter if provided (for single document query)
            if document_id:
                query = query.where(db_models.Document.id == document_id)

            result = await session.execute(query)
            rows = result.fetchall()

            # Group results by document and attach all index information
            documents_dict = {}
            for row in rows:
                doc = row.Document
                if doc.id not in documents_dict:
                    documents_dict[doc.id] = doc
                    # Initialize index information for all types
                    doc.indexes = {"VECTOR": None, "FULLTEXT": None, "GRAPH": None, "SUMMARY": None, "VISION": None}

                # Add index information if exists
                if row.index_type:
                    doc.indexes[row.index_type] = {
                        "index_type": row.index_type,
                        "status": row.index_status,
                        "created_at": row.index_created_at,
                        "updated_at": row.index_updated_at,
                        "error_message": row.index_error_message,
                        "index_data": row.index_data,
                    }

            return list(documents_dict.values())

        return await self.db_ops._execute_query(_execute_query)

    async def _build_document_response(self, document: db_models.Document) -> view_models.Document:
        """
        Build document response object with all index types information.
        """
        # Get all index information if available
        indexes = getattr(
            document, "indexes", {"VECTOR": None, "FULLTEXT": None, "GRAPH": None, "SUMMARY": None, "VISION": None}
        )

        # Parse summary from SUMMARY index's index_data
        summary = None
        summary_index = indexes.get("SUMMARY")
        if summary_index and summary_index.get("index_data"):
            try:
                index_data = json.loads(summary_index["index_data"]) if summary_index["index_data"] else None
                if index_data:
                    summary = index_data.get("summary")
            except Exception:
                summary = None

        return view_models.Document(
            id=document.id,
            name=document.name,
            status=document.status,
            # Vector index information
            vector_index_status=indexes["VECTOR"]["status"] if indexes["VECTOR"] else "SKIPPED",
            vector_index_updated=indexes["VECTOR"]["updated_at"] if indexes["VECTOR"] else None,
            # Fulltext index information
            fulltext_index_status=indexes["FULLTEXT"]["status"] if indexes["FULLTEXT"] else "SKIPPED",
            fulltext_index_updated=indexes["FULLTEXT"]["updated_at"] if indexes["FULLTEXT"] else None,
            # Graph index information
            graph_index_status=indexes["GRAPH"]["status"] if indexes["GRAPH"] else "SKIPPED",
            graph_index_updated=indexes["GRAPH"]["updated_at"] if indexes["GRAPH"] else None,
            # Summary index information
            summary_index_status=indexes["SUMMARY"]["status"] if indexes.get("SUMMARY") else "SKIPPED",
            summary_index_updated=indexes["SUMMARY"]["updated_at"] if indexes.get("SUMMARY") else None,
            vision_index_status=indexes["VISION"]["status"] if indexes.get("VISION") else "SKIPPED",
            vision_index_updated=indexes["VISION"]["updated_at"] if indexes.get("VISION") else None,
            summary=summary,  # Parse from index_data
            size=document.size,
            created=document.gmt_created,
            updated=document.gmt_updated,
        )

    async def create_documents(
        self, user: str, collection_id: str, files: List[UploadFile]
    ) -> view_models.DocumentList:
        if len(files) > 50:
            raise invalid_param("file_count", "documents are too many, add document failed")

        # Validate collection
        collection = await self._validate_collection(user, collection_id)

        # Prepare file data and validate all files before starting any database operations
        file_data = []
        for item in files:
            file_suffix = self._validate_file(item.filename, item.size)

            # Read file content from UploadFile
            file_content = await item.read()
            # Reset file pointer for potential future use
            await item.seek(0)

            file_data.append(
                {"filename": item.filename, "size": item.size, "suffix": file_suffix, "content": file_content}
            )

        # Process all files in a single transaction for atomicity
        async def _create_documents_atomically(session):
            # Check quotas
            await self._check_document_quotas(session, user, collection_id, len(files))

            documents_created = []
            collection_config = json.loads(collection.config)
            index_types = self._get_index_types_for_collection(collection_config)

            for file_info in file_data:
                # Create document and upload file
                document_instance = await self._create_document_record(
                    session=session,
                    user=user,
                    collection_id=collection.id,
                    filename=file_info["filename"],
                    size=file_info["size"],
                    status=db_models.DocumentStatus.PENDING,
                    file_suffix=file_info["suffix"],
                    file_content=file_info["content"],
                )

                # Create indexes
                await document_index_manager.create_or_update_document_indexes(
                    document_id=document_instance.id, index_types=index_types, session=session
                )

                # Build response object
                doc_response = await self._build_document_response(document_instance)
                documents_created.append(doc_response)

            return documents_created

        response = await self.db_ops.execute_with_transaction(_create_documents_atomically)

        # Trigger index reconciliation after successful document creation
        _trigger_index_reconciliation()

        return DocumentList(items=response)

    async def list_documents(
        self,
        user: str,
        collection_id: str,
        page: int = 1,
        page_size: int = 10,
        sort_by: str = None,
        sort_order: str = "desc",
        search: str = None,
    ) -> PaginatedResponse[view_models.Document]:
        """List documents with pagination, sorting and search capabilities."""

        # Define sort field mapping
        sort_mapping = {
            "name": db_models.Document.name,
            "created": db_models.Document.gmt_created,
            "updated": db_models.Document.gmt_updated,
            "size": db_models.Document.size,
            "status": db_models.Document.status,
        }

        # Define search fields mapping
        search_fields = {"name": db_models.Document.name}

        async def _execute_paginated_query(session):
            from sqlalchemy import and_, desc, select

            # Step 1: Build base document query for pagination (without indexes)
            base_query = select(db_models.Document).where(
                and_(
                    db_models.Document.user == user,
                    db_models.Document.collection_id == collection_id,
                    db_models.Document.status != db_models.DocumentStatus.DELETED,
                    db_models.Document.status != db_models.DocumentStatus.UPLOADED,
                    db_models.Document.status != db_models.DocumentStatus.EXPIRED,
                )
            )

            # Apply search filter
            if search:
                search_term = f"%{search}%"
                base_query = base_query.where(db_models.Document.name.ilike(search_term))

            # Build query parameters for documents
            params = ListParams(
                pagination=PaginationParams(page=page, page_size=page_size),
                sort=SortParams(sort_by=sort_by, sort_order=sort_order) if sort_by else None,
                search=SearchParams(search=search, search_fields=["name"]) if search else None,
            )

            # Use pagination helper for documents
            documents, total = await PaginationHelper.paginate_query(
                query=base_query,
                session=session,
                params=params,
                sort_mapping=sort_mapping,
                search_fields=search_fields,
                default_sort=desc(db_models.Document.gmt_created),
            )

            # Step 2: Batch load index information for the paginated documents
            if documents:
                document_ids = [doc.id for doc in documents]

                # Query all indexes for the paginated documents in one go
                index_query = select(db_models.DocumentIndex).where(
                    db_models.DocumentIndex.document_id.in_(document_ids)
                )
                index_result = await session.execute(index_query)
                indexes_data = index_result.scalars().all()

                # Group indexes by document_id
                indexes_by_doc = {}
                for index in indexes_data:
                    if index.document_id not in indexes_by_doc:
                        indexes_by_doc[index.document_id] = {}
                    indexes_by_doc[index.document_id][index.index_type] = {
                        "index_type": index.index_type,
                        "status": index.status,
                        "created_at": index.gmt_created,
                        "updated_at": index.gmt_updated,
                        "error_message": index.error_message,
                        "index_data": index.index_data,
                    }

                # Attach index information to documents
                for doc in documents:
                    # Initialize index information for all types
                    doc.indexes = {"VECTOR": None, "FULLTEXT": None, "GRAPH": None, "SUMMARY": None, "VISION": None}

                    # Add actual index data if exists
                    if doc.id in indexes_by_doc:
                        doc.indexes.update(indexes_by_doc[doc.id])

            # Step 3: Build document responses
            document_responses = []
            for doc in documents:
                doc_response = await self._build_document_response(doc)
                document_responses.append(doc_response)

            return PaginationHelper.build_response(
                items=document_responses, total=total, page=page, page_size=page_size
            )

        return await self.db_ops._execute_query(_execute_paginated_query)

    async def get_document(self, user: str, collection_id: str, document_id: str) -> view_models.Document:
        """Get a specific document by ID."""
        documents = await self._query_documents_with_indexes(user, collection_id, document_id)

        if not documents:
            raise DocumentNotFoundException(f"Document not found: {document_id}")

        document = documents[0]
        return await self._build_document_response(document)

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
        async_obj_store = get_async_object_store()
        metadata = json.loads(document.doc_metadata) if document.doc_metadata else {}
        if metadata.get("object_path"):
            try:
                # Use delete_objects_by_prefix to remove all related files (original, chunks, etc.)
                await async_obj_store.delete_objects_by_prefix(document.object_store_base_path())
                logger.info(f"Deleted objects from object store with prefix: {document.object_store_base_path()}")
            except Exception as e:
                logger.warning(f"Failed to delete objects for document {document.id} from object store: {e}")

        # Mark document as deleted
        document.status = db_models.DocumentStatus.DELETED
        document.gmt_deleted = utc_now()
        session.add(document)

        # Release quota within the same transaction
        from aperag.service.quota_service import quota_service

        await quota_service.release_quota(user, "max_document_count", 1, session)

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
        self, user_id: str, collection_id: str, document_id: str, index_types: list[str]
    ) -> dict:
        """
        Rebuild specified indexes for a document
        Args:
            user_id: User ID
            collection_id: Collection ID
            document_id: Document ID
            index_types: List of index types to rebuild ('VECTOR', 'FULLTEXT', 'GRAPH', 'SUMMARY')
        Returns:
            dict: Success response
        """
        if len(set(index_types)) != len(index_types):
            raise invalid_param("index_types", "duplicate index types are not allowed")

        logger.info(f"Rebuilding indexes for document {document_id} with types: {index_types}")

        from aperag.db.models import DocumentIndexType

        index_type_enums = []
        for index_type in index_types:
            if index_type == "VECTOR":
                index_type_enums.append(DocumentIndexType.VECTOR)
            elif index_type == "FULLTEXT":
                index_type_enums.append(DocumentIndexType.FULLTEXT)
            elif index_type == "GRAPH":
                index_type_enums.append(DocumentIndexType.GRAPH)
            elif index_type == "SUMMARY":
                index_type_enums.append(DocumentIndexType.SUMMARY)
            elif index_type == "VISION":
                index_type_enums.append(DocumentIndexType.VISION)
            else:
                raise invalid_param("index_type", f"Invalid index type: {index_type}")

        async def _rebuild_document_indexes_atomically(session):
            document = await self.db_ops.query_document(user_id, collection_id, document_id)
            if not document:
                raise DocumentNotFoundException(f"Document {document_id} not found")
            if document.collection_id != collection_id:
                raise ResourceNotFoundException(f"Document {document_id} not found in collection {collection_id}")
            collection = await self.db_ops.query_collection(user_id, collection_id)
            if not collection or collection.user != user_id:
                raise ResourceNotFoundException(f"Collection {collection_id} not found or access denied")
            collection_config = json.loads(collection.config)
            if not collection_config.get("enable_knowledge_graph", False):
                if db_models.DocumentIndexType.GRAPH in index_type_enums:
                    index_type_enums.remove(db_models.DocumentIndexType.GRAPH)
            # 支持 SUMMARY 类型的重建
            await document_index_manager.create_or_update_document_indexes(session, document_id, index_type_enums)
            logger.info(f"Successfully triggered rebuild for document {document_id} indexes: {index_types}")
            return {"code": "200", "message": f"Index rebuild initiated for types: {', '.join(index_types)}"}

        result = await self.db_ops.execute_with_transaction(_rebuild_document_indexes_atomically)
        _trigger_index_reconciliation()
        return result

    async def get_document_chunks(self, user_id: str, collection_id: str, document_id: str) -> List[Chunk]:
        """
        Get all chunks of a document.
        """

        # Use database operations with proper session management
        async def _get_document_chunks(session):
            # 1. Get the chunk IDs (ctx_ids) from the document_index table
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

            # 2. Retrieve chunks from Qdrant
            try:
                collection_name = generate_vector_db_collection_name(collection_id=collection_id)
                ctx = json.loads(settings.vector_db_context)
                ctx["collection"] = collection_name
                vector_store_adaptor = VectorStoreConnectorAdaptor(settings.vector_db_type, ctx=ctx)
                qdrant_client = vector_store_adaptor.connector.client

                points = qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=ctx_ids,
                    with_payload=True,
                )

                # 3. Format the response
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

        # Execute query with proper session management
        return await self.db_ops._execute_query(_get_document_chunks)

    async def get_document_vision_chunks(self, user_id: str, collection_id: str, document_id: str) -> List[VisionChunk]:
        """
        Get all vision chunks of a document.
        """

        async def _get_document_vision_chunks(session):
            # 1. Get the chunk IDs (ctx_ids) from the document_index table
            stmt = select(db_models.DocumentIndex).filter(
                db_models.DocumentIndex.document_id == document_id,
                db_models.DocumentIndex.index_type == db_models.DocumentIndexType.VISION,
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

            # 2. Retrieve chunks from Qdrant
            try:
                collection_name = generate_vector_db_collection_name(collection_id=collection_id)
                ctx = json.loads(settings.vector_db_context)
                ctx["collection"] = collection_name
                vector_store_adaptor = VectorStoreConnectorAdaptor(settings.vector_db_type, ctx=ctx)
                qdrant_client = vector_store_adaptor.connector.client

                points = qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=ctx_ids,
                    with_payload=True,
                )

                # 3. Format the response
                vision_chunks = []
                for point in points:
                    if point.payload:
                        # In llama-index-0.10.13, the payload is stored in _node_content
                        node_content = point.payload.get("_node_content")
                        if node_content and isinstance(node_content, str):
                            try:
                                payload_data = json.loads(node_content)
                                metadata = payload_data.get("metadata", {})
                                if metadata.get("index_method") == "vision_to_text":
                                    vision_chunks.append(
                                        VisionChunk(
                                            id=point.id,
                                            asset_id=metadata.get("asset_id"),
                                            text=payload_data.get("text", ""),
                                            metadata=metadata,
                                        )
                                    )
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse _node_content for point {point.id}")
                        else:
                            # Fallback for older or different data structures
                            metadata = point.payload.get("metadata", {})
                            if metadata.get("index_method") == "vision_to_text":
                                vision_chunks.append(
                                    VisionChunk(
                                        id=point.id,
                                        asset_id=metadata.get("asset_id"),
                                        text=point.payload.get("text", ""),
                                        metadata=metadata,
                                    )
                                )
                return vision_chunks
            except Exception as e:
                logger.error(
                    f"Failed to retrieve vision chunks from vector store for document {document_id}: {e}", exc_info=True
                )
                raise HTTPException(status_code=500, detail="Failed to retrieve vision chunks from vector store")

        return await self.db_ops._execute_query(_get_document_vision_chunks)

    async def get_document_preview(self, user_id: str, collection_id: str, document_id: str) -> DocumentPreview:
        """
        Get all preview-related information for a document.
        """

        # Use database operations with proper session management
        async def _get_document_preview(session: AsyncSession):
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

            # 2. Get chunks
            chunks = await self.get_document_chunks(user_id, collection_id, document_id)
            vision_chunks = await self.get_document_vision_chunks(user_id, collection_id, document_id)

            # 3. Get markdown content
            async_obj_store = get_async_object_store()
            markdown_content = ""
            # The parsed markdown file is stored with the name "parsed.md"
            markdown_path = f"{document.object_store_base_path()}/parsed.md"
            try:
                md_obj_result = await async_obj_store.get(markdown_path)
                if md_obj_result:
                    md_stream, _ = md_obj_result
                    content = b""
                    async for data in md_stream:
                        content += data
                    markdown_content = content.decode("utf-8")
            except Exception:
                logger.warning(f"Could not find or read markdown file at {markdown_path}")

            # 4. Determine paths
            doc_metadata = json.loads(document.doc_metadata) if document.doc_metadata else {}
            doc_object_path = doc_metadata.get("object_path")
            if doc_object_path:
                doc_object_path = os.path.basename(doc_object_path)

            # Return the converted PDF if it's available.
            converted_pdf_object_path = None
            converted_pdf_name = "converted.pdf"
            pdf_path = f"{document.object_store_base_path()}/{converted_pdf_name}"
            exists = await async_obj_store.obj_exists(pdf_path)
            if exists:
                converted_pdf_object_path = converted_pdf_name

            # 5. Construct and return response
            return DocumentPreview(
                doc_object_path=doc_object_path,
                doc_filename=document.name,
                converted_pdf_object_path=converted_pdf_object_path,
                markdown_content=markdown_content,
                chunks=chunks,
                vision_chunks=vision_chunks,
            )

        # Execute query with proper session management
        return await self.db_ops._execute_query(_get_document_preview)

    async def get_document_object(
        self, user_id: str, collection_id: str, document_id: str, path: str, range_header: str = None
    ):
        """
        Get a file object associated with a document from the object store.
        Supports HTTP Range requests.
        """

        # Use database operations with proper session management
        async def _get_document_object(session):
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
                async_obj_store = get_async_object_store()
                headers = {"Accept-Ranges": "bytes"}
                content_type, _ = mimetypes.guess_type(full_path)
                if content_type is None:
                    content_type = "application/octet-stream"
                headers["Content-Type"] = content_type

                if range_header:
                    # For range requests, we need the total size first.
                    total_size = await async_obj_store.get_obj_size(full_path)
                    if total_size is None:
                        raise HTTPException(status_code=404, detail="Object not found at specified path")

                    range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
                    if not range_match:
                        raise HTTPException(status_code=400, detail="Invalid range header format")

                    start_byte = int(range_match.group(1))
                    end_byte_str = range_match.group(2)
                    end_byte = int(end_byte_str) if end_byte_str else total_size - 1

                    if start_byte >= total_size or end_byte >= total_size or start_byte > end_byte:
                        headers["Content-Range"] = f"bytes */{total_size}"
                        raise HTTPException(status_code=416, headers=headers, detail="Requested range not satisfiable")

                    # Use stream_range to get the partial content
                    range_result = await async_obj_store.stream_range(full_path, start=start_byte, end=end_byte)
                    if not range_result:
                        raise HTTPException(status_code=404, detail="Object not found at specified path")

                    data_stream, content_length = range_result
                    headers["Content-Range"] = f"bytes {start_byte}-{end_byte}/{total_size}"
                    headers["Content-Length"] = str(content_length)
                    return StreamingResponse(data_stream, status_code=206, headers=headers)

                # Full content response - optimized to use size from get()
                get_obj_result = await async_obj_store.get(full_path)
                if not get_obj_result:
                    raise HTTPException(status_code=404, detail="Object not found at specified path")

                data_stream, file_size = get_obj_result
                headers["Content-Length"] = str(file_size)
                return StreamingResponse(data_stream, headers=headers)

            except Exception as e:
                logger.error(f"Failed to get object for document {document_id} at path {full_path}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to get object from store")

        # Execute query with proper session management
        return await self.db_ops._execute_query(_get_document_object)

    async def upload_document(
        self, user_id: str, collection_id: str, file: UploadFile
    ) -> view_models.UploadDocumentResponse:
        """Upload a single document file to temporary storage"""
        # Validate collection
        collection = await self._validate_collection(user_id, collection_id)

        # Validate file
        file_suffix = self._validate_file(file.filename, file.size)

        # Read file content
        file_content = await file.read()
        await file.seek(0)

        async def _upload_document_atomically(session):
            # Create document with UPLOADED status (temporary)
            document_instance = await self._create_document_record(
                session=session,
                user=user_id,
                collection_id=collection.id,
                filename=file.filename,
                size=file.size,
                status=db_models.DocumentStatus.UPLOADED,  # Temporary status
                file_suffix=file_suffix,
                file_content=file_content,
            )

            return view_models.UploadDocumentResponse(
                document_id=document_instance.id, filename=file.filename, size=file.size, status="UPLOADED"
            )

        return await self.db_ops.execute_with_transaction(_upload_document_atomically)

    async def confirm_documents(
        self, user_id: str, collection_id: str, document_ids: list[str]
    ) -> view_models.ConfirmDocumentsResponse:
        """Confirm uploaded documents and add them to the collection"""
        confirmed_count = 0
        failed_count = 0
        failed_documents = []

        async def _confirm_documents_atomically(session):
            nonlocal confirmed_count, failed_count, failed_documents

            # Check quotas
            await self._check_document_quotas(session, user_id, collection_id, len(document_ids))

            # Get collection config
            collection = await self.db_ops.query_collection(user_id, collection_id)
            collection_config = json.loads(collection.config)
            index_types = self._get_index_types_for_collection(collection_config)

            for document_id in document_ids:
                try:
                    # Get document (single query without status filter)
                    stmt = select(db_models.Document).where(
                        db_models.Document.id == document_id,
                        db_models.Document.user == user_id,
                        db_models.Document.collection_id == collection_id,
                    )
                    result = await session.execute(stmt)
                    document = result.scalars().first()

                    if not document:
                        # Document not found at all
                        failed_documents.append(
                            view_models.FailedDocument(document_id=document_id, name=None, error="DOCUMENT_NOT_FOUND")
                        )
                        failed_count += 1
                        continue

                    # Check document status
                    if document.status != db_models.DocumentStatus.UPLOADED:
                        # Document exists but not in correct status
                        if document.status == db_models.DocumentStatus.EXPIRED:
                            error_code = "DOCUMENT_EXPIRED"
                        else:
                            error_code = "DOCUMENT_NOT_UPLOADED"

                        failed_documents.append(
                            view_models.FailedDocument(document_id=document_id, name=document.name, error=error_code)
                        )
                        failed_count += 1
                        continue

                    # Change status to PENDING
                    document.status = db_models.DocumentStatus.PENDING
                    session.add(document)

                    # Create indexes
                    await document_index_manager.create_or_update_document_indexes(
                        document_id=document.id, index_types=index_types, session=session
                    )

                    confirmed_count += 1

                except Exception as e:
                    logger.error(f"Failed to confirm document {document_id}: {e}")
                    # Try to get document name for better error reporting
                    document_name = None
                    try:
                        stmt_name = select(db_models.Document.name).where(db_models.Document.id == document_id)
                        result_name = await session.execute(stmt_name)
                        document_name = result_name.scalar()
                    except Exception:
                        pass

                    failed_documents.append(
                        view_models.FailedDocument(
                            document_id=document_id, name=document_name, error="CONFIRMATION_FAILED"
                        )
                    )
                    failed_count += 1

        await self.db_ops.execute_with_transaction(_confirm_documents_atomically)

        # Trigger index reconciliation
        _trigger_index_reconciliation()

        return view_models.ConfirmDocumentsResponse(
            confirmed_count=confirmed_count, failed_count=failed_count, failed_documents=failed_documents
        )


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
document_service = DocumentService()
