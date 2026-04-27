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

"""Knowledge-base ``DocumentService`` (Phase 3 Step 5b3).

The service body moved from ``aperag/service/document_service.py`` with
the same G1 boundary pass Step 5b2b applied to ``collection_service``:

* ``aperag.db.models`` / ``aperag.schema.view_models`` /
  ``aperag.service.*`` imports rewritten to domain paths.
* ``marketplace_service`` / ``quota_service`` cross-service dependencies
  reached through the consumer-owned Protocol DI setters owned by the
  sibling ``collection_service`` module (both KB services share the
  same wire-up slot, so the app.py startup continues to wire one
  instance and both services see it).
* ``UploadDocumentResponse`` /
  ``ConfirmDocumentsResponse`` / ``FailedDocument`` /
  ``FetchUrlResultItem`` / ``FetchUrlResponse`` /
  ``StagedDocumentsResponse`` envelopes were carved out to
  ``aperag.domains.knowledge_base.schemas`` in the same commit (they
  belong to the KB document public surface); pre-migration callers
  continue to resolve via the ``view_models`` dual-hook re-export shim.
"""

import asyncio
import json
import logging
import mimetypes
import os
import re
from typing import List

from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.config import settings
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.docparser.doc_parser import DocParser
from aperag.domains.knowledge_base.db.models import (
    Collection,
    CollectionStatus,
    Document,
    DocumentStatus,
    _random_id,
)
from aperag.domains.knowledge_base.schemas import (
    ConfirmDocumentsResponse,
    DocumentList,
    DocumentPreview,
    FailedDocument,
    FetchUrlResponse,
    FetchUrlResultItem,
    StagedDocumentsResponse,
    UploadDocumentResponse,
)
from aperag.domains.knowledge_base.schemas import (
    Document as DocumentSchema,
)
from aperag.domains.knowledge_base.service.collection_service import (
    _get_marketplace_ops,
    _get_quota_ops,
)
from aperag.exceptions import (
    CollectionInactiveException,
    DocumentNameConflictException,
    DocumentNotFoundException,
    QuotaExceededException,
    ResourceNotFoundException,
    invalid_param,
)
from aperag.indexing.models import (
    DocumentIndex,
    Modality,
)
from aperag.objectstore.base import get_async_object_store
from aperag.schema.common import Chunk, VisionChunk
from aperag.schema.utils import parseCollectionConfig
from aperag.utils.pagination import (
    ListParams,
    PaginatedResponse,
    PaginationHelper,
    PaginationParams,
    SearchParams,
    SortParams,
)
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS
from aperag.utils.utils import calculate_file_hash, generate_vector_db_collection_name, utc_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# New-API wrappers — celery T3.1 chunk 3 (replace legacy
# ``document_index_manager.{create_or_update,delete}_document_indexes``).
# ---------------------------------------------------------------------
#
# The legacy ABC was hard-deleted in chunk 2; these two helpers are the
# minimum-blast-radius adapters that keep the existing 5 call sites
# compiling while routing to the new ``aperag.indexing`` surface
# (``dispatch_indexing()`` for INSERT, ``cleanup_for_deleted_documents()``
# for DELETE). Both consume the process-local
# :class:`aperag.indexing.runtime.IndexingRuntime` populated by the
# FastAPI lifespan; if the runtime is absent (test environment, or
# ``INDEXING_MODE != async``), they log + no-op rather than crash.


async def _create_or_update_document_indexes(
    *,
    document_id: str,
    index_types: list[Modality],
    session: AsyncSession,
) -> None:
    """Replacement for legacy ``document_index_manager.
    create_or_update_document_indexes``.

    Wave 3 T3.1 chunk 3: dispatches via the new
    :func:`aperag.indexing.dispatcher.dispatch_indexing` ASYNC mode.
    The ``parse_version`` is computed deterministically from the
    document content hash + canonical chunking config so the worker's
    re-derive path lands on the same value (per §E.2 hash). The
    ``source_path`` points at the document's object-store base path;
    the worker derives the per-modality artifact (chunks.jsonl /
    markdown.md / vision/manifest.jsonl) underneath.
    """
    if not index_types:
        return

    from aperag.indexing import DispatchRequest, IndexingMode, dispatch_indexing
    from aperag.indexing.parser import DEFAULT_PARSER_PIPELINE, ChunkingConfig
    from aperag.indexing.runtime import get_runtime
    from aperag.mcp.tools.parse_version import compute_parse_version

    runtime = get_runtime()
    if runtime is None:
        logger.warning(
            "_create_or_update_document_indexes(document=%s): IndexingRuntime not installed "
            "(INDEXING_MODE != async or pre-startup); skipping dispatch",
            document_id,
        )
        return

    document = await session.get(Document, document_id)
    if document is None:
        logger.warning(
            "_create_or_update_document_indexes(document=%s): Document row not found; skipping",
            document_id,
        )
        return

    parse_version = compute_parse_version(
        parser_pipeline=DEFAULT_PARSER_PIPELINE,
        document_md5=document.content_hash or "",
        chunking_config=ChunkingConfig().serialize(),
    )
    source_path = document.object_store_base_path()
    tenant_scope_key = f"user:{document.user}"

    # Wave 3 T3.1 chunk 3 fix-forward: ``rebuild_indexes`` re-invokes
    # this adapter with the same ``(document_id, parse_version,
    # modality)`` triple that already exists (content unchanged →
    # parse_version unchanged). The §F.1 ``uq_document_index_triple``
    # UNIQUE constraint then fails the dispatcher's INSERT with an
    # IntegrityError → 500 DATABASE_ERROR. Pre-DELETE matching rows
    # (any status / serving state) so the INSERT lands cleanly. The
    # cutover-on-sync-completion (§F.3) re-establishes the serving
    # state once the new dispatch's worker finishes; brief
    # unavailability between DELETE and cutover is acceptable for an
    # explicit rebuild op.
    from sqlalchemy import delete as sa_delete

    from aperag.indexing.models import DocumentIndex

    def _purge_existing_triples() -> None:
        from sqlalchemy.orm import Session

        with Session(runtime.engine) as sync_session, sync_session.begin():
            sync_session.execute(
                sa_delete(DocumentIndex).where(
                    DocumentIndex.document_id == document.id,
                    DocumentIndex.parse_version == parse_version,
                    DocumentIndex.modality.in_([m.value for m in index_types]),
                )
            )

    await asyncio.to_thread(_purge_existing_triples)

    await dispatch_indexing(
        engine=runtime.engine,
        queue=runtime.queue,
        workers=runtime.workers,
        request=DispatchRequest(
            collection_id=document.collection_id,
            document_id=document.id,
            parse_version=parse_version,
            source_path=source_path,
            tenant_scope_key=tenant_scope_key,
            modalities=tuple(index_types),
        ),
        mode=IndexingMode.ASYNC,
    )


async def _delete_document_indexes(*, document_id: str) -> None:
    """Replacement for legacy ``document_index_manager.
    delete_document_indexes``.

    Wave 3 T3.1 chunk 3: routes to
    :func:`aperag.indexing.cleanup.cleanup_for_deleted_documents` which
    handles the modality fan-out (graph lineage cleanup vs flat
    backend delete) + DELETEs the ``document_index`` rows.
    """
    from aperag.indexing.cleanup import cleanup_for_deleted_documents
    from aperag.indexing.runtime import get_runtime

    runtime = get_runtime()
    if runtime is None:
        logger.warning(
            "_delete_document_indexes(document=%s): IndexingRuntime not installed; skipping cleanup",
            document_id,
        )
        return

    await cleanup_for_deleted_documents(
        engine=runtime.engine,
        workers=runtime.workers,
        document_ids=[document_id],
    )


def _trigger_index_reconciliation():
    """No-op — Wave 3 T3.1 chunk 3.

    The legacy Celery beat-driven ``reconcile_indexes_task`` is gone;
    the new ``aperag.indexing.reconciler.run_reconcile_loop`` runs
    continuously inside the FastAPI process so manual triggering is
    unnecessary. Kept as a no-op shim so the existing call sites
    compile; the periodic 30-s loop picks up any newly-PENDING rows
    immediately.
    """
    return None


class DocumentService:
    """Document service that handles business logic for documents"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def _validate_collection(self, user: str, collection_id: str) -> Collection:
        """
        Validate that collection exists and is active.
        Returns the collection if valid, raises exception otherwise.
        """
        collection = await self.db_ops.query_collection(user, collection_id)
        if collection is None:
            raise ResourceNotFoundException("Collection", collection_id)
        if collection.status != CollectionStatus.ACTIVE:
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

    async def _check_duplicate_document(
        self, session: AsyncSession, user: str, collection_id: str, filename: str, file_hash: str
    ) -> Document | None:
        """
        Check if a document with the same name exists in the collection within the same transaction.
        Returns the existing document if found, None otherwise.

        Raises DocumentNameConflictException if same name but different file hash.

        Args:
            session: Database session for transaction isolation
            user: User ID
            collection_id: Collection ID
            filename: Document filename
            file_hash: File content hash for duplicate detection
        """
        # Query within the same transaction for proper isolation
        stmt = select(Document).where(
            Document.user == user,
            Document.collection_id == collection_id,
            Document.name == filename,
            Document.status != DocumentStatus.DELETED,
            Document.gmt_deleted.is_(None),  # Not soft deleted
        )
        result = await session.execute(stmt)
        existing_doc = result.scalars().first()

        if existing_doc:
            # If existing document has no hash (legacy document), skip hash check
            if existing_doc.content_hash is None:
                # Could calculate hash for legacy document here if needed
                logger.warning(f"Existing document {existing_doc.id} has no file hash, skipping hash comparison")
                return existing_doc

            # If file hashes match, it's a true duplicate (same file)
            if existing_doc.content_hash == file_hash:
                return existing_doc
            else:
                # Same name but different file content - conflict
                raise DocumentNameConflictException(filename, collection_id)

        return None

    async def _check_document_quotas(self, session: AsyncSession, user: str, collection_id: str, count: int):
        """
        Check and consume document quotas.
        Raises QuotaExceededException if quota would be exceeded.
        """
        from sqlalchemy import func, select

        # Check and consume user quota
        await _get_quota_ops().check_and_consume_quota(user, "max_document_count", count, session)

        # Check per-collection quota
        stmt = (
            select(func.count())
            .select_from(Document)
            .where(
                Document.collection_id == collection_id,
                Document.status != DocumentStatus.DELETED,
                Document.status != DocumentStatus.UPLOADED,  # Don't count temporary uploads
            )
        )
        existing_doc_count = await session.scalar(stmt)

        # Per-collection quota limit is an admin-controlled cap that
        # rarely mutates; read it through the QuotaOps Protocol (a
        # separate read session under the covers) rather than the
        # banned raw ``UserQuota`` ORM query. The ``COUNT(*)`` above
        # stays in the current transaction, which is the part of the
        # enforcement that needs write-isolation.
        all_quotas = await _get_quota_ops().get_user_quotas(user)
        per_collection_quota = all_quotas.get("max_document_count_per_collection")
        if per_collection_quota is not None:
            per_collection_limit = per_collection_quota.get("quota_limit")
            if per_collection_limit is not None and (existing_doc_count + count) > per_collection_limit:
                raise QuotaExceededException(
                    "max_document_count_per_collection", per_collection_limit, existing_doc_count
                )

    def _get_index_types_for_collection(self, collection_config: dict) -> list:
        """
        Get the list of :class:`Modality` values to create based on
        collection configuration. Wave 3 migrated the legacy
        ``DocumentIndexType`` enum to :class:`Modality`; the per-
        collection enable flags map 1-to-1 to modalities.

        ``enable_vector`` was historically implicit (vector was
        always created) but a collection without an embedding-model
        config cannot satisfy the Wave 3 production worker factory
        (factory raises :class:`WorkerFactoryError` → row finalises
        FAILED → reconciler retries forever). Honouring the
        ``enable_vector`` flag here turns "vector explicitly
        disabled" into a no-row state — the modality simply does not
        appear in the document_index table for this document.
        """
        parsed_config = parseCollectionConfig(json.dumps(collection_config))
        index_types: list = []

        if parsed_config.enable_vector is not False:
            index_types.append(Modality.VECTOR)
        if parsed_config.enable_fulltext is not False:
            index_types.append(Modality.FULLTEXT)

        if collection_config.get("enable_knowledge_graph", False):
            index_types.append(Modality.GRAPH)
        if collection_config.get("enable_summary", False):
            index_types.append(Modality.SUMMARY)
        if collection_config.get("enable_vision", False):
            index_types.append(Modality.VISION)

        return index_types

    async def _create_document_record(
        self,
        session: AsyncSession,
        user: str,
        collection_id: str,
        filename: str,
        size: int,
        status: DocumentStatus,
        file_suffix: str,
        file_content: bytes,
        custom_metadata: dict = None,
        content_hash: str = None,
    ) -> Document:
        """
        Create a document record in database and upload file to object store.
        Returns the created document instance.
        """
        # Calculate file hash if not provided
        if content_hash is None:
            content_hash = calculate_file_hash(file_content)

        # Create document in database
        document_instance = Document(
            user=user,
            name=filename,
            status=status,
            size=size,
            collection_id=collection_id,
            content_hash=content_hash,
        )
        session.add(document_instance)
        await session.flush()
        await session.refresh(document_instance)

        # Upload to object store
        async_obj_store = get_async_object_store()
        upload_path = f"{document_instance.object_store_base_path()}/original{file_suffix}"
        await async_obj_store.put(upload_path, file_content)

        # Update document with object path and custom metadata
        metadata = {"object_path": upload_path}
        if custom_metadata:
            metadata.update(custom_metadata)
        document_instance.doc_metadata = json.dumps(metadata)
        session.add(document_instance)
        await session.flush()
        await session.refresh(document_instance)

        return document_instance

    async def _query_documents_with_indexes(
        self, user: str, collection_id: str, document_id: str = None
    ) -> List[Document]:
        """
        Common function to query documents with their indexes using JOIN.
        If document_id is provided, query single document, otherwise query all documents.
        """

        async def _execute_query(session):
            from sqlalchemy import and_, outerjoin, select

            # Create JOIN query between Document and DocumentIndex tables
            # Use outerjoin to get all documents even if they don't have indexes.
            # Wave 3 §F.1 migration: ``modality`` column replaces
            # legacy ``index_type``; ``created_at``/``updated_at``
            # replace ``gmt_created``/``gmt_updated``; ``index_data``
            # JSON blob is gone (decomposed into per-modality
            # ``derived/`` artifacts on the object store), so the
            # surface returned to callers carries ``index_data=None``
            # for backward-compat with existing response shapes.
            query = (
                select(
                    Document,
                    DocumentIndex.modality.label("index_type"),
                    DocumentIndex.status.label("index_status"),
                    DocumentIndex.created_at.label("index_created_at"),
                    DocumentIndex.updated_at.label("index_updated_at"),
                    DocumentIndex.error_message.label("index_error_message"),
                )
                .select_from(
                    outerjoin(
                        Document,
                        DocumentIndex,
                        Document.id == DocumentIndex.document_id,
                    )
                )
                .where(
                    and_(
                        Document.user == user,
                        Document.collection_id == collection_id,
                        Document.status != DocumentStatus.DELETED,
                        Document.status != DocumentStatus.UPLOADED,  # Filter out temporary uploaded documents
                        Document.status != DocumentStatus.EXPIRED,  # Filter out temporary uploaded documents
                    )
                )
                .order_by(Document.gmt_created.desc())
            )

            # Add document_id filter if provided (for single document query)
            if document_id:
                query = query.where(Document.id == document_id)

            result = await session.execute(query)
            rows = result.fetchall()

            # Group results by document and attach all index information.
            # The new ``modality`` column carries lowercase strings
            # (``"vector"`` / ``"fulltext"`` / ``"graph"`` /
            # ``"summary"`` / ``"vision"``); the response shape uses
            # uppercase keys for backward-compat with HTTP clients,
            # so we translate via the :class:`Modality` enum.
            documents_dict = {}
            for row in rows:
                doc = row.Document
                if doc.id not in documents_dict:
                    documents_dict[doc.id] = doc
                    doc.indexes = {"VECTOR": None, "FULLTEXT": None, "GRAPH": None, "SUMMARY": None, "VISION": None}

                modality_value = row.index_type
                if modality_value:
                    response_key = modality_value.upper()
                    doc.indexes[response_key] = {
                        "index_type": response_key,
                        "status": row.index_status,
                        "created_at": row.index_created_at,
                        "updated_at": row.index_updated_at,
                        "error_message": row.index_error_message,
                        "index_data": None,
                    }

            return list(documents_dict.values())

        return await self.db_ops._execute_query(_execute_query)

    async def _build_document_response(self, document: Document) -> DocumentSchema:
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

        return DocumentSchema(
            id=document.id,
            name=document.name,
            status=document.status,
            # Per-modality status: ``None`` when the row does not exist
            # (modality not enabled for this collection — the dispatcher
            # never created a document_index row). Wave 3 §F.2 hard-cut
            # dropped the "SKIPPED" sentinel; absence is the canonical
            # NOT_ENABLED signal for the view-model layer. Friendly
            # client-facing mapping (NOT_ENABLED / INDEXING) lives in
            # §G.5 ``SearchResultMetadata.index_state_per_modality``.
            vector_index_status=indexes["VECTOR"]["status"] if indexes["VECTOR"] else None,
            vector_index_updated=indexes["VECTOR"]["updated_at"] if indexes["VECTOR"] else None,
            fulltext_index_status=indexes["FULLTEXT"]["status"] if indexes["FULLTEXT"] else None,
            fulltext_index_updated=indexes["FULLTEXT"]["updated_at"] if indexes["FULLTEXT"] else None,
            graph_index_status=indexes["GRAPH"]["status"] if indexes["GRAPH"] else None,
            graph_index_updated=indexes["GRAPH"]["updated_at"] if indexes["GRAPH"] else None,
            summary_index_status=indexes["SUMMARY"]["status"] if indexes.get("SUMMARY") else None,
            summary_index_updated=indexes["SUMMARY"]["updated_at"] if indexes.get("SUMMARY") else None,
            vision_index_status=indexes["VISION"]["status"] if indexes.get("VISION") else None,
            vision_index_updated=indexes["VISION"]["updated_at"] if indexes.get("VISION") else None,
            summary=summary,  # Parse from index_data
            size=document.size,
            created=document.gmt_created,
            updated=document.gmt_updated,
        )

    async def create_documents(
        self,
        user: str,
        collection_id: str,
        files: List[UploadFile],
        custom_metadata: dict = None,
        ignore_duplicate: bool = False,
    ) -> DocumentList:
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

            # Calculate original file hash for duplicate detection
            file_hash = calculate_file_hash(file_content)

            file_data.append(
                {
                    "filename": item.filename,
                    "size": item.size,
                    "suffix": file_suffix,
                    "content": file_content,
                    "file_hash": file_hash,
                }
            )

        # Process all files in a single transaction for atomicity
        async def _create_documents_atomically(session):
            # Check quotas
            await self._check_document_quotas(session, user, collection_id, len(files))

            documents_created = []
            collection_config = json.loads(collection.config)
            index_types = self._get_index_types_for_collection(collection_config)

            for file_info in file_data:
                # Check for duplicate document (same name and hash) within transaction
                existing_doc = await self._check_duplicate_document(
                    session, user, collection.id, file_info["filename"], file_info["file_hash"]
                )

                if existing_doc and not ignore_duplicate:
                    # Return existing document info (idempotent behavior)
                    logger.info(
                        f"Document '{file_info['filename']}' already exists with same content, returning existing document {existing_doc.id}"
                    )
                    doc_response = await self._build_document_response(existing_doc)
                    documents_created.append(doc_response)
                    continue

                # Create new document and upload file
                document_instance = await self._create_document_record(
                    session=session,
                    user=user,
                    collection_id=collection.id,
                    filename=file_info["filename"],
                    size=file_info["size"],
                    status=DocumentStatus.PENDING,
                    file_suffix=file_info["suffix"],
                    file_content=file_info["content"],
                    custom_metadata=custom_metadata,
                    content_hash=file_info["file_hash"],
                )

                # Create indexes (Wave 3 T3.1 chunk 3: dispatch via new
                # ``aperag.indexing.dispatcher.dispatch_indexing``).
                await _create_or_update_document_indexes(
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
    ) -> PaginatedResponse[DocumentSchema]:
        """List documents with pagination, sorting and search capabilities."""

        if not user:
            await _get_marketplace_ops().validate_marketplace_collection(collection_id)

        # Define sort field mapping
        sort_mapping = {
            "name": Document.name,
            "created": Document.gmt_created,
            "updated": Document.gmt_updated,
            "size": Document.size,
            "status": Document.status,
        }

        # Define search fields mapping
        search_fields = {"name": Document.name}

        async def _execute_paginated_query(session):
            from sqlalchemy import and_, desc, select

            # Step 1: Build base document query for pagination (without indexes)
            base_query = select(Document).where(
                and_(
                    Document.user == user,
                    Document.collection_id == collection_id,
                    Document.status != DocumentStatus.DELETED,
                    Document.status != DocumentStatus.UPLOADED,
                    Document.status != DocumentStatus.EXPIRED,
                )
            )

            # Apply search filter
            if search:
                search_term = f"%{search}%"
                base_query = base_query.where(Document.name.ilike(search_term))

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
                default_sort=desc(Document.gmt_created),
            )

            # Step 2: Batch load index information for the paginated documents
            if documents:
                document_ids = [doc.id for doc in documents]

                # Query all indexes for the paginated documents in one go
                index_query = select(DocumentIndex).where(DocumentIndex.document_id.in_(document_ids))
                index_result = await session.execute(index_query)
                indexes_data = index_result.scalars().all()

                # Group indexes by document_id. Wave 3 §F.1 schema
                # uses the lowercase ``modality`` column for the
                # discriminator + drops ``index_data``; HTTP response
                # keeps uppercase keys for backward compat so the
                # paginated index map retains its existing shape.
                indexes_by_doc: dict[str, dict[str, dict]] = {}
                for index in indexes_data:
                    if index.document_id not in indexes_by_doc:
                        indexes_by_doc[index.document_id] = {}
                    response_key = index.modality.upper()
                    indexes_by_doc[index.document_id][response_key] = {
                        "index_type": response_key,
                        "status": index.status,
                        "created_at": index.created_at,
                        "updated_at": index.updated_at,
                        "error_message": index.error_message,
                        "index_data": None,
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

    async def get_document(self, user: str, collection_id: str, document_id: str) -> DocumentSchema:
        """Get a specific document by ID."""
        if not user:
            await _get_marketplace_ops().validate_marketplace_collection(collection_id)

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

        # Cleanup all per-modality index rows + backend state (Wave 3
        # T3.1 chunk 3: routes to ``aperag.indexing.cleanup.
        # cleanup_for_deleted_documents`` which handles the modality
        # fan-out + DELETEs the ``document_index`` rows).
        await _delete_document_indexes(document_id=document.id)

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
        document.status = DocumentStatus.DELETED
        document.gmt_deleted = utc_now()
        session.add(document)

        # Release quota within the same transaction
        await _get_quota_ops().release_quota(user, "max_document_count", 1, session)

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

        index_type_enums: list[Modality] = []
        for index_type in index_types:
            if index_type == "VECTOR":
                index_type_enums.append(Modality.VECTOR)
            elif index_type == "FULLTEXT":
                index_type_enums.append(Modality.FULLTEXT)
            elif index_type == "GRAPH":
                index_type_enums.append(Modality.GRAPH)
            elif index_type == "SUMMARY":
                index_type_enums.append(Modality.SUMMARY)
            elif index_type == "VISION":
                index_type_enums.append(Modality.VISION)
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
            if not collection_config.get("enable_knowledge_graph", False) and Modality.GRAPH in index_type_enums:
                index_type_enums.remove(Modality.GRAPH)
            # Trigger rebuild for the requested modalities (Wave 3 T3.1
            # chunk 3: dispatch via the new dispatcher).
            await _create_or_update_document_indexes(
                document_id=document_id, index_types=index_type_enums, session=session
            )
            logger.info(f"Successfully triggered rebuild for document {document_id} indexes: {index_types}")
            return {"code": "200", "message": f"Index rebuild initiated for types: {', '.join(index_types)}"}

        result = await self.db_ops.execute_with_transaction(_rebuild_document_indexes_atomically)
        _trigger_index_reconciliation()
        return result

    async def rebuild_failed_indexes(self, user_id: str, collection_id: str) -> dict:
        """
        Rebuild all failed indexes for all documents in a collection
        Args:
            user_id: User ID
            collection_id: Collection ID
        Returns:
            dict: Success response with affected documents count
        """
        logger.info(f"Rebuilding failed indexes for collection {collection_id}")

        async def _rebuild_failed_indexes_atomically(session):
            # First verify collection access
            collection = await self.db_ops.query_collection(user_id, collection_id)
            if not collection or collection.user != user_id:
                raise ResourceNotFoundException(f"Collection {collection_id} not found or access denied")

            # Get collection config to check graph indexing
            collection_config = json.loads(collection.config)
            enable_knowledge_graph = collection_config.get("enable_knowledge_graph", False)

            # Query documents with failed indexes (no type filter)
            failed_docs = await self.db_ops.query_documents_with_failed_indexes(user_id, collection_id, None)

            if not failed_docs:
                return {"code": "200", "message": "No failed indexes found to rebuild", "affected_documents": 0}

            # Process each document with failed indexes
            affected_documents = 0
            for document_id, failed_index_types in failed_docs:
                # Filter out GRAPH type if not enabled in collection config
                rebuild_types = failed_index_types
                if not enable_knowledge_graph:
                    rebuild_types = [t for t in failed_index_types if t != Modality.GRAPH.value]

                if rebuild_types:
                    # Wave 3 T3.1 chunk 3: dispatch failed-rebuild via
                    # the new dispatcher. ``rebuild_types`` originates as
                    # raw enum-string values; coerce to ``Modality``.
                    rebuild_modalities = [rt if isinstance(rt, Modality) else Modality(rt) for rt in rebuild_types]
                    await _create_or_update_document_indexes(
                        document_id=document_id,
                        index_types=rebuild_modalities,
                        session=session,
                    )
                    affected_documents += 1
                    logger.info(f"Triggered rebuild for document {document_id} indexes: {[t for t in rebuild_types]}")

            return {
                "code": "200",
                "message": f"Failed indexes rebuild initiated for {affected_documents} documents",
                "affected_documents": affected_documents,
            }

        result = await self.db_ops.execute_with_transaction(_rebuild_failed_indexes_atomically)
        _trigger_index_reconciliation()
        return result

    async def get_document_chunks(self, user_id: str, collection_id: str, document_id: str) -> List[Chunk]:
        """
        Get all chunks of a document.
        """

        # Use database operations with proper session management
        async def _get_document_chunks(session):
            # Wave 3 §F.1 schema migration: legacy
            # ``DocumentIndex.index_data`` JSON blob (which used to
            # carry ``context_ids``) is gone. The chunk id list now
            # lives in the ``derived/parse_<v>/chunks.jsonl`` artifact
            # on the object store, addressed by the row's
            # ``derived_artifact_path``. Plumbing the object-store
            # read path into this HTTP handler is a chenyexuan T3.1
            # commit 4b follow-up; for now we exercise the §F.1
            # partial-unique invariant via a serving-row probe and
            # return an empty chunk list (degraded but safe — clients
            # see "no chunks indexed" until the read path lands).
            stmt = select(DocumentIndex.derived_artifact_path).filter(
                DocumentIndex.document_id == document_id,
                DocumentIndex.modality == Modality.VECTOR.value,
                DocumentIndex.is_serving.is_(True),
            )
            result = await session.execute(stmt)
            _ = result.scalars().first()
            ctx_ids: list[str] = []
            if not ctx_ids:
                return []

            # 2. Retrieve chunks via the vector-store connector. We go through
            # the connector (not the raw qdrant client) because in multitenant
            # mode it (a) routes to the correct global Qdrant collection based
            # on vector_size and (b) enforces the tenant-id guard.
            try:
                collection_obj = await self.db_ops.query_collection(user_id, collection_id)
                vector_size = None
                if collection_obj is not None:
                    try:
                        from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync

                        _, vector_size = get_collection_embedding_service_sync(collection_obj)
                    except Exception:
                        vector_size = None

                from aperag.config import get_vector_db_connector as _get_vdb

                vector_store_adaptor = _get_vdb(
                    collection=generate_vector_db_collection_name(collection_id=collection_id),
                    vector_size=vector_size,
                )
                points = vector_store_adaptor.connector.retrieve(ids=ctx_ids)

                # 3. Format the response using the shared payload flattener,
                # which understands both the modern {text, metadata} shape
                # and the legacy LlamaIndex _node_content JSON blob.
                from aperag.vectorstore.dto import flatten_node_payload

                chunks = []
                for point in points:
                    flat = flatten_node_payload(point.payload or {})
                    chunks.append(
                        Chunk(
                            id=point.id,
                            text=flat.get("text") or "",
                            metadata=flat.get("metadata") or {},
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
            # Wave 3 §F.1 migration: same ``index_data`` deprecation
            # as :meth:`get_document_chunks` above. Vision chunk ids
            # now live in the ``derived/parse_<v>/vision/manifest.jsonl``
            # artifact; plumbing the object-store read path is a
            # chenyexuan T3.1 commit 4b follow-up. Return empty for
            # now (degraded but safe).
            stmt = select(DocumentIndex.derived_artifact_path).filter(
                DocumentIndex.document_id == document_id,
                DocumentIndex.modality == Modality.VISION.value,
                DocumentIndex.is_serving.is_(True),
            )
            result = await session.execute(stmt)
            _ = result.scalars().first()
            ctx_ids: list[str] = []
            if not ctx_ids:
                return []

            # 2. Retrieve chunks via the connector for the same reasons as
            # get_document_chunks above (tenant-aware routing).
            try:
                collection_obj = await self.db_ops.query_collection(user_id, collection_id)
                vector_size = None
                if collection_obj is not None:
                    try:
                        from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync

                        _, vector_size = get_collection_embedding_service_sync(collection_obj)
                    except Exception:
                        vector_size = None

                from aperag.config import get_vector_db_connector as _get_vdb

                vector_store_adaptor = _get_vdb(
                    collection=generate_vector_db_collection_name(collection_id=collection_id),
                    vector_size=vector_size,
                )
                points = vector_store_adaptor.connector.retrieve(ids=ctx_ids)

                # Use the shared flattener; filter to vision-to-text
                # entries only (this endpoint's contract).
                from aperag.vectorstore.dto import flatten_node_payload

                vision_chunks = []
                for point in points:
                    flat = flatten_node_payload(point.payload or {})
                    metadata = flat.get("metadata") or {}
                    if metadata.get("index_method") == "vision_to_text":
                        vision_chunks.append(
                            VisionChunk(
                                id=point.id,
                                asset_id=metadata.get("asset_id"),
                                text=flat.get("text") or "",
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

        if not user_id:
            await _get_marketplace_ops().validate_marketplace_collection(collection_id)

        # Use database operations with proper session management
        async def _get_document_preview(session: AsyncSession):
            # 1. Get document and vector index in one go
            doc_stmt = select(Document).filter(
                Document.id == document_id,
                Document.collection_id == collection_id,
                Document.user == user_id,
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

    async def download_document(self, user_id: str, collection_id: str, document_id: str):
        """
        Download the original document file.
        Returns a StreamingResponse with the file content.
        """

        async def _download_document(session):
            # 1. Verify user has access to the document
            stmt = select(Document).filter(
                Document.id == document_id,
                Document.collection_id == collection_id,
                Document.user == user_id,
                Document.gmt_deleted.is_(None),  # Only allow downloading non-deleted documents
            )
            result = await session.execute(stmt)
            document = result.scalars().first()
            if not document:
                raise DocumentNotFoundException(document_id)

            # 2. Check document status - only disallow downloading expired/deleted documents
            # UPLOADED documents can be downloaded (before confirmation, within 24 hours)
            # Once expired or deleted, files may no longer exist in storage
            if document.status in [DocumentStatus.EXPIRED, DocumentStatus.DELETED]:
                raise HTTPException(
                    status_code=400, detail=f"Document status is {document.status.value}, cannot download"
                )

            # 3. Get object path from doc_metadata
            try:
                metadata = json.loads(document.doc_metadata) if document.doc_metadata else {}
                object_path = metadata.get("object_path")
                if not object_path:
                    raise HTTPException(status_code=404, detail="Document file not found in storage")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in doc_metadata for document {document_id}")
                raise HTTPException(status_code=500, detail="Document metadata is corrupted")

            # 4. Stream file from object store
            try:
                async_obj_store = get_async_object_store()

                # Get file stream and size
                get_result = await async_obj_store.get(object_path)
                if not get_result:
                    raise HTTPException(status_code=404, detail="Document file not found in object store")

                data_stream, file_size = get_result

                # Determine content type from filename
                content_type, _ = mimetypes.guess_type(document.name)
                if content_type is None:
                    content_type = "application/octet-stream"

                # Set headers for file download
                headers = {
                    "Content-Type": content_type,
                    "Content-Disposition": f'attachment; filename="{document.name}"',
                    "Content-Length": str(file_size),
                }

                logger.info(
                    f"User {user_id} downloading document {document_id} ({document.name}) "
                    f"from collection {collection_id}, size: {file_size} bytes"
                )

                return StreamingResponse(data_stream, headers=headers)

            except Exception as e:
                logger.error(f"Failed to download document {document_id} from path {object_path}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to download document from storage")

        # Execute query with proper session management
        return await self.db_ops._execute_query(_download_document)

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
            stmt = select(Document).filter(
                Document.id == document_id,
                Document.collection_id == collection_id,
                Document.user == user_id,
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

    async def upload_document(self, user_id: str, collection_id: str, file: UploadFile) -> UploadDocumentResponse:
        """Upload a single document file to temporary storage with duplicate detection"""
        # Validate collection
        collection = await self._validate_collection(user_id, collection_id)

        # Validate file
        file_suffix = self._validate_file(file.filename, file.size)

        # Read file content
        file_content = await file.read()
        await file.seek(0)

        # Calculate original file hash for duplicate detection
        file_hash = calculate_file_hash(file_content)

        async def _upload_document_atomically(session):
            from sqlalchemy.dialects.postgresql import insert

            # Try atomic insert first using INSERT ... ON CONFLICT
            # This prevents race condition at database level
            temp_doc_id = "doc" + _random_id()

            stmt = insert(Document).values(
                id=temp_doc_id,
                name=file.filename,
                user=user_id,
                collection_id=collection.id,
                status=DocumentStatus.UPLOADED,
                size=file.size,
                content_hash=file_hash,
                gmt_created=utc_now(),
                gmt_updated=utc_now(),
            )
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["collection_id", "name"], index_where=text("gmt_deleted IS NULL")
            )

            result = await session.execute(stmt)
            await session.flush()

            if result.rowcount == 0:
                # Document already exists, query and return it
                existing_doc = await self._check_duplicate_document(
                    session, user_id, collection.id, file.filename, file_hash
                )
                if existing_doc:
                    logger.info(
                        f"Document '{file.filename}' already exists with same content, returning existing document {existing_doc.id}"
                    )
                    return UploadDocumentResponse(
                        document_id=existing_doc.id,
                        filename=existing_doc.name,
                        size=existing_doc.size,
                        status=existing_doc.status,
                    )

            # Document created, now upload file to object store
            async_obj_store = get_async_object_store()
            document_instance = await session.get(Document, temp_doc_id)
            upload_path = f"{document_instance.object_store_base_path()}/original{file_suffix}"
            await async_obj_store.put(upload_path, file_content)

            # Update document with object path
            metadata = {"object_path": upload_path}
            document_instance.doc_metadata = json.dumps(metadata)
            session.add(document_instance)
            await session.flush()
            await session.refresh(document_instance)

            return UploadDocumentResponse(
                document_id=document_instance.id, filename=file.filename, size=file.size, status="UPLOADED"
            )

        return await self.db_ops.execute_with_transaction(_upload_document_atomically)

    async def confirm_documents(
        self, user_id: str, collection_id: str, document_ids: list[str]
    ) -> ConfirmDocumentsResponse:
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
                    stmt = select(Document).where(
                        Document.id == document_id,
                        Document.user == user_id,
                        Document.collection_id == collection_id,
                    )
                    result = await session.execute(stmt)
                    document = result.scalars().first()

                    if not document:
                        # Document not found at all
                        failed_documents.append(
                            FailedDocument(document_id=document_id, name=None, error="DOCUMENT_NOT_FOUND")
                        )
                        failed_count += 1
                        continue

                    # Check document status
                    if document.status != DocumentStatus.UPLOADED:
                        # Document exists but not in correct status
                        if document.status == DocumentStatus.EXPIRED:
                            error_code = "DOCUMENT_EXPIRED"
                        else:
                            error_code = "DOCUMENT_NOT_UPLOADED"

                        failed_documents.append(
                            FailedDocument(document_id=document_id, name=document.name, error=error_code)
                        )
                        failed_count += 1
                        continue

                    # Change status to PENDING
                    document.status = DocumentStatus.PENDING
                    session.add(document)

                    # Create indexes (Wave 3 T3.1 chunk 3: dispatch via
                    # new dispatcher post-confirm).
                    await _create_or_update_document_indexes(
                        document_id=document.id, index_types=index_types, session=session
                    )

                    confirmed_count += 1

                except Exception as e:
                    logger.error(f"Failed to confirm document {document_id}: {e}")
                    # Try to get document name for better error reporting
                    document_name = None
                    try:
                        stmt_name = select(Document.name).where(Document.id == document_id)
                        result_name = await session.execute(stmt_name)
                        document_name = result_name.scalar()
                    except Exception:
                        pass

                    failed_documents.append(
                        FailedDocument(document_id=document_id, name=document_name, error="CONFIRMATION_FAILED")
                    )
                    failed_count += 1

        await self.db_ops.execute_with_transaction(_confirm_documents_atomically)

        # Trigger index reconciliation
        _trigger_index_reconciliation()

        return ConfirmDocumentsResponse(
            confirmed_count=confirmed_count, failed_count=failed_count, failed_documents=failed_documents
        )

    async def get_staged_documents(self, user_id: str, collection_id: str) -> StagedDocumentsResponse:
        """Return all UPLOADED (staged) documents for the collection, ordered newest-first."""
        collection = await self._validate_collection(user_id, collection_id)

        async def _query(session: AsyncSession):
            stmt = (
                select(Document)
                .where(
                    Document.user == user_id,
                    Document.collection_id == collection.id,
                    Document.status == DocumentStatus.UPLOADED,
                    Document.gmt_deleted.is_(None),
                )
                .order_by(Document.gmt_created.asc())
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        docs = await self.db_ops.execute_with_transaction(_query)
        return StagedDocumentsResponse(
            documents=[
                UploadDocumentResponse(
                    document_id=doc.id,
                    filename=doc.name,
                    size=doc.size or 0,
                    status=doc.status,
                )
                for doc in docs
            ],
            total=len(docs),
        )

    async def fetch_url_documents(self, user_id: str, collection_id: str, urls: list) -> FetchUrlResponse:
        """
        Fetch web page content from URLs and create UPLOADED documents.

        For each URL, uses the web read service (JINA with Trafilatura fallback) to
        retrieve the page content as Markdown. The result is wrapped as a virtual
        UploadFile and passed to upload_document(), so the resulting documents are
        identical to file uploads and go through the same two-phase commit flow.
        """
        import io
        import re
        from urllib.parse import urlparse

        from fastapi import UploadFile
        from starlette.datastructures import Headers

        from aperag.domains.model_platform.service.model_service import model_platform_service
        from aperag.domains.web_access.reader.reader_service import read_with_jina_fallback
        from aperag.domains.web_access.schemas import WebReadRequest

        # Validate URL count
        if len(urls) > 10:
            raise HTTPException(status_code=400, detail="Too many URLs: maximum 10 URLs per request")

        url_strings = [str(u) for u in urls]

        # Determine which reader to use based on user's JINA API key
        jina_api_key = await model_platform_service.get_user_provider_api_key(
            user_id=user_id, provider_type="jina", fallback_to_public=True
        )
        logger.info(
            "Starting fetch-url import collection_id=%s urls=%s jina_configured=%s",
            collection_id,
            len(url_strings),
            bool(jina_api_key),
        )

        web_read_request = WebReadRequest(url_list=url_strings, timeout=30)

        try:
            web_response = await read_with_jina_fallback(
                web_read_request,
                jina_api_key,
                log_context=f"fetch_url_documents collection_id={collection_id}",
            )
        except Exception as e:
            logger.error(f"Web read service failed: {e}")
            # Return all URLs as failed
            results = [
                FetchUrlResultItem(
                    url=u,
                    fetch_status="error",
                    error=f"Web read service error: {str(e)}",
                )
                for u in url_strings
            ]
            return FetchUrlResponse(results=results, total=len(results), succeeded=0, failed=len(results))

        logger.info(
            "Fetch-url read stage completed collection_id=%s successful=%s failed=%s",
            collection_id,
            web_response.successful,
            web_response.failed,
        )

        results = []
        for item in web_response.results:
            if item.status != "success" or not item.content:
                results.append(
                    FetchUrlResultItem(
                        url=item.url,
                        fetch_status="error",
                        error=item.error or "Failed to fetch or empty content",
                    )
                )
                continue

            # Build a safe filename from the page title or URL path
            raw_name = item.title or urlparse(item.url).path.strip("/").replace("/", "_") or "page"
            safe_name = re.sub(r"[^\w\s\-.]", "", raw_name).strip()[:200] or "page"
            filename = f"{safe_name}.md"

            content_bytes = item.content.encode("utf-8")
            content_size = len(content_bytes)

            # Wrap Markdown content as a virtual UploadFile (same interface as real file upload)
            virtual_file = UploadFile(
                filename=filename,
                size=content_size,
                headers=Headers({"content-type": "text/markdown"}),
                file=io.BytesIO(content_bytes),
            )

            try:
                upload_response = await self.upload_document(user_id, collection_id, virtual_file)
                logger.info(
                    "Fetch-url imported successfully collection_id=%s url=%s document_id=%s",
                    collection_id,
                    item.url,
                    upload_response.document_id,
                )
                results.append(
                    FetchUrlResultItem(
                        url=item.url,
                        fetch_status="success",
                        document_id=upload_response.document_id,
                        filename=upload_response.filename,
                        size=upload_response.size,
                        status=str(
                            upload_response.status.value
                            if hasattr(upload_response.status, "value")
                            else upload_response.status
                        ),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to upload fetched content for {item.url}: {e}")
                results.append(
                    FetchUrlResultItem(
                        url=item.url,
                        fetch_status="error",
                        error=str(e),
                    )
                )

        succeeded = sum(1 for r in results if r.fetch_status == "success")
        logger.info(
            "Fetch-url import completed collection_id=%s total=%s succeeded=%s failed=%s",
            collection_id,
            len(results),
            succeeded,
            len(results) - succeeded,
        )
        return FetchUrlResponse(
            results=results,
            total=len(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
        )


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
document_service = DocumentService()
