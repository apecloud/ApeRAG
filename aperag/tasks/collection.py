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
from datetime import timedelta
from typing import Any

from asgiref.sync import Dict
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from aperag.config import get_vector_db_connector
from aperag.db import models as db_models
from aperag.db.ops import db_ops
from aperag.domains.indexing.fulltext_index import create_index, delete_collection_documents, delete_index
from aperag.domains.knowledge_base.db.models import CollectionStatus
from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
from aperag.objectstore.base import get_object_store
from aperag.schema.utils import parseCollectionConfig
from aperag.tasks.models import TaskResult
from aperag.utils.utils import (
    generate_fulltext_index_name,
    generate_legacy_fulltext_index_name,
    generate_vector_db_collection_name,
    utc_now,
)

logger = logging.getLogger(__name__)


class CollectionTask:
    """Collection workflow orchestrator"""

    def initialize_collection(self, collection_id: str, document_user_quota: int) -> TaskResult:
        """
        Initialize a new collection with all required components

        Args:
            collection_id: Collection ID to initialize
            document_user_quota: User quota for documents

        Returns:
            TaskResult: Result of the initialization
        """
        try:
            # Get collection from database
            collection = db_ops.query_collection_by_id(collection_id)

            if not collection or collection.status == CollectionStatus.DELETED:
                return TaskResult(success=False, error=f"Collection {collection_id} not found or deleted")

            # Initialize vector database connections
            self._initialize_vector_databases(collection_id, collection)

            config = parseCollectionConfig(collection.config)
            if config.enable_fulltext is not False:
                self._initialize_fulltext_index(collection_id)
            else:
                logger.info(
                    "Skipping fulltext index initialization for collection %s because enable_fulltext=false",
                    collection_id,
                )

            # No per-collection cutover flip here: graphindex v2 is the
            # only graph backend after the LightRAG removal, so the
            # "which store is the truth for this collection?" question
            # doesn't exist. A brand-new collection simply has no graph
            # rows yet — first document write populates them through
            # ``DocumentIndexTask._upsert_graph_index``.

            # Update collection status
            collection.status = CollectionStatus.ACTIVE
            db_ops.update_collection(collection)

            logger.info(f"Successfully initialized collection {collection_id}")

            return TaskResult(
                success=True,
                data={"collection_id": collection_id, "status": "initialized"},
                metadata={"document_user_quota": document_user_quota},
            )

        except Exception as e:
            logger.error(f"Failed to initialize collection {collection_id}: {str(e)}")
            return TaskResult(success=False, error=f"Collection initialization failed: {str(e)}")

    def delete_collection(self, collection_id: str) -> TaskResult:
        """
        Delete a collection and all its associated data

        Args:
            collection_id: Collection ID to delete

        Returns:
            TaskResult: Result of the deletion
        """
        try:
            # Get collection from database
            collection = db_ops.query_collection_by_id(collection_id, ignore_deleted=False)

            if not collection:
                return TaskResult(success=False, error=f"Collection {collection_id} not found")

            # Delete knowledge graph data if enabled
            deletion_stats = self._delete_knowledge_graph_data(collection)

            # Delete vector databases
            self._delete_vector_databases(collection_id)

            # Delete fulltext index
            self._delete_fulltext_index(collection_id)

            logger.info(f"Successfully deleted collection {collection_id}")

            return TaskResult(
                success=True, data={"collection_id": collection_id, "status": "deleted"}, metadata=deletion_stats
            )

        except Exception as e:
            logger.error(f"Failed to delete collection {collection_id}: {str(e)}")
            return TaskResult(success=False, error=f"Collection deletion failed: {str(e)}")

    def _initialize_vector_databases(self, collection_id: str, collection) -> None:
        """Ensure vector-store provisioning for this tenant.

        In multitenant mode this is essentially a no-op per tenant: the global
        Qdrant collection is created lazily on first use (idempotent inside
        the connector). We still call through so new deployments get their
        global collection primed at cluster-creation time rather than on the
        first user upload.

        Mirrors ``_initialize_fulltext_index``'s ``enable_fulltext`` skip:
        a collection with ``enable_vector=False`` does not require any
        embedding lookup, so we short-circuit before resolving the
        embedding provider. Without this guard, provider-independent
        collections (smoke tests, KG-only tenants) trigger a NoneType
        model lookup in ``base_embedding`` and a
        Celery retry storm.
        """
        config = parseCollectionConfig(collection.config)
        if not config.enable_vector:
            logger.info(
                "Skipping vector database initialization for collection %s because enable_vector=false",
                collection_id,
            )
            return

        # Get embedding service
        _, vector_size = get_collection_embedding_service_sync(collection)

        # Create main vector database collection (idempotent in multitenant mode).
        # The connector's __init__ calls ensure_collection() eagerly; this extra
        # ensure_collection() is a cheap explicit for operational clarity so
        # "did the cluster bootstrap create the physical shard?" has a clear
        # single call in the trace.
        vector_db_conn = get_vector_db_connector(
            collection=generate_vector_db_collection_name(collection_id=collection_id),
            vector_size=vector_size,
        )
        vector_db_conn.connector.ensure_collection()

        logger.debug(f"Initialized vector databases for collection {collection_id}")

    def _initialize_fulltext_index(self, collection_id: str) -> None:
        """Initialize the shared fulltext logical index."""
        index_name = generate_fulltext_index_name(collection_id)
        create_index(index_name)
        logger.debug(f"Initialized fulltext index {index_name}")

    def _delete_knowledge_graph_data(self, collection) -> Dict[str, Any]:
        """Wipe this collection's graphindex rows.

        Single tenant-scoped DELETE across the three ``graphindex_*``
        tables. No per-document loop is needed — every graphindex row
        is already tagged with ``collection_id``, so one transaction
        covers the lot. Failure is logged and swallowed so the overall
        collection-delete flow is not blocked by a transient graph
        issue; the DB row is tombstoned regardless.
        """
        config = parseCollectionConfig(collection.config)
        enable_knowledge_graph = config.enable_knowledge_graph or False

        deletion_stats = {"knowledge_graph_enabled": enable_knowledge_graph}
        if not enable_knowledge_graph:
            return deletion_stats

        from aperag.domains.knowledge_graph.graphindex.integration import run_drop_collection_sync
        from aperag.graph_curation.integration import run_purge_graph_curation_collection_sync

        try:
            run_drop_collection_sync(str(collection.id))
            run_purge_graph_curation_collection_sync(str(collection.id))
            deletion_stats["graphindex_dropped"] = True
            deletion_stats["graph_curation_purged"] = True
            logger.info(f"graphindex: dropped all rows for collection {collection.id}")
        except Exception as e:
            deletion_stats["graphindex_dropped"] = False
            deletion_stats["graphindex_error"] = str(e)
            deletion_stats["graph_curation_purged"] = False
            logger.warning(f"graphindex: failed to drop collection {collection.id}: {e}")

        return deletion_stats

    def _delete_vector_databases(self, collection_id: str) -> None:
        """Purge this tenant's vector data.

        * Multitenant mode (default): deletes only the points whose
          ``collection_id`` payload matches; the shared global Qdrant
          collection is left in place for other tenants.
        * Legacy mode: drops the whole per-tenant Qdrant collection.

        Routing in multitenant mode is ``vector_size``-aware (each
        ``(vector_size, distance)`` pair lives in a distinct global Qdrant
        collection). We try to resolve ``vector_size`` from the collection's
        embedding config first. If that fails — typically because the
        embedding provider/model has been removed from the LLM registry, or
        the collection row is already malformed — we fall back to
        ``purge_all_shards``: scan every ``aperag_vectors_*`` collection and
        delete any points tagged with this tenant. That avoids the silent
        "route-to-wrong-shard, leave orphans" failure mode we had before.
        """
        collection = db_ops.query_collection_by_id(collection_id, ignore_deleted=False)
        vector_size = None
        resolve_failed = False
        if collection is not None:
            try:
                _, vector_size = get_collection_embedding_service_sync(collection)
            except Exception as e:
                resolve_failed = True
                logger.warning(
                    "Could not resolve vector_size for collection %s during delete; "
                    "will purge across every global shard as a safety net: %s",
                    collection_id,
                    e,
                )
        else:
            resolve_failed = True

        vector_db_conn = get_vector_db_connector(
            collection=generate_vector_db_collection_name(collection_id=collection_id),
            vector_size=vector_size,
        )
        if resolve_failed:
            # Best-effort: iterate every aperag_vectors_* collection and drop
            # rows with this tenant_id. No-op on the legacy connector path.
            vector_db_conn.connector.drop_tenant(purge_all_shards=True)
        else:
            vector_db_conn.connector.drop_tenant()

        logger.debug(f"Deleted vector database data for collection {collection_id}")

    def _delete_fulltext_index(self, collection_id: str) -> None:
        """Delete a collection's documents from the shared fulltext index and prune legacy index."""
        deleted_shared = delete_collection_documents(collection_id, index=generate_fulltext_index_name(collection_id))
        logger.debug("Deleted %s shared fulltext docs for collection %s", deleted_shared, collection_id)

        legacy_index = generate_legacy_fulltext_index_name(collection_id)
        delete_index(legacy_index)
        logger.debug(f"Deleted legacy fulltext index {legacy_index}")

    def cleanup_expired_documents(self, collection_id: str):
        """
        Clean up documents that have been in UPLOADED status for more than 1 day.
        This function runs asynchronously and handles all database operations.
        Uses soft delete by marking documents as EXPIRED instead of deleting them.
        """
        logger.info("Starting cleanup of expired uploaded documents")

        def _cleanup_expired_documents(session: Session):
            # Calculate expiration time (1 day ago)
            current_time = utc_now()
            expiration_threshold = current_time - timedelta(days=1)

            # Query for expired documents
            stmt = select(db_models.Document).where(
                and_(
                    db_models.Document.collection_id == collection_id,
                    db_models.Document.status == db_models.DocumentStatus.UPLOADED,
                    db_models.Document.gmt_created < expiration_threshold,
                )
            )

            result = session.execute(stmt)
            expired_documents = result.scalars().all()

            if not expired_documents:
                logger.info("No expired documents found")
                return {"total_found": 0, "expired_count": 0, "failed_count": 0}

            logger.info(f"Found {len(expired_documents)} expired documents to clean up")

            expired_count = 0
            failed_count = 0
            obj_store = get_object_store()

            for document in expired_documents:
                try:
                    # Delete from object store
                    try:
                        obj_store.delete_objects_by_prefix(document.object_store_base_path())
                        logger.info(
                            f"Deleted objects from object store for expired document {document.id}: {document.object_store_base_path()}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete objects for expired document {document.id} from object store: {e}"
                        )

                    # Soft delete: Mark document as EXPIRED instead of deleting
                    document.status = db_models.DocumentStatus.EXPIRED
                    document.gmt_updated = current_time
                    session.add(document)
                    expired_count += 1
                    logger.info(
                        f"Marked document {document.id} as expired (name: {document.name}, created: {document.gmt_created})"
                    )

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to cleanup expired document {document.id}: {e}")

            session.commit()

            return {"expired_count": expired_count, "failed_count": failed_count, "total_found": len(expired_documents)}

        try:
            # Execute the cleanup with transaction
            result = db_ops._execute_transaction(_cleanup_expired_documents)

            logger.info(
                f"Cleanup completed - Expired: {result.get('expired_count', 0)}, "
                f"Failed: {result['failed_count']}, Total found: {result['total_found']}"
            )

            return result

        except Exception as e:
            logger.error(f"Error during expired documents cleanup: {e}", exc_info=True)
            return {"expired_count": 0, "failed_count": 0, "error": str(e)}


collection_task = CollectionTask()
