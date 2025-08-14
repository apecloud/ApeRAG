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

from datetime import timedelta
import logging
from typing import Any

from asgiref.sync import Dict, async_to_sync
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from aperag.config import get_vector_db_connector
from aperag.db import models as db_models
from aperag.db.models import CollectionStatus, Collection
from aperag.db.ops import db_ops
from aperag.graph import lightrag_manager
from aperag.index.fulltext_index import create_index, delete_index
from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
from aperag.objectstore.base import get_object_store
from aperag.schema.utils import parseCollectionConfig
from aperag.tasks.models import TaskResult
from aperag.utils.utils import (
    generate_fulltext_index_name,
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

            # Initialize fulltext index
            self._initialize_fulltext_index(collection_id)

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
        """Initialize vector database collections"""
        # Get embedding service
        _, vector_size = get_collection_embedding_service_sync(collection)

        # Create main vector database collection
        vector_db_conn = get_vector_db_connector(
            collection=generate_vector_db_collection_name(collection_id=collection_id)
        )
        vector_db_conn.connector.create_collection(vector_size=vector_size)

        logger.debug(f"Initialized vector databases for collection {collection_id}")

    def _initialize_fulltext_index(self, collection_id: str) -> None:
        """Initialize fulltext search index"""
        index_name = generate_fulltext_index_name(collection_id)
        create_index(index_name)
        logger.debug(f"Initialized fulltext index {index_name}")

    def _delete_knowledge_graph_data(self, collection) -> Dict[str, Any]:
        """Delete knowledge graph data for the collection"""
        config = parseCollectionConfig(collection.config)
        enable_knowledge_graph = config.enable_knowledge_graph or False

        deletion_stats = {"knowledge_graph_enabled": enable_knowledge_graph}

        if not enable_knowledge_graph:
            return deletion_stats

        async def _delete_lightrag():
            # Create new LightRAG instance
            rag = await lightrag_manager.create_lightrag_instance(collection)

            # Get all document IDs in this collection
            documents = db_ops.query_documents([collection.user], collection.id)
            document_ids = [doc.id for doc in documents]

            if document_ids:
                deleted_count = 0
                failed_count = 0

                for document_id in document_ids:
                    try:
                        await rag.adelete_by_doc_id(str(document_id))
                        deleted_count += 1
                        logger.debug(f"Deleted lightrag document for document ID: {document_id}")
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Failed to delete lightrag document for document ID {document_id}: {str(e)}")

                logger.info(
                    f"Completed lightrag document deletion for collection {collection.id}: "
                    f"{deleted_count} deleted, {failed_count} failed"
                )

                deletion_stats.update({"documents_deleted": deleted_count, "documents_failed": failed_count})
            else:
                logger.info(f"No documents found for collection {collection.id}")
                deletion_stats["documents_deleted"] = 0

            # Clean up resources
            await rag.finalize_storages()

        # Execute async deletion
        async_to_sync(_delete_lightrag)()

        return deletion_stats

    def _delete_vector_databases(self, collection_id: str) -> None:
        """Delete vector database collections"""
        # Delete main vector database collection
        vector_db_conn = get_vector_db_connector(
            collection=generate_vector_db_collection_name(collection_id=collection_id)
        )
        vector_db_conn.connector.delete_collection()

        logger.debug(f"Deleted vector database collections for collection {collection_id}")

    def _delete_fulltext_index(self, collection_id: str) -> None:
        """Delete fulltext search index"""
        index_name = generate_fulltext_index_name(collection_id)
        delete_index(index_name)
        logger.debug(f"Deleted fulltext index {index_name}")

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
                    db_models.Document.gmt_created < expiration_threshold
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
                        logger.info(f"Deleted objects from object store for expired document {document.id}: {document.object_store_base_path()}")
                    except Exception as e:
                        logger.warning(f"Failed to delete objects for expired document {document.id} from object store: {e}")

                    # Soft delete: Mark document as EXPIRED instead of deleting
                    document.status = db_models.DocumentStatus.EXPIRED
                    document.gmt_updated = current_time
                    session.add(document)
                    expired_count += 1
                    logger.info(f"Marked document {document.id} as expired (name: {document.name}, created: {document.gmt_created})")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to cleanup expired document {document.id}: {e}")

            session.commit()

            return {
                "expired_count": expired_count,
                "failed_count": failed_count,
                "total_found": len(expired_documents)
            }

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
