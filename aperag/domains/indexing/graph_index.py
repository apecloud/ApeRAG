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
from typing import Any, Dict, List

from aperag.db.ops import db_ops
from aperag.domains.indexing.base import AsyncIndexer, IndexResult, IndexType
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)


class GraphIndexer(AsyncIndexer):
    """Enable/disable gate for the per-collection graph index.

    Business code (search pipeline, document tasks) asks this indexer
    whether graph indexing is enabled for a collection and then routes
    to ``aperag.domains.knowledge_graph.graphindex`` for the actual work. The method bodies
    below only emit reconciliation-scheduling metadata; no graph
    writes happen here. Kept as a separate file because the
    reconciliation loop treats it uniformly with ``vector_index`` and
    ``fulltext_index``.
    """

    def __init__(self):
        super().__init__(IndexType.GRAPH)

    def is_enabled(self, collection) -> bool:
        """Check if graph indexing is enabled for the collection"""
        config = parseCollectionConfig(collection.config)
        return config.enable_knowledge_graph or False

    def create_index(self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """
        Create graph index for document (synchronous wrapper)

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of graph index creation
        """
        if not self.is_enabled(collection):
            return IndexResult(
                success=True,
                index_type=self.index_type,
                metadata={"message": "Graph indexing disabled", "status": "skipped"},
            )

        # For graph indexing, we use the async version
        return self.create_index_async(document_id, content, doc_parts, collection, **kwargs)

    def create_index_async(
        self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs
    ) -> IndexResult:
        """
        Schedule asynchronous graph indexing.

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result indicating async task was started
        """
        try:
            # Validate document and collection status
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")

            if document.status == "DELETED":
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "Document deleted, skipping graph indexing", "status": "skipped"},
                )

            if collection.status == "DELETED":
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "Collection deleted, skipping graph indexing", "status": "skipped"},
                )

            # Schedule async graph indexing task
            file_path = kwargs.get("file_path", f"document_{document_id}")

            # Graph indexing is now handled by the reconciliation system
            # No need to schedule tasks directly
            logger.info(f"Graph index task scheduled for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"task_scheduled": True, "document_id": document_id},
                metadata={
                    "status": "running",
                    "file_path": file_path,
                    "content_length": len(content) if content else 0,
                },
            )

        except Exception as e:
            logger.error(f"Graph index scheduling failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Graph index scheduling failed: {str(e)}"
            )

    def update_index(self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """
        Update graph index for document

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of graph index update
        """
        if not self.is_enabled(collection):
            return IndexResult(
                success=True,
                index_type=self.index_type,
                metadata={"message": "Graph indexing disabled", "status": "skipped"},
            )

        return self.update_index_async(document_id, content, doc_parts, collection, **kwargs)

    def update_index_async(
        self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs
    ) -> IndexResult:
        """
        Update graph index asynchronously

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result indicating async task was started
        """
        try:
            # For graph index update, we typically need to delete old data and create new
            file_path = kwargs.get("file_path", f"document_{document_id}")

            logger.info(f"Graph index update task scheduled for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"task_scheduled": True, "document_id": document_id},
                metadata={
                    "status": "running",
                    "operation": "update",
                    "file_path": file_path,
                    "content_length": len(content) if content else 0,
                },
            )

        except Exception as e:
            logger.error(f"Graph index update scheduling failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Graph index update scheduling failed: {str(e)}"
            )

    def delete_index(self, document_id: int, collection, **kwargs) -> IndexResult:
        """
        Delete graph index for document

        Args:
            document_id: Document ID
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of graph index deletion
        """
        try:
            if not self.is_enabled(collection):
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "Graph indexing disabled", "status": "skipped"},
                )

            # Graph deletion is now handled by the reconciliation system

            logger.info(f"Graph index deletion task scheduled for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"task_scheduled": True, "document_id": document_id},
                metadata={"status": "running", "operation": "delete"},
            )

        except Exception as e:
            logger.error(f"Graph index deletion scheduling failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Graph index deletion scheduling failed: {str(e)}"
            )

    def process_indexing_result(self, result: Dict[str, Any]) -> IndexResult:
        """Adapt a graphindex indexer result dict into an ``IndexResult``.

        Accepts the shape returned by
        ``DocumentIndexTask._upsert_graph_index`` (status / chunks_created
        / entities_extracted / relations_extracted). Kept as a thin
        adapter so the reconciliation layer doesn't need to know which
        graph backend produced the numbers.
        """
        try:
            status = result.get("status")
            if status == "success":
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    data={
                        "chunks_created": result.get("chunks_created", 0),
                        "entities_extracted": result.get("entities_extracted", 0),
                        "relations_extracted": result.get("relations_extracted", 0),
                    },
                    metadata={"status": "complete", "processing_time": result.get("processing_time")},
                )
            if status == "warning":
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    data={"warning_message": result.get("message")},
                    metadata={"status": "complete_with_warnings"},
                )
            return IndexResult(
                success=False,
                index_type=self.index_type,
                error=f"Graph indexing failed: {result.get('message', 'Unknown error')}",
            )
        except Exception as e:
            return IndexResult(
                success=False,
                index_type=self.index_type,
                error=f"Failed to process graph indexing result: {e}",
            )


# Global instance
graph_indexer = GraphIndexer()
