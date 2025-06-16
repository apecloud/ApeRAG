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

from aperag.db.models import CollectionStatus, DocumentStatus
from aperag.db.ops import db_ops
from aperag.index.base import AsyncIndexer, IndexResult, IndexType
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)


class GraphIndexer(AsyncIndexer):
    """Graph index implementation using LightRAG"""

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
        Create graph index asynchronously using LightRAG

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

            if document.status == DocumentStatus.DELETED:
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "Document deleted, skipping graph indexing", "status": "skipped"},
                )

            if collection.status == CollectionStatus.DELETED:
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

    def process_lightrag_result(self, result: Dict[str, Any]) -> IndexResult:
        """
        Process LightRAG processing result

        Args:
            result: Result from LightRAG processing

        Returns:
            IndexResult: Processed result
        """
        try:
            if result.get("status") == "success":
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
            elif result.get("status") == "warning":
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    data={"warning_message": result.get("message")},
                    metadata={"status": "complete_with_warnings"},
                )
            else:
                return IndexResult(
                    success=False,
                    index_type=self.index_type,
                    error=f"LightRAG processing failed: {result.get('message', 'Unknown error')}",
                )

        except Exception as e:
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Failed to process LightRAG result: {str(e)}"
            )


# Global instance
graph_indexer = GraphIndexer()
