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
from typing import Any, Dict, List, Optional

from aperag.db.models import DocumentIndexStatus
from aperag.db.ops import db_ops
from aperag.index.base import IndexResult, IndexType
from aperag.index.fulltext.indexer import fulltext_indexer
from aperag.index.graph.indexer import graph_indexer
from aperag.index.vector.indexer import vector_indexer
from aperag.tasks.async_interface import TaskResult

logger = logging.getLogger(__name__)


class IndexManager:
    """Centralized index management for all index types"""
    
    def __init__(self):
        self.logger = logger
        self.indexers = {
            IndexType.VECTOR: vector_indexer,
            IndexType.FULLTEXT: fulltext_indexer,
            IndexType.GRAPH: graph_indexer
        }
    
    def create_all_indexes(self, document_id: int, content: str, doc_parts: List[Any], 
                          collection, **kwargs) -> TaskResult:
        """
        Create all enabled indexes for a document
        
        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters
            
        Returns:
            TaskResult: Combined result of all index operations
        """
        try:
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")
            
            results = {}
            overall_success = True
            
            # Set initial status to RUNNING for all index types
            document.vector_index_status = DocumentIndexStatus.RUNNING
            document.fulltext_index_status = DocumentIndexStatus.RUNNING
            document.graph_index_status = DocumentIndexStatus.RUNNING
            db_ops.update_document(document)
            
            # Process each index type
            for index_type, indexer in self.indexers.items():
                try:
                    if indexer.is_enabled(collection):
                        self.logger.info(f"Creating {index_type.value} index for document {document_id}")
                        result = indexer.create_index(
                            document_id=document_id,
                            content=content,
                            doc_parts=doc_parts,
                            collection=collection,
                            **kwargs
                        )
                        results[index_type.value] = result.to_dict()
                        
                        # Update document status based on result
                        if result.success:
                            if index_type == IndexType.VECTOR:
                                document.vector_index_status = DocumentIndexStatus.COMPLETE
                            elif index_type == IndexType.FULLTEXT:
                                document.fulltext_index_status = DocumentIndexStatus.COMPLETE
                            elif index_type == IndexType.GRAPH:
                                # Graph indexing is async, status handled by async task
                                if result.metadata and result.metadata.get("status") == "running":
                                    document.graph_index_status = DocumentIndexStatus.RUNNING
                                else:
                                    document.graph_index_status = DocumentIndexStatus.COMPLETE
                        else:
                            overall_success = False
                            if index_type == IndexType.VECTOR:
                                document.vector_index_status = DocumentIndexStatus.FAILED
                            elif index_type == IndexType.FULLTEXT:
                                document.fulltext_index_status = DocumentIndexStatus.FAILED
                            elif index_type == IndexType.GRAPH:
                                document.graph_index_status = DocumentIndexStatus.FAILED
                    else:
                        # Index type is disabled
                        self.logger.info(f"Skipping {index_type.value} index for document {document_id} (disabled)")
                        if index_type == IndexType.VECTOR:
                            document.vector_index_status = DocumentIndexStatus.SKIPPED
                        elif index_type == IndexType.FULLTEXT:
                            document.fulltext_index_status = DocumentIndexStatus.SKIPPED
                        elif index_type == IndexType.GRAPH:
                            document.graph_index_status = DocumentIndexStatus.SKIPPED
                        
                        results[index_type.value] = {
                            "success": True,
                            "index_type": index_type.value,
                            "metadata": {"status": "skipped", "reason": "disabled"}
                        }
                        
                except Exception as e:
                    self.logger.error(f"Failed to create {index_type.value} index for document {document_id}: {str(e)}")
                    overall_success = False
                    results[index_type.value] = {
                        "success": False,
                        "index_type": index_type.value,
                        "error": str(e)
                    }
                    
                    # Set failed status
                    if index_type == IndexType.VECTOR:
                        document.vector_index_status = DocumentIndexStatus.FAILED
                    elif index_type == IndexType.FULLTEXT:
                        document.fulltext_index_status = DocumentIndexStatus.FAILED
                    elif index_type == IndexType.GRAPH:
                        document.graph_index_status = DocumentIndexStatus.FAILED
            
            # Update document status
            db_ops.update_document(document)
            
            return TaskResult(
                success=overall_success,
                data={"index_results": results},
                metadata={"document_id": document_id, "total_indexes": len(results)}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes for document {document_id}: {str(e)}")
            return TaskResult(
                success=False,
                error=f"Index creation failed: {str(e)}"
            )
    
    def update_all_indexes(self, document_id: int, content: str, doc_parts: List[Any],
                          collection, **kwargs) -> TaskResult:
        """
        Update all enabled indexes for a document
        
        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters
            
        Returns:
            TaskResult: Combined result of all index operations
        """
        try:
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")
            
            results = {}
            overall_success = True
            
            # Process each index type
            for index_type, indexer in self.indexers.items():
                try:
                    if indexer.is_enabled(collection):
                        self.logger.info(f"Updating {index_type.value} index for document {document_id}")
                        result = indexer.update_index(
                            document_id=document_id,
                            content=content,
                            doc_parts=doc_parts,
                            collection=collection,
                            **kwargs
                        )
                        results[index_type.value] = result.to_dict()
                        
                        if not result.success:
                            overall_success = False
                    else:
                        results[index_type.value] = {
                            "success": True,
                            "index_type": index_type.value,
                            "metadata": {"status": "skipped", "reason": "disabled"}
                        }
                        
                except Exception as e:
                    self.logger.error(f"Failed to update {index_type.value} index for document {document_id}: {str(e)}")
                    overall_success = False
                    results[index_type.value] = {
                        "success": False,
                        "index_type": index_type.value,
                        "error": str(e)
                    }
            
            return TaskResult(
                success=overall_success,
                data={"index_results": results},
                metadata={"document_id": document_id, "total_indexes": len(results)}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update indexes for document {document_id}: {str(e)}")
            return TaskResult(
                success=False,
                error=f"Index update failed: {str(e)}"
            )
    
    def delete_all_indexes(self, document_id: int, collection, **kwargs) -> TaskResult:
        """
        Delete all indexes for a document
        
        Args:
            document_id: Document ID
            collection: Collection object
            **kwargs: Additional parameters
            
        Returns:
            TaskResult: Combined result of all index operations
        """
        try:
            results = {}
            overall_success = True
            
            # Process each index type
            for index_type, indexer in self.indexers.items():
                try:
                    self.logger.info(f"Deleting {index_type.value} index for document {document_id}")
                    result = indexer.delete_index(
                        document_id=document_id,
                        collection=collection,
                        **kwargs
                    )
                    results[index_type.value] = result.to_dict()
                    
                    if not result.success:
                        overall_success = False
                        
                except Exception as e:
                    self.logger.error(f"Failed to delete {index_type.value} index for document {document_id}: {str(e)}")
                    overall_success = False
                    results[index_type.value] = {
                        "success": False,
                        "index_type": index_type.value,
                        "error": str(e)
                    }
            
            return TaskResult(
                success=overall_success,
                data={"index_results": results},
                metadata={"document_id": document_id, "total_indexes": len(results)}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to delete indexes for document {document_id}: {str(e)}")
            return TaskResult(
                success=False,
                error=f"Index deletion failed: {str(e)}"
            )
    
    def create_single_index(self, index_type: IndexType, document_id: int, content: str, 
                           doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """
        Create a single index type for a document
        
        Args:
            index_type: Type of index to create
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters
            
        Returns:
            IndexResult: Result of the index operation
        """
        if index_type not in self.indexers:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        indexer = self.indexers[index_type]
        return indexer.create_index(
            document_id=document_id,
            content=content,
            doc_parts=doc_parts,
            collection=collection,
            **kwargs
        )
    
    def delete_single_index(self, index_type: IndexType, document_id: int, 
                           collection, **kwargs) -> IndexResult:
        """
        Delete a single index type for a document
        
        Args:
            index_type: Type of index to delete
            document_id: Document ID
            collection: Collection object
            **kwargs: Additional parameters
            
        Returns:
            IndexResult: Result of the index operation
        """
        if index_type not in self.indexers:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        indexer = self.indexers[index_type]
        return indexer.delete_index(
            document_id=document_id,
            collection=collection,
            **kwargs
        )
    
    def get_enabled_index_types(self, collection) -> List[IndexType]:
        """
        Get list of enabled index types for a collection
        
        Args:
            collection: Collection object
            
        Returns:
            List of enabled index types
        """
        enabled_types = []
        for index_type, indexer in self.indexers.items():
            if indexer.is_enabled(collection):
                enabled_types.append(index_type)
        return enabled_types


# Global manager instance
index_manager = IndexManager() 