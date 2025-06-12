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
from pathlib import Path
from typing import Any, List

from aperag.core.processors.document_parser import document_parser
from aperag.db.models import DocumentStatus
from aperag.db.ops import db_ops
from aperag.docparser.doc_parser import DocParser
from aperag.index.manager import index_manager
from aperag.schema.utils import parseCollectionConfig
from aperag.source.base import get_source
from aperag.source.feishu.client import FeishuNoPermission, FeishuPermissionDenied
from aperag.tasks.async_interface import TaskResult, TaskScheduler
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS

logger = logging.getLogger(__name__)


class DocumentWorkflow:
    """Document workflow orchestrator for async tasks"""
    
    def __init__(self):
        self.logger = logger
    
    def add_document_index(self, document_id: int) -> TaskResult:
        """
        Add indexes for a document using declarative approach
        
        Args:
            document_id: Document ID to index
            
        Returns:
            TaskResult: Result of the index declaration creation
        """
        try:
            
            # Get document and collection
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")
            
            collection = db_ops.query_collection_by_id(document.collection_id)
            if not collection:
                raise Exception(f"Collection {document.collection_id} not found")
            
            # Set document status to indexing
            document.status = DocumentStatus.INDEXING
            db_ops.update_document(document)
            
            # Load document metadata for compressed file check
            metadata = json.loads(document.doc_metadata or "{}")
            metadata["doc_id"] = document_id
            supported_file_extensions = DocParser().supported_extensions()
            supported_file_extensions += SUPPORTED_COMPRESSED_EXTENSIONS
            
            # Check if file is compressed - handle separately for now
            if document.object_path and Path(document.object_path).suffix in SUPPORTED_COMPRESSED_EXTENSIONS:
                config = parseCollectionConfig(collection.config)
                if config.source != "system":
                    return TaskResult(
                        success=False,
                        error="Compressed files only supported for system source"
                    )
                
                # Handle compressed file - fall back to direct processing for now
                # TODO: Add compressed file support to declarative indexing
                scheduler = TaskScheduler()
                result = document_parser.handle_compressed_file(
                    document, supported_file_extensions, scheduler
                )
                return result
            
            # Create index specifications using the new K8s-inspired system
            try:
                # Import here to avoid circular dependencies
                import asyncio
                from aperag.document_index_manager import document_index_manager
                from aperag.db.ops import get_session
                
                # Create index specs asynchronously
                async def create_specs():
                    async with get_session() as session:
                        await document_index_manager.create_document_indexes(
                            session, str(document_id), "system"
                        )
                        await session.commit()
                
                asyncio.run(create_specs())
                self.logger.info(f"Created index specifications for document {document_id}. "
                                f"Reconciliation controller will process the actual indexing.")
                return TaskResult(success=True)
            except Exception as e:
                # Update document status to failed
                document.status = DocumentStatus.FAILED
                db_ops.update_document(document)
                self.logger.error(f"Failed to create index specifications for document {document_id}: {str(e)}")
                return TaskResult(success=False, error=str(e))
                
        except FeishuNoPermission:
            error_msg = f"No permission to access document {document.name}"
            self.logger.error(error_msg)
            return TaskResult(success=False, error=error_msg)
        except FeishuPermissionDenied:
            error_msg = f"Permission denied to access document {document.name}"
            self.logger.error(error_msg)
            return TaskResult(success=False, error=error_msg)
        except Exception as e:
            error_msg = f"Error creating index declarations for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            # Update document status to failed
            document = db_ops.query_document_by_id(document_id)
            if document:
                document.status = DocumentStatus.FAILED
                db_ops.update_document(document)
            return TaskResult(success=False, error=error_msg)
    
    def update_document_index(self, document_id: int) -> TaskResult:
        """
        Update indexes for a document
        
        Args:
            document_id: Document ID to update
            
        Returns:
            TaskResult: Result of the update operation
        """
        try:
            # Get document and collection
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")
            
            collection = db_ops.query_collection_by_id(document.collection_id)
            if not collection:
                raise Exception(f"Collection {document.collection_id} not found")
            
            # Set document status to running
            document.status = DocumentStatus.RUNNING
            db_ops.update_document(document)
            
            # Prepare document
            source = get_source(parseCollectionConfig(collection.config))
            metadata = json.loads(document.doc_metadata or "{}")
            metadata["doc_id"] = document_id
            local_doc = source.prepare_document(name=document.name, metadata=metadata)
            
            try:
                # Parse document
                parsing_result = document_parser.process_document_parsing(
                    local_doc.path,
                    local_doc.metadata,
                    document.object_store_base_path()
                )
                
                # Update indexes using index manager
                index_result = index_manager.update_all_indexes(
                    document_id=document_id,
                    content=parsing_result.content,
                    doc_parts=parsing_result.doc_parts,
                    collection=collection,
                    file_path=local_doc.path
                )
                
                if not index_result.success:
                    raise Exception(index_result.error)
                
                self.logger.info(f"Document {document_id} updated successfully")
                
                return index_result
                
            finally:
                # Cleanup local document
                source.cleanup_document(local_doc.path)
                
        except FeishuNoPermission:
            error_msg = f"No permission to access document {document.name}"
            self.logger.error(error_msg)
            return TaskResult(success=False, error=error_msg)
        except FeishuPermissionDenied:
            error_msg = f"Permission denied to access document {document.name}"
            self.logger.error(error_msg)
            return TaskResult(success=False, error=error_msg)
        except Exception as e:
            error_msg = f"Error updating document {document.name}: {str(e)}"
            self.logger.error(error_msg)
            return TaskResult(success=False, error=error_msg)
    
    def remove_document_index(self, document_id: int) -> TaskResult:
        """
        Remove indexes for a document using declarative approach
        
        Args:
            document_id: Document ID to remove
            
        Returns:
            TaskResult: Result of the removal declaration
        """
        try:
            # Mark document indexes for deletion using the new K8s-inspired system
            try:
                # Import here to avoid circular dependencies
                import asyncio
                from aperag.document_index_manager import document_index_manager
                from aperag.db.ops import get_session
                
                # Delete index specs asynchronously
                async def delete_specs():
                    async with get_session() as session:
                        await document_index_manager.delete_document_indexes(
                            session, str(document_id)
                        )
                        await session.commit()
                
                asyncio.run(delete_specs())
                self.logger.info(f"Marked document {document_id} indexes for deletion. "
                                f"Reconciliation controller will handle the actual cleanup.")
                return TaskResult(success=True)
            except Exception as e:
                self.logger.error(f"Failed to mark document {document_id} indexes for deletion: {str(e)}")
                return TaskResult(success=False, error=str(e))
            
        except Exception as e:
            error_msg = f"Error marking document {document_id} indexes as deleted: {str(e)}"
            self.logger.error(error_msg)
            return TaskResult(success=False, error=error_msg)
    
    def add_lightrag_index(self, content: str, document_id: int, file_path: str) -> TaskResult:
        """
        Add LightRAG index for a document
        
        Args:
            content: Document content
            document_id: Document ID
            file_path: File path for logging
            
        Returns:
            TaskResult: Result of the LightRAG indexing
        """
        try:
            # Get document and check status
            document = db_ops.query_document_by_id(document_id)
            if not document:
                return TaskResult(
                    success=False,
                    error=f"Document {document_id} not found"
                )
            
            if document.status == DocumentStatus.DELETED:
                return TaskResult(
                    success=True,
                    metadata={"message": "Document deleted, skipping LightRAG indexing"}
                )
            
            # Get collection and check status
            collection = db_ops.query_collection_by_id(document.collection_id)
            if not collection:
                return TaskResult(
                    success=False,
                    error=f"Collection {document.collection_id} not found"
                )
            
            # Use the stateless wrapper for document processing
            from aperag.graph.lightrag_manager import process_document_for_celery
            
            result = process_document_for_celery(
                collection=collection,
                content=content,
                doc_id=str(document_id),
                file_path=file_path
            )
            
            # Process result
            if result.get("status") == "success":
                self.logger.info(
                    f"LightRAG indexing completed for document {document_id}: "
                    f"Chunks: {result.get('chunks_created', 0)}, "
                    f"Entities: {result.get('entities_extracted', 0)}, "
                    f"Relations: {result.get('relations_extracted', 0)}"
                )
                return TaskResult(
                    success=True,
                    data=result,
                    metadata={"operation": "lightrag_index_created"}
                )
            elif result.get("status") == "warning":
                self.logger.warning(
                    f"LightRAG indexing completed with warnings for document {document_id}: "
                    f"{result.get('message', 'Unknown warning')}"
                )
                return TaskResult(
                    success=True,
                    data=result,
                    metadata={"operation": "lightrag_index_created_with_warnings"}
                )
            else:
                error_msg = f"LightRAG indexing failed: {result.get('message', 'Unknown error')}"
                self.logger.error(f"Document {document_id}: {error_msg}")
                return TaskResult(success=False, error=error_msg)
                
        except Exception as e:
            error_msg = f"LightRAG indexing failed for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            return TaskResult(success=False, error=error_msg)
    
    def remove_lightrag_index(self, document_id: int, collection_id: str) -> TaskResult:
        """
        Remove LightRAG index for a document
        
        Args:
            document_id: Document ID
            collection_id: Collection ID
            
        Returns:
            TaskResult: Result of the LightRAG removal
        """
        try:
            # Get collection
            collection = db_ops.query_collection_by_id(collection_id)
            if not collection:
                return TaskResult(
                    success=False,
                    error=f"Collection {collection_id} not found"
                )
            
            # Use the stateless wrapper for document deletion
            from aperag.graph.lightrag_manager import delete_document_for_celery
            
            result = delete_document_for_celery(
                collection=collection,
                doc_id=str(document_id)
            )
            
            if result.get("status") == "success":
                self.logger.info(f"LightRAG deletion completed for document {document_id}")
                return TaskResult(
                    success=True,
                    data=result,
                    metadata={"operation": "lightrag_index_deleted"}
                )
            else:
                error_msg = f"LightRAG deletion failed: {result.get('message', 'Unknown error')}"
                self.logger.error(f"Document {document_id}: {error_msg}")
                return TaskResult(success=False, error=error_msg)
                
        except Exception as e:
            error_msg = f"LightRAG deletion failed for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            return TaskResult(success=False, error=error_msg)


# Global workflow instance
document_workflow = DocumentWorkflow() 