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

from aperag.context.full_text import insert_document, remove_document
from aperag.db.ops import db_ops
from aperag.index.base import BaseIndexer, IndexResult, IndexType
from aperag.utils.utils import generate_fulltext_index_name

logger = logging.getLogger(__name__)


class FulltextIndexer(BaseIndexer):
    """Fulltext index implementation"""
    
    def __init__(self):
        super().__init__(IndexType.FULLTEXT)
        self.logger = logger
    
    def is_enabled(self, collection) -> bool:
        """Fulltext indexing is always enabled"""
        return True
    
    def create_index(self, document_id: int, content: str, doc_parts: List[Any], 
                    collection, **kwargs) -> IndexResult:
        """
        Create fulltext index for document
        
        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters
            
        Returns:
            IndexResult: Result of fulltext index creation
        """
        try:
            # Only create fulltext index when there is content
            if not content or not content.strip():
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "No content to index", "status": "skipped"}
                )
            
            # Get document for name
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")
            
            # Insert into fulltext index
            index_name = generate_fulltext_index_name(collection.id)
            insert_document(index_name, document_id, document.name, content)
            
            self.logger.info(f"Fulltext index created for document {document_id}")
            
            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"index_name": index_name, "document_name": document.name},
                metadata={
                    "content_length": len(content),
                    "content_words": len(content.split()) if content else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Fulltext index creation failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False,
                index_type=self.index_type,
                error=f"Fulltext index creation failed: {str(e)}"
            )
    
    def update_index(self, document_id: int, content: str, doc_parts: List[Any],
                    collection, **kwargs) -> IndexResult:
        """
        Update fulltext index for document
        
        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters
            
        Returns:
            IndexResult: Result of fulltext index update
        """
        try:
            # Get document for name
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")
            
            index_name = generate_fulltext_index_name(collection.id)
            
            # Remove old index
            try:
                remove_document(index_name, document_id)
                self.logger.debug(f"Removed old fulltext index for document {document_id}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old fulltext index for document {document_id}: {str(e)}")
            
            # Create new index if there is content
            if content and content.strip():
                insert_document(index_name, document_id, document.name, content)
                self.logger.info(f"Fulltext index updated for document {document_id}")
                
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    data={"index_name": index_name, "document_name": document.name},
                    metadata={
                        "content_length": len(content),
                        "content_words": len(content.split()),
                        "operation": "updated"
                    }
                )
            else:
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "No content to index", "status": "skipped"}
                )
            
        except Exception as e:
            self.logger.error(f"Fulltext index update failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False,
                index_type=self.index_type,
                error=f"Fulltext index update failed: {str(e)}"
            )
    
    def delete_index(self, document_id: int, collection, **kwargs) -> IndexResult:
        """
        Delete fulltext index for document
        
        Args:
            document_id: Document ID
            collection: Collection object
            **kwargs: Additional parameters
            
        Returns:
            IndexResult: Result of fulltext index deletion
        """
        try:
            index_name = generate_fulltext_index_name(collection.id)
            remove_document(index_name, document_id)
            
            self.logger.info(f"Fulltext index deleted for document {document_id}")
            
            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"index_name": index_name},
                metadata={"operation": "deleted"}
            )
            
        except Exception as e:
            self.logger.error(f"Fulltext index deletion failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False,
                index_type=self.index_type,
                error=f"Fulltext index deletion failed: {str(e)}"
            )


# Global instance
fulltext_indexer = FulltextIndexer() 