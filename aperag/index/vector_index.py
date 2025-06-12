import logging
from aperag.config import settings
from aperag.db.ops import db_ops
from aperag.embed.base_embedding import get_collection_embedding_service_sync
from aperag.embed.embedding_utils import create_embeddings_and_store
from aperag.index.base import BaseIndexer, IndexResult, IndexType
from aperag.utils.tokenizer import get_default_tokenizer
from aperag.utils.utils import generate_vector_db_collection_name
from config.vector_db import get_vector_db_connector


import json
from typing import Any, List

logger = logging.getLogger(__name__)


class VectorIndexer(BaseIndexer):
    """Vector index implementation"""

    def __init__(self):
        super().__init__(IndexType.VECTOR)
        self.logger = logger

    def is_enabled(self, collection) -> bool:
        """Vector indexing is always enabled"""
        return True

    def create_index(self, document_id: int, content: str, doc_parts: List[Any],
                    collection, **kwargs) -> IndexResult:
        """
        Create vector index for document

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of vector index creation
        """
        try:
            # Get embedding model and create embeddings
            embedding_model, vector_size = get_collection_embedding_service_sync(collection)
            vector_store_adaptor = get_vector_db_connector(
                collection=generate_vector_db_collection_name(collection_id=collection.id)
            )

            # Generate embeddings and store in vector database
            ctx_ids = create_embeddings_and_store(
                parts=doc_parts,
                vector_store_adaptor=vector_store_adaptor,
                embedding_model=embedding_model,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap_size,
                tokenizer=get_default_tokenizer(),
            )

            # Update document with related IDs
            document = db_ops.query_document_by_id(document_id)
            if document:
                relate_ids = json.loads(document.relate_ids) if document.relate_ids else {}
                relate_ids["ctx"] = ctx_ids
                document.relate_ids = json.dumps(relate_ids)
                db_ops.update_document(document)

            self.logger.info(f"Vector index created for document {document_id}: {len(ctx_ids)} vectors")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"context_ids": ctx_ids},
                metadata={
                    "vector_count": len(ctx_ids),
                    "vector_size": vector_size,
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap_size
                }
            )

        except Exception as e:
            self.logger.error(f"Vector index creation failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False,
                index_type=self.index_type,
                error=f"Vector index creation failed: {str(e)}"
            )

    def update_index(self, document_id: int, content: str, doc_parts: List[Any],
                    collection, **kwargs) -> IndexResult:
        """
        Update vector index for document

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of vector index update
        """
        try:
            # Get document and existing relate_ids
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")

            relate_ids = json.loads(document.relate_ids) if document.relate_ids else {}
            old_ctx_ids = relate_ids.get("ctx", [])

            # Get vector store adaptor
            vector_store_adaptor = get_vector_db_connector(
                collection=generate_vector_db_collection_name(collection_id=collection.id)
            )

            # Delete old vectors
            if old_ctx_ids:
                vector_store_adaptor.connector.delete(ids=old_ctx_ids)
                self.logger.info(f"Deleted {len(old_ctx_ids)} old vectors for document {document_id}")

            # Create new vectors
            embedding_model, vector_size = get_collection_embedding_service_sync(collection)
            ctx_ids = create_embeddings_and_store(
                parts=doc_parts,
                vector_store_adaptor=vector_store_adaptor,
                embedding_model=embedding_model,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap_size,
                tokenizer=get_default_tokenizer(),
            )

            # Update relate_ids
            relate_ids["ctx"] = ctx_ids
            document.relate_ids = json.dumps(relate_ids)
            db_ops.update_document(document)

            self.logger.info(f"Vector index updated for document {document_id}: {len(ctx_ids)} vectors")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"context_ids": ctx_ids},
                metadata={
                    "vector_count": len(ctx_ids),
                    "old_vector_count": len(old_ctx_ids),
                    "vector_size": vector_size
                }
            )

        except Exception as e:
            self.logger.error(f"Vector index update failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False,
                index_type=self.index_type,
                error=f"Vector index update failed: {str(e)}"
            )

    def delete_index(self, document_id: int, collection, **kwargs) -> IndexResult:
        """
        Delete vector index for document

        Args:
            document_id: Document ID
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of vector index deletion
        """
        try:
            # Get document and relate_ids
            document = db_ops.query_document_by_id(document_id)
            if not document or not document.relate_ids:
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "No vector index to delete"}
                )

            relate_ids = json.loads(document.relate_ids)
            ctx_ids = relate_ids.get("ctx", [])

            if not ctx_ids:
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "No context IDs to delete"}
                )

            # Delete vectors from vector database
            vector_db = get_vector_db_connector(
                collection=generate_vector_db_collection_name(collection_id=collection.id)
            )
            vector_db.connector.delete(ids=ctx_ids)

            self.logger.info(f"Deleted {len(ctx_ids)} vectors for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"deleted_context_ids": ctx_ids},
                metadata={"deleted_vector_count": len(ctx_ids)}
            )

        except Exception as e:
            self.logger.error(f"Vector index deletion failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False,
                index_type=self.index_type,
                error=f"Vector index deletion failed: {str(e)}"
            )


# Global instance
vector_indexer = VectorIndexer()