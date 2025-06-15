import json
import logging
from typing import Any, List

from sqlalchemy import and_, select

from aperag.config import settings

# NOTE: This file has been updated to use DocumentIndex table instead of Document.relate_ids field
# The relate_ids field has been removed from Document model and replaced with index_data in DocumentIndex
from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
from aperag.llm.embed.embedding_utils import create_embeddings_and_store
from aperag.index.base import BaseIndexer, IndexResult, IndexType
from aperag.utils.tokenizer import get_default_tokenizer
from aperag.utils.utils import generate_vector_db_collection_name
from config.vector_db import get_vector_db_connector

logger = logging.getLogger(__name__)


class VectorIndexer(BaseIndexer):
    """Vector index implementation"""

    def __init__(self):
        super().__init__(IndexType.VECTOR)

    def is_enabled(self, collection) -> bool:
        """Vector indexing is always enabled"""
        return True

    def create_index(self, document_id: str, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
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

            # Update DocumentIndex with vector IDs
            from aperag.config import get_sync_session
            from aperag.db.models import DocumentIndex, DocumentIndexType

            # Use sync session to update DocumentIndex
            for session in get_sync_session():
                stmt = select(DocumentIndex).where(
                    and_(DocumentIndex.document_id == document_id, DocumentIndex.index_type == DocumentIndexType.VECTOR)
                )
                result = session.execute(stmt)
                doc_index = result.scalar_one_or_none()

                if doc_index:
                    index_data = {"ctx": ctx_ids}
                    doc_index.index_data = json.dumps(index_data)
                    session.commit()

            logger.info(f"Vector index created for document {document_id}: {len(ctx_ids)} vectors")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"context_ids": ctx_ids},
                metadata={
                    "vector_count": len(ctx_ids),
                    "vector_size": vector_size,
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap_size,
                },
            )

        except Exception as e:
            logger.error(f"Vector index creation failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Vector index creation failed: {str(e)}"
            )

    def update_index(self, document_id: str, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
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
            # Get existing vector index data from DocumentIndex
            from aperag.config import get_sync_session
            from aperag.db.models import DocumentIndex, DocumentIndexType

            old_ctx_ids = []
            doc_index = None
            for session in get_sync_session():
                stmt = select(DocumentIndex).where(
                    and_(DocumentIndex.document_id == document_id, DocumentIndex.index_type == DocumentIndexType.VECTOR)
                )
                result = session.execute(stmt)
                doc_index = result.scalar_one_or_none()

                if doc_index and doc_index.index_data:
                    index_data = json.loads(doc_index.index_data)
                    old_ctx_ids = index_data.get("ctx", [])

            # Get vector store adaptor
            vector_store_adaptor = get_vector_db_connector(
                collection=generate_vector_db_collection_name(collection_id=collection.id)
            )

            # Delete old vectors
            if old_ctx_ids:
                vector_store_adaptor.connector.delete(ids=old_ctx_ids)
                logger.info(f"Deleted {len(old_ctx_ids)} old vectors for document {document_id}")

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

            # Update DocumentIndex with new vector IDs
            if doc_index:
                index_data = {"ctx": ctx_ids}
                doc_index.index_data = json.dumps(index_data)
                session.commit()

            logger.info(f"Vector index updated for document {document_id}: {len(ctx_ids)} vectors")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"context_ids": ctx_ids},
                metadata={
                    "vector_count": len(ctx_ids),
                    "old_vector_count": len(old_ctx_ids),
                    "vector_size": vector_size,
                },
            )

        except Exception as e:
            logger.error(f"Vector index update failed for document {document_id}: {str(e)}")
            return IndexResult(success=False, index_type=self.index_type, error=f"Vector index update failed: {str(e)}")

    def delete_index(self, document_id: str, collection, **kwargs) -> IndexResult:
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
            # Get existing vector index data from DocumentIndex
            from aperag.config import get_sync_session
            from aperag.db.models import DocumentIndex, DocumentIndexType

            ctx_ids = []
            for session in get_sync_session():
                stmt = select(DocumentIndex).where(
                    and_(DocumentIndex.document_id == document_id, DocumentIndex.index_type == DocumentIndexType.VECTOR)
                )
                result = session.execute(stmt)
                doc_index = result.scalar_one_or_none()

                if not doc_index or not doc_index.index_data:
                    return IndexResult(
                        success=True, index_type=self.index_type, metadata={"message": "No vector index to delete"}
                    )

                index_data = json.loads(doc_index.index_data)
                ctx_ids = index_data.get("ctx", [])

            if not ctx_ids:
                return IndexResult(
                    success=True, index_type=self.index_type, metadata={"message": "No context IDs to delete"}
                )

            # Delete vectors from vector database
            vector_db = get_vector_db_connector(
                collection=generate_vector_db_collection_name(collection_id=collection.id)
            )
            vector_db.connector.delete(ids=ctx_ids)

            logger.info(f"Deleted {len(ctx_ids)} vectors for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"deleted_context_ids": ctx_ids},
                metadata={"deleted_vector_count": len(ctx_ids)},
            )

        except Exception as e:
            logger.error(f"Vector index deletion failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Vector index deletion failed: {str(e)}"
            )


# Global instance
vector_indexer = VectorIndexer()
