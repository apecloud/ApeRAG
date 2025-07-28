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
from typing import Any, List

from aperag.config import get_vector_db_connector
from aperag.db.ops import db_ops
from aperag.docparser.base import TextPart
from aperag.index.base import BaseIndexer, IndexResult, IndexType
from aperag.llm.completion.base_completion import get_collection_completion_service_sync
from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
from aperag.llm.embed.embedding_utils import create_embeddings_and_store
from aperag.llm.llm_error_types import CompletionError, InvalidConfigurationError
from aperag.utils.utils import generate_vector_db_collection_name

logger = logging.getLogger(__name__)


class SummaryIndexer(BaseIndexer):
    """Summary index implementation using map-reduce strategy"""

    def __init__(self):
        super().__init__(IndexType.SUMMARY)

    def is_enabled(self, collection) -> bool:
        """Summary indexing is enabled by default if completion service is configured"""
        try:
            get_collection_completion_service_sync(collection)
            return True
        except (InvalidConfigurationError, CompletionError):
            return False

    def create_index(self, document_id: str, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """
        Create summary index for document using map-reduce strategy

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of summary index creation
        """
        try:
            # Check if summary indexing is enabled
            if not self.is_enabled(collection):
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "Summary indexing disabled", "status": "skipped"},
                )

            # Get document for name
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")

            # Generate summary using map-reduce strategy
            summary = self._generate_document_summary(content, doc_parts, collection)

            if not summary:
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "Empty summary generated", "status": "skipped"},
                )

            # Vectorize and store summary in vector database
            summary_ctx_ids = []
            try:
                # Get embedding model and vector store
                embedding_model, vector_size = get_collection_embedding_service_sync(collection)
                vector_store_adaptor = get_vector_db_connector(
                    collection=generate_vector_db_collection_name(collection_id=collection.id)
                )

                # Create a TextPart for the summary
                summary_part = TextPart(
                    content=summary,
                    metadata={
                        "document_id": document_id,
                        "document_name": document.name,
                        "name": f"{document.name} - Summary",
                        "indexer": "summary",
                        "index_method": "summary",
                        "collection_id": collection.id,
                        "content_type": "summary",
                    }
                )

                # Store summary vector in vector database
                summary_ctx_ids = create_embeddings_and_store(
                    parts=[summary_part],
                    vector_store_adaptor=vector_store_adaptor,
                    embedding_model=embedding_model,
                )

                logger.info(f"Summary vectorized and stored for document {document_id}: {len(summary_ctx_ids)} vectors")

            except Exception as e:
                logger.warning(f"Failed to vectorize summary for document {document_id}: {str(e)}")
                # Continue without failing the entire summary indexing process

            # Store summary data
            summary_data = {
                "summary": summary,
                "document_name": document.name,
                "chunk_count": len(doc_parts) if doc_parts else 0,
                "content_length": len(content) if content else 0,
                "summary_context_ids": summary_ctx_ids,
            }

            logger.info(f"Summary index created for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data=summary_data,
                metadata={
                    "summary_length": len(summary),
                    "chunk_count": len(doc_parts) if doc_parts else 0,
                    "content_length": len(content) if content else 0,
                    "summary_vector_count": len(summary_ctx_ids),
                },
            )

        except Exception as e:
            logger.error(f"Summary index creation failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Summary index creation failed: {str(e)}"
            )

    def update_index(self, document_id: str, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """
        Update summary index for document

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of summary index update
        """
        try:
            # Get existing summary index data from DocumentIndex to find old vector IDs
            from sqlalchemy import and_, select

            from aperag.config import get_sync_session
            from aperag.db.models import DocumentIndex, DocumentIndexType

            old_summary_ctx_ids = []
            for session in get_sync_session():
                stmt = select(DocumentIndex).where(
                    and_(
                        DocumentIndex.document_id == document_id, DocumentIndex.index_type == DocumentIndexType.SUMMARY
                    )
                )
                result = session.execute(stmt)
                doc_index = result.scalar_one_or_none()

                if doc_index and doc_index.index_data:
                    index_data = json.loads(doc_index.index_data)
                    old_summary_ctx_ids = index_data.get("summary_context_ids", [])

            # Delete old summary vectors from vector database if they exist
            if old_summary_ctx_ids:
                try:
                    vector_store_adaptor = get_vector_db_connector(
                        collection=generate_vector_db_collection_name(collection_id=collection.id)
                    )
                    vector_store_adaptor.connector.delete(ids=old_summary_ctx_ids)
                    logger.info(f"Deleted {len(old_summary_ctx_ids)} old summary vectors for document {document_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete old summary vectors for document {document_id}: {str(e)}")

            # Create new summary index (which includes vectorization)
            result = self.create_index(document_id, content, doc_parts, collection, **kwargs)
            
            # Update metadata to include old vector count
            if result.success and result.metadata:
                result.metadata["old_summary_vector_count"] = len(old_summary_ctx_ids)

            return result

        except Exception as e:
            logger.error(f"Summary index update failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Summary index update failed: {str(e)}"
            )

    def delete_index(self, document_id: str, collection, **kwargs) -> IndexResult:
        """
        Delete summary index for document

        Args:
            document_id: Document ID
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of summary index deletion
        """
        try:
            # Get existing summary index data from DocumentIndex to find vector IDs
            from sqlalchemy import and_, select

            from aperag.config import get_sync_session
            from aperag.db.models import DocumentIndex, DocumentIndexType

            summary_ctx_ids = []
            for session in get_sync_session():
                stmt = select(DocumentIndex).where(
                    and_(
                        DocumentIndex.document_id == document_id, DocumentIndex.index_type == DocumentIndexType.SUMMARY
                    )
                )
                result = session.execute(stmt)
                doc_index = result.scalar_one_or_none()

                if doc_index and doc_index.index_data:
                    index_data = json.loads(doc_index.index_data)
                    summary_ctx_ids = index_data.get("summary_context_ids", [])

            # Delete summary vectors from vector database if they exist
            if summary_ctx_ids:
                try:
                    vector_store_adaptor = get_vector_db_connector(
                        collection=generate_vector_db_collection_name(collection_id=collection.id)
                    )
                    vector_store_adaptor.connector.delete(ids=summary_ctx_ids)
                    logger.info(f"Deleted {len(summary_ctx_ids)} summary vectors for document {document_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete summary vectors for document {document_id}: {str(e)}")

            logger.info(f"Summary index deleted for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                metadata={
                    "operation": "deleted",
                    "deleted_vector_count": len(summary_ctx_ids),
                },
            )

        except Exception as e:
            logger.error(f"Summary index deletion failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Summary index deletion failed: {str(e)}"
            )

    def _generate_document_summary(self, content: str, doc_parts: List[Any], collection) -> str:
        """
        Generate document summary using map-reduce strategy

        Args:
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object

        Returns:
            str: Generated summary
        """
        try:
            completion_service = get_collection_completion_service_sync(collection)

            # Filter out non-text parts
            doc_parts = [part for part in doc_parts if hasattr(part, "content") and part.content]

            # If no doc_parts or content is short, summarize directly
            if not doc_parts or len(content) < 4000:
                return self._summarize_text(content, completion_service)

            # Map phase: summarize each chunk
            chunk_summaries = []
            for part in doc_parts:
                if hasattr(part, "content") and part.content:
                    chunk_text = part.content
                elif hasattr(part, "text") and part.text:
                    chunk_text = part.text
                else:
                    # If part is a dict or other format, try to extract text
                    chunk_text = str(part)

                if chunk_text.strip():
                    chunk_summary = self._summarize_text(chunk_text, completion_service, is_chunk=True)
                    if chunk_summary:
                        chunk_summaries.append(chunk_summary)

            # If we have chunk summaries, reduce them
            if chunk_summaries:
                # Combine chunk summaries
                combined_summaries = "\n\n".join(chunk_summaries)

                # Reduce phase: create final summary from chunk summaries
                return self._reduce_summaries(combined_summaries, completion_service)
            else:
                # Fallback to direct summarization
                return self._summarize_text(content, completion_service)

        except Exception as e:
            logger.error(f"Failed to generate document summary: {str(e)}")
            return ""

    def _summarize_text(self, text: str, completion_service, is_chunk: bool = False) -> str:
        """
        Summarize a single text using LLM

        Args:
            text: Text to summarize
            completion_service: Completion service instance
            is_chunk: Whether this is a chunk summary (affects prompt)

        Returns:
            str: Generated summary
        """
        try:
            if not text.strip():
                return ""

            # Create appropriate prompt based on whether it's a chunk or full document
            if is_chunk:
                prompt = f"""Summarize this text chunk concisely. Requirements:
1. Use the same language as the original text for the summary
2. Keep it within 1-2 sentences
3. Extract only the most important core information
4. Stay objective and accurate, do not add content not present in the original text
5. Output ONLY the summary content, no additional text, explanations, or formatting

Text content:
{text}

Summary:"""
            else:
                prompt = f"""Generate a concise summary of this document. Requirements:
1. Use the same language as the original text for the summary
2. Keep it within 2-3 sentences
3. Summarize the main topic and key insights of the document
4. Stay objective and accurate, do not add content not present in the original text
5. If it's a technical document, highlight the technical points
6. Output ONLY the summary content, no additional text, explanations, or formatting

Document content:
{text}

Summary:"""

            # Generate summary
            summary = completion_service.generate(history=[], prompt=prompt)
            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to summarize text: {str(e)}")
            return ""

    def _reduce_summaries(self, combined_summaries: str, completion_service) -> str:
        """
        Reduce multiple chunk summaries into a final document summary

        Args:
            combined_summaries: Combined chunk summaries
            completion_service: Completion service instance

        Returns:
            str: Final document summary
        """
        try:
            prompt = f"""Combine these section summaries into a comprehensive final document summary. Requirements:
1. Use the same language as the original summaries for the final summary
2. Keep it within 3-4 sentences
3. Integrate the core content from all sections into a coherent overall summary
4. Highlight the main topic and most important insights of the document
5. Maintain logical clarity and avoid repetitive content
6. If technical content is involved, maintain accuracy of technical terminology
7. Output ONLY the final summary content, no additional text, explanations, or formatting

Section summaries:
{combined_summaries}

Final summary:"""

            # Generate final summary
            final_summary = completion_service.generate(history=[], prompt=prompt)
            return final_summary.strip()

        except Exception as e:
            logger.error(f"Failed to reduce summaries: {str(e)}")
            return ""

    def get_document_summary(self, document_id: str) -> str:
        """
        Get the summary for a document from the index

        Args:
            document_id: Document ID

        Returns:
            str: Document summary or empty string if not found
        """
        try:
            from sqlalchemy import and_, select

            from aperag.config import get_sync_session
            from aperag.db.models import DocumentIndex, DocumentIndexType

            for session in get_sync_session():
                stmt = select(DocumentIndex).where(
                    and_(
                        DocumentIndex.document_id == document_id, DocumentIndex.index_type == DocumentIndexType.SUMMARY
                    )
                )
                result = session.execute(stmt)
                doc_index = result.scalar_one_or_none()

                if doc_index and doc_index.index_data:
                    index_data = json.loads(doc_index.index_data)
                    return index_data.get("summary", "")

            return ""

        except Exception as e:
            logger.error(f"Failed to get document summary for {document_id}: {str(e)}")
            return ""


# Global instance
summary_indexer = SummaryIndexer()
