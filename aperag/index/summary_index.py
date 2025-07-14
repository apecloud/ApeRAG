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

from aperag.db.ops import db_ops
from aperag.index.base import BaseIndexer, IndexResult, IndexType
from aperag.llm.completion.base_completion import get_collection_completion_service_sync
from aperag.llm.llm_error_types import CompletionError, InvalidConfigurationError

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

            # Store summary data
            summary_data = {
                "summary": summary,
                "document_name": document.name,
                "chunk_count": len(doc_parts) if doc_parts else 0,
                "content_length": len(content) if content else 0,
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
        # For summary index, update is the same as create
        return self.create_index(document_id, content, doc_parts, collection, **kwargs)

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
            # For summary index, deletion is just removing the stored data
            # The actual data is stored in DocumentIndex.index_data
            logger.info(f"Summary index deleted for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                metadata={"operation": "deleted"},
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
                prompt = f"""Please provide a concise summary of the following text chunk. Focus on the key points and main ideas:

{text}

Summary:"""
            else:
                prompt = f"""Please provide a comprehensive summary of the following document. Include the main themes, key points, and important details:

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
            prompt = f"""The following are summaries of different sections of a document. Please create a comprehensive final summary that captures the main themes, key points, and important details from all sections:

{combined_summaries}

Please provide a well-structured final summary:"""

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
