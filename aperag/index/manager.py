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
from typing import List, Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import DocumentIndex, DocumentIndexStatus, DocumentIndexType, utc_now

logger = logging.getLogger(__name__)


class DocumentIndexManager:
    """Simple manager for document index specs (frontend chain)"""

    async def create_or_update_document_indexes(
        self, session: AsyncSession, document_id: str, index_types: Optional[List[DocumentIndexType]] = None
    ):
        """
        Create or update index records for a document (called when document is created or index isupdated)

        Args:
            session: Database session
            document_id: Document ID
            index_types: List of index types to create (defaults to all)
        """
        if index_types is None:
            index_types = [
                DocumentIndexType.VECTOR,
                DocumentIndexType.FULLTEXT,
                DocumentIndexType.GRAPH,
                DocumentIndexType.SUMMARY,
            ]

        for index_type in index_types:
            # Check if index already exists
            stmt = select(DocumentIndex).where(
                and_(DocumentIndex.document_id == document_id, DocumentIndex.index_type == index_type)
            )
            result = await session.execute(stmt)
            existing_index = result.scalar_one_or_none()

            if existing_index:
                # Update existing index to pending and increment version
                existing_index.status = DocumentIndexStatus.PENDING
                existing_index.update_version()
                logger.debug(f"Updated index for {document_id}:{index_type.value} to version {existing_index.version}")
            else:
                # Create new index
                doc_index = DocumentIndex(
                    document_id=document_id,
                    index_type=index_type,
                    status=DocumentIndexStatus.PENDING,
                    version=1,
                    observed_version=0,
                )
                session.add(doc_index)
                logger.debug(f"Created new index for {document_id}:{index_type.value}")

    async def delete_document_indexes(
        self, session: AsyncSession, document_id: str, index_types: Optional[List[DocumentIndexType]] = None
    ):
        """
        Delete document indexes (called when document is deleted)

        Args:
            session: Database session
            document_id: Document ID
            index_types: List of index types to delete (defaults to all)
        """
        if index_types is None:
            index_types = [DocumentIndexType.VECTOR, DocumentIndexType.FULLTEXT, DocumentIndexType.GRAPH]

        for index_type in index_types:
            stmt = select(DocumentIndex).where(
                and_(DocumentIndex.document_id == document_id, DocumentIndex.index_type == index_type)
            )
            result = await session.execute(stmt)
            doc_index = result.scalar_one_or_none()

            if doc_index:
                # Mark for deletion
                doc_index.status = DocumentIndexStatus.DELETING
                doc_index.gmt_updated = utc_now()
                logger.debug(f"Marked index {document_id}:{index_type.value} for deletion")


# Global instance
document_index_manager = DocumentIndexManager()
