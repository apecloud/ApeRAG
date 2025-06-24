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
from typing import List, Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import DocumentIndex, DocumentIndexType, DocumentIndexStatus, utc_now

logger = logging.getLogger(__name__)


class DocumentIndexManager:
    """Manager for document index lifecycle using single status model"""

    async def create_document_indexes(
        self, session: AsyncSession, document_id: str, user: str, index_types: Optional[List[DocumentIndexType]] = None
    ):
        """
        Create index records for a document (called when document is created)

        Args:
            session: Database session
            document_id: Document ID
            user: User creating the document
            index_types: List of index types to create (defaults to vector and fulltext)
        """
        if index_types is None:
            index_types = [DocumentIndexType.VECTOR, DocumentIndexType.FULLTEXT, DocumentIndexType.GRAPH]

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
                existing_index.update_version(user)
                logger.debug(f"Updated index for {document_id}:{index_type.value} to version {existing_index.version}")
            else:
                # Create new index
                doc_index = DocumentIndex(
                    document_id=document_id,
                    index_type=index_type,
                    status=DocumentIndexStatus.PENDING,
                    version=1,
                    observed_version=0,
                    created_by=user,
                )
                session.add(doc_index)
                logger.debug(f"Created new index for {document_id}:{index_type.value}")

    async def update_document_indexes(self, session: AsyncSession, document_id: str, user: str = None):
        """
        Update document indexes (called when document content is updated)

        This increments the version of all active indexes to trigger reconciliation.

        Args:
            session: Database session
            document_id: Document ID
            user: User triggering the update (optional)
        """
        stmt = select(DocumentIndex).where(DocumentIndex.document_id == document_id)
        result = await session.execute(stmt)
        indexes = result.scalars().all()

        for index in indexes:
            # Only update active indexes, failed indexes can be manually rebuilt
            if index.status in [DocumentIndexStatus.ACTIVE, DocumentIndexStatus.FAILED]:
                index.status = DocumentIndexStatus.PENDING
                index.update_version(user)
                logger.debug(f"Updated index {document_id}:{index.index_type} to version {index.version}")

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

    async def rebuild_document_indexes(
        self, session: AsyncSession, document_id: str, index_types: List[DocumentIndexType], user: str = None
    ):
        """
        Rebuild specified document indexes (called when user requests index rebuild)

        This increments the version of specified indexes to trigger reconciliation.

        Args:
            session: Database session
            document_id: Document ID
            index_types: List of index types to rebuild
            user: User triggering the rebuild (optional)
        """
        if len(set(index_types)) != len(index_types):
            raise ValueError("Duplicate index types are not allowed")

        for index_type in index_types:
            stmt = select(DocumentIndex).where(
                and_(DocumentIndex.document_id == document_id, DocumentIndex.index_type == index_type)
            )
            result = await session.execute(stmt)
            doc_index = result.scalar_one_or_none()

            if doc_index:
                # Reset to pending and increment version to trigger rebuild
                doc_index.status = DocumentIndexStatus.PENDING
                doc_index.update_version(user)
                doc_index.error_message = None  # Clear any previous error
                logger.info(f"Triggered rebuild for {index_type.value} index of document {document_id} (v{doc_index.version})")
            else:
                # Create new index if it doesn't exist
                doc_index = DocumentIndex(
                    document_id=document_id,
                    index_type=index_type,
                    status=DocumentIndexStatus.PENDING,
                    version=1,
                    observed_version=0,
                    created_by=user or "system",
                )
                session.add(doc_index)
                logger.info(f"Created new {index_type.value} index for document {document_id}")

# Global instance
document_index_manager = DocumentIndexManager()
