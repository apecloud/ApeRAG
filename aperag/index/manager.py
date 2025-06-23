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

from aperag.db.models import DocumentIndex, DocumentIndexType, IndexActualState, IndexDesiredState, utc_now

logger = logging.getLogger(__name__)


class FrontendIndexManager:
    """Simple manager for document index specs (frontend chain)"""

    async def create_document_indexes(
        self, session: AsyncSession, document_id: str, user: str, index_types: Optional[List[DocumentIndexType]] = None
    ):
        """
        Create index specs for a document (called when document is created)

        Args:
            session: Database session
            document_id: Document ID
            user: User creating the document
            index_types: List of index types to create (defaults to all)
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
                # Update existing index
                existing_index.update_spec(IndexDesiredState.PRESENT, user)
                logger.debug(f"Updated index for {document_id}:{index_type} to version {existing_index.version}")
            else:
                # Create new index
                doc_index = DocumentIndex(
                    document_id=document_id,
                    index_type=index_type,
                    desired_state=IndexDesiredState.PRESENT,
                    version=1,
                    created_by=user,
                )
                session.add(doc_index)

    async def update_document_indexes(self, session: AsyncSession, document_id: str):
        """
        Update document indexes (called when document content is updated)

        This increments the version of all indexes to trigger reconciliation.

        Args:
            session: Database session
            document_id: Document ID
        """
        stmt = select(DocumentIndex).where(DocumentIndex.document_id == document_id)
        result = await session.execute(stmt)
        indexes = result.scalars().all()

        for index in indexes:
            if index.desired_state == IndexDesiredState.PRESENT:
                index.version += 1  # Increment version to trigger re-indexing
                index.gmt_updated = utc_now()

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
                doc_index.update_spec(IndexDesiredState.ABSENT)

    async def rebuild_document_indexes(
        self, session: AsyncSession, document_id: str, index_types: List[DocumentIndexType]
    ):
        """
        Rebuild specified document indexes (called when user requests index rebuild)

        This increments the version of specified indexes to trigger reconciliation.

        Args:
            session: Database session
            document_id: Document ID
            index_types: List of index types to rebuild
        """
        if len(set(index_types)) != len(index_types):
            raise Exception("Duplicate index types are not allowed")

        for index_type in index_types:
            stmt = select(DocumentIndex).where(
                and_(DocumentIndex.document_id == document_id, DocumentIndex.index_type == index_type)
            )
            result = await session.execute(stmt)
            doc_index = result.scalar_one_or_none()

            if doc_index:
                # Only rebuild if the index is present or failed
                if doc_index.desired_state == IndexDesiredState.PRESENT:
                    doc_index.version += 1  # Increment version to trigger re-indexing
                    doc_index.gmt_updated = utc_now()
                    logger.info(f"Triggered rebuild for {index_type.value} index of document {document_id}")
                else:
                    logger.warning(
                        f"Cannot rebuild {index_type.value} index for document {document_id}: index not present"
                    )
            else:
                logger.warning(f"No {index_type.value} index found for document {document_id}")

    async def get_document_index_status(self, session: AsyncSession, document_id: str) -> dict:
        """
        Get current index status for a document

        Args:
            session: Database session
            document_id: Document ID

        Returns:
            Dictionary with index status information
        """
        # Get all indexes for the document
        stmt = select(DocumentIndex).where(DocumentIndex.document_id == document_id)
        result = await session.execute(stmt)
        indexes = result.scalars().all()

        # Build result
        result = {"document_id": document_id, "indexes": {}, "overall_status": "complete"}

        has_creating = False
        has_failed = False

        for index in indexes:
            index_info = {
                "type": index.index_type,
                "desired_state": index.desired_state,
                "actual_state": index.actual_state,
                "in_sync": index.is_in_sync(),
            }

            if index.actual_state == IndexActualState.CREATING:
                has_creating = True
            elif index.actual_state == IndexActualState.FAILED:
                has_failed = True
                index_info["error"] = index.error_message

            result["indexes"][index.index_type] = index_info

        # Determine overall status
        if has_failed:
            result["overall_status"] = "failed"
        elif has_creating:
            result["overall_status"] = "running"
        else:
            result["overall_status"] = "complete"

        return result


# Global instance
document_index_manager = FrontendIndexManager()
