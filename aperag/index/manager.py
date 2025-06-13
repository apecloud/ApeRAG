import logging
from typing import List, Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import DocumentIndexSpec, DocumentIndexStatus, DocumentIndexType, IndexDesiredState, utc_now

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
            # Check if spec already exists
            stmt = select(DocumentIndexSpec).where(
                and_(DocumentIndexSpec.document_id == document_id, DocumentIndexSpec.index_type == index_type)
            )
            result = await session.execute(stmt)
            existing_spec = result.scalar_one_or_none()

            if existing_spec:
                # Update existing spec
                existing_spec.desired_state = IndexDesiredState.PRESENT
                existing_spec.version += 1  # Increment version to trigger reconciliation
                existing_spec.gmt_updated = utc_now()
                logger.debug(f"Updated spec for {document_id}:{index_type} to version {existing_spec.version}")
            else:
                # Create new spec
                spec = DocumentIndexSpec(
                    document_id=document_id,
                    index_type=index_type,
                    desired_state=IndexDesiredState.PRESENT,
                    version=1,
                    created_by=user,
                )
                session.add(spec)
                logger.info(f"Created spec for {document_id}:{index_type}")

    async def update_document_indexes(self, session: AsyncSession, document_id: str):
        """
        Update document indexes (called when document content is updated)

        This increments the version of all specs to trigger reconciliation.

        Args:
            session: Database session
            document_id: Document ID
        """
        stmt = select(DocumentIndexSpec).where(DocumentIndexSpec.document_id == document_id)
        result = await session.execute(stmt)
        specs = result.scalars().all()

        for spec in specs:
            if spec.desired_state == IndexDesiredState.PRESENT:
                spec.version += 1  # Increment version to trigger re-indexing
                spec.gmt_updated = utc_now()
                logger.info(f"Updated spec version for {document_id}:{spec.index_type} to {spec.version}")

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
            stmt = select(DocumentIndexSpec).where(
                and_(DocumentIndexSpec.document_id == document_id, DocumentIndexSpec.index_type == index_type)
            )
            result = await session.execute(stmt)
            spec = result.scalar_one_or_none()

            if spec:
                # Option 1: Mark as absent (let reconciliation handle cleanup)
                spec.desired_state = IndexDesiredState.ABSENT
                spec.version += 1
                spec.gmt_updated = utc_now()
                logger.info(f"Marked spec for deletion: {document_id}:{index_type}")

    async def get_document_index_status(self, session: AsyncSession, document_id: str) -> dict:
        """
        Get current index status for a document

        Args:
            session: Database session
            document_id: Document ID

        Returns:
            Dictionary with index status information
        """
        # Get specs
        spec_stmt = select(DocumentIndexSpec).where(DocumentIndexSpec.document_id == document_id)
        spec_result = await session.execute(spec_stmt)
        specs = {spec.index_type: spec for spec in spec_result.scalars().all()}

        # Get statuses
        status_stmt = select(DocumentIndexStatus).where(DocumentIndexStatus.document_id == document_id)
        status_result = await session.execute(status_stmt)
        statuses = {status.index_type: status for status in status_result.scalars().all()}

        # Combine information
        result = {"document_id": document_id, "indexes": {}, "overall_status": "complete"}

        all_index_types = set(specs.keys()) | set(statuses.keys())
        has_creating = False
        has_failed = False

        for index_type in all_index_types:
            spec = specs.get(index_type)
            status = statuses.get(index_type)

            index_info = {
                "type": index_type,
                "desired_state": spec.desired_state if spec else "absent",
                "actual_state": status.actual_state if status else "absent",
                "in_sync": False,
            }

            # Check if in sync (version matches and states are consistent)
            if spec and status:
                index_info["in_sync"] = status.observed_version == spec.version and not status.needs_reconciliation(
                    spec
                )

            if status:
                if status.actual_state == "creating":
                    has_creating = True
                elif status.actual_state == "failed":
                    has_failed = True
                    index_info["error"] = status.error_message

            result["indexes"][index_type] = index_info

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
