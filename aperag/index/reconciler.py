
import logging

from aperag.db.models import DocumentIndexSpec, DocumentIndexStatus, DocumentIndexType, IndexActualState, IndexDesiredState, utc_now
from aperag.db.ops import get_session
from aperag.tasks.scheduler import TaskScheduler, create_task_scheduler

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class BackendIndexReconciler:
    """Simple reconciler for document indexes (backend chain)"""

    def __init__(self, task_scheduler: Optional[TaskScheduler] = None, scheduler_type: str = "celery"):
        self.running = False
        self.task_scheduler = task_scheduler or create_task_scheduler(scheduler_type)

    async def reconcile_all(self):
        """
        Main reconciliation loop - scan all specs and reconcile differences
        Groups operations by document to enable batch processing
        """
        async with get_session() as session:
            # Get all specs and their corresponding statuses
            spec_status_pairs = await self._get_specs_needing_reconciliation(session)

            if not spec_status_pairs:
                logger.debug("No indexes need reconciliation")
                return

            logger.info(f"Found {len(spec_status_pairs)} indexes needing reconciliation")

            # Group by document ID and operation type for batch processing
            await self._reconcile_grouped(session, spec_status_pairs)

            await session.commit()

    async def _reconcile_grouped(self, session: AsyncSession, spec_status_pairs):
        """Group reconciliation operations by document for batch processing"""
        from collections import defaultdict

        # Group by document_id and operation type
        doc_operations = defaultdict(lambda: {
            'create': [], 'update': [], 'delete': []
        })

        for spec, status in spec_status_pairs:
            # Skip if currently processing
            if status.actual_state in [IndexActualState.CREATING, IndexActualState.DELETING]:
                logger.debug(f"Skipping {spec.document_id}:{spec.index_type.value} - already processing")
                continue

            # Update observed_version to mark as being processed
            status.observed_version = spec.version
            status.gmt_last_reconciled = utc_now()

            # Determine operation type
            if spec.desired_state == IndexDesiredState.PRESENT:
                if status.actual_state == IndexActualState.ABSENT:
                    doc_operations[spec.document_id]['create'].append((spec, status))
                elif status.actual_state == IndexActualState.FAILED and status.can_retry():
                    doc_operations[spec.document_id]['create'].append((spec, status))
                elif status.actual_state == IndexActualState.PRESENT:
                    doc_operations[spec.document_id]['update'].append((spec, status))
            elif spec.desired_state == IndexDesiredState.ABSENT:
                if status.actual_state in [IndexActualState.CREATING, IndexActualState.PRESENT]:
                    doc_operations[spec.document_id]['delete'].append((spec, status))

        # Process each document's operations
        for document_id, operations in doc_operations.items():
            try:
                await self._reconcile_document_operations(session, document_id, operations)
            except Exception as e:
                logger.error(f"Failed to reconcile operations for document {document_id}: {e}")

    async def _reconcile_document_operations(self, session: AsyncSession, document_id: str, operations: dict):
        """Reconcile operations for a single document, using batch processing when possible"""

        create_index_types = []
        for spec, status in operations['create']:
            status.actual_state = IndexActualState.CREATING
            status.gmt_updated = utc_now()
            create_index_types.append(spec.index_type.value)
        await self.task_scheduler.schedule_create_index(
            index_types=create_index_types,
            document_id=document_id
        )

        update_index_types = []
        for spec, status in operations['update']:
            status.actual_state = IndexActualState.CREATING
            status.gmt_updated = utc_now()
            update_index_types.append(spec.index_type.value)
        await self.task_scheduler.schedule_update_index(
            index_types=update_index_types,
            document_id=document_id
        )

        delete_index_types = []
        for spec, status in operations['delete']:
            status.actual_state = IndexActualState.DELETING
            status.gmt_updated = utc_now()
            delete_index_types.append(spec.index_type.value)
        await self.task_scheduler.schedule_delete_index(
            index_types=delete_index_types,
            document_id=document_id,
            index_data=status.index_data if status.index_data else None
        )

    async def _get_specs_needing_reconciliation(self, session: AsyncSession) -> List[Tuple[DocumentIndexSpec, DocumentIndexStatus]]:
        """Get all spec/status pairs that need reconciliation"""
        # Get all specs
        spec_stmt = select(DocumentIndexSpec)
        spec_result = await session.execute(spec_stmt)
        specs = spec_result.scalars().all()

        # Get all statuses
        status_stmt = select(DocumentIndexStatus)
        status_result = await session.execute(status_stmt)
        statuses = {(s.document_id, s.index_type): s for s in status_result.scalars().all()}

        pairs_needing_reconciliation = []

        for spec in specs:
            key = (spec.document_id, spec.index_type)
            status = statuses.get(key)

            # Create status if it doesn't exist
            if status is None:
                status = DocumentIndexStatus(
                    document_id=spec.document_id,
                    index_type=spec.index_type,
                    actual_state=IndexActualState.ABSENT,
                    observed_version=0
                )
                session.add(status)
                await session.flush()  # Get the ID

            # Check if reconciliation is needed
            if status.needs_reconciliation(spec):
                pairs_needing_reconciliation.append((spec, status))

        return pairs_needing_reconciliation


# Index task completion callbacks
class IndexTaskCallbacks:
    """Callbacks for index task completion"""

    @staticmethod
    async def on_index_created(document_id: str, index_types: List[str], index_data: str = None):
        """Called when index creation succeeds"""
        async with get_session() as session:
            for index_type in index_types:
                stmt = select(DocumentIndexStatus).where(
                    and_(
                        DocumentIndexStatus.document_id == document_id,
                        DocumentIndexStatus.index_type == DocumentIndexType(index_type)
                    )
                )
                result = await session.execute(stmt)
                status = result.scalar_one_or_none()

                if status:
                    status.actual_state = IndexActualState.PRESENT
                    status.index_data = index_data
                    status.error_message = None
                    status.retry_count = 0
                    status.gmt_updated = utc_now()
                logger.info(f"Index creation completed: {document_id}:{index_type}")
            await session.commit()

    @staticmethod
    async def on_index_failed(document_id: str, index_types: List[str], error_message: str):
        """Called when index operation fails"""
        async with get_session() as session:
            for index_type in index_types:
                stmt = select(DocumentIndexStatus).where(
                    and_(
                        DocumentIndexStatus.document_id == document_id,
                        DocumentIndexStatus.index_type == DocumentIndexType(index_type)
                    )
                )
                result = await session.execute(stmt)
                status = result.scalar_one_or_none()

                if status:
                    status.actual_state = IndexActualState.FAILED
                    status.error_message = error_message
                    status.retry_count += 1
                    status.gmt_updated = utc_now()
                logger.error(f"Index operation failed: {document_id}:{index_type} - {error_message}")
            await session.commit()

    @staticmethod
    async def on_index_deleted(document_id: str, index_types: List[str]):
        """Called when index deletion succeeds"""
        async with get_session() as session:
            for index_type in index_types:
                stmt = select(DocumentIndexStatus).where(
                    and_(
                        DocumentIndexStatus.document_id == document_id,
                        DocumentIndexStatus.index_type == DocumentIndexType(index_type)
                    )
                )
                result = await session.execute(stmt)
                status = result.scalar_one_or_none()

                if status:
                    status.actual_state = IndexActualState.ABSENT
                    status.index_data = None
                    status.error_message = None
                    status.retry_count = 0
                    status.gmt_updated = utc_now()
                logger.info(f"Index deletion completed: {document_id}:{index_type}")
            await session.commit()


# Global instance
index_reconciler = BackendIndexReconciler()
index_task_callbacks = IndexTaskCallbacks()