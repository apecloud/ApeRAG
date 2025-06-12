import logging
from aperag.db.models import DocumentIndexSpec, DocumentIndexStatus, DocumentIndexType, IndexActualState, IndexDesiredState, utc_now
from aperag.db.ops import get_session


from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession


from typing import List, Tuple

logger = logging.getLogger(__name__)


class DocumentIndexBackendReconciler:
    """Simple reconciler for document indexes (backend chain)"""

    def __init__(self):
        self.running = False

    async def reconcile_all(self):
        """
        Main reconciliation loop - scan all specs and reconcile differences
        """
        async with get_session() as session:
            # Get all specs and their corresponding statuses
            spec_status_pairs = await self._get_specs_needing_reconciliation(session)

            if not spec_status_pairs:
                logger.debug("No indexes need reconciliation")
                return

            logger.info(f"Found {len(spec_status_pairs)} indexes needing reconciliation")

            for spec, status in spec_status_pairs:
                try:
                    await self._reconcile_single(session, spec, status)
                except Exception as e:
                    logger.error(f"Failed to reconcile {spec.document_id}:{spec.index_type.value}: {e}")

            await session.commit()

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

    async def _reconcile_single(self, session: AsyncSession, spec: DocumentIndexSpec, status: DocumentIndexStatus):
        """Reconcile a single spec/status pair"""
        logger.info(f"Reconciling {spec.document_id}:{spec.index_type.value} - "
                   f"desired: {spec.desired_state.value}, actual: {status.actual_state.value}, "
                   f"spec_version: {spec.version}, observed_version: {status.observed_version}")

        # Skip if currently processing (avoid duplicate tasks)
        if status.actual_state in [IndexActualState.CREATING, IndexActualState.DELETING]:
            logger.debug(f"Skipping {spec.document_id}:{spec.index_type.value} - already processing")
            return

        # Update observed_version to mark as being processed
        status.observed_version = spec.version
        status.gmt_last_reconciled = utc_now()

        if spec.desired_state == IndexDesiredState.PRESENT:
            await self._ensure_index_present(session, spec, status)
        elif spec.desired_state == IndexDesiredState.ABSENT:
            await self._ensure_index_absent(session, spec, status)

    async def _ensure_index_present(self, session: AsyncSession, spec: DocumentIndexSpec, status: DocumentIndexStatus):
        """Ensure the index exists"""
        if status.actual_state == IndexActualState.ABSENT:
            # Start creating the index
            status.actual_state = IndexActualState.CREATING
            status.gmt_updated = utc_now()
            await self._dispatch_create_task(spec.document_id, spec.index_type)

        elif status.actual_state == IndexActualState.FAILED:
            # Retry failed index if possible
            if status.can_retry():
                logger.info(f"Retrying failed index {spec.document_id}:{spec.index_type.value} "
                           f"(attempt {status.retry_count + 1})")
                status.actual_state = IndexActualState.CREATING
                status.gmt_updated = utc_now()
                await self._dispatch_create_task(spec.document_id, spec.index_type)
            else:
                logger.warning(f"Index {spec.document_id}:{spec.index_type.value} has failed too many times")

        elif status.actual_state == IndexActualState.PRESENT:
            # Index exists but version changed - need to update
            logger.info(f"Updating existing index {spec.document_id}:{spec.index_type.value}")
            status.actual_state = IndexActualState.CREATING  # Use same state for updates
            status.gmt_updated = utc_now()
            await self._dispatch_update_task(spec.document_id, spec.index_type)

    async def _ensure_index_absent(self, session: AsyncSession, spec: DocumentIndexSpec, status: DocumentIndexStatus):
        """Ensure the index does not exist"""
        if status.actual_state in [IndexActualState.CREATING, IndexActualState.PRESENT]:
            # Start deleting the index
            status.actual_state = IndexActualState.DELETING
            status.gmt_updated = utc_now()
            await self._dispatch_delete_task(spec.document_id, spec.index_type, status.index_data)

    async def _dispatch_create_task(self, document_id: str, index_type: DocumentIndexType):
        """Dispatch async task to create index"""
        # Import here to avoid circular imports
        from aperag.tasks.index_tasks import create_index_task

        # Dispatch the task asynchronously
        create_index_task.delay(document_id, index_type.value)
        logger.info(f"Dispatched create task for {document_id}:{index_type.value}")

    async def _dispatch_update_task(self, document_id: str, index_type: DocumentIndexType):
        """Dispatch async task to update index"""
        from aperag.tasks.index_tasks import update_index_task

        update_index_task.delay(document_id, index_type.value)
        logger.info(f"Dispatched update task for {document_id}:{index_type.value}")

    async def _dispatch_delete_task(self, document_id: str, index_type: DocumentIndexType, index_data: str = None):
        """Dispatch async task to delete index"""
        from aperag.tasks.index_tasks import delete_index_task

        delete_index_task.delay(document_id, index_type.value, index_data)
        logger.info(f"Dispatched delete task for {document_id}:{index_type.value}")


# Index task completion callbacks
class IndexTaskCallbacks:
    """Callbacks for index task completion"""

    @staticmethod
    async def on_index_created(document_id: str, index_type: str, index_data: str = None):
        """Called when index creation succeeds"""
        async with get_session() as session:
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
                await session.commit()
                logger.info(f"Index creation completed: {document_id}:{index_type}")

    @staticmethod
    async def on_index_failed(document_id: str, index_type: str, error_message: str):
        """Called when index operation fails"""
        async with get_session() as session:
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
                await session.commit()
                logger.error(f"Index operation failed: {document_id}:{index_type} - {error_message}")

    @staticmethod
    async def on_index_deleted(document_id: str, index_type: str):
        """Called when index deletion succeeds"""
        async with get_session() as session:
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
                await session.commit()
                logger.info(f"Index deletion completed: {document_id}:{index_type}")


# Global instance
index_reconciler = DocumentIndexBackendReconciler()
index_task_callbacks = IndexTaskCallbacks()