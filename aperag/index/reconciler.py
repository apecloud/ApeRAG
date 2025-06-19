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
import time
from typing import List, Optional

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session

from aperag.config import get_sync_session
from aperag.db.models import (
    Document,
    DocumentIndex,
    DocumentIndexType,
    DocumentStatus,
    IndexActualState,
    IndexDesiredState,
)
from aperag.tasks.scheduler import TaskScheduler, create_task_scheduler

logger = logging.getLogger(__name__)


class BackendIndexReconciler:
    """Simple reconciler for document indexes (backend chain)"""

    def __init__(self, task_scheduler: Optional[TaskScheduler] = None, scheduler_type: str = "celery"):
        self.task_scheduler = task_scheduler or create_task_scheduler(scheduler_type)

    def reconcile_all(self, document_ids: List[str] = None):
        """
        Main reconciliation loop - scan specs and reconcile differences
        Groups operations by document to enable batch processing

        Args:
            document_ids: Optional list of specific document IDs to reconcile. If None, reconcile all.
        """
        for session in get_sync_session():
            # Get indexes needing reconciliation with atomic state checking
            indexes_needing_reconciliation = self._get_and_lock_indexes_needing_reconciliation(session, document_ids)

            if not indexes_needing_reconciliation:
                logger.debug("No indexes need reconciliation")
                return

            # Group by document ID and operation type for batch processing
            self._reconcile_grouped(session, indexes_needing_reconciliation)

            session.commit()

    def _get_and_lock_indexes_needing_reconciliation(
        self, session: Session, document_ids: List[str] = None
    ) -> List[DocumentIndex]:
        """
        Get all indexes that need reconciliation and atomically update their state to prevent
        duplicate processing in concurrent scenarios. All filtering and state updates are done
        at the database level for maximum efficiency and concurrency safety.

        Args:
            document_ids: Optional list of specific document IDs to process
        """
        indexes_to_process = []

        # Process indexes that need CREATING (desired=PRESENT, actual=ABSENT/FAILED, not currently processing)
        create_claimed = self._claim_indexes_for_creating(session, document_ids)
        indexes_to_process.extend(create_claimed)

        # Process indexes that need DELETING (desired=ABSENT, actual=CREATING/PRESENT, not currently processing)
        delete_claimed = self._claim_indexes_for_deleting(session, document_ids)
        indexes_to_process.extend(delete_claimed)

        logger.info(f"Claimed {len(indexes_to_process)} indexes for reconciliation")
        return indexes_to_process

    def _claim_indexes_for_creating(self, session: Session, document_ids: List[str] = None) -> List[DocumentIndex]:
        """
        Atomically claim indexes that need CREATING by updating their state to CREATING.
        Uses database-level filtering and atomic updates for maximum efficiency.
        """
        # Build the WHERE conditions for indexes that need creating
        conditions = [
            DocumentIndex.desired_state == IndexDesiredState.PRESENT,
            # Need reconciliation: either version mismatch or state mismatch
            or_(
                DocumentIndex.observed_version < DocumentIndex.version,
                DocumentIndex.actual_state.in_([IndexActualState.ABSENT, IndexActualState.FAILED]),
            ),
            # Not currently being processed
            DocumentIndex.actual_state.notin_([IndexActualState.CREATING, IndexActualState.DELETING]),
        ]

        if document_ids:
            conditions.append(DocumentIndex.document_id.in_(document_ids))

        # Atomic update: claim these indexes by setting state to CREATING
        update_stmt = (
            update(DocumentIndex)
            .where(and_(*conditions))
            .values(actual_state=IndexActualState.CREATING, gmt_updated=func.now(), gmt_last_reconciled=func.now())
            .returning(DocumentIndex)
        )

        result = session.execute(update_stmt)
        claimed_indexes = result.scalars().all()
        session.flush()  # Ensure changes are visible

        logger.debug(f"Claimed {len(claimed_indexes)} indexes for CREATING")
        return claimed_indexes

    def _claim_indexes_for_deleting(self, session: Session, document_ids: List[str] = None) -> List[DocumentIndex]:
        """
        Atomically claim indexes that need DELETING by updating their state to DELETING.
        Uses database-level filtering and atomic updates for maximum efficiency.
        """
        # Build the WHERE conditions for indexes that need deleting
        conditions = [
            DocumentIndex.desired_state == IndexDesiredState.ABSENT,
            # Need reconciliation: either version mismatch or state mismatch
            or_(
                DocumentIndex.observed_version < DocumentIndex.version,
                DocumentIndex.actual_state.in_([IndexActualState.CREATING, IndexActualState.PRESENT]),
            ),
            # Not currently being processed
            DocumentIndex.actual_state.notin_([IndexActualState.CREATING, IndexActualState.DELETING]),
        ]

        if document_ids:
            conditions.append(DocumentIndex.document_id.in_(document_ids))

        # Atomic update: claim these indexes by setting state to DELETING
        update_stmt = (
            update(DocumentIndex)
            .where(and_(*conditions))
            .values(actual_state=IndexActualState.DELETING, gmt_updated=func.now(), gmt_last_reconciled=func.now())
            .returning(DocumentIndex)
        )

        result = session.execute(update_stmt)
        claimed_indexes = result.scalars().all()
        session.flush()  # Ensure changes are visible

        logger.debug(f"Claimed {len(claimed_indexes)} indexes for DELETING")
        return claimed_indexes

    def _reconcile_grouped(self, session: Session, indexes_needing_reconciliation):
        """Group reconciliation operations by document for batch processing"""
        from collections import defaultdict

        # Group by document_id and operation type
        doc_operations = defaultdict(lambda: {"create": [], "update": [], "delete": []})

        for doc_index in indexes_needing_reconciliation:
            # At this point, states should already be updated to CREATING/DELETING
            # Determine operation type based on the nature of the reconciliation needed
            if doc_index.desired_state == IndexDesiredState.PRESENT:
                if doc_index.actual_state == IndexActualState.CREATING:
                    # Check if this is an update (version mismatch with existing index data) or creation
                    if doc_index.index_data and doc_index.observed_version > 0:
                        # Index has data and was observed before - this is an update
                        operation_type = "update"
                    else:
                        # No existing data or never observed - this is a creation
                        operation_type = "create"
                    doc_operations[doc_index.document_id][operation_type].append(doc_index)
            elif doc_index.desired_state == IndexDesiredState.ABSENT:
                if doc_index.actual_state == IndexActualState.DELETING:
                    doc_operations[doc_index.document_id]["delete"].append(doc_index)

        logger.info(f"Found {len(doc_operations)} documents need to be reconciled")

        # Process each document's operations
        for document_id, operations in doc_operations.items():
            try:
                self._reconcile_document_operations(document_id, operations)
            except Exception as e:
                logger.error(f"Failed to reconcile operations for document {document_id}: {e}")
                # Revert states for failed operations
                self._revert_failed_operations(session, document_id, operations, e)

    def _revert_failed_operations(self, session: Session, document_id: str, operations: dict, error: Exception):
        """Revert states for operations that failed to schedule"""
        for operation_type, doc_indexes in operations.items():
            for doc_index in doc_indexes:
                try:
                    if operation_type == "create":
                        doc_index.mark_failed(f"Failed to schedule task: {str(error)}")
                    elif operation_type == "update":
                        # Revert to PRESENT state - it was updating an existing index
                        doc_index.actual_state = IndexActualState.PRESENT
                    elif operation_type == "delete":
                        # Revert to previous state - assume it was PRESENT
                        doc_index.actual_state = IndexActualState.PRESENT
                    session.add(doc_index)
                except Exception as revert_error:
                    logger.error(
                        f"Failed to revert state for {doc_index.document_id}:{doc_index.index_type}: {revert_error}"
                    )

    def _reconcile_document_operations(self, document_id: str, operations: dict):
        """
        Reconcile operations for a single document, using batch processing when possible
        States are already updated to CREATING/DELETING before calling this method
        """

        create_index_types = []
        for doc_index in operations["create"]:
            create_index_types.append(doc_index.index_type)
        if create_index_types:
            # Add document_id to task for better idempotency checking
            task_id = f"create_index_{document_id}_{int(time.time())}"
            self.task_scheduler.schedule_create_index(
                index_types=create_index_types, document_id=document_id, task_id=task_id
            )
            logger.info(
                f"Scheduled create index task {task_id} for document {document_id} with types {create_index_types}"
            )

        update_index_types = []
        for doc_index in operations["update"]:
            update_index_types.append(doc_index.index_type)
        if update_index_types:
            task_id = f"update_index_{document_id}_{int(time.time())}"
            self.task_scheduler.schedule_update_index(
                index_types=update_index_types, document_id=document_id, task_id=task_id
            )
            logger.info(
                f"Scheduled update index task {task_id} for document {document_id} with types {update_index_types}"
            )

        delete_index_types = []
        for doc_index in operations["delete"]:
            delete_index_types.append(doc_index.index_type)
        if delete_index_types:
            # Use the last index_data for the delete operation
            index_data = operations["delete"][-1].index_data if operations["delete"] else None
            task_id = f"delete_index_{document_id}_{int(time.time())}"
            self.task_scheduler.schedule_delete_index(
                index_types=delete_index_types,
                document_id=document_id,
                index_data=index_data,
                task_id=task_id,
            )
            logger.info(
                f"Scheduled delete index task {task_id} for document {document_id} with types {delete_index_types}"
            )


# Index task completion callbacks
class IndexTaskCallbacks:
    """Callbacks for index task completion"""

    @staticmethod
    def _update_document_status(document_id: str, session: Session):
        stmt = select(Document).where(Document.id == document_id, Document.status != DocumentStatus.DELETED)
        result = session.execute(stmt)
        document = result.scalar_one_or_none()
        if not document:
            return
        document.status = document.get_overall_index_status(session)
        session.add(document)

    @staticmethod
    def on_index_created(document_id: str, index_type: str, index_data: str = None):
        """Called when index creation succeeds"""
        for session in get_sync_session():
            stmt = select(DocumentIndex).where(
                and_(
                    DocumentIndex.document_id == document_id,
                    DocumentIndex.index_type == DocumentIndexType(index_type),
                )
            )
            result = session.execute(stmt)
            doc_index = result.scalar_one_or_none()

            if doc_index:
                doc_index.mark_present(index_data)
            IndexTaskCallbacks._update_document_status(document_id, session)

            logger.info(f"{index_type} index creation completed for document {document_id}")
            session.commit()

    @staticmethod
    def on_index_failed(document_id: str, index_type: str, error_message: str):
        """Called when index operation fails"""
        for session in get_sync_session():
            stmt = select(DocumentIndex).where(
                and_(
                    DocumentIndex.document_id == document_id,
                    DocumentIndex.index_type == DocumentIndexType(index_type),
                )
            )
            result = session.execute(stmt)
            doc_index = result.scalar_one_or_none()

            if doc_index:
                doc_index.mark_failed(error_message)
            IndexTaskCallbacks._update_document_status(document_id, session)

            logger.error(f"{index_type} index operation failed for document {document_id}: {error_message}")
            session.commit()

    @staticmethod
    def on_index_deleted(document_id: str, index_type: str):
        """Called when index deletion succeeds"""
        for session in get_sync_session():
            stmt = select(DocumentIndex).where(
                and_(
                    DocumentIndex.document_id == document_id,
                    DocumentIndex.index_type == DocumentIndexType(index_type),
                )
            )
            result = session.execute(stmt)
            doc_index = result.scalar_one_or_none()

            if doc_index:
                doc_index.mark_absent()
            IndexTaskCallbacks._update_document_status(document_id, session)

            logger.info(f"{index_type} index deletion completed for document {document_id}")
            session.commit()


# Global instance
index_reconciler = BackendIndexReconciler()
index_task_callbacks = IndexTaskCallbacks()
