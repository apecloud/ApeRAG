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
from aperag.config import settings
from aperag.tasks.scheduler import TaskScheduler, create_task_scheduler
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)


class BackendIndexReconciler:
    """Simple reconciler for document indexes (backend chain)"""

    def __init__(self, task_scheduler: Optional[TaskScheduler] = None, scheduler_type: str = "celery"):
        self.task_scheduler = task_scheduler or create_task_scheduler(scheduler_type)

    @staticmethod
    def _get_reconciliation_conditions(operation_type: str, document_ids: List[str] = None):
        """
        Get all conditions for indexes that need reconciliation based on operation type.
        This is the authoritative source for determining which indexes can be processed.

        Args:
            operation_type: 'create', 'update', or 'delete'
            document_ids: Optional list of document IDs to filter by

        Returns:
            List of SQLAlchemy conditions
        """
        if operation_type in ["create", "update"]:
            conditions = [
                DocumentIndex.desired_state == IndexDesiredState.PRESENT,
                # Need reconciliation: either version mismatch or state mismatch
                or_(
                    DocumentIndex.observed_version < DocumentIndex.version,
                    DocumentIndex.actual_state.in_([IndexActualState.ABSENT, IndexActualState.FAILED]),
                ),
                # For create/update operations, exclude both CREATING and DELETING states
                DocumentIndex.actual_state.notin_([IndexActualState.CREATING, IndexActualState.DELETING]),
            ]
        elif operation_type == "delete":
            conditions = [
                DocumentIndex.desired_state == IndexDesiredState.ABSENT,
                # Only delete indexes that actually exist or are being created
                DocumentIndex.actual_state.in_([IndexActualState.CREATING, IndexActualState.PRESENT]),
                # For delete operations, allow claiming indexes in CREATING state to enable deletion override
                # Only exclude DELETING state to prevent concurrent deletions
                DocumentIndex.actual_state != IndexActualState.DELETING,
            ]
        else:
            raise ValueError(f"Unknown operation_type: {operation_type}")

        if document_ids:
            conditions.append(DocumentIndex.document_id.in_(document_ids))

        return conditions

    def reconcile_all(self, document_ids: List[str] = None):
        """
        Main reconciliation loop - scan specs and reconcile differences
        Groups operations by document to enable batch processing

        Args:
            document_ids: Optional list of specific document IDs to reconcile. If None, reconcile all.
        """
        # Get all indexes that need reconciliation first
        all_indexes_needing_reconciliation = []
        for session in get_sync_session():
            all_indexes_needing_reconciliation = self._get_indexes_needing_reconciliation(session, document_ids)
            break  # Only need to query once

        if not all_indexes_needing_reconciliation:
            logger.debug("No indexes need reconciliation")
            return

        # Group by document ID and operation type for batch processing
        from collections import defaultdict

        doc_operations = defaultdict(lambda: {"create": [], "update": [], "delete": []})

        for doc_index in all_indexes_needing_reconciliation:
            # Group operations by document and type
            if doc_index.desired_state == IndexDesiredState.PRESENT:
                # Check if this is an update (version mismatch with existing index data) or creation
                if doc_index.index_data and doc_index.observed_version > 0:
                    # Index has data and was observed before - this is an update
                    operation_type = "update"
                else:
                    # No existing data or never observed - this is a creation
                    operation_type = "create"
                doc_operations[doc_index.document_id][operation_type].append(doc_index)
            elif doc_index.desired_state == IndexDesiredState.ABSENT:
                doc_operations[doc_index.document_id]["delete"].append(doc_index)

        logger.info(f"Found {len(doc_operations)} documents need to be reconciled")

        # Process each document with its own transaction
        successful_docs = 0
        failed_docs = 0
        for document_id, operations in doc_operations.items():
            try:
                self._reconcile_single_document(document_id, operations)
                successful_docs += 1
            except Exception as e:
                failed_docs += 1
                logger.error(f"Failed to reconcile document {document_id}: {e}", exc_info=True)
                # Continue processing other documents - don't let one failure stop everything

        logger.info(f"Reconciliation completed: {successful_docs} successful, {failed_docs} failed")

    def _get_indexes_needing_reconciliation(
        self, session: Session, document_ids: List[str] = None
    ) -> List[DocumentIndex]:
        """
        Get all indexes that need reconciliation without modifying their state.
        State modifications will happen in individual document transactions.
        """
        # Use shared reconciliation conditions
        create_conditions = self._get_reconciliation_conditions("create", document_ids)
        delete_conditions = self._get_reconciliation_conditions("delete", document_ids)

        # Query for indexes that need creating
        create_stmt = select(DocumentIndex).where(and_(*create_conditions))
        create_result = session.execute(create_stmt)
        create_indexes = create_result.scalars().all()

        # Query for indexes that need deleting
        delete_stmt = select(DocumentIndex).where(and_(*delete_conditions))
        delete_result = session.execute(delete_stmt)
        delete_indexes = delete_result.scalars().all()

        all_indexes = list(create_indexes) + list(delete_indexes)
        logger.debug(f"Found {len(all_indexes)} indexes needing reconciliation")
        return all_indexes

    def _reconcile_single_document(self, document_id: str, operations: dict):
        """
        Reconcile operations for a single document within its own transaction
        """
        for session in get_sync_session():
            # Get the specific indexes for this document and claim them atomically
            indexes_to_claim = []

            # Collect indexes for this document that need claiming
            for operation_type, doc_indexes in operations.items():
                for doc_index in doc_indexes:
                    indexes_to_claim.append((doc_index.id, operation_type))

            # Atomically claim the indexes for this document
            claimed_successfully = self._claim_document_indexes(session, document_id, indexes_to_claim)

            if claimed_successfully:
                # Schedule tasks for successfully claimed indexes
                self._reconcile_document_operations(document_id, operations)
                session.commit()
            else:
                # Some indexes couldn't be claimed (likely already being processed), skip this document
                logger.debug(f"Skipping document {document_id} - indexes already being processed")

    def _claim_document_indexes(self, session: Session, document_id: str, indexes_to_claim: List[tuple]) -> bool:
        """
        Atomically claim indexes for a document by updating their state.
        Returns True if all indexes were successfully claimed, False otherwise.
        """
        try:
            for index_id, operation_type in indexes_to_claim:
                if operation_type in ["create", "update"]:
                    target_state = IndexActualState.CREATING
                elif operation_type == "delete":
                    target_state = IndexActualState.DELETING
                else:
                    continue

                # Try to claim this specific index
                # Use all reconciliation conditions plus specific index/document filters
                # This ensures claiming conditions are a superset of reconciliation conditions
                base_conditions = self._get_reconciliation_conditions(operation_type)
                where_conditions = [
                    DocumentIndex.id == index_id,
                    DocumentIndex.document_id == document_id,
                ] + base_conditions

                update_stmt = (
                    update(DocumentIndex)
                    .where(and_(*where_conditions))
                    .values(
                        actual_state=target_state,
                        gmt_updated=utc_now(),
                        gmt_last_reconciled=utc_now()
                    )
                )

                result = session.execute(update_stmt)
                if result.rowcount == 0:
                    # This index couldn't be claimed (already being processed)
                    logger.debug(f"Could not claim index {index_id} for document {document_id}")
                    return False

            session.flush()  # Ensure changes are visible
            return True
        except Exception as e:
            logger.error(f"Failed to claim indexes for document {document_id}: {e}")
            return False

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
            # Use atomic update with state validation
            update_stmt = (
                update(DocumentIndex)
                .where(
                    and_(
                        DocumentIndex.document_id == document_id,
                        DocumentIndex.index_type == DocumentIndexType(index_type),
                        DocumentIndex.actual_state == IndexActualState.CREATING,  # Only allow transition from CREATING
                    )
                )
                .values(
                    actual_state=IndexActualState.PRESENT,
                    observed_version=DocumentIndex.version,  # Mark as processed
                    index_data=index_data,
                    error_message=None,
                    gmt_updated=utc_now(),
                    gmt_last_reconciled=utc_now(),
                )
            )
            
            result = session.execute(update_stmt)
            if result.rowcount > 0:
                IndexTaskCallbacks._update_document_status(document_id, session)
                logger.info(f"{index_type} index creation completed for document {document_id}")
                session.commit()
            else:
                logger.warning(f"Index creation callback ignored for document {document_id} type {index_type} - not in CREATING state")
                session.rollback()

    @staticmethod
    def on_index_failed(document_id: str, index_type: str, error_message: str):
        """Called when index operation fails"""
        for session in get_sync_session():
            # Use atomic update with state validation
            update_stmt = (
                update(DocumentIndex)
                .where(
                    and_(
                        DocumentIndex.document_id == document_id,
                        DocumentIndex.index_type == DocumentIndexType(index_type),
                        # Only allow transition from CREATING or DELETING states
                        DocumentIndex.actual_state.in_([IndexActualState.CREATING, IndexActualState.DELETING]),
                    )
                )
                .values(
                    actual_state=IndexActualState.FAILED,
                    error_message=error_message,
                    gmt_updated=utc_now(),
                    gmt_last_reconciled=utc_now(),
                )
            )
            
            result = session.execute(update_stmt)
            if result.rowcount > 0:
                IndexTaskCallbacks._update_document_status(document_id, session)
                logger.error(f"{index_type} index operation failed for document {document_id}: {error_message}")
                session.commit()
            else:
                logger.warning(f"Index failure callback ignored for document {document_id} type {index_type} - not in CREATING or DELETING state")
                session.rollback()

    @staticmethod
    def on_index_deleted(document_id: str, index_type: str):
        """Called when index deletion succeeds"""
        for session in get_sync_session():
            # Use atomic update with state validation
            update_stmt = (
                update(DocumentIndex)
                .where(
                    and_(
                        DocumentIndex.document_id == document_id,
                        DocumentIndex.index_type == DocumentIndexType(index_type),
                        DocumentIndex.actual_state == IndexActualState.DELETING,  # Only allow transition from DELETING
                    )
                )
                .values(
                    actual_state=IndexActualState.ABSENT,
                    observed_version=DocumentIndex.version,  # Mark as processed
                    index_data=None,
                    error_message=None,
                    gmt_updated=utc_now(),
                    gmt_last_reconciled=utc_now(),
                )
            )
            
            result = session.execute(update_stmt)
            if result.rowcount > 0:
                IndexTaskCallbacks._update_document_status(document_id, session)
                logger.info(f"{index_type} index deletion completed for document {document_id}")
                session.commit()
            else:
                logger.warning(f"Index deletion callback ignored for document {document_id} type {index_type} - not in DELETING state")
                session.rollback()


# Global instance
index_reconciler = BackendIndexReconciler(scheduler_type=settings.task_scheduler_type)
index_task_callbacks = IndexTaskCallbacks()
