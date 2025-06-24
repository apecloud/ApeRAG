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

from sqlalchemy import and_, select, update
from sqlalchemy.orm import Session

from aperag.config import get_sync_session
from aperag.db.models import (
    Document,
    DocumentIndex,
    DocumentIndexStatus,
    DocumentIndexType,
    DocumentStatus,
)
from aperag.tasks.scheduler import TaskScheduler, create_task_scheduler
from aperag.utils.constant import IndexAction
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)


class DocumentIndexReconciler:
    """Reconciler for document indexes using single status model"""

    def __init__(self, task_scheduler: Optional[TaskScheduler] = None, scheduler_type: str = "celery"):
        self.task_scheduler = task_scheduler or create_task_scheduler(scheduler_type)

    def reconcile_all(self):
        """
        Main reconciliation loop - scan indexes and reconcile differences
        Groups operations by document and index type for atomic processing
        """
        # Get all indexes that need reconciliation
        for session in get_sync_session():
            operations = self._get_indexes_needing_reconciliation(session)

        logger.info(f"Found {len(operations)} documents need to be reconciled")

        # Process each document with its own transaction
        successful_docs = 0
        failed_docs = 0
        for document_id, doc_operations in operations.items():
            try:
                self._reconcile_single_document(document_id, doc_operations)
                successful_docs += 1
            except Exception as e:
                failed_docs += 1
                logger.error(f"Failed to reconcile document {document_id}: {e}", exc_info=True)
                # Continue processing other documents - don't let one failure stop everything

        logger.info(f"Reconciliation completed: {successful_docs} successful, {failed_docs} failed")

    def _get_indexes_needing_reconciliation(self, session: Session) -> List[DocumentIndex]:
        """
        Get all indexes that need reconciliation without modifying their state.
        State modifications will happen in individual document transactions.
        """
        from collections import defaultdict

        operations = defaultdict(lambda: {IndexAction.CREATE: [], IndexAction.UPDATE: [], IndexAction.DELETE: []})

        conditions = {
            IndexAction.CREATE: and_(
                DocumentIndex.status == DocumentIndexStatus.PENDING,
                DocumentIndex.observed_version < DocumentIndex.version,
                DocumentIndex.version == 1,
            ),
            IndexAction.UPDATE: and_(
                DocumentIndex.status == DocumentIndexStatus.PENDING,
                DocumentIndex.observed_version < DocumentIndex.version,
                DocumentIndex.version > 1,
            ),
            IndexAction.DELETE: and_(
                DocumentIndex.status == DocumentIndexStatus.DELETING,
            ),
        }

        for action, condition in conditions.items():
            stmt = select(DocumentIndex).where(condition)
            result = session.execute(stmt)
            indexes = result.scalars().all()
            for index in indexes:
                operations[index.document_id][action].append(index)

        return operations

    def _reconcile_single_document(self, document_id: str, operations: dict):
        """
        Reconcile operations for a single document within its own transaction
        """
        for session in get_sync_session():
            # Collect indexes for this document that need claiming
            indexes_to_claim = []

            for action, doc_indexes in operations.items():
                for doc_index in doc_indexes:
                    indexes_to_claim.append((doc_index.id, doc_index.index_type, action))

            # Atomically claim the indexes for this document
            claimed_indexes = self._claim_document_indexes(session, document_id, indexes_to_claim)

            if claimed_indexes:
                # Schedule tasks for successfully claimed indexes
                self._reconcile_document_operations(document_id, claimed_indexes)
                session.commit()
            else:
                # Some indexes couldn't be claimed (likely already being processed), skip this document
                logger.debug(f"Skipping document {document_id} - indexes already being processed")

    def _claim_document_indexes(self, session: Session, document_id: str, indexes_to_claim: List[tuple]) -> List[dict]:
        """
        Atomically claim indexes for a document by updating their state.
        Returns list of successfully claimed indexes with their details.
        """
        claimed_indexes = []

        try:
            for index_id, index_type, action in indexes_to_claim:
                if action in [IndexAction.CREATE, IndexAction.UPDATE]:
                    target_state = DocumentIndexStatus.CREATING
                elif action == IndexAction.DELETE:
                    target_state = DocumentIndexStatus.DELETION_IN_PROGRESS
                else:
                    continue

                # Get the current index record to extract version info
                stmt = select(DocumentIndex).where(DocumentIndex.id == index_id)
                result = session.execute(stmt)
                current_index = result.scalar_one_or_none()

                if not current_index:
                    continue

                # Build appropriate claiming conditions based on operation type
                if action == IndexAction.CREATE:
                    claiming_conditions = [
                        DocumentIndex.id == index_id,
                        DocumentIndex.status == DocumentIndexStatus.PENDING,
                        DocumentIndex.observed_version < DocumentIndex.version,
                        DocumentIndex.version == 1,
                    ]
                elif action == IndexAction.UPDATE:
                    claiming_conditions = [
                        DocumentIndex.id == index_id,
                        DocumentIndex.status == DocumentIndexStatus.PENDING,
                        DocumentIndex.observed_version < DocumentIndex.version,
                        DocumentIndex.version > 1,
                    ]
                elif action == IndexAction.DELETE:
                    claiming_conditions = [
                        DocumentIndex.id == index_id,
                        DocumentIndex.status == DocumentIndexStatus.DELETING,
                    ]

                # Try to claim this specific index
                update_stmt = (
                    update(DocumentIndex)
                    .where(and_(*claiming_conditions))
                    .values(status=target_state, gmt_updated=utc_now(), gmt_last_reconciled=utc_now())
                )

                result = session.execute(update_stmt)
                if result.rowcount > 0:
                    # Successfully claimed this index
                    claimed_indexes.append(
                        {
                            "index_id": index_id,
                            "document_id": document_id,
                            "index_type": index_type,
                            "action": action,
                            "target_version": current_index.version
                            if action in [IndexAction.CREATE, IndexAction.UPDATE]
                            else None,
                        }
                    )
                    logger.debug(f"Claimed index {index_id} for document {document_id} ({action})")
                else:
                    logger.debug(f"Could not claim index {index_id} for document {document_id}")

            session.flush()  # Ensure changes are visible
            return claimed_indexes
        except Exception as e:
            logger.error(f"Failed to claim indexes for document {document_id}: {e}")
            return []

    def _reconcile_document_operations(self, document_id: str, claimed_indexes: List[dict]):
        """
        Reconcile operations for a single document, batching same operation types together
        """
        from collections import defaultdict

        # Group by operation type to batch operations
        operations_by_type = defaultdict(list)
        for claimed_index in claimed_indexes:
            action = claimed_index["action"]
            operations_by_type[action].append(claimed_index)

        # Process create operations as a batch
        if IndexAction.CREATE in operations_by_type:
            create_indexes = operations_by_type[IndexAction.CREATE]
            create_types = [claimed_index["index_type"] for claimed_index in create_indexes]
            context = {}

            for claimed_index in create_indexes:
                index_type = claimed_index["index_type"]
                target_version = claimed_index.get("target_version")

                # Store version info in context
                if target_version is not None:
                    context[f"{index_type}_version"] = target_version

            self.task_scheduler.schedule_create_index(
                document_id=document_id, index_types=create_types, context=context
            )
            logger.info(f"Scheduled create task for document {document_id}, types: {create_types}")

        # Process update operations as a batch
        if IndexAction.UPDATE in operations_by_type:
            update_indexes = operations_by_type[IndexAction.UPDATE]
            update_types = [claimed_index["index_type"] for claimed_index in update_indexes]
            context = {}

            for claimed_index in update_indexes:
                index_type = claimed_index["index_type"]
                target_version = claimed_index.get("target_version")

                # Store version info in context
                if target_version is not None:
                    context[f"{index_type}_version"] = target_version

            self.task_scheduler.schedule_update_index(
                document_id=document_id, index_types=update_types, context=context
            )
            logger.info(f"Scheduled update task for document {document_id}, types: {update_types}")

        # Process delete operations as a batch
        if IndexAction.DELETE in operations_by_type:
            delete_indexes = operations_by_type[IndexAction.DELETE]
            delete_types = [claimed_index["index_type"] for claimed_index in delete_indexes]

            self.task_scheduler.schedule_delete_index(document_id=document_id, index_types=delete_types)
            logger.info(f"Scheduled delete task for document {document_id}, types: {delete_types}")


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
    def on_index_created(document_id: str, index_type: str, target_version: int, index_data: str = None):
        """Called when index creation/update succeeds"""
        for session in get_sync_session():
            # Use atomic update with version validation
            update_stmt = (
                update(DocumentIndex)
                .where(
                    and_(
                        DocumentIndex.document_id == document_id,
                        DocumentIndex.index_type == DocumentIndexType(index_type),
                        DocumentIndex.status == DocumentIndexStatus.CREATING,
                        DocumentIndex.version == target_version,  # Critical: validate version
                    )
                )
                .values(
                    status=DocumentIndexStatus.ACTIVE,
                    observed_version=target_version,  # Mark this version as processed
                    index_data=index_data,
                    error_message=None,
                    gmt_updated=utc_now(),
                    gmt_last_reconciled=utc_now(),
                )
            )

            result = session.execute(update_stmt)
            if result.rowcount > 0:
                IndexTaskCallbacks._update_document_status(document_id, session)
                logger.info(f"{index_type} index creation completed for document {document_id} (v{target_version})")
                session.commit()
            else:
                logger.warning(
                    f"Index creation callback ignored for document {document_id} type {index_type} v{target_version} - not in expected state"
                )
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
                        # Allow transition from any in-progress state
                        DocumentIndex.status.in_(
                            [DocumentIndexStatus.CREATING, DocumentIndexStatus.DELETION_IN_PROGRESS]
                        ),
                    )
                )
                .values(
                    status=DocumentIndexStatus.FAILED,
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
                logger.warning(
                    f"Index failure callback ignored for document {document_id} type {index_type} - not in expected state"
                )
                session.rollback()

    @staticmethod
    def on_index_deleted(document_id: str, index_type: str):
        """Called when index deletion succeeds - hard delete the record"""
        for session in get_sync_session():
            # Delete the record entirely
            from sqlalchemy import delete

            delete_stmt = delete(DocumentIndex).where(
                and_(
                    DocumentIndex.document_id == document_id,
                    DocumentIndex.index_type == DocumentIndexType(index_type),
                    DocumentIndex.status == DocumentIndexStatus.DELETION_IN_PROGRESS,
                )
            )

            result = session.execute(delete_stmt)
            if result.rowcount > 0:
                IndexTaskCallbacks._update_document_status(document_id, session)
                logger.info(f"{index_type} index deleted for document {document_id}")
                session.commit()
            else:
                logger.warning(
                    f"Index deletion callback ignored for document {document_id} type {index_type} - not in expected state"
                )
                session.rollback()


# Global instances
index_reconciler = DocumentIndexReconciler()
index_task_callbacks = IndexTaskCallbacks()
