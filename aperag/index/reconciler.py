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
from sqlalchemy.orm import Session

from aperag.config import get_sync_session
from aperag.db.models import (
    Document,
    DocumentIndex,
    DocumentIndexType,
    IndexActualState,
    IndexDesiredState,
)
from aperag.tasks.scheduler import TaskScheduler, create_task_scheduler

logger = logging.getLogger(__name__)


class BackendIndexReconciler:
    """Simple reconciler for document indexes (backend chain)"""

    def __init__(self, task_scheduler: Optional[TaskScheduler] = None, scheduler_type: str = "celery"):
        self.running = False
        self.task_scheduler = task_scheduler or create_task_scheduler(scheduler_type)

    def reconcile_all(self):
        """
        Main reconciliation loop - scan all specs and reconcile differences
        Groups operations by document to enable batch processing
        """
        for session in get_sync_session():
            # Get all indexes needing reconciliation
            indexes_needing_reconciliation = self._get_indexes_needing_reconciliation(session)

            if not indexes_needing_reconciliation:
                logger.debug("No indexes need reconciliation")
                return

            # Group by document ID and operation type for batch processing
            self._reconcile_grouped(indexes_needing_reconciliation)

            session.commit()

    def _reconcile_grouped(self, indexes_needing_reconciliation):
        """Group reconciliation operations by document for batch processing"""
        from collections import defaultdict

        # Group by document_id and operation type
        doc_operations = defaultdict(lambda: {"create": [], "update": [], "delete": []})

        for doc_index in indexes_needing_reconciliation:
            # Skip if currently processing
            if doc_index.actual_state in [IndexActualState.CREATING, IndexActualState.DELETING]:
                logger.debug(
                    f"Skipping reconcile for {doc_index.document_id}:{doc_index.index_type} - already processing"
                )
                continue

            # Determine operation type
            if doc_index.desired_state == IndexDesiredState.PRESENT:
                if doc_index.actual_state == IndexActualState.ABSENT:
                    doc_operations[doc_index.document_id]["create"].append(doc_index)
                elif doc_index.actual_state == IndexActualState.FAILED:
                    doc_operations[doc_index.document_id]["create"].append(doc_index)
                elif doc_index.actual_state == IndexActualState.PRESENT:
                    doc_operations[doc_index.document_id]["update"].append(doc_index)
            elif doc_index.desired_state == IndexDesiredState.ABSENT:
                if doc_index.actual_state in [IndexActualState.CREATING, IndexActualState.PRESENT]:
                    doc_operations[doc_index.document_id]["delete"].append(doc_index)
        logger.info(f"Found {len(doc_operations)} documents need to be reconciled")

        # Process each document's operations
        for document_id, operations in doc_operations.items():
            try:
                self._reconcile_document_operations(document_id, operations)
            except Exception as e:
                logger.error(f"Failed to reconcile operations for document {document_id}: {e}")

    def _reconcile_document_operations(self, document_id: str, operations: dict):
        """Reconcile operations for a single document, using batch processing when possible"""

        create_index_types = []
        for doc_index in operations["create"]:
            create_index_types.append(doc_index.index_type)
        if create_index_types:
            self.task_scheduler.schedule_create_index(index_types=create_index_types, document_id=document_id)
        for doc_index in operations["create"]:
            doc_index.mark_creating()

        update_index_types = []
        for doc_index in operations["update"]:
            update_index_types.append(doc_index.index_type)
        if update_index_types:
            self.task_scheduler.schedule_update_index(index_types=update_index_types, document_id=document_id)
        for doc_index in operations["update"]:
            doc_index.mark_creating()

        delete_index_types = []
        for doc_index in operations["delete"]:
            delete_index_types.append(doc_index.index_type)
        if delete_index_types:
            # Use the last index_data for the delete operation
            index_data = operations["delete"][-1].index_data if operations["delete"] else None
            self.task_scheduler.schedule_delete_index(
                index_types=delete_index_types,
                document_id=document_id,
                index_data=index_data,
            )
        for doc_index in operations["delete"]:
            doc_index.mark_deleting()

    def _get_indexes_needing_reconciliation(self, session: Session) -> List[DocumentIndex]:
        """Get all indexes that need reconciliation"""
        stmt = select(DocumentIndex)
        result = session.execute(stmt)
        all_indexes = result.scalars().all()

        indexes_needing_reconciliation = []
        for doc_index in all_indexes:
            if doc_index.needs_reconciliation():
                indexes_needing_reconciliation.append(doc_index)

        return indexes_needing_reconciliation


# Index task completion callbacks
class IndexTaskCallbacks:
    """Callbacks for index task completion"""

    @staticmethod
    def _update_document_status(document_id: str, session: Session):
        stmt = select(Document).where(Document.id == document_id)
        result = session.execute(stmt)
        document = result.scalar_one_or_none()
        if document:
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
