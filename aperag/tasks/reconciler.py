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

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from aperag.config import get_sync_session
from aperag.db.models import (
    Collection,
    CollectionStatus,
    CollectionSummary,
    CollectionSummaryStatus,
    Document,
    DocumentIndex,
    DocumentIndexStatus,
    DocumentIndexType,
    DocumentStatus,
)
from aperag.schema.utils import parseCollectionConfig
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
        stmt = select(Document).where(
            Document.id == document_id,
            Document.status.not_in([DocumentStatus.DELETED, DocumentStatus.UPLOADED, DocumentStatus.EXPIRED]),
        )
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


class CollectionSummaryReconciler:
    """Reconciler for collection summaries using reconcile pattern"""

    def __init__(self, scheduler_type: str = "celery"):
        self.scheduler_type = scheduler_type

    def reconcile_all(self):
        """
        Main reconciliation loop - scan collections and reconcile summary differences
        """
        for session in get_sync_session():
            summaries_to_reconcile = self._get_summaries_needing_reconciliation(session)
            logger.info(f"Found {len(summaries_to_reconcile)} collection summaries need reconciliation")

            successful_reconciliations = 0
            failed_reconciliations = 0
            for summary in summaries_to_reconcile:
                try:
                    self._reconcile_single_summary(session, summary)
                    successful_reconciliations += 1
                except Exception as e:
                    failed_reconciliations += 1
                    logger.error(f"Failed to reconcile collection summary {summary.id}: {e}", exc_info=True)

            if successful_reconciliations > 0 or failed_reconciliations > 0:
                logger.info(
                    f"Summary reconciliation completed: {successful_reconciliations} successful, {failed_reconciliations} failed"
                )

    def _get_summaries_needing_reconciliation(self, session: Session) -> List[CollectionSummary]:
        """
        Get all collection summaries that need reconciliation
        """
        stmt = select(CollectionSummary).where(
            or_(
                CollectionSummary.version != CollectionSummary.observed_version,
                CollectionSummary.status != CollectionSummaryStatus.GENERATING,
            )
        )
        result = session.execute(stmt)
        return result.scalars().all()

    def _reconcile_single_summary(self, session: Session, summary: CollectionSummary):
        """
        Reconcile summary generation for a single collection summary
        """
        claimed = self._claim_summary_for_processing(session, summary.id, summary.version)

        if claimed:
            self._schedule_summary_generation(summary.id, summary.collection_id, summary.version)
            session.commit()
        else:
            logger.debug(
                f"Skipping summary {summary.id} - could not be claimed (likely already processing or version mismatch)"
            )

    def _claim_summary_for_processing(self, session: Session, summary_id: str, version: int) -> bool:
        """Atomically claim a summary for processing by updating its state and observed_version"""
        try:
            update_stmt = (
                update(CollectionSummary)
                .where(
                    and_(
                        CollectionSummary.id == summary_id,
                        CollectionSummary.status != CollectionSummaryStatus.GENERATING,
                        CollectionSummary.version == version,
                    )
                )
                .values(
                    status=CollectionSummaryStatus.GENERATING,
                    gmt_last_reconciled=utc_now(),
                    gmt_updated=utc_now(),
                )
            )
            result = session.execute(update_stmt)
            if result.rowcount > 0:
                logger.debug(f"Claimed summary {summary_id} (v{version}) for processing")
                session.flush()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to claim summary {summary_id}: {e}")
            session.rollback()
            return False

    def _schedule_summary_generation(self, summary_id: str, collection_id: str, target_version: int):
        """
        Schedule summary generation task
        """
        try:
            from config.celery_tasks import collection_summary_task

            task_result = collection_summary_task.delay(summary_id, collection_id, target_version)
            logger.info(
                f"Collection summary generation task scheduled for summary {summary_id} "
                f"(collection: {collection_id}, version: {target_version}), task ID: {task_result.id}"
            )
        except Exception as e:
            logger.error(f"Failed to schedule summary generation for {summary_id}: {e}")
            raise


class CollectionSummaryCallbacks:
    """Callbacks for collection summary task completion"""

    @staticmethod
    def on_summary_generated(summary_id: str, summary_content: str, target_version: int):
        """Called when summary generation succeeds"""
        try:
            for session in get_sync_session():
                # First, get the collection summary record to get collection_id
                summary_query = select(CollectionSummary).where(
                    and_(
                        CollectionSummary.id == summary_id,
                        CollectionSummary.status == CollectionSummaryStatus.GENERATING,
                        CollectionSummary.version == target_version,
                    )
                )
                summary_result = session.execute(summary_query)
                summary_record = summary_result.scalar_one_or_none()
                
                if not summary_record:
                    logger.warning(
                        f"Summary completion callback ignored for {summary_id} (v{target_version}) - not in expected state"
                    )
                    return

                collection_id = summary_record.collection_id
                
                # Get collection info to check if summary is enabled and get current gmt_updated
                collection_query = select(Collection).where(
                    and_(
                        Collection.id == collection_id,
                        Collection.gmt_deleted.is_(None)
                    )
                )
                collection_result = session.execute(collection_query)
                collection_record = collection_result.scalar_one_or_none()
                
                if not collection_record:
                    logger.error(f"Collection {collection_id} not found during summary completion")
                    return
                
                # Check if summary is enabled in collection config
                try:
                    config = parseCollectionConfig(collection_record.config)
                    is_summary_enabled = config.enable_summary
                except Exception as e:
                    logger.error(f"Failed to parse collection config for {collection_id}: {e}")
                    is_summary_enabled = False
                
                current_time = utc_now()
                collection_updated_time = collection_record.gmt_updated
                
                # Update collection_summary table
                summary_update_stmt = (
                    update(CollectionSummary)
                    .where(
                        and_(
                            CollectionSummary.id == summary_id,
                            CollectionSummary.status == CollectionSummaryStatus.GENERATING,
                            CollectionSummary.version == target_version,
                        )
                    )
                    .values(
                        status=CollectionSummaryStatus.COMPLETE,
                        summary=summary_content,
                        error_message=None,
                        observed_version=target_version,
                        gmt_updated=current_time,
                    )
                )
                summary_update_result = session.execute(summary_update_stmt)
                
                if summary_update_result.rowcount == 0:
                    session.rollback()
                    logger.warning(
                        f"Summary completion callback ignored for {summary_id} (v{target_version}) - summary not in expected state"
                    )
                    return
                
                # Update collection table if summary is enabled and collection hasn't been updated since we read it
                if is_summary_enabled and summary_content:
                    collection_update_stmt = (
                        update(Collection)
                        .where(
                            and_(
                                Collection.id == collection_id,
                                Collection.gmt_updated == collection_updated_time,  # Race condition prevention
                                Collection.gmt_deleted.is_(None)
                            )
                        )
                        .values(
                            description=summary_content,
                            gmt_updated=current_time,
                        )
                    )
                    collection_update_result = session.execute(collection_update_stmt)
                    
                    if collection_update_result.rowcount > 0:
                        logger.info(f"Updated collection {collection_id} description with generated summary")
                    else:
                        logger.warning(
                            f"Failed to update collection {collection_id} description - collection may have been modified concurrently"
                        )
                
                session.commit()
                logger.info(f"Collection summary generation completed for {summary_id} (v{target_version})")

        except Exception as e:
            logger.error(f"Failed to update collection summary completion for {summary_id}: {e}")
            try:
                session.rollback()
            except:
                pass

    @staticmethod
    def on_summary_failed(summary_id: str, error_message: str, target_version: int):
        """Called when summary generation fails"""
        try:
            for session in get_sync_session():
                update_stmt = (
                    update(CollectionSummary)
                    .where(
                        and_(
                            CollectionSummary.id == summary_id,
                            CollectionSummary.status == CollectionSummaryStatus.GENERATING,
                            CollectionSummary.version == target_version,
                        )
                    )
                    .values(
                        status=CollectionSummaryStatus.FAILED,
                        error_message=error_message,
                        gmt_updated=utc_now(),
                    )
                )
                result = session.execute(update_stmt)
                if result.rowcount > 0:
                    session.commit()
                    logger.error(
                        f"Collection summary generation failed for {summary_id} (v{target_version}): {error_message}"
                    )
                else:
                    session.rollback()
                    logger.warning(
                        f"Summary failure callback ignored for {summary_id} (v{target_version}) - not in expected state"
                    )
        except Exception as e:
            logger.error(f"Failed to update collection summary failure for {summary_id}: {e}")


class CollectionGCReconciler:
    def __init__(self, scheduler_type: str = "celery"):
        self.scheduler_type = scheduler_type

    def reconcile_all(self):
        collections = None
        for session in get_sync_session():
            stmt = select(Collection).where(
                or_(
                    Collection.status == CollectionStatus.ACTIVE,
                )
            )
            result = session.execute(stmt)
            collections = result.scalars().all()

        if not collections:
            return

        from aperag.tasks.collection import collection_task

        for collection in collections:
            collection_task.cleanup_expired_documents(collection.id)


# Global instances
index_reconciler = DocumentIndexReconciler()
index_task_callbacks = IndexTaskCallbacks()
collection_summary_reconciler = CollectionSummaryReconciler()
collection_summary_callbacks = CollectionSummaryCallbacks()
collection_gc_reconciler = CollectionGCReconciler()
