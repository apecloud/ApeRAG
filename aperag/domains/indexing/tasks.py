"""
Celery Task System for Document Indexing - Dynamic Workflow Architecture

Domain-owned tasks for the indexing domain. Moved from ``config/celery_tasks.py``
as part of phase-3 infra absorption (task #37 D4a). Pure move — no behavior
change. Task ``name="..."`` strings are pinned to ``config.celery_tasks.<name>``
to preserve task identity for in-flight queue messages.

This module implements a dynamic task system for document indexing with runtime workflow orchestration.
All tasks use structured data classes for parameter passing and result handling.

## Architecture Overview

The new task system is designed with the following principles:
1. **Fine-grained tasks**: Each operation (parse, create index, delete index, update index) is a separate task
2. **Dynamic workflow orchestration**: Tasks are composed at runtime using trigger tasks
3. **Parallel execution**: Index creation/update/deletion tasks run in parallel for better performance
4. **Individual retries**: Each task has its own retry mechanism with configurable parameters
5. **Runtime decision making**: Workflows can adapt based on document content and parsing results

## Task Flow Architecture

### Sequential Phase (Chain):
```
parse_document_task -> trigger_indexing_workflow
```

### Parallel Phase (Group + Chord):
```
[create_index_task(vector), create_index_task(fulltext), create_index_task(graph)] -> notify_workflow_complete
```

### Key Innovation: Dynamic Fan-out
The `trigger_indexing_workflow` task receives parsed document data and dynamically creates
the parallel index tasks, solving the static parameter passing limitation.

## Task Hierarchy

### Core Tasks:
- `parse_document_task`: Parse document content and extract metadata
- `create_index_task`: Create a single type of index (vector/fulltext/graph)
- `delete_index_task`: Delete a single type of index
- `update_index_task`: Update a single type of index

### Workflow Orchestration Tasks:
- `trigger_create_indexes_workflow`: Dynamic fan-out for index creation
- `trigger_delete_indexes_workflow`: Dynamic fan-out for index deletion
- `trigger_update_indexes_workflow`: Dynamic fan-out for index updates
- `notify_workflow_complete`: Aggregation task for workflow completion

### Workflow Entry Points:
- `create_document_indexes_workflow()`: Chain composition function
- `delete_document_indexes_workflow()`: Chain composition function
- `update_document_indexes_workflow()`: Chain composition function
"""

import json
import logging
from typing import Any, Callable, List

from celery import Task, chain, chord, current_app, group

from aperag.docparser.base import ParserError
from aperag.tasks.document import document_index_task
from aperag.tasks.models import (
    IndexTaskResult,
    ParsedDocumentData,
    TaskStatus,
    WorkflowResult,
)
from aperag.tasks.processing_lease import (
    DEFAULT_PROCESSING_LEASE_RENEW_INTERVAL_SECONDS,
    DEFAULT_PROCESSING_LEASE_TTL_SECONDS,
    ProcessingLeaseRenewer,
    build_lease_expires_at,
)
from aperag.utils.constant import IndexAction
from aperag.utils.utils import utc_now

logger = logging.getLogger()


def _build_skipped_payload(reason: str, **payload) -> dict:
    payload.update({"status": "skipped", "reason": reason})
    return payload


def _build_skipped_task_result(document_id: str, index_type: str, reason: str) -> dict:
    return _build_skipped_payload(reason, document_id=document_id, index_type=index_type)


def _is_skipped_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("status") == "skipped"


def _validate_task_relevance(
    document_id: str,
    index_type: str,
    target_version: int,
    expected_status,
    processing_token: str = None,
):
    """
    Double-check the database to ensure the task is still valid.

    Returns a dictionary with a 'skipped' status if the task is no longer relevant,
    otherwise returns None.
    """
    from sqlalchemy import and_, select

    from aperag.config import get_sync_session
    from aperag.domains.indexing.db.models import (
        DocumentIndex,
        DocumentIndexType,
    )
    from aperag.domains.knowledge_base.db.models import (
        Document,
        DocumentStatus,
    )

    for session in get_sync_session():
        # Check document index status
        stmt = select(DocumentIndex).where(
            and_(DocumentIndex.document_id == document_id, DocumentIndex.index_type == DocumentIndexType(index_type))
        )
        result = session.execute(stmt)
        db_index = result.scalar_one_or_none()

        if not db_index:
            logger.info(f"Index record not found for {document_id}:{index_type}, skipping task.")
            return _build_skipped_task_result(document_id, index_type, "index_record_not_found")

        if db_index.status != expected_status:
            logger.info(
                f"Index status for {document_id}:{index_type} changed to {db_index.status} (expected {expected_status}), skipping task."
            )
            return _build_skipped_task_result(document_id, index_type, f"status_changed_to_{db_index.status}")

        if target_version and db_index.version != target_version:
            logger.info(
                f"Version mismatch for {document_id}:{index_type}, expected: {target_version}, current: {db_index.version}, skipping task."
            )
            return _build_skipped_task_result(
                document_id, index_type, f"version_mismatch_expected_{target_version}_current_{db_index.version}"
            )

        if processing_token and db_index.processing_token != processing_token:
            logger.info(
                "Processing token mismatch for %s:%s, expected: %s, current: %s, skipping task.",
                document_id,
                index_type,
                processing_token,
                db_index.processing_token,
            )
            return _build_skipped_task_result(document_id, index_type, "token_mismatch")

        # Check document status - if document is UPLOADED or EXPIRED, task should be skipped
        doc_stmt = select(Document).where(Document.id == document_id)
        doc_result = session.execute(doc_stmt)
        document = doc_result.scalar_one_or_none()

        if not document:
            logger.info(f"Document {document_id} not found, skipping task.")
            return _build_skipped_task_result(document_id, index_type, "document_not_found")

        if document.status in [DocumentStatus.UPLOADED, DocumentStatus.EXPIRED]:
            logger.info(f"Document {document_id} status is {document.status}, skipping task.")
            return _build_skipped_task_result(document_id, index_type, f"document_status_{document.status}")

        return None  # Task is still relevant


def _get_index_context_value(context: dict, index_type: str, key_suffix: str):
    context = context or {}
    return context.get(f"{index_type}_{key_suffix}")


def _require_index_processing_context(index_type: str, context: dict, *, require_version: bool = True) -> dict:
    target_version = _get_index_context_value(context, index_type, "version")
    processing_token = _get_index_context_value(context, index_type, "processing_token")
    index_id = _get_index_context_value(context, index_type, "index_id")

    missing_fields = []
    if require_version and target_version is None:
        missing_fields.append("version")
    if not processing_token:
        missing_fields.append("processing_token")
    if index_id is None:
        missing_fields.append("index_id")

    if missing_fields:
        raise ValueError(f"Missing processing context for {index_type}: {', '.join(missing_fields)}")

    return {
        "target_version": target_version,
        "processing_token": processing_token,
        "index_id": index_id,
    }


def _build_document_index_lease_targets(index_types: List[str], context: dict, expected_status) -> List[dict]:
    targets = []
    for index_type in index_types:
        target_context = _require_index_processing_context(
            index_type,
            context,
            require_version=expected_status.name != "DELETION_IN_PROGRESS",
        )
        target = {
            "index_type": index_type,
            "index_id": target_context["index_id"],
            "processing_token": target_context["processing_token"],
            "target_version": target_context["target_version"],
            "expected_status": expected_status,
        }
        targets.append(target)
    return targets


def _validate_index_batch_relevance(document_id: str, index_types: List[str], context: dict, expected_status):
    for index_type in index_types:
        target_context = _require_index_processing_context(
            index_type,
            context,
            require_version=expected_status.name != "DELETION_IN_PROGRESS",
        )
        skip_reason = _validate_task_relevance(
            document_id,
            index_type,
            target_context["target_version"],
            expected_status,
            processing_token=target_context["processing_token"],
        )
        if skip_reason:
            return skip_reason
    return None


def _renew_document_index_leases(targets: List[dict]) -> bool:
    from sqlalchemy import and_, update

    from aperag.config import get_sync_session
    from aperag.domains.indexing.db.models import DocumentIndex

    if not targets:
        return False

    current_time = utc_now()
    next_expiry = build_lease_expires_at(DEFAULT_PROCESSING_LEASE_TTL_SECONDS)

    for session in get_sync_session():
        for target in targets:
            conditions = [
                DocumentIndex.id == target["index_id"],
                DocumentIndex.status == target["expected_status"],
                DocumentIndex.processing_token == target["processing_token"],
            ]
            if target.get("target_version") is not None:
                conditions.append(DocumentIndex.version == target["target_version"])

            renew_stmt = (
                update(DocumentIndex)
                .where(and_(*conditions))
                .values(
                    lease_expires_at=next_expiry,
                    gmt_updated=current_time,
                )
            )
            result = session.execute(renew_stmt)
            if result.rowcount == 0:
                session.rollback()
                return False

        session.commit()
        return True
    return False


def _make_document_index_lease_renewer(targets: List[dict], description: str) -> ProcessingLeaseRenewer:
    return ProcessingLeaseRenewer(
        lambda: _renew_document_index_leases(targets),
        interval_seconds=DEFAULT_PROCESSING_LEASE_RENEW_INTERVAL_SECONDS,
        description=description,
    )


def _build_dispatched_workflow_result(async_result) -> dict:
    """Return a small, JSON-serializable handoff payload for downstream workflow tracking."""
    return {
        "status": "dispatched",
        "workflow_id": async_result.id,
    }


def _handle_ownership_lost(*, payload_factory: Callable[[], dict], log_message: str):
    logger.warning("%s", log_message)
    return payload_factory()


class BaseIndexTask(Task):
    """
    Base class for all index tasks
    """

    abstract = True

    def _handle_index_success(
        self,
        document_id: str,
        index_type: str,
        target_version: int,
        processing_token: str,
        index_data: dict = None,
    ):
        try:
            from aperag.tasks.reconciler import index_task_callbacks

            index_data_json = json.dumps(index_data) if index_data else None
            index_task_callbacks.on_index_created(
                document_id,
                index_type,
                target_version,
                processing_token,
                index_data_json,
            )
            logger.info(
                "Index success callback executed for %s index of document %s (v%s)",
                index_type,
                document_id,
                target_version,
            )
        except Exception as e:
            logger.warning(
                "Failed to execute index success callback for %s of %s v%s: %s",
                index_type,
                document_id,
                target_version,
                e,
                exc_info=True,
            )

    def _handle_index_deletion_success(self, document_id: str, index_type: str, processing_token: str):
        try:
            from aperag.tasks.reconciler import index_task_callbacks

            index_task_callbacks.on_index_deleted(document_id, index_type, processing_token)
            logger.info(f"Index deletion callback executed for {index_type} index of document {document_id}")
        except Exception as e:
            logger.warning(
                f"Failed to execute index deletion callback for {index_type} of {document_id}: {e}", exc_info=True
            )

    def _handle_index_failure(
        self,
        document_id: str,
        index_types: List[str],
        error_msg: str,
        *,
        context: dict = None,
        expected_status=None,
    ):
        try:
            from aperag.domains.indexing.db.models import DocumentIndexStatus
            from aperag.tasks.reconciler import index_task_callbacks

            expected_status = expected_status or DocumentIndexStatus.CREATING
            for index_type in index_types:
                target_context = _require_index_processing_context(
                    index_type,
                    context,
                    require_version=expected_status != DocumentIndexStatus.DELETION_IN_PROGRESS,
                )
                index_task_callbacks.on_index_failed(
                    document_id,
                    index_type,
                    error_msg,
                    target_context["processing_token"],
                    target_version=target_context["target_version"],
                    expected_status=expected_status,
                )
            logger.info(f"Index failure callback executed for {index_types} indexes of document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to execute index failure callback for {document_id}: {e}", exc_info=True)


# ========== Core Document Processing Tasks ==========


@current_app.task(
    bind=True,
    base=BaseIndexTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
    name="config.celery_tasks.parse_document_task",
)
def parse_document_task(self, document_id: str, index_types: List[str], context: dict = None) -> dict:
    """
    Parse document content task

    Args:
        document_id: Document ID to parse

    Returns:
        Serialized ParsedDocumentData
    """
    from aperag.domains.indexing.db.models import DocumentIndexStatus

    context = context or {}
    renewer = None

    try:
        skip_reason = _validate_index_batch_relevance(
            document_id,
            index_types,
            context,
            DocumentIndexStatus.CREATING,
        )
        if skip_reason:
            return skip_reason

        renewer = _make_document_index_lease_renewer(
            _build_document_index_lease_targets(index_types, context, DocumentIndexStatus.CREATING),
            f"parse-document:{document_id}",
        )
        renewer.start()

        logger.info(f"Starting to parse document {document_id}")
        parsed_data = document_index_task.parse_document(document_id)

        if renewer.ownership_lost:
            return _handle_ownership_lost(
                payload_factory=lambda: _build_skipped_payload("ownership_lost", document_id=document_id),
                log_message=(
                    f"Processing ownership lost while parsing document {document_id}; " "dropping downstream callbacks"
                ),
            )

        logger.info(f"Successfully parsed document {document_id}")
        return parsed_data.to_dict()
    except ParserError as e:
        error_msg = f"Failed to parse document {document_id}: {e.diagnostic_message()}"
        logger.error(error_msg, exc_info=True)

        # Only mark as failed if all retries are exhausted
        if self.request.retries >= self.max_retries:
            self._handle_index_failure(document_id, index_types, error_msg)

        raise
    except Exception as e:
        if renewer and renewer.ownership_lost:
            return _handle_ownership_lost(
                payload_factory=lambda: _build_skipped_payload("ownership_lost", document_id=document_id),
                log_message=(
                    f"Processing ownership lost while parsing document {document_id}; "
                    "suppressing parse failure callback"
                ),
            )

        error_msg = f"Failed to parse document {document_id}: source=runtime, code=parse_failed, detail={str(e)}"
        logger.error(error_msg, exc_info=True)

        # Only mark as failed if all retries are exhausted
        if self.request.retries >= self.max_retries:
            self._handle_index_failure(document_id, index_types, error_msg, context=context)

        raise
    finally:
        if renewer:
            renewer.stop()


@current_app.task(
    bind=True,
    base=BaseIndexTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
    name="config.celery_tasks.create_index_task",
)
def create_index_task(self, document_id: str, index_type: str, parsed_data_dict: dict, context: dict = None) -> dict:
    """
    Create a single index for a document with distributed locking

    Args:
        document_id: Document ID to process
        index_type: Type of index to create ('vector', 'fulltext', 'graph')
        parsed_data_dict: Serialized ParsedDocumentData from parse_document_task
        context: Task context including index version

    Returns:
        Serialized IndexTaskResult
    """
    from aperag.domains.indexing.db.models import DocumentIndexStatus

    context = context or {}
    target_context = _require_index_processing_context(index_type, context)
    target_version = target_context["target_version"]
    processing_token = target_context["processing_token"]
    renewer = None

    try:
        logger.info(f"Starting to create {index_type} index for document {document_id} (v{target_version})")

        # Double-check: verify task is still valid
        skip_reason = _validate_task_relevance(
            document_id,
            index_type,
            target_version,
            DocumentIndexStatus.CREATING,
            processing_token=processing_token,
        )
        if skip_reason:
            return skip_reason

        renewer = _make_document_index_lease_renewer(
            _build_document_index_lease_targets([index_type], context, DocumentIndexStatus.CREATING),
            f"create-index:{document_id}:{index_type}",
        )
        renewer.start()

        # Convert dict back to structured data
        parsed_data = ParsedDocumentData.from_dict(parsed_data_dict)

        # Execute index creation
        result = document_index_task.create_index(document_id, index_type, parsed_data)

        # Check if the operation failed and raise exception to trigger retry
        if not result.success:
            error_msg = f"Failed to create {index_type} index for document {document_id}: {result.error}"
            logger.error(error_msg)
            raise Exception(error_msg)

        if renewer.ownership_lost:
            return _handle_ownership_lost(
                payload_factory=lambda: _build_skipped_task_result(document_id, index_type, "ownership_lost"),
                log_message=(
                    f"Processing ownership lost for create {index_type} index on document {document_id}; "
                    "dropping success callback"
                ),
            )

        # Handle success callback with version validation
        logger.info(f"Successfully created {index_type} index for document {document_id} (v{target_version})")
        self._handle_index_success(document_id, index_type, target_version, processing_token, result.data)

        return result.to_dict()

    except Exception as e:
        if renewer and renewer.ownership_lost:
            return _handle_ownership_lost(
                payload_factory=lambda: _build_skipped_task_result(document_id, index_type, "ownership_lost"),
                log_message=(
                    f"Processing ownership lost for create {index_type} index on document {document_id}; "
                    "suppressing failure callback"
                ),
            )

        error_msg = f"Failed to create {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Only mark as failed if all retries are exhausted
        if self.request.retries >= self.max_retries:
            self._handle_index_failure(document_id, [index_type], error_msg, context=context)

        raise
    finally:
        if renewer:
            renewer.stop()


@current_app.task(
    bind=True,
    base=BaseIndexTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
    name="config.celery_tasks.delete_index_task",
)
def delete_index_task(self, document_id: str, index_type: str, context: dict = None) -> dict:
    """
    Delete a single index for a document

    Args:
        document_id: Document ID to process
        index_type: Type of index to delete ('vector', 'fulltext', 'graph')

    Returns:
        Serialized IndexTaskResult
    """
    from aperag.domains.indexing.db.models import DocumentIndexStatus

    context = context or {}
    target_context = _require_index_processing_context(index_type, context, require_version=False)
    processing_token = target_context["processing_token"]
    renewer = None

    try:
        logger.info(f"Starting to delete {index_type} index for document {document_id}")

        skip_reason = _validate_task_relevance(
            document_id,
            index_type,
            None,
            DocumentIndexStatus.DELETION_IN_PROGRESS,
            processing_token=processing_token,
        )
        if skip_reason:
            return skip_reason

        renewer = _make_document_index_lease_renewer(
            _build_document_index_lease_targets([index_type], context, DocumentIndexStatus.DELETION_IN_PROGRESS),
            f"delete-index:{document_id}:{index_type}",
        )
        renewer.start()

        # Execute index deletion
        result = document_index_task.delete_index(document_id, index_type)

        # Check if the operation failed and raise exception to trigger retry
        if not result.success:
            error_msg = f"Failed to delete {index_type} index for document {document_id}: {result.error}"
            logger.error(error_msg)
            raise Exception(error_msg)

        if renewer.ownership_lost:
            return _handle_ownership_lost(
                payload_factory=lambda: _build_skipped_task_result(document_id, index_type, "ownership_lost"),
                log_message=(
                    f"Processing ownership lost for delete {index_type} index on document {document_id}; "
                    "dropping success callback"
                ),
            )

        # Handle success callback
        logger.info(f"Successfully deleted {index_type} index for document {document_id}")
        self._handle_index_deletion_success(document_id, index_type, processing_token)

        return result.to_dict()

    except Exception as e:
        if renewer and renewer.ownership_lost:
            return _handle_ownership_lost(
                payload_factory=lambda: _build_skipped_task_result(document_id, index_type, "ownership_lost"),
                log_message=(
                    f"Processing ownership lost for delete {index_type} index on document {document_id}; "
                    "suppressing failure callback"
                ),
            )

        error_msg = f"Failed to delete {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Only mark as failed if all retries are exhausted
        if self.request.retries >= self.max_retries:
            self._handle_index_failure(
                document_id,
                [index_type],
                error_msg,
                context=context,
                expected_status=DocumentIndexStatus.DELETION_IN_PROGRESS,
            )

        raise
    finally:
        if renewer:
            renewer.stop()


@current_app.task(
    bind=True,
    base=BaseIndexTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
    name="config.celery_tasks.update_index_task",
)
def update_index_task(self, document_id: str, index_type: str, parsed_data_dict: dict, context: dict = None) -> dict:
    """
    Update a single index for a document with distributed locking

    Args:
        document_id: Document ID to process
        index_type: Type of index to update ('vector', 'fulltext', 'graph')
        parsed_data_dict: Serialized ParsedDocumentData from parse_document_task
        context: Task context including index version

    Returns:
        Serialized IndexTaskResult
    """
    from aperag.domains.indexing.db.models import DocumentIndexStatus

    context = context or {}
    target_context = _require_index_processing_context(index_type, context)
    target_version = target_context["target_version"]
    processing_token = target_context["processing_token"]
    renewer = None

    try:
        logger.info(f"Starting to update {index_type} index for document {document_id} (v{target_version})")

        # Double-check: verify task is still valid
        skip_reason = _validate_task_relevance(
            document_id,
            index_type,
            target_version,
            DocumentIndexStatus.CREATING,
            processing_token=processing_token,
        )
        if skip_reason:
            return skip_reason

        renewer = _make_document_index_lease_renewer(
            _build_document_index_lease_targets([index_type], context, DocumentIndexStatus.CREATING),
            f"update-index:{document_id}:{index_type}",
        )
        renewer.start()

        # Convert dict back to structured data
        parsed_data = ParsedDocumentData.from_dict(parsed_data_dict)

        # Execute index update
        result = document_index_task.update_index(document_id, index_type, parsed_data)

        # Check if the operation failed and raise exception to trigger retry
        if not result.success:
            error_msg = f"Failed to update {index_type} index for document {document_id}: {result.error}"
            logger.error(error_msg)
            raise Exception(error_msg)

        if renewer.ownership_lost:
            return _handle_ownership_lost(
                payload_factory=lambda: _build_skipped_task_result(document_id, index_type, "ownership_lost"),
                log_message=(
                    f"Processing ownership lost for update {index_type} index on document {document_id}; "
                    "dropping success callback"
                ),
            )

        # Handle success callback with version validation
        logger.info(f"Successfully updated {index_type} index for document {document_id} (v{target_version})")
        self._handle_index_success(document_id, index_type, target_version, processing_token, result.data)

        return result.to_dict()

    except Exception as e:
        if renewer and renewer.ownership_lost:
            return _handle_ownership_lost(
                payload_factory=lambda: _build_skipped_task_result(document_id, index_type, "ownership_lost"),
                log_message=(
                    f"Processing ownership lost for update {index_type} index on document {document_id}; "
                    "suppressing failure callback"
                ),
            )

        error_msg = f"Failed to update {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Only mark as failed if all retries are exhausted
        if self.request.retries >= self.max_retries:
            self._handle_index_failure(document_id, [index_type], error_msg, context=context)

        raise
    finally:
        if renewer:
            renewer.stop()


# ========== Dynamic Workflow Orchestration Tasks ==========


@current_app.task(bind=True, name="config.celery_tasks.trigger_create_indexes_workflow")
def trigger_create_indexes_workflow(
    self, parsed_data_dict: dict, document_id: str, index_types: List[str], context: dict = None
) -> Any:
    """
    Dynamic orchestration task for index creation workflow.

    This task acts as a fan-out point, receiving parsed document data and dynamically
    creating parallel index creation tasks based on the actual parsed content.

    Args:
        parsed_data_dict: Serialized ParsedDocumentData from parse_document_task
        document_id: Document ID to process
        index_types: List of index types to create

    Returns:
        Chord signature for parallel index creation + completion notification
    """
    try:
        if _is_skipped_payload(parsed_data_dict):
            logger.warning(
                "Skipping create-index workflow fan-out for document %s because parse stage returned %s",
                document_id,
                parsed_data_dict.get("reason"),
            )
            return parsed_data_dict

        logger.info(f"Triggering parallel index creation for document {document_id} with types: {index_types}")

        # Dynamically create parallel index creation tasks
        parallel_index_tasks = group(
            [create_index_task.s(document_id, index_type, parsed_data_dict, context) for index_type in index_types]
        )

        # Create a chord that executes the completion notification after all create tasks are done
        workflow_chord = chord(
            parallel_index_tasks, notify_workflow_complete.s(document_id, IndexAction.CREATE, index_types)
        )

        # Execute the chord
        chord_async_result = workflow_chord.apply_async()

        return _build_dispatched_workflow_result(chord_async_result)

    except Exception as e:
        error_msg = f"Failed to trigger create indexes workflow: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@current_app.task(bind=True, name="config.celery_tasks.trigger_delete_indexes_workflow")
def trigger_delete_indexes_workflow(self, document_id: str, index_types: List[str], context: dict = None) -> Any:
    """
    Dynamic orchestration task for index deletion workflow.

    Args:
        document_id: Document ID to process
        index_types: List of index types to delete

    Returns:
        Chord signature for parallel index deletion + completion notification
    """
    try:
        logger.info(f"Triggering parallel index deletion for document {document_id} with types: {index_types}")

        # Create parallel index deletion tasks
        parallel_delete_tasks = group(
            [delete_index_task.s(document_id, index_type, context) for index_type in index_types]
        )

        # Create a chord that executes the completion notification after all delete tasks are done
        workflow_chord = chord(
            parallel_delete_tasks, notify_workflow_complete.s(document_id, IndexAction.DELETE, index_types)
        )

        # Execute the chord
        chord_async_result = workflow_chord.apply_async()

        return _build_dispatched_workflow_result(chord_async_result)

    except Exception as e:
        error_msg = f"Failed to trigger delete indexes workflow: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@current_app.task(bind=True, name="config.celery_tasks.trigger_update_indexes_workflow")
def trigger_update_indexes_workflow(
    self, parsed_data_dict: dict, document_id: str, index_types: List[str], context: dict = None
) -> Any:
    """
    Dynamic orchestration task for index update workflow.

    Args:
        parsed_data_dict: Serialized ParsedDocumentData from parse_document_task
        document_id: Document ID to process
        index_types: List of index types to update

    Returns:
        Chord signature for parallel index update + completion notification
    """
    try:
        if _is_skipped_payload(parsed_data_dict):
            logger.warning(
                "Skipping update-index workflow fan-out for document %s because parse stage returned %s",
                document_id,
                parsed_data_dict.get("reason"),
            )
            return parsed_data_dict

        logger.info(f"Triggering parallel index update for document {document_id} with types: {index_types}")

        # Create parallel index update tasks
        parallel_update_tasks = group(
            [update_index_task.s(document_id, index_type, parsed_data_dict, context) for index_type in index_types]
        )

        # Create chord: parallel tasks + completion notification
        workflow_chord = chord(
            parallel_update_tasks, notify_workflow_complete.s(document_id, IndexAction.UPDATE, index_types)
        )

        chord_async_result = workflow_chord.apply_async()

        return _build_dispatched_workflow_result(chord_async_result)

    except Exception as e:
        error_msg = f"Failed to trigger update indexes workflow: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@current_app.task(bind=True, base=BaseIndexTask, name="config.celery_tasks.notify_workflow_complete")
def notify_workflow_complete(
    self, index_results: List[dict], document_id: str, operation: str, index_types: List[str]
) -> dict:
    """
    Workflow completion notification task.

    This task is called after all parallel index operations complete,
    aggregating results and providing final workflow status.

    Args:
        index_results: List of IndexTaskResult dicts from parallel tasks
        document_id: Document ID that was processed
        operation: Operation type ('create', 'delete', 'update')
        index_types: List of index types that were processed

    Returns:
        Serialized WorkflowResult
    """
    try:
        logger.info(f"Workflow {operation} completed for document {document_id}")
        logger.info(f"Index results: {index_results}")

        # Analyze results
        successful_tasks = []
        failed_tasks = []
        skipped_tasks = []
        normalized_results = []

        for result_dict in index_results:
            if isinstance(result_dict, dict) and result_dict.get("status") == "skipped":
                skipped_tasks.append(result_dict.get("index_type", "unknown"))
                continue
            try:
                result = IndexTaskResult.from_dict(result_dict)
                normalized_results.append(result)
                if result.success:
                    successful_tasks.append(result.index_type)
                else:
                    failed_tasks.append(f"{result.index_type}: {result.error}")
            except Exception as e:
                failed_tasks.append(f"unknown: {str(e)}")

        # Determine overall status
        if not failed_tasks:
            status = TaskStatus.SUCCESS
            processed_indexes = successful_tasks if successful_tasks else skipped_tasks
            status_message = (
                f"Document {document_id} {operation} COMPLETED SUCCESSFULLY! "
                f"Processed indexes: {', '.join(processed_indexes)}"
            )
            if skipped_tasks:
                status_message += f". Skipped: {', '.join(skipped_tasks)}"
            logger.info(status_message)
        elif successful_tasks:
            status = TaskStatus.PARTIAL_SUCCESS
            status_message = (
                f"Document {document_id} {operation} COMPLETED with WARNINGS. "
                f"Success: {', '.join(successful_tasks)}. Failures: {'; '.join(failed_tasks)}"
            )
            if skipped_tasks:
                status_message += f". Skipped: {', '.join(skipped_tasks)}"
            logger.warning(status_message)
        else:
            status = TaskStatus.FAILED
            status_message = f"Document {document_id} {operation} FAILED. All tasks failed: {'; '.join(failed_tasks)}"
            logger.error(status_message)

        # Create workflow result
        workflow_result = WorkflowResult(
            workflow_id=f"{document_id}_{operation}",
            document_id=document_id,
            operation=operation,
            status=status,
            message=status_message,
            successful_indexes=successful_tasks,
            failed_indexes=[f.split(":")[0] for f in failed_tasks],
            total_indexes=len(index_types),
            index_results=normalized_results,
        )

        return workflow_result.to_dict()

    except Exception as e:
        error_msg = f"Failed to process workflow completion for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Return failure result
        workflow_result = WorkflowResult(
            workflow_id=f"{document_id}_{operation}",
            document_id=document_id,
            operation=operation,
            status=TaskStatus.FAILED,
            message=error_msg,
            successful_indexes=[],
            failed_indexes=index_types,
            total_indexes=len(index_types),
            index_results=[],
        )

        return workflow_result.to_dict()


# ========== Workflow Entry Point Functions ==========


def create_document_indexes_workflow(document_id: str, index_types: List[str], context: dict = None):
    """
    Create indexes for a document using dynamic workflow orchestration.

    This function composes a chain that:
    1. Parses the document
    2. Dynamically triggers parallel index creation based on parsed content
    3. Aggregates results and notifies completion

    Args:
        document_id: Document ID to process
        index_types: List of index types to create

    Returns:
        AsyncResult for the workflow chain
    """
    logger.info(f"Starting create indexes workflow for document {document_id} with types: {index_types}")
    # Create the workflow chain: parse -> dynamic trigger
    workflow_chain = chain(
        parse_document_task.s(document_id, index_types, context),
        trigger_create_indexes_workflow.s(document_id, index_types, context),
    )

    # Submit the workflow
    workflow_result = workflow_chain.delay()
    logger.info(f"Create indexes workflow submitted for document {document_id}, workflow ID: {workflow_result.id}")

    return workflow_result


def delete_document_indexes_workflow(document_id: str, index_types: List[str], context: dict = None):
    """
    Delete indexes for a document using dynamic workflow orchestration.

    Args:
        document_id: Document ID to process
        index_types: List of index types to delete

    Returns:
        AsyncResult for the workflow
    """
    logger.info(f"Starting delete indexes workflow for document {document_id} with types: {index_types}")

    # For deletion, we don't need parsing, so we directly trigger the delete workflow
    workflow_result = trigger_delete_indexes_workflow.delay(document_id, index_types, context)
    logger.info(f"Delete indexes workflow submitted for document {document_id}, workflow ID: {workflow_result.id}")

    return workflow_result


def update_document_indexes_workflow(document_id: str, index_types: List[str], context: dict = None):
    """
    Update indexes for a document using dynamic workflow orchestration.

    This function composes a chain that:
    1. Re-parses the document to get updated content
    2. Dynamically triggers parallel index updates based on parsed content
    3. Aggregates results and notifies completion

    Args:
        document_id: Document ID to process
        index_types: List of index types to update

    Returns:
        AsyncResult for the workflow chain
    """
    logger.info(f"Starting update indexes workflow for document {document_id} with types: {index_types}")

    # Create the workflow chain: parse -> dynamic trigger
    workflow_chain = chain(
        parse_document_task.s(document_id, index_types, context),
        trigger_update_indexes_workflow.s(document_id, index_types, context),
    )

    # Submit the workflow
    workflow_result = workflow_chain.delay()
    logger.info(f"Update indexes workflow submitted for document {document_id}, workflow ID: {workflow_result.id}")

    return workflow_result


# ========== Reconcile Tasks ==========


@current_app.task(name="config.celery_tasks.reconcile_indexes_task")
def reconcile_indexes_task():
    """Periodic task to reconcile index specs with statuses"""
    try:
        logger.info("Starting index reconciliation")

        # Import here to avoid circular dependencies
        from aperag.tasks.reconciler import index_reconciler

        # Run reconciliation
        index_reconciler.reconcile_all()

        logger.info("Index reconciliation completed")

    except Exception as e:
        logger.error(f"Index reconciliation failed: {e}", exc_info=True)
        raise
