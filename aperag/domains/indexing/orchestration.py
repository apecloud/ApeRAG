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

"""Pure orchestration helpers for indexing Celery workflow tasks.

These functions encapsulate the orchestration / aggregation logic that
used to live inside the ``trigger_*_workflow`` and
``notify_workflow_complete`` Celery tasks in this domain. They are
extracted as plain sync helpers so the logic can be reasoned about,
unit-tested, and reused without spinning up a Celery worker.

Per Phase 8 D4 (refined) canonical:
- Thin Celery task wrappers in ``aperag/domains/indexing/tasks.py``
  keep their ``@app.task`` decorators (chain/chord composition + chord
  callback contract require broker-registered tasks) but delegate
  their bodies to these helpers.
- chain/chord composition, reconciler/scheduler call sites, task name
  strings, and beat schedule entries are unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, List

from celery import chord, group

from aperag.tasks.models import IndexTaskResult, TaskStatus, WorkflowResult

logger = logging.getLogger(__name__)


def is_skipped_payload(payload: Any) -> bool:
    """A payload is "skipped" when carrying the sentinel ``status == "skipped"``.

    Public mirror of the previous private ``_is_skipped_payload`` helper.
    """
    return isinstance(payload, dict) and payload.get("status") == "skipped"


def build_dispatched_workflow_result(async_result) -> dict:
    """Return a small, JSON-serializable handoff payload for downstream tracking."""
    return {
        "status": "dispatched",
        "workflow_id": async_result.id,
    }


def build_index_workflow_chord(
    *,
    document_id: str,
    index_types: List[str],
    per_index_signature_factory,
    completion_callback_signature,
):
    """Build a ``chord(group(parallel_index_tasks), completion_callback)``.

    Pure orchestration: no I/O, no broker call. The caller decides when
    to ``.apply_async()`` the returned chord.

    Args:
        document_id: Document being indexed (for logging only).
        index_types: List of index types to fan out to.
        per_index_signature_factory: Callable ``(index_type) -> Signature``
            that produces a Celery signature for a single index_type.
            Encapsulates the per-task arguments (e.g. ``parsed_data``).
        completion_callback_signature: Celery signature for the chord
            callback (typically ``notify_workflow_complete.s(...)``).

    Returns:
        A ``celery.canvas.chord`` object ready to be ``.apply_async()``.
    """
    parallel_tasks = group([per_index_signature_factory(index_type) for index_type in index_types])
    workflow_chord = chord(parallel_tasks, completion_callback_signature)
    logger.debug(
        "Built index workflow chord for document %s with %d parallel index tasks",
        document_id,
        len(index_types),
    )
    return workflow_chord


def aggregate_workflow_results(
    *,
    index_results: List[dict],
    document_id: str,
    operation: str,
    index_types: List[str],
) -> WorkflowResult:
    """Aggregate per-index results from a chord body into a ``WorkflowResult``.

    Pure logic: parses/normalizes ``IndexTaskResult`` dicts, classifies
    them into successful / failed / skipped, derives an overall
    ``TaskStatus``, and constructs the final ``WorkflowResult`` payload.

    No I/O, no broker call. Safe to call without a running Celery worker.
    """
    successful_tasks: List[str] = []
    failed_tasks: List[str] = []
    skipped_tasks: List[str] = []
    normalized_results: List[IndexTaskResult] = []

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

    return WorkflowResult(
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


def build_workflow_failure_result(
    *,
    document_id: str,
    operation: str,
    index_types: List[str],
    error_message: str,
) -> WorkflowResult:
    """Construct a uniform failure ``WorkflowResult`` for the unexpected path
    in ``notify_workflow_complete``."""
    return WorkflowResult(
        workflow_id=f"{document_id}_{operation}",
        document_id=document_id,
        operation=operation,
        status=TaskStatus.FAILED,
        message=error_message,
        successful_indexes=[],
        failed_indexes=index_types,
        total_indexes=len(index_types),
        index_results=[],
    )
