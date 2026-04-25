"""Celery tasks owned by the knowledge_base domain.

Domain-owned tasks for the knowledge_base domain. Moved from
``config/celery_tasks.py`` as part of phase-3 infra absorption (task #37 D4a).
Pure move — no behavior change. Task ``name="..."`` strings are pinned to
``config.celery_tasks.<name>`` to preserve task identity for in-flight queue
messages.

Scope:
- Collection lifecycle tasks (init / delete)
- Collection summary generation + reconciliation
- Document GC (cleanup of expired uploads)

Note: Collection export packaging (``export_collection_task``) remains in
``config/export_tasks.py`` until #39 carves the ``ExportTask`` ORM into the
KB domain (G1 boundary blocks moving the task body before the ORM mirror
exists in ``aperag/domains/knowledge_base/db/models.py``).
"""

import logging
from typing import Any, Callable

from celery import current_app

from aperag.tasks.collection import collection_task
from aperag.tasks.processing_lease import (
    DEFAULT_PROCESSING_LEASE_RENEW_INTERVAL_SECONDS,
    DEFAULT_PROCESSING_LEASE_TTL_SECONDS,
    ProcessingLeaseRenewer,
    build_lease_expires_at,
)
from aperag.tasks.utils import TaskConfig
from aperag.utils.utils import utc_now
from config.celery import app

logger = logging.getLogger(__name__)


# ========== Internal helpers ==========


def _build_skipped_payload(reason: str, **payload) -> dict:
    payload.update({"status": "skipped", "reason": reason})
    return payload


def _handle_ownership_lost(*, payload_factory: Callable[[], dict], log_message: str):
    logger.warning("%s", log_message)
    return payload_factory()


def _renew_collection_summary_lease(summary_id: str, target_version: int, processing_token: str) -> bool:
    from sqlalchemy import and_, update

    from aperag.config import get_sync_session
    from aperag.domains.knowledge_base.db.models import (
        CollectionSummary,
        CollectionSummaryStatus,
    )

    current_time = utc_now()
    next_expiry = build_lease_expires_at(DEFAULT_PROCESSING_LEASE_TTL_SECONDS)

    for session in get_sync_session():
        renew_stmt = (
            update(CollectionSummary)
            .where(
                and_(
                    CollectionSummary.id == summary_id,
                    CollectionSummary.status == CollectionSummaryStatus.GENERATING,
                    CollectionSummary.version == target_version,
                    CollectionSummary.processing_token == processing_token,
                )
            )
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


def _make_collection_summary_lease_renewer(
    summary_id: str, target_version: int, processing_token: str
) -> ProcessingLeaseRenewer:
    return ProcessingLeaseRenewer(
        lambda: _renew_collection_summary_lease(summary_id, target_version, processing_token),
        interval_seconds=DEFAULT_PROCESSING_LEASE_RENEW_INTERVAL_SECONDS,
        description=f"collection-summary:{summary_id}",
    )


def _validate_collection_summary_relevance(summary_id: str, target_version: int, processing_token: str):
    from sqlalchemy import select

    from aperag.config import get_sync_session
    from aperag.domains.knowledge_base.db.models import (
        CollectionSummary,
        CollectionSummaryStatus,
    )

    for session in get_sync_session():
        stmt = select(CollectionSummary).where(CollectionSummary.id == summary_id)
        result = session.execute(stmt)
        summary = result.scalar_one_or_none()

        if not summary:
            logger.info("Collection summary %s not found, skipping task.", summary_id)
            return _build_skipped_payload("summary_record_not_found", summary_id=summary_id)

        if summary.status != CollectionSummaryStatus.GENERATING:
            logger.info(
                "Collection summary %s status changed to %s (expected %s), skipping task.",
                summary_id,
                summary.status,
                CollectionSummaryStatus.GENERATING,
            )
            return _build_skipped_payload(f"status_changed_to_{summary.status}", summary_id=summary_id)

        if summary.version != target_version:
            logger.info(
                "Collection summary %s version mismatch, expected %s current %s, skipping task.",
                summary_id,
                target_version,
                summary.version,
            )
            return _build_skipped_payload(
                f"version_mismatch_expected_{target_version}_current_{summary.version}",
                summary_id=summary_id,
            )

        if summary.processing_token != processing_token:
            logger.info(
                "Collection summary %s token mismatch, expected %s current %s, skipping task.",
                summary_id,
                processing_token,
                summary.processing_token,
            )
            return _build_skipped_payload("token_mismatch", summary_id=summary_id)

        return None


# ========== Collection Tasks ==========


@current_app.task(name="config.celery_tasks.reconcile_collection_summaries_task")
def reconcile_collection_summaries_task():
    """Periodic task to reconcile collection summary specs with statuses"""
    try:
        logger.info("Starting collection summary reconciliation")

        # Import here to avoid circular dependencies
        from aperag.tasks.reconciler import collection_summary_reconciler

        # Run reconciliation
        collection_summary_reconciler.reconcile_all()

        logger.info("Collection summary reconciliation completed")

    except Exception as e:
        logger.error(f"Collection summary reconciliation failed: {e}", exc_info=True)
        raise


@app.task(bind=True, name="config.celery_tasks.collection_delete_task")
def collection_delete_task(self, collection_id: str) -> Any:
    """
    Delete collection task entry point

    Args:
        collection_id: Collection ID to delete
    """
    try:
        result = collection_task.delete_collection(collection_id)

        if not result.success:
            raise Exception(result.error)

        logger.info(f"Collection {collection_id} deleted successfully")
        return result.to_dict()

    except Exception as e:
        logger.error(f"Collection deletion failed for {collection_id}: {str(e)}")
        raise self.retry(
            exc=e,
            countdown=TaskConfig.RETRY_COUNTDOWN_COLLECTION,
            max_retries=TaskConfig.RETRY_MAX_RETRIES_COLLECTION,
        )


@app.task(bind=True, name="config.celery_tasks.collection_init_task")
def collection_init_task(self, collection_id: str, document_user_quota: int) -> Any:
    """
    Initialize collection task entry point

    Args:
        collection_id: Collection ID to initialize
        document_user_quota: User quota for documents
    """
    try:
        result = collection_task.initialize_collection(collection_id, document_user_quota)

        if not result.success:
            raise Exception(result.error)

        logger.info(f"Collection {collection_id} initialized successfully")
        return result.to_dict()

    except Exception as e:
        logger.error(f"Collection initialization failed for {collection_id}: {str(e)}")
        raise self.retry(
            exc=e,
            countdown=TaskConfig.RETRY_COUNTDOWN_COLLECTION,
            max_retries=TaskConfig.RETRY_MAX_RETRIES_COLLECTION,
        )


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
    name="config.celery_tasks.collection_summary_task",
)
def collection_summary_task(
    self, summary_id: str, collection_id: str, target_version: int, processing_token: str
) -> Any:
    """
    Generate collection summary task entry point

    Args:
        summary_id: Summary ID to generate
        collection_id: Collection ID to generate summary for
    """
    renewer = None

    try:
        from aperag.domains.knowledge_base.service.collection_summary_service import collection_summary_service

        skip_reason = _validate_collection_summary_relevance(summary_id, target_version, processing_token)
        if skip_reason:
            return skip_reason

        renewer = _make_collection_summary_lease_renewer(summary_id, target_version, processing_token)
        renewer.start()

        collection_summary_service.generate_collection_summary_task(
            summary_id,
            collection_id,
            target_version,
            processing_token,
            callback_allowed=lambda: not renewer.ownership_lost,
        )

        if renewer.ownership_lost:
            return _handle_ownership_lost(
                payload_factory=lambda: _build_skipped_payload(
                    "ownership_lost",
                    summary_id=summary_id,
                    collection_id=collection_id,
                ),
                log_message=(
                    f"Processing ownership lost for collection summary {summary_id}; " "suppressing success handling"
                ),
            )

        logger.info(f"Collection summary task completed for {collection_id}")
        return {"success": True, "collection_id": collection_id}

    except Exception as e:
        if renewer and renewer.ownership_lost:
            return _handle_ownership_lost(
                payload_factory=lambda: _build_skipped_payload(
                    "ownership_lost",
                    summary_id=summary_id,
                    collection_id=collection_id,
                ),
                log_message=(
                    f"Processing ownership lost for collection summary {summary_id}; " "suppressing failure callback"
                ),
            )

        logger.error(f"Collection summary generation failed for {collection_id}: {str(e)}")

        # Mark as failed using callback if we've exhausted retries
        if self.request.retries >= self.max_retries:
            from aperag.tasks.reconciler import collection_summary_callbacks

            collection_summary_callbacks.on_summary_failed(summary_id, str(e), target_version, processing_token)

        raise self.retry(
            exc=e,
            countdown=TaskConfig.RETRY_COUNTDOWN_COLLECTION,
            max_retries=TaskConfig.RETRY_MAX_RETRIES_COLLECTION,
        )
    finally:
        if renewer:
            renewer.stop()


@current_app.task(name="config.celery_tasks.cleanup_expired_documents_task")
def cleanup_expired_documents_task():
    """
    Celery task to clean up expired uploaded documents.
    This task should be scheduled to run periodically (e.g., every hour).
    """
    logger.info("Starting Celery task: cleanup_expired_documents")

    # Import here to avoid circular dependencies
    from aperag.tasks.reconciler import collection_gc_reconciler

    result = collection_gc_reconciler.reconcile_all()

    logger.info(f"Celery task completed with result: {result}")
    return result
