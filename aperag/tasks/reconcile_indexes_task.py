import logging
from celery import current_app

import asyncio

logger = logging.getLogger(__name__)


@current_app.task
def reconcile_indexes_task():
    """Periodic task to reconcile index specs with statuses"""
    try:
        logger.info("Starting index reconciliation")

        # Import here to avoid circular dependencies
        from aperag.index.reconciler import index_reconciler

        # Run reconciliation
        asyncio.run(index_reconciler.reconcile_all())

        logger.info("Index reconciliation completed")

    except Exception as e:
        logger.error(f"Index reconciliation failed: {e}")
        raise