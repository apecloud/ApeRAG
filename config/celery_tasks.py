"""Legacy evaluation Celery tasks.

This module is the residual of the original ``config/celery_tasks.py`` after
phase-3 infra absorption (task #37 D4a) split the indexing / knowledge_base /
knowledge_graph tasks into domain-owned modules. The four evaluation tasks
below are scheduled for deletion in #39 evaluation legacy cleanup.
"""

# scheduled for deletion in #39 evaluation legacy cleanup

import logging
from contextlib import asynccontextmanager
from typing import Any

from celery import current_app

from config.celery import app

logger = logging.getLogger()


# ========== Evaluation Tasks ==========


# By default, get_async_session() uses a global AsyncEngine object.
# Since we also use asyncio.run() to execute async functions, old connections
# in the AsyncEngine connection pool cannot work in the new event loop,
# which will raise an exception like "xxx attached to a different loop".
# Therefore, using a dedicated AsyncEngine to avoid issues from connection reuse.
@asynccontextmanager
async def _new_async_engine():
    from aperag.config import new_async_engine

    engine = new_async_engine()
    try:
        yield engine
    finally:
        await engine.dispose()


@current_app.task
def reconcile_evaluations_task():
    """Periodic task to reconcile evaluations."""
    try:

        async def execute():
            from aperag.service.evaluation_service import EvaluationExecutor

            async with _new_async_engine() as engine:
                executor = EvaluationExecutor(engine)
                await executor.schedule_evaluations()

        import asyncio

        asyncio.run(execute())

        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to reconcile evaluations: {e}", exc_info=True)
        raise


@app.task(bind=True)
def initialize_evaluation_task(self, evaluation_id: str) -> Any:
    """Task to initialize a specific evaluation."""
    try:

        async def execute():
            from aperag.service.evaluation_service import EvaluationExecutor

            async with _new_async_engine() as engine:
                executor = EvaluationExecutor(engine)
                await executor.initialize_evaluation(evaluation_id)

        import asyncio

        asyncio.run(execute())

        return {"success": True, "evaluation_id": evaluation_id}
    except Exception as e:
        logger.error(f"Failed to initialize evaluation {evaluation_id}: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=3)


@app.task(bind=True)
def process_evaluation_batch_task(self, evaluation_id: str) -> Any:
    """Task to process a batch of items for an evaluation."""
    try:

        async def execute():
            from aperag.service.evaluation_service import EvaluationExecutor

            async with _new_async_engine() as engine:
                executor = EvaluationExecutor(engine)
                await executor.process_evaluation_batch(evaluation_id)

        import asyncio

        asyncio.run(execute())

        return {"success": True, "evaluation_id": evaluation_id}
    except Exception as e:
        logger.error(f"Failed to process batch for evaluation {evaluation_id}: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=3)


@app.task(bind=True)
def process_evaluation_item_task(self, evaluation_id: str, item_id: str) -> Any:
    """Task to process a single evaluation item."""
    try:

        async def execute():
            from aperag.service.evaluation_service import EvaluationExecutor

            async with _new_async_engine() as engine:
                executor = EvaluationExecutor(engine)
                await executor.process_evaluation_item(evaluation_id, item_id)

        import asyncio

        asyncio.run(execute())

        return {"success": True, "item_id": item_id}
    except Exception as e:
        logger.error(f"Failed to process item {item_id}: {e}", exc_info=True)
        # You might want a different retry policy for item tasks
        raise self.retry(exc=e, countdown=60, max_retries=3)
