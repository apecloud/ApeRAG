"""Celery tasks owned by the knowledge_graph domain.

Domain-owned tasks for the knowledge_graph domain. Moved from
``config/celery_tasks.py`` as part of phase-3 infra absorption (task #37 D4a).
Pure move — no behavior change. Task ``name="..."`` strings are pinned to
``config.celery_tasks.<name>`` to preserve task identity for in-flight queue
messages.
"""

import logging
from typing import Any

from config.celery import app

logger = logging.getLogger(__name__)


@app.task(bind=True, name="config.celery_tasks.generate_graph_curation_run_task")
def generate_graph_curation_run_task(self, run_id: str, collection_id: str) -> Any:
    """Execute one graph-curation scan run."""
    try:
        from aperag.graph_curation.integration import run_graph_curation_run_sync

        run_graph_curation_run_sync(run_id, collection_id)
        return {"success": True, "run_id": run_id, "collection_id": collection_id}
    except Exception as e:
        logger.error(f"Graph curation run failed for {collection_id}/{run_id}: {e}", exc_info=True)
        raise
