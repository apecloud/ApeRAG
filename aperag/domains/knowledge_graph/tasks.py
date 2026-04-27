"""Plain-Python tasks owned by the knowledge_graph domain.

Wave 3 T3.1 chunk 2 (per architect msg=3890c9d7 Pattern A/B/C): the
legacy ``@app.task`` decorator + ``config.celery`` import are gone.
``generate_graph_curation_run_task`` is now a plain Python sync
function — callers schedule it directly (Pattern C fire-and-forget via
``asyncio.create_task(asyncio.to_thread(generate_graph_curation_run_task,
run_id, collection_id))``).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_graph_curation_run_task(run_id: str, collection_id: str) -> Any:
    """Execute one graph-curation scan run."""
    try:
        from aperag.graph_curation.integration import run_graph_curation_run_sync

        run_graph_curation_run_sync(run_id, collection_id)
        return {"success": True, "run_id": run_id, "collection_id": collection_id}
    except Exception as e:
        logger.error(f"Graph curation run failed for {collection_id}/{run_id}: {e}", exc_info=True)
        raise
