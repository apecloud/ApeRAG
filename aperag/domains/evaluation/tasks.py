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

"""Async worker pipeline for evaluation-v3 (#evaluation #20 / PR-1b).

Wave 3 T3.1 chunk 2 + post-pass-6 fix: the legacy Celery decorators
+ ``config.celery`` import are gone (per architect msg=3890c9d7
Pattern A/B/C). ``run_evaluation_run`` is now a true coroutine —
callers schedule it directly (Pattern C fire-and-forget via
``asyncio.create_task(run_evaluation_run(run_id))``).

Why an awaitable rather than the prior sync wrapper around
``asyncio.run``: the wrapper started a *fresh* event loop on a
worker thread, so any ``asyncpg`` connection borrowed from the
process-wide pool (which is bound to the FastAPI lifespan loop) was
"a Future attached to a different loop" — corrupting the pool and
cascading into 500s on every later DB call. Running the coroutine
on the FastAPI loop keeps the connection-pool affinity correct.

All state-machine logic still lives in :mod:`aperag.domains.
evaluation.worker` so this module stays a thin scheduling shim safe
to import during test collection.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def run_evaluation_run(run_id: str) -> dict:
    """Async entrypoint — schedules on the caller's event loop.

    Pattern C fire-and-forget callers do
    ``asyncio.create_task(run_evaluation_run(run_id))``; the task
    runs concurrently with the request handler and shares the same
    event loop, so any DB session it opens borrows from the same
    asyncpg pool the rest of the process uses. Idempotent: the
    orchestration layer short-circuits unknown / already-terminal
    runs.
    """

    # Lazy import: keeps this module import-safe when the agent runtime /
    # chat service / DB are not configured (e.g. unit-test collection).
    from aperag.domains.evaluation.worker import execute_evaluation_run

    logger.info("evaluation worker picking up run %s", run_id)
    final_status = await execute_evaluation_run(run_id)
    final_status_value = final_status.value if hasattr(final_status, "value") else str(final_status)
    logger.info("evaluation worker finished run %s with status %s", run_id, final_status_value)
    return {"run_id": run_id, "status": final_status_value}
