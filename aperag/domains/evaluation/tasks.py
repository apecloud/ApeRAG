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

Celery producer seam. ``run_evaluation_run(run_id)`` is the task name that
``EvaluationRunService.launch_run`` dispatches via ``.delay(...)``; all
state-machine logic lives in :mod:`aperag.evaluation_v2.worker` to keep
this module a thin sync wrapper that is safe to import during test
collection even when Celery / the agent runtime / Redis are unavailable.
"""

from __future__ import annotations

import asyncio
import logging

from config.celery import app

logger = logging.getLogger(__name__)


@app.task(bind=True, name="aperag.evaluation_v2.tasks.run_evaluation_run")
def run_evaluation_run(self, run_id: str) -> dict:
    """Celery entrypoint. Runs :func:`execute_evaluation_run` in a fresh
    event loop and returns a small status payload for worker logging.

    The task is intentionally short-circuited when the run_id is unknown
    or already terminal — the orchestration layer handles both cases
    idempotently, so a re-dispatched Celery message is safe to replay.
    """

    # Lazy import: keeps this module import-safe when the agent runtime /
    # chat service / DB are not configured (e.g. unit-test collection).
    from aperag.evaluation_v2.worker import execute_evaluation_run

    logger.info("evaluation worker picking up run %s", run_id)
    final_status = asyncio.run(execute_evaluation_run(run_id))
    final_status_value = final_status.value if hasattr(final_status, "value") else str(final_status)
    logger.info("evaluation worker finished run %s with status %s", run_id, final_status_value)
    return {"run_id": run_id, "status": final_status_value}
