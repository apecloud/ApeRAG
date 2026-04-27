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

Wave 3 T3.1 chunk 2: the legacy Celery decorators + ``config.celery``
import are gone (per architect msg=3890c9d7 Pattern A/B/C). The
``run_evaluation_run`` function is now a plain Python sync wrapper —
callers schedule it directly (Pattern C fire-and-forget via
``asyncio.create_task(asyncio.to_thread(run_evaluation_run, run_id))``).
All state-machine logic still lives in :mod:`aperag.domains.evaluation.
worker` so this module stays a thin sync wrapper that is safe to import
during test collection.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def run_evaluation_run(run_id: str) -> dict:
    """Plain Python entrypoint (Wave 3 T3.1 chunk 2 — formerly Celery).

    Runs :func:`execute_evaluation_run` in a fresh event loop and
    returns a small status payload for worker logging. Idempotent: the
    orchestration layer short-circuits unknown / already-terminal runs.
    """

    # Lazy import: keeps this module import-safe when the agent runtime /
    # chat service / DB are not configured (e.g. unit-test collection).
    from aperag.domains.evaluation.worker import execute_evaluation_run

    logger.info("evaluation worker picking up run %s", run_id)
    final_status = asyncio.run(execute_evaluation_run(run_id))
    final_status_value = final_status.value if hasattr(final_status, "value") else str(final_status)
    logger.info("evaluation worker finished run %s with status %s", run_id, final_status_value)
    return {"run_id": run_id, "status": final_status_value}
