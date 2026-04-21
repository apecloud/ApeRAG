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

"""Async worker pipeline for evaluation-v2.

Phase 1 only scaffolds the module. The Phase 3 implementation will:

1. Pick up queued runs, transition them to RUNNING.
2. For each run item, create an isolated agent chat and dispatch a
   Runtime V3 turn via `agent_runtime.runtime.agent_runtime_manager`.
3. Wait for the turn to finalize, fetch the answer / reference artifacts,
   and invoke the configured judge (`aperag.evaluation_v2.judges`).
4. Persist attempt + item results and update run summary metrics.
5. Handle cancellation and retries by respecting run status transitions.

Keep this module import-safe; register Celery tasks here so they are
discoverable by the worker autodiscover pass.
"""

from __future__ import annotations


async def run_evaluation_run(run_id: str) -> None:
    """Phase 3 entrypoint. No-op until the worker pipeline lands."""
    return None
