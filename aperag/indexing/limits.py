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

"""Bulkhead limits — celery T2.2.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §H.6, every
worker process applies tenant-blind hard ceilings on three resources:

* **LLM call timeout** — 60 seconds. A long-running graph LLM call
  is the most expensive operation per §E.4 and the most likely to
  hang on rate-limit / retry storms; a 60s ceiling keeps a single
  stuck call from monopolising the worker slot.
* **Embedding call timeout** — 30 seconds. Embeddings are smaller +
  cheaper than LLM completions, and a longer wait usually means an
  upstream-saturated provider rather than a single slow request.
  The shorter ceiling lets the orchestrator's retry / backpressure
  loop kick in faster.
* **Upload size cap** — 50 MB. Documents above this size will not
  fit through the parser → chunks → embedding pipeline within the
  per-document SLO; rejecting at upload boundary is cheaper than
  hitting an OOM mid-parse.

These are §H.6 "defense in depth" limits — they apply *in addition*
to the §H.5 per-tenant quota gates in :mod:`aperag.indexing.quota`.
A quota acquire might let an LLM call through; the bulkhead timeout
still kicks in if the call itself stalls. The two layers compose.

The module exposes the constants, plus :func:`bulkhead_timeout` —
an async context manager that wraps :func:`asyncio.timeout` so
worker code can lock the timeout in one line. Centralising the
context manager here means a future change to the timeout strategy
(e.g., per-modality overrides, structured cancellation telemetry)
lands in one place rather than scattered across modality workers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# §H.6 hard ceilings — overridable per deployment via env / config but
# *not* per tenant. These are bulkheads, not quotas; quotas live in
# :mod:`aperag.indexing.quota`.
# ---------------------------------------------------------------------


LLM_CALL_TIMEOUT_SECONDS: float = 60.0
"""Maximum wall time for a single LLM completion call.

The graph modality's entity / relation extraction dominates this
budget; vector / fulltext / summary / vision modalities never approach
it. A worker that exceeds the budget surfaces a :class:`TimeoutError`
which the orchestrator's failure-handling path treats as a transient
failure (retry with §I.2 backoff) rather than a permanent one.
"""

EMBEDDING_CALL_TIMEOUT_SECONDS: float = 30.0
"""Maximum wall time for a single embedding API call.

Shorter than the LLM ceiling because embeddings are cheaper + more
predictable; a long wait usually means provider-side saturation that
the §H.5 quota system or the orchestrator's backoff should handle
rather than the worker holding a slot open.
"""

UPLOAD_MAX_BYTES: int = 50 * 1024 * 1024
"""Maximum source-document size accepted at the upload boundary.

The parser → chunks → embedding pipeline assumes per-document body
fits in single-process memory plus a few hundred MB of headroom for
LLM context windows. Documents above this size should be split at
ingestion before they enter the pipeline.
"""


# ---------------------------------------------------------------------
# Async context manager wrapper — keeps callers' code one-liner.
# ---------------------------------------------------------------------


@asynccontextmanager
async def bulkhead_timeout(seconds: float, *, label: str | None = None) -> AsyncIterator[None]:
    """Run the inner block with a hard wall-time ceiling.

    On timeout, the ``asyncio.timeout`` context manager cancels the
    inner task and surfaces a :class:`TimeoutError`. The caller
    typically catches this in the modality worker's ``derive`` /
    ``sync`` and re-raises after recording the failure metric — see
    :func:`aperag.indexing.observability.emit_index_failure`.

    ``label`` is included in the timeout log line so a flood of
    timeouts in one modality is identifiable from a single metric.
    Callers should pass a stable identifier such as
    ``"graph.derive.llm_extraction"``.

    Example::

        async with bulkhead_timeout(LLM_CALL_TIMEOUT_SECONDS, label="graph.derive.llm"):
            await llm_client.complete(prompt)
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError:
        logger.warning(
            "bulkhead timeout — %s exceeded %.1fs",
            label or "<unlabeled>",
            seconds,
        )
        raise


# ---------------------------------------------------------------------
# Boundary check for upload size.
# ---------------------------------------------------------------------


def reject_if_oversize(content_length: int, *, label: str | None = None) -> None:
    """Raise :class:`ValueError` if ``content_length`` exceeds the §H.6
    upload ceiling.

    Surface the cap at the upload boundary rather than mid-parser so
    a 100 MB PDF does not trigger an OOM after the parser has already
    started loading it. The caller is the upload-handler request
    path, *not* the modality workers (they trust the boundary).
    """
    if content_length > UPLOAD_MAX_BYTES:
        logger.warning(
            "upload rejected — %s exceeds %d bytes (got %d)",
            label or "<unlabeled>",
            UPLOAD_MAX_BYTES,
            content_length,
        )
        raise ValueError(
            f"upload exceeds {UPLOAD_MAX_BYTES} byte ceiling: {content_length} bytes ({label or 'document'})"
        )


__all__ = [
    "LLM_CALL_TIMEOUT_SECONDS",
    "EMBEDDING_CALL_TIMEOUT_SECONDS",
    "UPLOAD_MAX_BYTES",
    "bulkhead_timeout",
    "reject_if_oversize",
]
