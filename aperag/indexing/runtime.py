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

"""Process-local indexing runtime singleton — celery T3.1 chunk 3.

The FastAPI lifespan owns the canonical ``engine`` + ``queue`` +
``workers`` triple (see ``aperag/app.py:combined_lifespan``). It also
stashes them on ``app.state.indexing_*`` for HTTP routes that have a
``Request`` handle. Service-layer code (``aperag/domains/**``) does not
have a ``Request`` and shouldn't import FastAPI, so this module is the
back-channel: the lifespan calls :func:`set_runtime` after building the
triple, and service-layer callers use :func:`get_runtime` (returns
``None`` when ``INDEXING_MODE != async`` — the legacy synchronous code
path is dead and any caller hitting that case should log + no-op).

The runtime is intentionally a single mutable global rather than a
``ContextVar`` because the indexing fan-out is one process-wide queue
+ engine pair, not per-request. Tests that need to stub a different
runtime can call :func:`set_runtime` with a fixture and reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Mapping, Optional

from sqlalchemy import Engine

from aperag.indexing.base import ModalityWorker
from aperag.indexing.models import DocumentIndex, Modality
from aperag.indexing.observability import MetricsEmitter, NoopMetricsEmitter
from aperag.indexing.orchestrator import WorkQueue
from aperag.indexing.quota import InMemoryQuotaBackend, QuotaBackend, QuotaPolicyRegistry


def _default_quota_backend() -> QuotaBackend:
    return InMemoryQuotaBackend(QuotaPolicyRegistry())


@dataclass(frozen=True)
class IndexingRuntime:
    """The runtime triple required by ``dispatch_indexing()`` and
    ``cleanup_for_deleted_documents()``, plus the §J.1 metrics emitter
    and Wave 4 T5 quota backend.

    ``workers`` may be empty when the deployment runs in ASYNC mode and
    cleanup is intentionally non-cascading (e.g. tests). ``queue`` may
    be ``None`` when the deployment runs in INLINE mode (the
    dispatcher fans out via ``workers`` directly).

    ``cleanup_worker_factory`` (Wave 4 T2) is the per-row resolver the
    cleanup path uses to materialise the right per-(collection,
    modality) ``_backend`` (or ``_store + _entity_lock`` for graph).
    Production wires
    :meth:`aperag.indexing.worker_factory.ProductionWorkerFactory.build_for_cleanup_row`;
    tests typically leave it ``None`` and pass a static ``workers``
    map directly to ``cleanup_for_deleted_documents``.

    ``metrics_emitter`` defaults to :class:`NoopMetricsEmitter` so older
    callers that build the runtime without specifying one keep working.
    The lifespan in ``aperag/app.py`` swaps in
    :class:`OTLPMetricsEmitter` when ``INDEXING_METRICS_EMITTER=otlp``.

    ``quota_backend`` (Wave 4 T5) is the §H.5 token-bucket the
    LLM/embedding-heavy modalities can consume to enforce per-tenant
    rate limits before issuing the upstream API call. Default is
    :class:`InMemoryQuotaBackend` (per-process, single-pod scope).
    Production multi-pod sets ``INDEXING_QUOTA_BACKEND=redis`` to wire
    :class:`RedisQuotaBackend` so worker processes share token state
    via the §H.5.1 logical-db assignment (db=3).
    """

    engine: Engine
    queue: Optional[WorkQueue]
    workers: Mapping[Modality, ModalityWorker]
    metrics_emitter: MetricsEmitter = field(default_factory=NoopMetricsEmitter)
    cleanup_worker_factory: Optional[Callable[[DocumentIndex], Awaitable[ModalityWorker]]] = None
    quota_backend: QuotaBackend = field(default_factory=_default_quota_backend)


_runtime: Optional[IndexingRuntime] = None


def set_runtime(runtime: Optional[IndexingRuntime]) -> None:
    """Install the process-wide indexing runtime (called by the
    FastAPI lifespan). Pass ``None`` on shutdown to clear it."""

    global _runtime
    _runtime = runtime


def get_runtime() -> Optional[IndexingRuntime]:
    """Return the installed runtime, or ``None`` if the lifespan has
    not run (test environment, sync-only mode, or pre-startup boot
    sequence). Service-layer callers should treat ``None`` as "indexing
    disabled" and log + no-op rather than crashing."""

    return _runtime


__all__ = ["IndexingRuntime", "get_runtime", "set_runtime"]
