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

"""Parse worker pool orchestrator — celery Wave 4 T3 chunk 2.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §E.2, the
parse stage is **its own queue + worker pool**, separate from the 5
per-modality lanes. Without this split the upload HTTP handler blocks
on the full ``DocParser`` (MarkItDown / MinerU / OCR) latency — a 30s+
freeze for PDFs and images that destroys responsiveness.

This module ships the dispatch surface that promotes parse to async:

* :class:`ParseDispatchPayload` — the JSON-serialisable envelope
  pushed onto ``q:parse`` by the upload handler.
* :func:`process_one_parse_task` — single-payload happy / failure
  paths. Reads the source artifact, runs :func:`parse_document`, then
  fans out via :func:`dispatch_indexing` to the 5 per-modality queues.
* :func:`run_parse_worker_loop` + :func:`run_parse_worker` — the
  long-lived BLPOP loop wired into the FastAPI lifespan, mirroring the
  per-modality ``run_*_worker`` entrypoints.

Why no DocumentIndex row for parse: the per-modality rows are inserted
**after** parse completes (so they can carry the real
``parse_version``). Tracking parse-in-progress on a separate row would
require either a sentinel ``parse_version`` (collides with the
``UNIQUE(document_id, parse_version, modality)`` constraint when the
real version lands) or a new table; both buy little for chunk 2's
minimal scope. A failed / lost parse currently surfaces as a document
that never sprouted ``document_index`` rows — Wave 5 follow-up will
extend the reconciler to re-enqueue parse jobs for stuck documents.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from sqlalchemy import Engine, and_
from sqlalchemy import delete as sa_delete
from sqlalchemy.orm import Session

from aperag.indexing.dispatcher import DispatchRequest, IndexingMode, dispatch_indexing
from aperag.indexing.models import DocumentIndex, Modality
from aperag.indexing.orchestrator import WorkQueue
from aperag.indexing.parser import ParseConfig, parse_document
from aperag.objectstore.base import ObjectStore as _SyncObjectStore

logger = logging.getLogger(__name__)


# Default poll timeout for the parse worker loop — short so a
# ``shutdown`` event is responsive without busy-looping. Mirrors the
# per-modality :class:`OrchestratorConfig` default.
DEFAULT_POLL_TIMEOUT_SECONDS = 1.0

# Per design pack §E.2 ASCII diagram: parse_worker (1 process,
# asyncio concurrency = 8). Parse latency is dominated by external
# OCR / MinerU / MarkItDown calls, so 8 concurrent in-flight parses
# is the throughput sweet spot before disk + LLM rate-limit pressure
# kicks in.
DEFAULT_PARSE_CONCURRENCY = 8


# ---------------------------------------------------------------------
# Dispatch payload (round-trips through Redis as JSON).
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ParseDispatchPayload:
    """Decoded parse-queue payload — the unit of work the parse worker runs.

    Carries enough context to (a) read the source artifact from the
    object store, (b) run :func:`parse_document` with the right
    parser config, and (c) fan out to the per-modality queues with
    the correct ``modalities`` subset + ``tenant_scope_key`` for the
    quota / bulkhead lane.

    No ``parse_version`` field — that is what parsing produces. The
    upload handler does not know it yet, which is precisely why parse
    is async.
    """

    document_id: str
    collection_id: str
    object_path: str
    tenant_scope_key: str
    modalities: tuple[str, ...]
    parser_config: dict[str, Any] | None = None
    purge_existing_triples: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ParseDispatchPayload":
        modalities_raw = raw.get("modalities") or ()
        if isinstance(modalities_raw, str):
            modalities_raw = json.loads(modalities_raw)
        modalities = tuple(str(m) for m in modalities_raw)
        parser_config = raw.get("parser_config")
        if parser_config is not None and not isinstance(parser_config, dict):
            raise TypeError(
                f"ParseDispatchPayload.parser_config must be a dict or None, got {type(parser_config).__name__}"
            )
        return cls(
            document_id=str(raw["document_id"]),
            collection_id=str(raw["collection_id"]),
            object_path=str(raw["object_path"]),
            tenant_scope_key=str(raw["tenant_scope_key"]),
            modalities=modalities,
            parser_config=parser_config,
            purge_existing_triples=bool(raw.get("purge_existing_triples", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "collection_id": self.collection_id,
            "object_path": self.object_path,
            "tenant_scope_key": self.tenant_scope_key,
            "modalities": list(self.modalities),
            "parser_config": self.parser_config,
            "purge_existing_triples": self.purge_existing_triples,
        }


# ---------------------------------------------------------------------
# Single-task entrypoint — used directly by tests, wrapped by the loop.
# ---------------------------------------------------------------------


def _read_source_bytes_sync(store: _SyncObjectStore, object_path: str) -> bytes | None:
    """Read source bytes from the object store, returning ``None`` if missing.

    Returns ``None`` rather than raising so the caller can surface
    ``failed_read`` instead of bubbling an opaque exception. The
    ``read_or_none`` helper in :mod:`object_store` is for *derived*
    artifacts (which legitimately may be missing mid-derive); source
    artifacts should always exist by the time the upload handler
    enqueues the parse job, so a missing source genuinely indicates a
    bug or an external delete.
    """
    handle = store.get(object_path)
    if handle is None:
        return None
    try:
        body = handle.read()
    finally:
        try:
            handle.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
    return body


def _purge_existing_triples_sync(
    engine: Engine,
    document_id: str,
    parse_version: str,
    modalities: tuple[Modality, ...],
) -> None:
    """Delete any existing ``(document_id, parse_version, modality)`` rows.

    Mirrors the rebuild-path purge that
    ``_create_or_update_document_indexes`` ran inline before chunk 2.
    Required when a rebuild is dispatched on a document whose content
    has not changed: the parse_version is content-derived, so a
    re-enqueue would trip the ``uq_document_index_triple`` UNIQUE
    constraint when the dispatcher INSERTs.
    """
    if not modalities:
        return
    modality_values = [m.value for m in modalities]
    with Session(engine) as session, session.begin():
        session.execute(
            sa_delete(DocumentIndex).where(
                and_(
                    DocumentIndex.document_id == document_id,
                    DocumentIndex.parse_version == parse_version,
                    DocumentIndex.modality.in_(modality_values),
                )
            )
        )


async def process_one_parse_task(
    *,
    engine: Engine,
    queue: WorkQueue,
    object_store: _SyncObjectStore,
    payload: ParseDispatchPayload,
    parse_config: ParseConfig | None = None,
) -> str:
    """Run the full read-source → parse → dispatch cycle for one payload.

    Returns one of ``"completed"``, ``"failed_read"``, ``"failed_parse"``,
    ``"failed_dispatch"`` so the run-loop / tests can assert on the
    per-task outcome without scraping logs.

    Failure modes are recorded via ``logger.exception`` and the parse
    job is dropped (no retry — a failed parse leaves the document
    without ``document_index`` rows; the operator surfaces it via
    ``document.status`` and the Wave 5 reconciler extension). Modality
    workers handle their own retry via §I.2 backoff once parse
    succeeds.
    """
    # ---- 1. Read source from the object store ----
    try:
        source_bytes = await asyncio.to_thread(_read_source_bytes_sync, object_store, payload.object_path)
    except Exception as exc:  # noqa: BLE001 — surface via log, drop job
        logger.exception(
            "parse_worker source read failure document_id=%s object_path=%s: %s",
            payload.document_id,
            payload.object_path,
            exc,
        )
        return "failed_read"

    if source_bytes is None:
        logger.error(
            "parse_worker dropping payload — source missing in object store: document_id=%s object_path=%s",
            payload.document_id,
            payload.object_path,
        )
        return "failed_read"

    # ---- 2. Parse: produce derived/parse_<version>/{markdown.md,outline.json,chunks.jsonl} ----
    cfg = parse_config or ParseConfig()
    try:
        parsed = await asyncio.to_thread(
            parse_document,
            store=object_store,
            collection_id=payload.collection_id,
            document_id=payload.document_id,
            source_bytes=source_bytes,
            source_filename=payload.object_path,
            parser_config=payload.parser_config,
            config=cfg,
        )
    except Exception as exc:  # noqa: BLE001 — surface via log, drop job
        logger.exception(
            "parse_worker parse failure document_id=%s object_path=%s: %s",
            payload.document_id,
            payload.object_path,
            exc,
        )
        return "failed_parse"

    # ---- 3. Optional rebuild-path purge (idempotent re-dispatch) ----
    try:
        modalities = tuple(Modality(m) for m in payload.modalities)
    except ValueError as exc:
        logger.error(
            "parse_worker dropping payload — unknown modality value document_id=%s modalities=%r: %s",
            payload.document_id,
            payload.modalities,
            exc,
        )
        return "failed_dispatch"

    if payload.purge_existing_triples and modalities:
        try:
            await asyncio.to_thread(
                _purge_existing_triples_sync,
                engine,
                payload.document_id,
                parsed.parse_version,
                modalities,
            )
        except Exception as exc:  # noqa: BLE001 — surface via log, drop job
            logger.exception(
                "parse_worker triple purge failure document_id=%s parse_version=%s: %s",
                payload.document_id,
                parsed.parse_version,
                exc,
            )
            return "failed_dispatch"

    # ---- 4. Dispatch fan-out to the 5 per-modality queues ----
    if not modalities:
        # Empty modality set is a degenerate parse-only enqueue (e.g.
        # tests that exercise parse without dispatch). The artifacts
        # are written and the document is parsed; we just have nothing
        # to fan out.
        logger.info(
            "parse_worker parse_only — no modalities to dispatch for document_id=%s parse_version=%s",
            payload.document_id,
            parsed.parse_version,
        )
        return "completed"

    try:
        await dispatch_indexing(
            engine=engine,
            queue=queue,
            workers=None,
            request=DispatchRequest(
                collection_id=payload.collection_id,
                document_id=payload.document_id,
                parse_version=parsed.parse_version,
                source_path=parsed.chunks_path,
                tenant_scope_key=payload.tenant_scope_key,
                modalities=modalities,
            ),
            mode=IndexingMode.ASYNC,
        )
    except Exception as exc:  # noqa: BLE001 — surface via log, drop job
        logger.exception(
            "parse_worker dispatch_indexing failure document_id=%s parse_version=%s: %s",
            payload.document_id,
            parsed.parse_version,
            exc,
        )
        return "failed_dispatch"

    logger.info(
        "parse_worker completed document_id=%s parse_version=%s modalities=%d",
        payload.document_id,
        parsed.parse_version,
        len(modalities),
    )
    return "completed"


# ---------------------------------------------------------------------
# Run loop (parse worker process).
# ---------------------------------------------------------------------


@dataclass
class ParseOrchestratorConfig:
    """Per-process parse-worker tuning. Mirrors :class:`OrchestratorConfig`.

    ``concurrency`` is bounded by an :class:`asyncio.Semaphore` inside
    the run loop so a slow OCR call cannot starve the BLPOP loop.
    """

    concurrency: int = DEFAULT_PARSE_CONCURRENCY
    poll_timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS


# A factory so the run loop can resolve a per-process object store
# without baking a concrete backend into the orchestrator. Production
# wires :class:`aperag.objectstore.factories.get_object_store`; tests
# inject :class:`InMemoryObjectStore`. Async to leave room for backends
# that need an async resolution step (e.g. S3 client warm-up).
ObjectStoreFactory = Callable[[], Awaitable[_SyncObjectStore]]


async def run_parse_worker_loop(
    *,
    config: ParseOrchestratorConfig,
    engine: Engine,
    queue: WorkQueue,
    object_store_factory: ObjectStoreFactory,
    shutdown: asyncio.Event,
) -> None:
    """Parse worker run loop — BLPOP ``q:parse`` + dispatch with concurrency cap.

    Pops parse payloads, decodes them, and runs each through
    :func:`process_one_parse_task` under an :class:`asyncio.Semaphore`
    whose permit count == ``config.concurrency``. Exits cleanly when
    ``shutdown`` is set, draining in-flight tasks first.

    Mirrors :func:`run_worker_loop` for the per-modality lanes — same
    shutdown discipline, same in-flight set housekeeping, same
    malformed-payload drop semantics. The only structural difference
    is that there is no per-task DB row to claim before work starts:
    parse jobs are claim-free (the document_id is the natural idempotency
    key, and at-most-once delivery is provided by Redis BLPOP).
    """
    semaphore = asyncio.Semaphore(config.concurrency)
    in_flight: set[asyncio.Task[str]] = set()
    object_store = await object_store_factory()

    async def _runner(payload: ParseDispatchPayload) -> str:
        async with semaphore:
            return await process_one_parse_task(
                engine=engine,
                queue=queue,
                object_store=object_store,
                payload=payload,
            )

    while not shutdown.is_set():
        raw = await queue.pop_parse(timeout_seconds=config.poll_timeout_seconds)
        if raw is None:
            in_flight = {t for t in in_flight if not t.done()}
            continue

        try:
            payload = ParseDispatchPayload.from_dict(raw)
        except (KeyError, ValueError, TypeError) as exc:
            logger.error(
                "parse_worker dropping malformed payload: %r (%s)",
                raw,
                exc,
            )
            continue

        task = asyncio.create_task(_runner(payload))
        in_flight.add(task)

    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)


async def run_parse_worker(
    *,
    engine: Engine,
    queue: WorkQueue,
    object_store_factory: ObjectStoreFactory,
    shutdown: asyncio.Event,
    config: ParseOrchestratorConfig | None = None,
) -> None:
    """Thin entrypoint mirroring ``run_*_worker`` for the modality lanes.

    Production wires this as a single asyncio task in the FastAPI
    lifespan alongside the 5 per-modality workers + reconciler +
    cleanup loop. The default :class:`ParseOrchestratorConfig` matches
    design pack §E.2 (concurrency = 8).
    """
    await run_parse_worker_loop(
        config=config or ParseOrchestratorConfig(),
        engine=engine,
        queue=queue,
        object_store_factory=object_store_factory,
        shutdown=shutdown,
    )


__all__ = [
    "DEFAULT_PARSE_CONCURRENCY",
    "DEFAULT_POLL_TIMEOUT_SECONDS",
    "ObjectStoreFactory",
    "ParseDispatchPayload",
    "ParseOrchestratorConfig",
    "process_one_parse_task",
    "run_parse_worker",
    "run_parse_worker_loop",
]
