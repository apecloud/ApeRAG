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

"""Graph modality — celery T1.2.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §D.3, the
graph modality is the only one of the five whose backend rows can be
shared across documents — entity ``Linus Torvalds`` may be cited by 100
different docs, and a doc-level retry / delete must NOT wipe the other
99 docs' contribution. The simple "DELETE-by-(doc, parse_version) +
re-INSERT" contract that vector / fulltext / summary / vision rely on
would either drop cross-doc lineage (under-delete) or leave doc_A's
description forever concatenated with doc_B's after a doc_A retry
(under-clean). This is the bug Bryce v2 review msg=7ccb176f #3 caught
and architect msg=cc555e33 / msg=f2921ae0 locked the lineage model for.

The §D.3.2 algorithm — implemented by :class:`GraphModalityWorker` —
sweeps in two phases:

1. **Lineage cleanup**. Every entity / relation that was last touched
   by ``(document_id, parse_version)`` has the corresponding member
   removed from its ``source_lineage`` / ``description_parts`` /
   ``evidence_lineage`` set. Rows whose lineage set goes empty are
   garbage-collected.

2. **Lineage rebuild**. Every entity / relation in the ``kg.jsonl``
   artifact is upserted with a fresh ``(document_id, parse_version)``
   member added to its lineage set. Existing other-doc / other-parse_v
   lineage members are preserved.

The contract is then idempotent at the lineage level: re-running
``sync(doc, v1, kg.jsonl)`` collapses cleanly, whether or not it ran
before, and whether or not other docs have already written to the
same entities.

Backends differ in how they implement the lineage SET (§D.3.5):

* **Neo4j**: native list ops + APOC; one Cypher call per phase.
* **Nebula 3.x**: list ops are limited, so the application reads the
  current lineage list, mutates in Python, writes back. That
  read-modify-write loop is racy unless we serialize per entity ID.
  The :class:`EntityLock` abstraction takes a Redis lock keyed by the
  entity name so two concurrent ``sync`` calls touching the same
  entity never overlap (msg=f2921ae0 architect-locked invariant).
* **In-memory**: :class:`InMemoryLineageGraphStore` provides a deterministic
  Python implementation usable from tests and as a reference for
  what the backends MUST behave like.

The kg.jsonl artifact is the canonical (and most expensive)
modality output — re-running ``derive`` would reissue every LLM
extraction call. So ``derive`` writes once and ``sync`` only ever
reads. The actual entity / relation extraction is provided by a
:class:`GraphExtractor` callable injected at construction time so that
T2.x can wire the production LLM-driven extractor without
touching this module.

T1.2 ships:

* the lineage data model (:class:`EntityRecord`, :class:`RelationRecord`,
  :class:`LineageMember`, :class:`DescriptionPart`),
* the storage Protocol (:class:`LineageGraphStore`) plus the
  :class:`InMemoryLineageGraphStore` reference implementation,
* the :class:`EntityLock` abstraction (in-memory + a thin Redis adapter),
* the :class:`GraphModalityWorker` implementing the §D.3.2 algorithm,
* JSONL serialization helpers,

and the §D.3.6 five-step idempotency self-test plus a Nebula-
race-style concurrent-sync test in
``tests/unit_test/indexing/test_t1_2_graph.py``.

Concrete Nebula / Neo4j backends bind to this Protocol via their own
adapter modules in Wave 2 (T2.1) when the worker pool wires real
connections; that integration is intentionally out of scope for T1.2
which focuses on the algorithm + race-protection contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol
from uuid import NAMESPACE_DNS, uuid5

from aperag.indexing.base import DeriveResult, ModalityWorker
from aperag.indexing.models import Modality
from aperag.indexing.object_store import (
    derived_artifact,
    read_or_none_async,
    write_atomic_async,
)
from aperag.indexing.parser import read_chunks
from aperag.objectstore.base import ObjectStore as _SyncObjectStore

if TYPE_CHECKING:  # pragma: no cover — TYPE_CHECKING-only imports avoid circular deps
    from aperag.indexing.graph_compactor import GraphIndexCompactor
    from aperag.indexing.merge_candidate_detector import MergeCandidateDetector
    from aperag.vectorstore.base import VectorStoreConnector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Lineage data model — design pack §D.3.1
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LineageMember:
    """One ``(document_id, parse_version)`` slice of an entity / relation's lineage.

    The ``chunk_ids`` tuple records which chunks of THAT specific
    parse contributed to the entity / relation. On a doc_A v1 retry,
    we replace exactly the lineage member with
    ``document_id == "doc_A" AND parse_version == "v1"`` and leave
    every other ``(doc, ver)`` slice untouched.

    ``tenant_scope_key`` propagates the §H.2 quota / tenant-isolation
    key into every lineage SET element so that a read-path can filter
    a shared entity's contributions by tenant ACL (e.g., "show me
    only the lineage members from my tenant plus public") without
    forcing the entity row itself to belong to a single tenant.
    Architect ruling msg=c3b0ba5b leaves the placement decision to
    the graph implementer; SET-element level is the placement that
    preserves the shared-entity model the §D.3 lineage design relies
    on.
    """

    document_id: str
    parse_version: str
    tenant_scope_key: str
    chunk_ids: tuple[str, ...]

    def key(self) -> tuple[str, str]:
        """The natural key used to dedup lineage members across syncs."""
        return (self.document_id, self.parse_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "parse_version": self.parse_version,
            "tenant_scope_key": self.tenant_scope_key,
            "chunk_ids": list(self.chunk_ids),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LineageMember":
        return cls(
            document_id=str(raw["document_id"]),
            parse_version=str(raw["parse_version"]),
            tenant_scope_key=str(raw["tenant_scope_key"]),
            chunk_ids=tuple(str(cid) for cid in raw.get("chunk_ids", [])),
        )


@dataclass(frozen=True)
class DescriptionPart:
    """One ``(document_id, parse_version)`` slice of an entity's description.

    Per §D.3.3 Option A the read path returns the *full set* of these
    parts and lets the upstream LLM dedup / summarise, so each doc's
    contribution to the description is preserved verbatim instead of
    being concatenated into a monolithic blob (which is what the v1
    ``upsert_entities`` append-on-conflict path did, and which broke
    incremental retries — see the regression Bryce surfaced in
    msg=7ccb176f #3).
    """

    document_id: str
    parse_version: str
    text: str

    def key(self) -> tuple[str, str]:
        return (self.document_id, self.parse_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "parse_version": self.parse_version,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DescriptionPart":
        return cls(
            document_id=str(raw["document_id"]),
            parse_version=str(raw["parse_version"]),
            text=str(raw.get("text", "")),
        )


@dataclass(frozen=True)
class EntityRecord:
    """One entity row read from / written to ``kg.jsonl``.

    The ``source_chunk_ids`` are the chunk ids that mention the
    entity within THIS document's parse. They contribute to the new
    lineage member that ``sync`` adds during phase-2 rebuild.

    Wave 6 #36 (per §K.10 item 7 + architect ruling): the field is
    named ``entity_type`` (not ``type``) to free the bare ``type``
    name across the §D.3 Protocol surface so callers can introduce
    Python type-hint metadata or Cypher ``TYPE()`` keyword usage on
    relation edges without shadow conflicts. The on-wire ``to_dict``
    JSON key is also ``entity_type`` to keep storage and Python
    surface identical (hard-cut per earayu2 msg=30c81478 — no
    production data on the legacy ``type`` shape).
    """

    name: str
    entity_type: str
    description: str
    source_chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "entity",
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "source_chunk_ids": list(self.source_chunk_ids),
        }


@dataclass(frozen=True)
class RelationRecord:
    """One relation row read from / written to ``kg.jsonl``.

    Wave 6 #36: ``relation_type`` (was ``type``) frees the bare
    ``type`` name; cross-backend column / property / tag-prop renames
    follow in `graph_storage/{postgres,neo4j,nebula}.py`.
    """

    source: str
    target: str
    relation_type: str
    description: str
    source_chunk_ids: tuple[str, ...]

    def relation_key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.relation_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "relation",
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "description": self.description,
            "source_chunk_ids": list(self.source_chunk_ids),
        }


@dataclass
class EntityWithLineage:
    """The full lineage-aware view of one stored entity row.

    Backends materialise this from their underlying
    storage representation (Nebula JSON STRING, Neo4j list-of-MAP,
    in-memory dict) and the algorithm in :class:`GraphModalityWorker`
    only ever sees this canonical form.

    Wave 7 (W7-1) adds ``compacted_description``: a derived cache
    column written by ``GraphIndexCompactor`` (Wave 7 W7-2) when the
    aggregate ``description_parts`` text grows past
    ``max_description_chars`` / ``summarize_at_fragments`` thresholds.
    The compactor calls an LLM to summarise the union of the parts and
    persists the result here so the vector embedding step (Wave 7 W7-3)
    has a bounded, coherent text source. The field is **derived**:
    ``description_parts`` remains the per-doc source of truth, and the
    compacted value can be reproduced by re-running the compactor over
    the parts at any time. NULL means "not yet compacted" — readers
    should fall back to joining ``description_parts.text``.
    """

    name: str
    entity_type: str
    source_lineage: tuple[LineageMember, ...]
    description_parts: tuple[DescriptionPart, ...]
    compacted_description: str | None = None


@dataclass
class RelationWithLineage:
    source: str
    target: str
    relation_type: str
    evidence_lineage: tuple[LineageMember, ...]
    description_parts: tuple[DescriptionPart, ...]
    compacted_description: str | None = None


# ---------------------------------------------------------------------
# Per-entity lock abstraction — Nebula read-modify-write race guard
# (architect msg=f2921ae0 invariant).
# ---------------------------------------------------------------------


class EntityLock(Protocol):
    """Async context manager factory keyed by entity name.

    Two concurrent ``sync`` invocations that touch the same entity
    must not overlap on the read-modify-write inside Nebula
    backends; the lock makes that serialization explicit. Neo4j
    backends with native list ops can use a no-op lock implementation
    without losing correctness.
    """

    def acquire(self, entity_id: str) -> "AbstractAsyncContextManager[None]":
        """Return an async context manager that holds the lock for the
        duration of the ``async with`` block.

        ``entity_id`` is the natural identifier (e.g., entity name);
        the implementation is free to hash-route it onto a finite
        slot pool to bound memory.
        """


# ``AbstractAsyncContextManager`` lives in ``contextlib`` but typing
# uses the ``typing.AsyncContextManager`` alias; use the runtime
# alternative via ``Awaitable``-friendly Protocol so ``EntityLock``
# stays decoupled from the concrete implementation.
from contextlib import AbstractAsyncContextManager  # noqa: E402  (after Protocol on purpose)


class InMemoryEntityLock:
    """``EntityLock`` backed by per-key :class:`asyncio.Lock` instances.

    Suitable for tests and single-process deployments. Production
    multi-process workloads (Wave 2 worker pool) bind to
    :class:`RedisEntityLock` instead.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_guard = asyncio.Lock()

    async def _get_lock(self, entity_id: str) -> asyncio.Lock:
        async with self._registry_guard:
            lock = self._locks.get(entity_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[entity_id] = lock
            return lock

    @asynccontextmanager
    async def _acquire(self, entity_id: str) -> AsyncIterator[None]:
        lock = await self._get_lock(entity_id)
        async with lock:
            yield

    def acquire(self, entity_id: str) -> AbstractAsyncContextManager[None]:
        return self._acquire(entity_id)


# Abstract factory for the Redis adapter — concrete redis-py wiring
# lands in Wave 2 alongside the Redis queue infrastructure (T2.1).
class RedisEntityLock:
    """Redis-backed implementation of :class:`EntityLock`.

    The lock key is ``f"{key_prefix}:{slot}"`` where ``slot`` is the
    32-bit hash of ``entity_id`` modulo ``slot_count``. Bounding the
    key space avoids unbounded growth in Redis when entity ids have
    high cardinality.
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        key_prefix: str = "indexing:graph:entity",
        slot_count: int = 4096,
        lock_timeout_seconds: int = 30,
    ) -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._slot_count = max(1, slot_count)
        self._lock_timeout_seconds = lock_timeout_seconds

    def _slot_key(self, entity_id: str) -> str:
        # ``crc32`` is part of stdlib and gives a stable 32-bit hash;
        # ``hash()`` is process-randomised since Python 3.3 so it is
        # unsuitable for sharding across worker processes.
        from binascii import crc32

        slot = crc32(entity_id.encode("utf-8")) % self._slot_count
        return f"{self._key_prefix}:{slot:04x}"

    @asynccontextmanager
    async def _acquire(self, entity_id: str) -> AsyncIterator[None]:
        # ``redis.asyncio.Redis.lock`` returns an async context manager
        # in modern redis-py; we delegate so any blocking happens on
        # the redis client's own event loop.
        lock = self._redis.lock(
            self._slot_key(entity_id),
            timeout=self._lock_timeout_seconds,
            blocking=True,
        )
        await lock.acquire()
        try:
            yield
        finally:
            try:
                await lock.release()
            except Exception:  # noqa: BLE001
                # If the lock has already expired the release will
                # fail; we tolerate that since we cannot do anything
                # more constructive at this point and the caller's
                # work is already done.
                pass

    def acquire(self, entity_id: str) -> AbstractAsyncContextManager[None]:
        return self._acquire(entity_id)


# ---------------------------------------------------------------------
# Storage Protocol — backend abstraction
# ---------------------------------------------------------------------


class LineageGraphStore(Protocol):
    """Per-collection lineage-aware graph backend.

    The methods are written so that
    :class:`GraphModalityWorker` can implement the §D.3.2 algorithm
    against any backend. Each method MUST be safe to call from the
    asyncio event loop (typically by offloading blocking work to
    ``asyncio.to_thread``).

    Backends are expected to maintain the invariant that a row's
    lineage SET deduplicates by ``(document_id, parse_version)``;
    callers that try to add a member that already exists may either
    no-op or replace, but two members with the same key MUST NOT
    coexist after an upsert.
    """

    async def find_entity_ids_with_lineage(
        self,
        *,
        document_id: str,
    ) -> list[str]:
        """Return entity ids whose ``source_lineage`` contains ANY
        member with the given ``document_id`` (across any
        ``parse_version``).

        Per §D.3.6 step 3 ("doc_A v2 写入（覆盖 doc_A 旧 lineage）"),
        when a new parse_version supersedes an old one, ALL of the
        document's lineage members must be cleared so the rebuild
        phase can write the new ``(document_id, parse_version)`` slice
        without leaving stale members from older parses behind. The
        Protocol therefore filters by document_id only; per-parse_v
        granularity is preserved on the SET elements themselves but
        not exposed at the cleanup query.
        """

    async def find_relation_keys_with_lineage(
        self,
        *,
        document_id: str,
    ) -> list[tuple[str, str, str]]:
        """Return relation ``(source, target, type)`` keys whose
        ``evidence_lineage`` contains ANY member with the given
        ``document_id``."""

    async def remove_entity_lineage_member(
        self,
        *,
        entity_name: str,
        document_id: str,
    ) -> None:
        """Remove ALL ``LineageMember`` rows from ``source_lineage``
        AND all ``DescriptionPart`` rows from ``description_parts``
        whose ``document_id`` matches. Other documents' lineage is
        left untouched.

        Removing by document_id (not (doc, parse_version)) is what
        makes §D.3.6 step 3 work without an out-of-band orchestrator
        cleanup: when doc_A re-parses to v2, the v1 lineage is
        cleared as part of the same sync call.
        """

    async def remove_relation_lineage_member(
        self,
        *,
        source: str,
        target: str,
        type: str,
        document_id: str,
    ) -> None:
        """Remove ALL ``LineageMember`` rows from ``evidence_lineage``
        and matching ``DescriptionPart`` rows whose ``document_id``
        matches."""

    async def gc_entity_if_orphan(self, entity_name: str) -> bool:
        """Delete the entity row iff its ``source_lineage`` is empty.

        Returns True when a delete actually ran.
        """

    async def gc_relation_if_orphan(self, source: str, target: str, type: str) -> bool:
        """Same as :meth:`gc_entity_if_orphan` for a relation edge."""

    async def delete_entity(self, entity_name: str) -> bool:
        """Wave 7 W7-1: unconditionally delete the entity row.

        Used by :class:`GraphCurationService.merge_entities` (W7-6) to
        remove the source entities after a user-driven merge — we want
        the row gone regardless of any remaining lineage members,
        because the canonical entity has absorbed those parts. The
        ``gc_*_if_orphan`` family is for indexer-side garbage
        collection (only delete when lineage is empty); curation merge
        needs an explicit unconditional delete that complements it.

        Returns ``True`` when a delete actually ran (the row existed),
        ``False`` when the name was not present (idempotent retry).
        """

    async def delete_relation(self, source: str, target: str, type: str) -> bool:
        """Wave 7 W7-1: unconditionally delete the relation row.

        Symmetric to :meth:`delete_entity` for a relation edge —
        same use case (merge of relations under a curation flow).
        """

    async def upsert_entity_with_lineage(
        self,
        *,
        record: EntityRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        """Add ``lineage`` to ``source_lineage`` and a corresponding
        ``DescriptionPart`` to ``description_parts``. Creates the
        entity if absent. Replaces an existing member with the same
        ``(document_id, parse_version)`` key.

        Wave 7 (W7-1) ``compacted_description`` semantics: ``None``
        (default) means "preserve any existing compacted_description on
        the row" — Wave 4 indexer-side per-chunk upserts MUST NOT clear
        a compacted value computed by ``GraphIndexCompactor`` (W7-2) on
        a previous sync. A non-None string overwrites the column.
        Backends implement this with a ``COALESCE``-style update so the
        Wave 4 hot path stays a single SQL statement (per Wave 4 §C.3
        forward-only retry contract).
        """

    async def upsert_relation_with_lineage(
        self,
        *,
        record: RelationRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        """Symmetric to :meth:`upsert_entity_with_lineage` for relations.

        Wave 7 (W7-1) ``compacted_description`` semantics match the
        entity variant: ``None`` preserves the existing column value;
        a non-None string overwrites.
        """

    async def bulk_upsert_entity_with_lineage_parts(
        self,
        *,
        parts: Sequence[tuple[EntityRecord, LineageMember]],
    ) -> None:
        """Wave 8 W8-2: atomic bulk variant of
        :meth:`upsert_entity_with_lineage`.

        Each ``(record, lineage)`` tuple lands as a separate description
        part with the same ``(document_id, parse_version)`` dedup key
        the single upsert uses; semantically equivalent to looping
        :meth:`upsert_entity_with_lineage` but executed in **one
        transaction / one round-trip** so callers consolidating N×M
        parts (e.g. :class:`LineageEntityMerger.merge_entities` step 6a)
        get O(1) network round-trips instead of O(N×M).

        Contract:

        * All ``record.name`` values MUST share the same string —
          backends MAY assert and raise ``ValueError`` if they don't.
        * Empty ``parts`` is a no-op.
        * Per-part ``record.entity_type`` follows the single-upsert
          "most recently observed value wins" rule (last tuple's type
          is the post-write entity_type for the row).
        * ``compacted_description`` is intentionally **not** a
          parameter — the bulk path never touches the column. Callers
          that need to set it run a separate single
          :meth:`upsert_entity_with_lineage` afterwards (the merger's
          step 6b sentinel write does exactly this).
        * Forward-only retry safety: the dedup key is per-part, so a
          mid-flight crash + retry replays each tuple idempotently.
        """

    async def get_entity(self, entity_name: str) -> EntityWithLineage | None:
        """Read-path helper used by tests / read primitives. Returns
        the canonical lineage view, or ``None`` if the row was GC'd.
        """

    async def get_relation(self, source: str, target: str, type: str) -> RelationWithLineage | None:
        """Read-path helper for relations."""

    # ------------------------------------------------------------------
    # Wave 6 #33 — graph-RAG retrieval query layer. The retrieval
    # pipeline (§G.5) and graph-curation flows consume these to compose
    # a graph-recall context for a user query without going through the
    # legacy ``GraphIndexService.query_context``.
    #
    # Design notes:
    # * Returns are the canonical :class:`EntityWithLineage` /
    #   :class:`RelationWithLineage` shapes already used by ``get_*`` —
    #   keeps the read surface uniform; callers compose context text
    #   themselves rather than relying on a backend-formatted string.
    # * ``expand_neighbors_n_hops`` returns BOTH neighbour entities and
    #   the relation edges that connect them (1-hop or multi-hop). Per
    #   simple-stable directive the multi-hop frontier is bounded by
    #   ``hops`` (caller picks; default 1) so backends don't need a
    #   query planner.
    # * Vector-recall (``query_entities_by_vector``) was originally
    #   declared in chunk 1 but dropped per architect ruling
    #   msg=54eac595 (Option A): the Wave 4 lineage schema does not
    #   carry an entity-vector column, so a Protocol method on this
    #   store would either invite a Qdrant cross-concern coupling or
    #   silently no-op. Wave 7+ may add it via a separate
    #   dedicated query service if real evidence surfaces; until then
    #   no dead code lives here. See §K.11 amend.
    # ------------------------------------------------------------------

    async def query_entities_by_keyword(
        self,
        *,
        query: str,
        top_k: int,
    ) -> list[EntityWithLineage]:
        """Return up to ``top_k`` entities whose ``name`` (or backend
        equivalent text-searchable field) matches ``query``.

        Match semantics are backend-defined (case-insensitive substring
        on Postgres / Cypher CONTAINS on Neo4j / etc.); the contract
        is "best-effort lexical recall, not strict equality". An empty
        or whitespace-only ``query`` returns ``[]`` (no spurious recall).
        ``top_k <= 0`` returns ``[]``.
        """

    async def expand_neighbors_n_hops(
        self,
        *,
        entity_names: list[str],
        hops: int = 1,
    ) -> tuple[list[EntityWithLineage], list[RelationWithLineage]]:
        """Return entities + relations reachable from any of
        ``entity_names`` within ``hops`` graph traversal steps.

        ``hops=1`` returns immediate neighbours plus the connecting
        relations. ``hops=N`` returns the N-hop frontier; the result
        UNION includes the seed entities themselves so callers can
        feed it directly to a context formatter without a separate
        join step.

        Implementations MUST bound the result by ``hops`` to keep the
        traversal cost predictable; per simple-stable directive
        operators don't manage tunable per-collection traversal
        budgets — this is the only knob. ``hops <= 0`` returns just
        the requested seed entities (no relations). Empty
        ``entity_names`` returns ``([], [])``.
        """

    async def list_entity_labels(self) -> list[str]:
        """Return the sorted distinct ``EntityRecord.type`` values
        present in this collection.

        Used by the UI ``GET /collections/{id}/graphs/labels`` endpoint
        to populate the entity-type filter. Returns ``[]`` when the
        collection has not been indexed yet — that is the *correct*
        answer, not a cue to fall back to the legacy graphindex path.

        Wave 6 #40 narrow-replacement (per architect ruling
        msg=3efdf906): replaces the legacy
        ``GraphIndexService.list_labels(collection_id)`` call. The
        store is already collection-scoped per-instance so no
        ``collection_id`` parameter is needed at this layer.
        """

    async def list_entities(
        self,
        *,
        label: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[EntityWithLineage]:
        """Wave 7 W7-10: paginated list of entities in this collection,
        optionally filtered by ``entity_type`` (label).

        Sorted deterministically by ``name`` so paged callers
        (``offset += limit`` loop) never miss / duplicate rows. ``limit
        <= 0`` returns ``[]``; ``offset < 0`` is treated as 0.

        Replaces the legacy
        ``GraphIndexService.list_entities_for_curation`` /
        ``GraphIndexService.get_knowledge_graph`` enumerate-by-label
        paths now that the legacy package is being deleted. Used by:

        * ``aperag/domains/knowledge_graph/service.py:get_knowledge_graph``
          — UI graph view, single-page call with ``limit=max_nodes``
          followed by ``GraphSearchService.get_subgraph`` for the
          edge expansion (per architect Q1 ratify msg=f3216dfc).
        * ``aperag/graph_curation/service.py:start_run`` —
          user-triggered cross-doc candidate sweep, paged loop with
          ``limit=1000`` until fewer than ``limit`` rows return.

        Mirror ``query_entities_by_keyword`` / ``expand_neighbors_n_hops``
        Protocol pattern (W6 #33 chunk 2).
        """


# ---------------------------------------------------------------------
# In-memory reference implementation — usable by tests and as the
# canonical correctness oracle.
# ---------------------------------------------------------------------


@dataclass
class _InMemoryEntityRow:
    name: str
    entity_type: str
    description_parts: dict[tuple[str, str], DescriptionPart] = field(default_factory=dict)
    source_lineage: dict[tuple[str, str], LineageMember] = field(default_factory=dict)
    compacted_description: str | None = None


@dataclass
class _InMemoryRelationRow:
    source: str
    target: str
    relation_type: str
    description_parts: dict[tuple[str, str], DescriptionPart] = field(default_factory=dict)
    evidence_lineage: dict[tuple[str, str], LineageMember] = field(default_factory=dict)
    compacted_description: str | None = None


class InMemoryLineageGraphStore:
    """Process-local Python implementation of :class:`LineageGraphStore`.

    The state is plain dicts so behaviour is fully deterministic and
    tests can introspect lineage members directly. The reference
    implementation is also the one the §D.3.6 self-test runs against,
    so any divergence between the algorithm spec and the
    implementation will fail there.
    """

    def __init__(self) -> None:
        self._entities: dict[str, _InMemoryEntityRow] = {}
        self._relations: dict[tuple[str, str, str], _InMemoryRelationRow] = {}
        # Coarse guard around the dictionaries themselves so concurrent
        # ``find_*`` calls observe a consistent state. Production
        # backends serialize per-entity via :class:`EntityLock`; this
        # in-memory store is intentionally simpler and fully serial.
        self._guard = asyncio.Lock()

    # ---- queries -----------------------------------------------------

    async def find_entity_ids_with_lineage(
        self,
        *,
        document_id: str,
    ) -> list[str]:
        async with self._guard:
            return [
                name
                for name, row in self._entities.items()
                if any(member.document_id == document_id for member in row.source_lineage.values())
            ]

    async def find_relation_keys_with_lineage(
        self,
        *,
        document_id: str,
    ) -> list[tuple[str, str, str]]:
        async with self._guard:
            return [
                rel_key
                for rel_key, row in self._relations.items()
                if any(member.document_id == document_id for member in row.evidence_lineage.values())
            ]

    # ---- lineage member removal -------------------------------------

    async def remove_entity_lineage_member(
        self,
        *,
        entity_name: str,
        document_id: str,
    ) -> None:
        async with self._guard:
            row = self._entities.get(entity_name)
            if row is None:
                return
            stale_keys = [key for key in row.source_lineage if key[0] == document_id]
            for key in stale_keys:
                row.source_lineage.pop(key, None)
                row.description_parts.pop(key, None)

    async def remove_relation_lineage_member(
        self,
        *,
        source: str,
        target: str,
        type: str,
        document_id: str,
    ) -> None:
        rel_key = (source, target, type)
        async with self._guard:
            row = self._relations.get(rel_key)
            if row is None:
                return
            stale_keys = [key for key in row.evidence_lineage if key[0] == document_id]
            for key in stale_keys:
                row.evidence_lineage.pop(key, None)
                row.description_parts.pop(key, None)

    # ---- garbage collection -----------------------------------------

    async def gc_entity_if_orphan(self, entity_name: str) -> bool:
        async with self._guard:
            row = self._entities.get(entity_name)
            if row is None:
                return False
            if row.source_lineage:
                return False
            del self._entities[entity_name]
            return True

    async def gc_relation_if_orphan(self, source: str, target: str, type: str) -> bool:
        rel_key = (source, target, type)
        async with self._guard:
            row = self._relations.get(rel_key)
            if row is None:
                return False
            if row.evidence_lineage:
                return False
            del self._relations[rel_key]
            return True

    # ---- unconditional delete (Wave 7 W7-1) -------------------------

    async def delete_entity(self, entity_name: str) -> bool:
        async with self._guard:
            if entity_name not in self._entities:
                return False
            del self._entities[entity_name]
            return True

    async def delete_relation(self, source: str, target: str, type: str) -> bool:
        rel_key = (source, target, type)
        async with self._guard:
            if rel_key not in self._relations:
                return False
            del self._relations[rel_key]
            return True

    # ---- upserts ----------------------------------------------------

    async def upsert_entity_with_lineage(
        self,
        *,
        record: EntityRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        async with self._guard:
            row = self._entities.get(record.name)
            if row is None:
                row = _InMemoryEntityRow(name=record.name, entity_type=record.entity_type)
                self._entities[record.name] = row
            else:
                # Type may evolve as new docs refine the entity; keep
                # the most recently observed value.
                row.entity_type = record.entity_type
            row.source_lineage[lineage.key()] = lineage
            row.description_parts[lineage.key()] = DescriptionPart(
                document_id=lineage.document_id,
                parse_version=lineage.parse_version,
                text=record.description,
            )
            # Wave 7 W7-1: ``None`` preserves; non-None overwrites.
            if compacted_description is not None:
                row.compacted_description = compacted_description

    async def upsert_relation_with_lineage(
        self,
        *,
        record: RelationRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        rel_key = record.relation_key()
        async with self._guard:
            row = self._relations.get(rel_key)
            if row is None:
                row = _InMemoryRelationRow(
                    source=record.source,
                    target=record.target,
                    relation_type=record.relation_type,
                )
                self._relations[rel_key] = row
            row.evidence_lineage[lineage.key()] = lineage
            row.description_parts[lineage.key()] = DescriptionPart(
                document_id=lineage.document_id,
                parse_version=lineage.parse_version,
                text=record.description,
            )
            # Wave 7 W7-1: ``None`` preserves; non-None overwrites.
            if compacted_description is not None:
                row.compacted_description = compacted_description

    async def bulk_upsert_entity_with_lineage_parts(
        self,
        *,
        parts: Sequence[tuple[EntityRecord, LineageMember]],
    ) -> None:
        if not parts:
            return
        target_name = parts[0][0].name
        if any(record.name != target_name for record, _ in parts):
            raise ValueError("bulk_upsert_entity_with_lineage_parts: all records must share the same name")
        async with self._guard:
            row = self._entities.get(target_name)
            if row is None:
                row = _InMemoryEntityRow(name=target_name, entity_type=parts[0][0].entity_type)
                self._entities[target_name] = row
            for record, lineage in parts:
                # Type may evolve as new docs refine the entity; keep
                # the most recently observed value (mirror single upsert).
                row.entity_type = record.entity_type
                row.source_lineage[lineage.key()] = lineage
                row.description_parts[lineage.key()] = DescriptionPart(
                    document_id=lineage.document_id,
                    parse_version=lineage.parse_version,
                    text=record.description,
                )

    # ---- read path --------------------------------------------------

    async def get_entity(self, entity_name: str) -> EntityWithLineage | None:
        async with self._guard:
            row = self._entities.get(entity_name)
            if row is None:
                return None
            return EntityWithLineage(
                name=row.name,
                entity_type=row.entity_type,
                source_lineage=tuple(_sorted_lineage(row.source_lineage.values())),
                description_parts=tuple(_sorted_description_parts(row.description_parts.values())),
                compacted_description=row.compacted_description,
            )

    async def get_relation(self, source: str, target: str, type: str) -> RelationWithLineage | None:
        rel_key = (source, target, type)
        async with self._guard:
            row = self._relations.get(rel_key)
            if row is None:
                return None
            return RelationWithLineage(
                source=row.source,
                target=row.target,
                relation_type=row.relation_type,
                evidence_lineage=tuple(_sorted_lineage(row.evidence_lineage.values())),
                description_parts=tuple(_sorted_description_parts(row.description_parts.values())),
                compacted_description=row.compacted_description,
            )

    # ------------------------------------------------------------------
    # Wave 6 #33 chunk 2 — graph-RAG query layer real impls.
    # ------------------------------------------------------------------

    async def query_entities_by_keyword(
        self,
        *,
        query: str,
        top_k: int,
    ) -> list[EntityWithLineage]:
        if not query or not query.strip() or top_k <= 0:
            return []
        needle = query.strip().lower()
        matches: list[EntityWithLineage] = []
        async with self._guard:
            for name, row in self._entities.items():
                if needle not in name.lower():
                    continue
                matches.append(
                    EntityWithLineage(
                        name=row.name,
                        entity_type=row.entity_type,
                        source_lineage=tuple(_sorted_lineage(row.source_lineage.values())),
                        description_parts=tuple(_sorted_description_parts(row.description_parts.values())),
                        compacted_description=row.compacted_description,
                    )
                )
        # Deterministic ordering for stable test output.
        matches.sort(key=lambda e: e.name)
        return matches[:top_k]

    async def expand_neighbors_n_hops(
        self,
        *,
        entity_names: list[str],
        hops: int = 1,
    ) -> tuple[list[EntityWithLineage], list[RelationWithLineage]]:
        if not entity_names:
            return ([], [])

        seen_entities: dict[str, EntityWithLineage] = {}
        seen_relations: dict[tuple[str, str, str], RelationWithLineage] = {}
        frontier: set[str] = {n for n in entity_names if n}

        async with self._guard:
            # Always include the seed entities (whether or not they
            # exist in the store).
            for name in frontier:
                row = self._entities.get(name)
                if row is None:
                    continue
                seen_entities[name] = EntityWithLineage(
                    name=row.name,
                    entity_type=row.entity_type,
                    source_lineage=tuple(_sorted_lineage(row.source_lineage.values())),
                    description_parts=tuple(_sorted_description_parts(row.description_parts.values())),
                    compacted_description=row.compacted_description,
                )

            current = set(frontier)
            for _ in range(max(hops, 0)):
                next_frontier: set[str] = set()
                for rel_key, rel_row in self._relations.items():
                    src, tgt, rtype = rel_key
                    if src in current or tgt in current:
                        if rel_key not in seen_relations:
                            seen_relations[rel_key] = RelationWithLineage(
                                source=rel_row.source,
                                target=rel_row.target,
                                relation_type=rel_row.relation_type,
                                evidence_lineage=tuple(_sorted_lineage(rel_row.evidence_lineage.values())),
                                description_parts=tuple(_sorted_description_parts(rel_row.description_parts.values())),
                                compacted_description=rel_row.compacted_description,
                            )
                        for neighbour in (src, tgt):
                            if neighbour in seen_entities:
                                continue
                            row = self._entities.get(neighbour)
                            if row is None:
                                continue
                            seen_entities[neighbour] = EntityWithLineage(
                                name=row.name,
                                entity_type=row.entity_type,
                                source_lineage=tuple(_sorted_lineage(row.source_lineage.values())),
                                description_parts=tuple(_sorted_description_parts(row.description_parts.values())),
                                compacted_description=row.compacted_description,
                            )
                            next_frontier.add(neighbour)
                if not next_frontier:
                    break
                current = next_frontier

        entities = sorted(seen_entities.values(), key=lambda e: e.name)
        relations = sorted(seen_relations.values(), key=lambda r: (r.source, r.target, r.relation_type))
        return (entities, relations)

    async def list_entity_labels(self) -> list[str]:
        async with self._guard:
            labels = {row.entity_type for row in self._entities.values() if row.entity_type}
        return sorted(labels)

    async def list_entities(
        self,
        *,
        label: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[EntityWithLineage]:
        if limit <= 0:
            return []
        offset = max(0, offset)
        async with self._guard:
            rows = sorted(self._entities.values(), key=lambda r: r.name)
            if label is not None:
                rows = [r for r in rows if r.entity_type == label]
            sliced = rows[offset : offset + limit]
            return [
                EntityWithLineage(
                    name=row.name,
                    entity_type=row.entity_type,
                    source_lineage=tuple(_sorted_lineage(row.source_lineage.values())),
                    description_parts=tuple(_sorted_description_parts(row.description_parts.values())),
                    compacted_description=row.compacted_description,
                )
                for row in sliced
            ]


def _sorted_lineage(members: Iterable[LineageMember]) -> list[LineageMember]:
    return sorted(members, key=lambda m: (m.document_id, m.parse_version))


def _sorted_description_parts(parts: Iterable[DescriptionPart]) -> list[DescriptionPart]:
    return sorted(parts, key=lambda p: (p.document_id, p.parse_version))


# ---------------------------------------------------------------------
# Extractor — the LLM-extraction injection point
# ---------------------------------------------------------------------


GraphExtractor = Callable[
    [Sequence[dict[str, Any]]],
    Awaitable[tuple[list[EntityRecord], list[RelationRecord]]],
]
"""``derive`` calls the extractor with the chunk records read from
``chunks.jsonl`` and gets back the raw entity / relation list to
serialise into ``kg.jsonl``.

The production extractor (LLM-driven, per-chunk) is wired in
T2.x; tests pass deterministic stubs so the §D.3.6 self-test does not
depend on any LLM.
"""


# ---------------------------------------------------------------------
# kg.jsonl serialization
# ---------------------------------------------------------------------


KG_ARTIFACT_FILENAME = "kg.jsonl"


def serialize_kg_jsonl(
    entities: Iterable[EntityRecord],
    relations: Iterable[RelationRecord],
) -> bytes:
    """Encode ``entities`` and ``relations`` into the canonical
    line-oriented ``kg.jsonl`` byte format.

    Order: entities first, then relations. Each line is one JSON
    object with a ``"kind"`` discriminator so a future schema
    evolution can add new record kinds without breaking the file
    format.

    The output ALWAYS contains at least one byte (a newline) so
    the §C.7 read contract "empty bytes → derive not finished →
    reschedule" stays unambiguous: a deliberate "no records produced"
    payload is encoded as ``b"\\n"`` and reads back as a non-empty
    artifact whose record lists are both empty. This is what makes
    the §D.3.6 step 4 / step 5 deletion flows work — the orchestrator
    can explicitly publish an empty kg.jsonl to clear a document's
    lineage contribution without that empty payload colliding with
    the "derive crashed mid-write" sentinel.
    """
    lines: list[str] = []
    for entity in entities:
        lines.append(json.dumps(entity.to_dict(), ensure_ascii=False))
    for relation in relations:
        lines.append(json.dumps(relation.to_dict(), ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_kg_jsonl(body: bytes) -> tuple[list[EntityRecord], list[RelationRecord]]:
    """Decode a ``kg.jsonl`` byte payload back into entity / relation
    record lists. Lines whose ``"kind"`` is unknown are skipped with
    a warning so a forward-compatible extension does not crash an
    older worker mid-sync.
    """
    entities: list[EntityRecord] = []
    relations: list[RelationRecord] = []
    for line_no, line in enumerate(body.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record.get("kind")
        if kind == "entity":
            entities.append(
                EntityRecord(
                    name=str(record["name"]),
                    entity_type=str(record["entity_type"]),
                    description=str(record.get("description", "")),
                    source_chunk_ids=tuple(str(cid) for cid in record.get("source_chunk_ids", [])),
                )
            )
        elif kind == "relation":
            relations.append(
                RelationRecord(
                    source=str(record["source"]),
                    target=str(record["target"]),
                    relation_type=str(record["relation_type"]),
                    description=str(record.get("description", "")),
                    source_chunk_ids=tuple(str(cid) for cid in record.get("source_chunk_ids", [])),
                )
            )
        else:
            logger.warning("kg.jsonl: skipping unknown record kind=%r at line=%d", kind, line_no)
    return entities, relations


# ---------------------------------------------------------------------
# GraphModalityWorker — implements §D.3.2 algorithm
# ---------------------------------------------------------------------


class GraphModalityWorker(ModalityWorker):
    """Graph modality :class:`ModalityWorker` implementation.

    The worker is parameterised over a :class:`LineageGraphStore`
    backend, an :class:`EntityLock` for race-protection, and a
    :class:`GraphExtractor` callable that performs the actual LLM
    entity / relation extraction from chunks. This keeps the
    algorithm code backend-agnostic so the same class works against
    Nebula, Neo4j, and the in-memory reference store; production
    deployments swap the backend and lock at construction time.
    """

    modality = Modality.GRAPH

    def __init__(
        self,
        *,
        store: LineageGraphStore,
        extractor: GraphExtractor,
        entity_lock: EntityLock,
        object_store: _SyncObjectStore,
        collection_id: str,
        tenant_scope_key: str,
        compactor: "GraphIndexCompactor | None" = None,
        embedder: Callable[[str], list[float]] | None = None,
        vector_connector: "VectorStoreConnector | None" = None,
        merge_detector: "MergeCandidateDetector | None" = None,
    ) -> None:
        """Construct a graph worker bound to a single tenant scope.

        ``tenant_scope_key`` is the §H.2 quota / ACL key supplied by
        the orchestrator at job-dispatch time. It is captured into
        every :class:`LineageMember` this worker writes during
        ``sync`` so the SET-level tenant attribution survives the
        artifact round-trip and is available to read-path ACL
        filtering.

        Wave 7 W7-3 dependencies (all optional — when omitted the
        sync() worker degrades to Wave 6 lineage-only behaviour and
        skips Phase 3 entirely):

        * ``compactor`` (Wave 7 W7-2): produces a single bounded LLM
          summary from the per-doc ``description_parts`` list. Skipped
          when ``None`` — the entity's existing ``compacted_description``
          (potentially set by a prior sync or curation merge) is left
          alone.
        * ``embedder``: sync ``(text -> list[float])`` callable; mirrors
          the ``EmbeddingService.embed_query`` signature used by the
          vector worker. Required for Step B (vector upsert) and the
          merge detector to function.
        * ``vector_connector``: a :class:`VectorStoreConnector` already
          per-tenant bound (the connector enforces the tenant guard so
          the per-point payload is 3-field — see §K.12.5 ratify
          msg=acbd0003 / msg=d3f4e6f8). Required for Steps B + C.
        * ``merge_detector`` (Wave 7 W7-4): writes PENDING auto-detect
          ``GraphCurationSuggestion`` rows. Skipped when ``None`` —
          the detector is a write-only auxiliary, not on the lineage
          critical path.
        """
        self._store = store
        self._extractor = extractor
        self._entity_lock = entity_lock
        self._object_store = object_store
        self._collection_id = collection_id
        self._tenant_scope_key = tenant_scope_key
        self._compactor = compactor
        self._embedder = embedder
        self._vector_connector = vector_connector
        self._merge_detector = merge_detector

    # ---- derive -----------------------------------------------------

    async def derive(
        self,
        *,
        document_id: str,
        parse_version: str,
        source_path: str,
    ) -> DeriveResult:
        """Read ``chunks.jsonl`` for the parse, run the extractor, and
        write ``kg.jsonl`` atomically. ``source_path`` is accepted to
        keep the ABC contract uniform but unused — the parser already
        produced everything we need under ``derived/parse_<v>/``.
        """
        del source_path  # see docstring; ABC contract uniformity
        chunks_path = derived_artifact(
            collection_id=self._collection_id,
            document_id=document_id,
            parse_version=parse_version,
            filename="chunks.jsonl",
        )
        chunks = await asyncio.to_thread(read_chunks, self._object_store, chunks_path)
        entities, relations = await self._extractor(chunks)
        body = serialize_kg_jsonl(entities, relations)
        kg_path = derived_artifact(
            collection_id=self._collection_id,
            document_id=document_id,
            parse_version=parse_version,
            filename=KG_ARTIFACT_FILENAME,
        )
        await write_atomic_async(self._object_store, kg_path, body)
        logger.info(
            "graph.derive collection=%s document=%s parse_version=%s entities=%d relations=%d",
            self._collection_id,
            document_id,
            parse_version,
            len(entities),
            len(relations),
        )
        return DeriveResult(derived_artifact_path=kg_path)

    # ---- sync -------------------------------------------------------

    async def sync(
        self,
        *,
        document_id: str,
        parse_version: str,
        derived_artifact_path: str,
    ) -> None:
        """Apply the §D.3.2 algorithm so that the backend's lineage
        state for ``(document_id, parse_version)`` matches the
        artifact, while preserving every other doc / parse_v's
        contribution to shared entities."""
        body = await read_or_none_async(self._object_store, derived_artifact_path)
        if body is None:
            # §C.7 contract: empty / missing artifact means "derive
            # not yet finished"; surface as a no-op so the orchestrator
            # reschedules instead of trapping.
            logger.info(
                "graph.sync skipped — derived artifact missing or empty path=%s",
                derived_artifact_path,
            )
            return

        entities, relations = parse_kg_jsonl(body)

        # ---- Phase 1: lineage cleanup --------------------------------
        # Per §D.3.6 step 3 + step 4, Phase 1 removes ALL lineage
        # members for the document (across any parse_version), so a
        # new parse_version supersedes the old one cleanly within one
        # sync call. The schema still tracks (doc, parse_v) members
        # so a future multi-version coexistence (if requested) is
        # unblocked at the storage level.

        affected_entity_ids = await self._store.find_entity_ids_with_lineage(document_id=document_id)
        for entity_name in affected_entity_ids:
            async with self._entity_lock.acquire(entity_name):
                await self._store.remove_entity_lineage_member(
                    entity_name=entity_name,
                    document_id=document_id,
                )
                await self._store.gc_entity_if_orphan(entity_name)

        affected_relation_keys = await self._store.find_relation_keys_with_lineage(document_id=document_id)
        for source, target, rel_type in affected_relation_keys:
            async with self._entity_lock.acquire(_relation_lock_key(source, target, rel_type)):
                await self._store.remove_relation_lineage_member(
                    source=source,
                    target=target,
                    type=rel_type,
                    document_id=document_id,
                )
                await self._store.gc_relation_if_orphan(source, target, rel_type)

        # ---- Phase 2: lineage rebuild -------------------------------

        for entity_record in entities:
            lineage = LineageMember(
                document_id=document_id,
                parse_version=parse_version,
                tenant_scope_key=self._tenant_scope_key,
                chunk_ids=entity_record.source_chunk_ids,
            )
            async with self._entity_lock.acquire(entity_record.name):
                await self._store.upsert_entity_with_lineage(record=entity_record, lineage=lineage)

        for relation_record in relations:
            lineage = LineageMember(
                document_id=document_id,
                parse_version=parse_version,
                tenant_scope_key=self._tenant_scope_key,
                chunk_ids=relation_record.source_chunk_ids,
            )
            lock_key = _relation_lock_key(
                relation_record.source,
                relation_record.target,
                relation_record.relation_type,
            )
            async with self._entity_lock.acquire(lock_key):
                await self._store.upsert_relation_with_lineage(record=relation_record, lineage=lineage)

        logger.info(
            "graph.sync collection=%s document=%s parse_version=%s rebuilt_entities=%d rebuilt_relations=%d",
            self._collection_id,
            document_id,
            parse_version,
            len(entities),
            len(relations),
        )

        # ---- Phase 3: Wave 7 W7-3 — compact + embed + vector upsert
        # + snapshot-diff vector cleanup + merge candidate detect.
        #
        # The phase is opt-in: a worker constructed without
        # ``embedder`` / ``vector_connector`` (e.g. unit tests, dev
        # scripts) skips all four steps and degrades to Wave 6
        # lineage-only behaviour. Production deployments wire the
        # dependencies in ``worker_factory._build_graph_worker``.
        #
        # Step ordering (per spec §K.12.3 + huangheng msg=16a38734
        # 7-invariant): Compactor → embed → vector upsert →
        # snapshot-diff delete → merge detector. Out-of-order would
        # break invariant #3 (Compactor must run before vector embed)
        # or invariant #7 (snapshot-diff entity-name set must be
        # captured *after* lineage rebuild + gc).
        await self._post_sync_vector_pipeline(
            document_id=document_id,
            parse_version=parse_version,
            kg_entity_records=entities,
            kg_relation_records=relations,
            pre_sync_entity_names=set(affected_entity_ids),
            pre_sync_relation_keys=set(affected_relation_keys),
        )

    # ---- Phase 3 helpers (Wave 7 W7-3) ------------------------------

    async def _post_sync_vector_pipeline(
        self,
        *,
        document_id: str,
        parse_version: str,
        kg_entity_records: Sequence[EntityRecord],
        kg_relation_records: Sequence[RelationRecord],
        pre_sync_entity_names: set[str],
        pre_sync_relation_keys: set[tuple[str, str, str]],
    ) -> None:
        """Phase 3: compact → embed → vector upsert → snapshot-diff
        delete → merge detector. See ``sync`` docstring for invariant
        ordering rationale.

        Skips entirely when both ``embedder`` and ``vector_connector``
        are unwired — the post-Wave-7 production wiring always pairs
        them, but unit tests for Phase 1+2 algorithm coverage can
        construct the worker without these dependencies.
        """
        if self._vector_connector is None or self._embedder is None:
            return  # Wave 6 backward-compat: lineage-only sync.

        # Step A — Compactor: write ``compacted_description`` for each
        # entity / relation that this sync just touched. The
        # ``upsert_*_with_lineage`` kwarg ``compacted_description``
        # uses COALESCE-style preserve, so passing ``None`` (when
        # compactor opts out or is unwired) does NOT clear an existing
        # cache; only a real summary string overwrites.
        affected_entities_for_phase3: list[EntityWithLineage] = []
        for entity_record in kg_entity_records:
            entity = await self._store.get_entity(entity_record.name)
            if entity is None:
                # Entity was extracted in this sync but is not in the
                # store anymore — should be impossible after Phase 2,
                # but guard against partial-write races.
                continue
            compacted = await self._maybe_compact(entity)
            if compacted is not None:
                lineage = LineageMember(
                    document_id=document_id,
                    parse_version=parse_version,
                    tenant_scope_key=self._tenant_scope_key,
                    chunk_ids=entity_record.source_chunk_ids,
                )
                async with self._entity_lock.acquire(entity_record.name):
                    await self._store.upsert_entity_with_lineage(
                        record=entity_record,
                        lineage=lineage,
                        compacted_description=compacted,
                    )
                # Refresh the in-memory copy so Step B embeds the new
                # compacted string, not the stale read-before-write.
                entity = EntityWithLineage(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    source_lineage=entity.source_lineage,
                    description_parts=entity.description_parts,
                    compacted_description=compacted,
                )
            affected_entities_for_phase3.append(entity)

        affected_relations_for_phase3: list[RelationWithLineage] = []
        for relation_record in kg_relation_records:
            relation = await self._store.get_relation(
                relation_record.source,
                relation_record.target,
                relation_record.relation_type,
            )
            if relation is None:
                continue
            compacted_rel = await self._maybe_compact_relation(relation)
            if compacted_rel is not None:
                lineage = LineageMember(
                    document_id=document_id,
                    parse_version=parse_version,
                    tenant_scope_key=self._tenant_scope_key,
                    chunk_ids=relation_record.source_chunk_ids,
                )
                lock_key = _relation_lock_key(
                    relation_record.source,
                    relation_record.target,
                    relation_record.relation_type,
                )
                async with self._entity_lock.acquire(lock_key):
                    await self._store.upsert_relation_with_lineage(
                        record=relation_record,
                        lineage=lineage,
                        compacted_description=compacted_rel,
                    )
                relation = RelationWithLineage(
                    source=relation.source,
                    target=relation.target,
                    relation_type=relation.relation_type,
                    evidence_lineage=relation.evidence_lineage,
                    description_parts=relation.description_parts,
                    compacted_description=compacted_rel,
                )
            affected_relations_for_phase3.append(relation)

        # Step B — Embed + vector upsert (3-field payload, uuid5 id
        # includes collection_id for cross-collection uniqueness in a
        # shared backing store; per-tenant guard is the connector's
        # responsibility — see spec §K.12.5 lock).
        for entity in affected_entities_for_phase3:
            await self._upsert_entity_vector_point(entity)
        for relation in affected_relations_for_phase3:
            await self._upsert_relation_vector_point(relation)

        # Step C — Snapshot-diff delete: drop vector points for any
        # entity / relation that pre-sync had a row but post-sync does
        # not (gc_*_if_orphan deleted it during Phase 1). Computing the
        # diff on lineage-store names instead of an ANN list-all is
        # invariant #7 — the vector backend is treated as derivable
        # state, the lineage store is the source of truth.
        post_sync_entity_names = await self._capture_entity_names(
            kg_entity_records=kg_entity_records,
            pre_sync_entity_names=pre_sync_entity_names,
        )
        gc_entity_names = pre_sync_entity_names - post_sync_entity_names
        if gc_entity_names:
            ids = [self._entity_vector_id(name) for name in gc_entity_names]
            await asyncio.to_thread(self._vector_connector.delete, ids)

        post_sync_relation_keys = await self._capture_relation_keys(
            kg_relation_records=kg_relation_records,
            pre_sync_relation_keys=pre_sync_relation_keys,
        )
        gc_relation_keys = pre_sync_relation_keys - post_sync_relation_keys
        if gc_relation_keys:
            ids = [self._relation_vector_id(*key) for key in gc_relation_keys]
            await asyncio.to_thread(self._vector_connector.delete, ids)

        # Step D — Merge candidate detection (write-only, D-3 lock —
        # detector writes PENDING ``GraphCurationSuggestion`` rows for
        # the curator UI; never auto-merges).
        if self._merge_detector is not None:
            affected_names = [e.name for e in affected_entities_for_phase3]
            sync_run_id = f"{self._collection_id}:{document_id}:{parse_version}"
            try:
                await self._merge_detector.detect_for_sync(
                    sync_run_id=sync_run_id,
                    affected_entity_names=affected_names,
                )
            except Exception:  # noqa: BLE001 — detector failure must not break sync
                logger.warning(
                    "graph.sync merge_detector failed (non-fatal) collection=%s document=%s",
                    self._collection_id,
                    document_id,
                    exc_info=True,
                )

    async def _maybe_compact(self, entity: EntityWithLineage) -> str | None:
        """Run the compactor over an entity's ``description_parts``;
        return the new summary or ``None`` (caller leaves the column
        alone — COALESCE preserve)."""
        if self._compactor is None:
            return None
        parts_text = [p.text for p in entity.description_parts if p.text]
        try:
            return await self._compactor.compact_if_oversized(parts_text)
        except Exception:  # noqa: BLE001 — compactor LLM may flake
            logger.warning(
                "graph.sync compactor failed for entity=%r (non-fatal)",
                entity.name,
                exc_info=True,
            )
            return None

    async def _maybe_compact_relation(self, relation: RelationWithLineage) -> str | None:
        if self._compactor is None:
            return None
        parts_text = [p.text for p in relation.description_parts if p.text]
        try:
            return await self._compactor.compact_if_oversized(parts_text)
        except Exception:  # noqa: BLE001
            logger.warning(
                "graph.sync compactor failed for relation=%s->%s:%s (non-fatal)",
                relation.source,
                relation.target,
                relation.relation_type,
                exc_info=True,
            )
            return None

    async def _upsert_entity_vector_point(self, entity: EntityWithLineage) -> None:
        """Embed the entity's compacted description (or fallback to
        ``name + parts`` concat when compactor declined or unwired) and
        upsert one Qdrant-style point. Connector handles tenant guard;
        payload is the 3-field shape locked at spec §K.12.5."""
        text = self._embed_text_for_entity(entity)
        if not text:
            return
        try:
            embedding = await asyncio.to_thread(self._embedder, text)
        except Exception:  # noqa: BLE001
            logger.warning(
                "graph.sync embedder failed for entity=%r (non-fatal)",
                entity.name,
                exc_info=True,
            )
            return
        from aperag.vectorstore.dto import VectorPoint

        point = VectorPoint(
            id=self._entity_vector_id(entity.name),
            vector=list(embedding),
            payload={
                "indexer": "graph_entity",
                "entity_name": entity.name,
                "entity_type": entity.entity_type,
            },
        )
        try:
            await asyncio.to_thread(self._vector_connector.upsert, [point])
        except Exception:  # noqa: BLE001
            logger.warning(
                "graph.sync vector upsert failed for entity=%r (non-fatal)",
                entity.name,
                exc_info=True,
            )

    async def _upsert_relation_vector_point(self, relation: RelationWithLineage) -> None:
        text = self._embed_text_for_relation(relation)
        if not text:
            return
        try:
            embedding = await asyncio.to_thread(self._embedder, text)
        except Exception:  # noqa: BLE001
            logger.warning(
                "graph.sync embedder failed for relation=%s->%s:%s (non-fatal)",
                relation.source,
                relation.target,
                relation.relation_type,
                exc_info=True,
            )
            return
        from aperag.vectorstore.dto import VectorPoint

        point = VectorPoint(
            id=self._relation_vector_id(relation.source, relation.target, relation.relation_type),
            vector=list(embedding),
            payload={
                "indexer": "graph_relation",
                "entity_name": f"{relation.source}->{relation.target}",
                "entity_type": relation.relation_type,
            },
        )
        try:
            await asyncio.to_thread(self._vector_connector.upsert, [point])
        except Exception:  # noqa: BLE001
            logger.warning(
                "graph.sync vector upsert failed for relation=%s->%s:%s (non-fatal)",
                relation.source,
                relation.target,
                relation.relation_type,
                exc_info=True,
            )

    @staticmethod
    def _embed_text_for_entity(entity: EntityWithLineage) -> str:
        # Prefer compactor's summary; fallback joins ``name`` + raw
        # description parts so newly-extracted entities (compactor
        # below threshold) still get vectorised.
        if entity.compacted_description:
            return entity.compacted_description
        body = "\n\n".join(p.text for p in entity.description_parts if p.text)
        if body:
            return f"{entity.name}\n\n{body}"
        return entity.name

    @staticmethod
    def _embed_text_for_relation(relation: RelationWithLineage) -> str:
        if relation.compacted_description:
            return relation.compacted_description
        body = "\n\n".join(p.text for p in relation.description_parts if p.text)
        head = f"{relation.source} -[{relation.relation_type}]-> {relation.target}"
        if body:
            return f"{head}\n\n{body}"
        return head

    def _entity_vector_id(self, entity_name: str) -> str:
        return str(uuid5(NAMESPACE_DNS, f"graph_entity:{self._collection_id}:{entity_name}"))

    def _relation_vector_id(self, source: str, target: str, relation_type: str) -> str:
        return str(
            uuid5(
                NAMESPACE_DNS,
                f"graph_relation:{self._collection_id}:{source}->{target}:{relation_type}",
            )
        )

    async def _capture_entity_names(
        self,
        *,
        kg_entity_records: Sequence[EntityRecord],
        pre_sync_entity_names: set[str],
    ) -> set[str]:
        """Build the post-sync entity-name set: every entity that was
        in the sync's kg.jsonl plus any pre-existing entity from
        ``pre_sync_entity_names`` that survived ``gc_entity_if_orphan``
        (it had lineage from another doc / parse_version)."""
        post: set[str] = {r.name for r in kg_entity_records}
        # Pre-existing entities — only include those still present after
        # Phase 1 gc (other documents' lineage kept them alive).
        for name in pre_sync_entity_names:
            if name in post:
                continue
            row = await self._store.get_entity(name)
            if row is not None:
                post.add(name)
        return post

    async def _capture_relation_keys(
        self,
        *,
        kg_relation_records: Sequence[RelationRecord],
        pre_sync_relation_keys: set[tuple[str, str, str]],
    ) -> set[tuple[str, str, str]]:
        post: set[tuple[str, str, str]] = {r.relation_key() for r in kg_relation_records}
        for key in pre_sync_relation_keys:
            if key in post:
                continue
            row = await self._store.get_relation(*key)
            if row is not None:
                post.add(key)
        return post


def _relation_lock_key(source: str, target: str, type: str) -> str:
    """Lock key that serialises read-modify-write on a relation edge.

    Using a structured prefix keeps relation locks disjoint from
    entity locks (an entity named ``"rel:foo->bar:type"`` would
    otherwise clobber the lock).
    """
    return f"rel::{source}::{target}::{type}"


__all__ = [
    # Lineage data model
    "LineageMember",
    "DescriptionPart",
    "EntityRecord",
    "RelationRecord",
    "EntityWithLineage",
    "RelationWithLineage",
    # Per-entity lock abstraction
    "EntityLock",
    "InMemoryEntityLock",
    "RedisEntityLock",
    # Storage Protocol + reference implementation
    "LineageGraphStore",
    "InMemoryLineageGraphStore",
    # Extractor injection point
    "GraphExtractor",
    # kg.jsonl helpers
    "KG_ARTIFACT_FILENAME",
    "serialize_kg_jsonl",
    "parse_kg_jsonl",
    # Modality worker
    "GraphModalityWorker",
]
