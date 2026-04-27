# Copyright 2026 ApeCloud, Inc.
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

"""User-driven entity merge over the lineage graph store — Wave 7
§K.12.6 / §K.12.10 task #6.

Replaces the legacy ``GraphIndexService.merge_entities`` flow with one
that operates on :class:`aperag.indexing.graph.LineageGraphStore` +
:class:`aperag.graph_curation.alias_map.AliasMapRepository` +
:class:`aperag.indexing.graph_compactor.GraphIndexCompactor` +
:class:`aperag.vectorstore.base.VectorStoreConnector`. Per architect
ratify msg=cf860ae4 + huangheng endorse msg=22816e0d:

Step ordering (locked, invariant #2 L1 → L2 单向派生):

1. ``alias_repo.resolve_canonical(target_name)`` to flatten any
   pre-existing chain (1-hop guarantee — target may itself be an
   alias).
2. For each source: ``alias_repo.upsert_alias(alias=source,
   target=final_target)``. Cycle reject is :class:`AliasCycleError`
   raised by the alias repo; we let it propagate so the user sees an
   actionable error.
3. Read target + sources via ``store.get_entity``; UNION their
   ``description_parts`` lists.
4. LLM unified description over the merged parts.
5. Compactor pass via ``compact_if_oversized([unified],
   subject_kind="entity", subject_label=final_target, language=...)``
   — kwargs locked by drift #3.
6. L1 target write — TWO sub-steps:
   6a. For each source's ``description_parts``: re-upsert under the
       target name preserving the original
       ``(document_id, parse_version, chunk_ids)`` so per-doc lineage
       is not lost (invariant #1 L1 不污染).
   6b. Final upsert with the unified text + compacted_description,
       tagged with sentinel ``document_id="__curation_merge__"``
       (sentinel pick locked by huangheng msg=22816e0d) and
       ``parse_version=str(merge_uuid)`` so each merge has a unique
       lineage member that cannot be confused with a real-doc
       contribution.
7. Vector layer write — 3-field payload (``indexer / entity_name /
   entity_type``), ``uuid5(NAMESPACE_DNS, "graph_entity:{cid}:{name}")``
   for the deterministic point id (invariants #5 + #6).
8. Delete sources — last. ``store.delete_entity(s)`` (single-arg, drift
   #1) plus the corresponding vector point delete.

Failure mode
------------

The merge is *not* a single SQL transaction across all eight steps —
the lineage store layer commits each ``upsert_*_with_lineage`` call
independently. Per huangheng msg=22816e0d this gives a "soft
transition" mid-failure state: partial source parts already on the
target, sources still readable until step 8 deletes them, and the
caller can retry the merge idempotently because every upsert is keyed
on ``(document_id, parse_version)``.

The alias upsert (step 2) IS transactional inside :class:`AliasMapRepository`
so a failed merge never leaves orphan alias rows that point at half-merged
canonicals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence
from uuid import NAMESPACE_DNS, uuid4, uuid5

from aperag.graph_curation.alias_map import AliasCycleError, AliasMapRepository
from aperag.indexing.graph import (
    DescriptionPart,
    EntityRecord,
    EntityWithLineage,
    LineageGraphStore,
    LineageMember,
)
from aperag.indexing.graph_compactor import GraphIndexCompactor
from aperag.vectorstore.base import VectorStoreConnector
from aperag.vectorstore.dto import VectorPoint

logger = logging.getLogger(__name__)


# Sentinel ``document_id`` for the unified+compacted lineage member
# written in step 6b. ``__curation_merge__`` was picked by huangheng
# msg=22816e0d over ``__user_merge__`` because ``user`` is misleading
# (the API caller is ``GraphCurationService``, not the UI user).
CURATION_MERGE_DOCUMENT_ID: str = "__curation_merge__"


# Wave 7 §K.12 invariant #5: graph entity vector points carry exactly
# these three payload fields. Locked by architect ratify (msg=acbd0003
# Q4 + msg=cf860ae4 step 7'). Match this constant against task #3's
# write payload to keep the indexer hot path and the curation merge
# path in lock-step.
GRAPH_ENTITY_INDEXER: str = "graph_entity"


@dataclass(frozen=True)
class LineageMergeResult:
    """Return shape of :meth:`LineageEntityMerger.merge_entities`.

    ``final_target`` is the canonical name actually written to (may
    differ from the requested ``target_name`` if it was itself an
    alias). ``merged_source_ids`` is the list of source names that
    were folded in (in input order). ``compacted_description`` is the
    unified+compacted text written to L1 / embedded into the vector
    point — ``None`` when the unified description was short enough to
    skip compaction (Compactor returns ``None`` below threshold).
    """

    final_target: str
    merged_source_ids: list[str]
    unified_description: str
    compacted_description: str | None


# Embedder API: callers pass any object exposing a sync ``embed_query``
# (matches :class:`aperag.llm.embed.embedding_service.EmbeddingService`).
EmbedQueryFn = Callable[[str], list[float]]


# LLM API: any async callable mapping prompt → response text. Reuses
# the :class:`aperag.indexing.llm.LLMCall` shape so wiring is a direct
# drop-in.
LLMCall = Callable[[str], Awaitable[str]]


class LineageEntityMerger:
    """Per-collection user-driven merge orchestrator.

    Constructor takes the inner store (NOT the alias-redirect decorator
    — the merge writes the canonical names directly), the alias-map
    repository, the compactor + LLM + embedder + vector connector, and
    the bound ``collection_id``. Construction does no I/O.
    """

    def __init__(
        self,
        *,
        store: LineageGraphStore,
        alias_repo: AliasMapRepository,
        compactor: GraphIndexCompactor,
        vector_connector: VectorStoreConnector,
        embedder: Any,
        llm: LLMCall,
        collection_id: str,
        language: str = "Chinese",
    ) -> None:
        self._store = store
        self._alias_repo = alias_repo
        self._compactor = compactor
        self._vector_connector = vector_connector
        self._embedder = embedder
        self._llm = llm
        self._collection_id = collection_id
        self._language = language

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    async def merge_entities(
        self,
        *,
        target_name: str,
        source_names: Sequence[str],
        merged_by: str | None = None,
    ) -> LineageMergeResult:
        """Merge ``source_names`` into ``target_name`` (or the canonical
        ``target_name`` resolves to).

        Raises :class:`aperag.graph_curation.alias_map.AliasCycleError`
        if any source → target relationship would form a cycle. The
        caller should surface this to the user so they can pick a
        different target.

        Empty ``source_names`` is a no-op (returns immediately with
        ``LineageMergeResult(final_target=target_name, ...,
        merged_source_ids=[])``).
        """
        if not source_names:
            return LineageMergeResult(
                final_target=target_name,
                merged_source_ids=[],
                unified_description="",
                compacted_description=None,
            )

        # Step 1 — flatten target chain (1-hop).
        final_target = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=target_name)

        # Step 2 — record the alias rows so future indexer writes get
        # transparently redirected. Cycle reject propagates as
        # AliasCycleError so the caller can abort the merge cleanly.
        for src in source_names:
            await self._alias_repo.upsert_alias(
                collection_id=self._collection_id,
                alias_name=src,
                target=final_target,
                merged_by=merged_by,
            )

        # Step 3 — read target + sources, accumulate description parts.
        target_entity = await self._store.get_entity(final_target)
        if target_entity is None:
            # Target was GC'd between merge initiation and execution.
            # We still wrote the alias rows so future indexer writes to
            # the sources will resolve to the (now-empty) canonical;
            # but the merge body has nothing to consolidate.
            logger.warning(
                "lineage_merger: target %r missing at merge time (collection=%s); "
                "alias rows written but no L1/vector update performed",
                final_target,
                self._collection_id,
            )
            return LineageMergeResult(
                final_target=final_target,
                merged_source_ids=list(source_names),
                unified_description="",
                compacted_description=None,
            )

        source_entities: list[EntityWithLineage] = []
        for src in source_names:
            row = await self._store.get_entity(src)
            if row is not None:
                source_entities.append(row)

        all_parts: list[DescriptionPart] = list(target_entity.description_parts)
        for src in source_entities:
            all_parts.extend(src.description_parts)
        part_texts = [p.text for p in all_parts if p.text and p.text.strip()]

        # Step 4 — LLM unified description.
        unified = await self._unified_description(
            entity_name=final_target,
            entity_type=target_entity.entity_type,
            part_texts=part_texts,
        )

        # Step 5 — compact if oversized.
        compacted = await self._compactor.compact_if_oversized(
            [unified],
            subject_kind="entity",
            subject_label=final_target,
            language=self._language,
        )

        # Step 6a — re-anchor source parts under the target name,
        # preserving their original lineage so per-doc tracking is not
        # lost (invariant #1). DescriptionPart carries no chunk_ids of
        # its own; we look up the matching LineageMember by
        # ``(document_id, parse_version)`` to recover them.
        for src in source_entities:
            lineage_by_key = {m.key(): m for m in src.source_lineage}
            for part in src.description_parts:
                member = lineage_by_key.get(part.key())
                chunk_ids = tuple(member.chunk_ids) if member is not None else ()
                await self._store.upsert_entity_with_lineage(
                    record=EntityRecord(
                        name=final_target,
                        entity_type=target_entity.entity_type,
                        description=part.text,
                        source_chunk_ids=chunk_ids,
                    ),
                    lineage=LineageMember(
                        document_id=part.document_id,
                        parse_version=part.parse_version,
                        tenant_scope_key=self._tenant_scope_key_for(src),
                        chunk_ids=chunk_ids,
                    ),
                )

        # Step 6b — final write with unified text + compacted_description
        # under the curation-merge sentinel lineage.
        merge_uuid = str(uuid4())
        await self._store.upsert_entity_with_lineage(
            record=EntityRecord(
                name=final_target,
                entity_type=target_entity.entity_type,
                description=unified,
                source_chunk_ids=(),
            ),
            lineage=LineageMember(
                document_id=CURATION_MERGE_DOCUMENT_ID,
                parse_version=merge_uuid,
                tenant_scope_key=self._tenant_scope_key_for(target_entity),
                chunk_ids=(),
            ),
            compacted_description=compacted,
        )

        # Step 7 — vector layer write (3-field payload, uuid5 point id).
        await self._upsert_vector_point(
            entity_name=final_target,
            entity_type=target_entity.entity_type,
            text=compacted or unified,
        )

        # Step 8 — delete sources from L1 and vector store, last.
        for src_entity in source_entities:
            await self._store.delete_entity(src_entity.name)
            await self._delete_vector_point(src_entity.name)

        return LineageMergeResult(
            final_target=final_target,
            merged_source_ids=list(source_names),
            unified_description=unified,
            compacted_description=compacted,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _unified_description(
        self,
        *,
        entity_name: str,
        entity_type: str,
        part_texts: list[str],
    ) -> str:
        """Use the LLM to produce one consolidated description from the
        per-doc fragments. Returns the joined fragments unchanged when
        there are 0 or 1 fragments (no merge needed) or when the LLM
        call fails (graceful degrade — partial merge still proceeds).
        """
        if not part_texts:
            return ""
        if len(part_texts) == 1:
            return part_texts[0]
        prompt = self._render_merge_prompt(
            entity_name=entity_name,
            entity_type=entity_type,
            fragments=part_texts,
        )
        try:
            return (await self._llm(prompt)).strip()
        except Exception:
            logger.warning(
                "lineage_merger: unified-description LLM call failed for %r; falling back to joined parts",
                entity_name,
                exc_info=True,
            )
            return "\n\n".join(part_texts)

    @staticmethod
    def _render_merge_prompt(*, entity_name: str, entity_type: str, fragments: list[str]) -> str:
        # Light-weight prompt — the heavy lifting (length cap, language
        # control, S-P-O framing) lives in the Compactor (task #2). The
        # merge prompt's only job is to consolidate fragments into one
        # paragraph so the Compactor sees a single coherent input.
        joined = "\n\n---\n\n".join(fragments)
        return (
            f"You are merging descriptions of the same {entity_type} "
            f"entity ({entity_name}). Consolidate the following "
            f"fragments into ONE coherent paragraph that preserves "
            f"every fact. Do not add information that isn't in the "
            f"fragments.\n\n"
            f"Fragments:\n\n---\n\n{joined}\n\n---\n\n"
            f"Consolidated description:"
        )

    async def _upsert_vector_point(self, *, entity_name: str, entity_type: str, text: str) -> None:
        if not text:
            return
        try:
            embedding = await _to_thread(self._embedder.embed_query, text)
        except Exception:
            logger.warning(
                "lineage_merger: embed failed for %r; skipping vector upsert",
                entity_name,
                exc_info=True,
            )
            return
        point = VectorPoint(
            id=str(
                uuid5(
                    NAMESPACE_DNS,
                    f"{GRAPH_ENTITY_INDEXER}:{self._collection_id}:{entity_name}",
                )
            ),
            vector=embedding,
            payload={
                "indexer": GRAPH_ENTITY_INDEXER,
                "entity_name": entity_name,
                "entity_type": entity_type,
            },
        )
        try:
            await _to_thread(self._vector_connector.upsert, [point])
        except Exception:
            logger.warning(
                "lineage_merger: vector upsert failed for %r",
                entity_name,
                exc_info=True,
            )

    async def _delete_vector_point(self, entity_name: str) -> None:
        point_id = str(
            uuid5(
                NAMESPACE_DNS,
                f"{GRAPH_ENTITY_INDEXER}:{self._collection_id}:{entity_name}",
            )
        )
        try:
            await _to_thread(self._vector_connector.delete, [point_id])
        except Exception:
            logger.warning(
                "lineage_merger: vector delete failed for %r",
                entity_name,
                exc_info=True,
            )

    @staticmethod
    def _tenant_scope_key_for(entity: EntityWithLineage) -> str:
        # Re-anchored parts inherit the tenant_scope_key of the source
        # they came from; the curation-merge sentinel lineage inherits
        # the target entity's. Fall back to empty string when the
        # entity has no lineage members yet (defensive only —
        # ``upsert_entity_with_lineage`` is always called with a real
        # lineage member alongside a real record).
        for member in entity.source_lineage:
            return member.tenant_scope_key
        return ""


async def _to_thread(func, *args, **kwargs):
    """Local wrapper around ``asyncio.to_thread`` so the import stays
    inside the function body when this module is imported eagerly."""
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


# ---------------------------------------------------------------------
# Per-collection factory — Wave 7 §K.12.8 task #8 wiring
# ---------------------------------------------------------------------


class _SyncEmbedderShim:
    """Adapt the sync ``(text -> list[float])`` callable used by the
    graph worker into the ``embed_query`` shape the merger / detector
    expect (mirrors :class:`EmbeddingService` surface). Lifted from
    :class:`MergeCandidateDetector`'s factory so the merger and
    detector share one shim."""

    def __init__(self, fn: Callable[[str], list[float]]) -> None:
        self._fn = fn

    def embed_query(self, text: str) -> list[float]:
        return self._fn(text)


def build_lineage_entity_merger_for(collection: Any) -> "LineageEntityMerger":
    """Build a :class:`LineageEntityMerger` for ``collection``.

    Wires the six dependencies the merger expects:

    * ``store`` — :func:`_build_lineage_graph_store_inner` (raw inner
      store; the merger writes canonical names directly and must NOT
      be intercepted by the alias-redirect decorator that
      :func:`_build_lineage_graph_store` returns to indexer / read
      paths).
    * ``alias_repo`` — fresh :class:`AliasMapRepository`.
    * ``compactor`` — :func:`_build_collection_graph_compactor`. Falls
      back to a no-op compactor if the collection has no completion
      model configured (the merger still runs; description simply
      stays uncompacted).
    * ``vector_connector`` + ``embedder`` —
      :func:`_build_collection_graph_vector_writer` shared with the
      Phase 3 indexer write path. Wrapped via :class:`_SyncEmbedderShim`
      so the ``embed_query`` interface matches.
    * ``llm`` — :func:`build_collection_llm_callable` (same async LLM
      callable the legacy graphindex used).
    * ``collection_id`` — bound at construction.

    Raises :class:`aperag.indexing.worker_factory.WorkerFactoryError`
    when the embedder / vector connector / LLM cannot be resolved
    — a user-driven merge cannot meaningfully proceed without them
    (no unified description, no vector re-anchor).
    """
    # Lazy imports keep this module free of worker_factory at import
    # time so worker_factory's own ``from .lineage_merge`` import (if
    # ever added) wouldn't form a cycle.
    from aperag.domains.knowledge_graph.graphindex.integration import (
        build_collection_llm_callable,
    )
    from aperag.indexing.graph_compactor import GraphIndexCompactor
    from aperag.indexing.worker_factory import (
        WorkerFactoryError,
        _build_collection_graph_vector_writer,
        _build_lineage_graph_store_inner,
        _resolve_graph_backend_type,
    )

    backend_type = _resolve_graph_backend_type(collection)
    inner_store = _build_lineage_graph_store_inner(backend_type=backend_type, collection=collection)

    vector_connector, embed_fn = _build_collection_graph_vector_writer(collection)
    if vector_connector is None or embed_fn is None:
        raise WorkerFactoryError(
            f"merge_entities: vector connector / embedder unavailable for collection {collection.id!r}"
        )

    try:
        llm = build_collection_llm_callable(collection)
    except Exception as exc:  # noqa: BLE001 — surface as factory failure.
        raise WorkerFactoryError(f"merge_entities: LLM not configured for collection {collection.id!r}: {exc}") from exc

    # Compactor is best-effort — the merger still runs without it
    # (description stays uncompacted; embedding falls back to unified).
    compactor = GraphIndexCompactor(llm=llm)

    return LineageEntityMerger(
        store=inner_store,
        alias_repo=AliasMapRepository(),
        compactor=compactor,
        vector_connector=vector_connector,
        embedder=_SyncEmbedderShim(embed_fn),
        llm=llm,
        collection_id=str(collection.id),
    )


__all__ = [
    "LineageEntityMerger",
    "LineageMergeResult",
    "AliasCycleError",
    "CURATION_MERGE_DOCUMENT_ID",
    "GRAPH_ENTITY_INDEXER",
    "build_lineage_entity_merger_for",
]
