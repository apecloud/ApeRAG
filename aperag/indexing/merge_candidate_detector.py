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

"""Sync-driven merge candidate detection — Wave 7 §K.12.4 task #4.

The detector runs at the end of :meth:`GraphModalityWorker.sync` (task #3
will wire it in). For each entity touched in the sync run it embeds the
description, ANN-searches the graph-entity vector index for similar
neighbours, then scores ``(affected, neighbour)`` pairs with the existing
:func:`aperag.graph_curation.candidate_generation.build_candidate_pairs`
algorithm. Any pair scoring above the threshold is persisted as a
``PENDING`` :class:`GraphCurationSuggestion` for human review — never
auto-merged (D-3 invariant locked by earayu2).

Per the §K.12.4 amend2 ratify (msg=6ab89fbb) this writes into the existing
``GraphCurationSuggestion`` infrastructure rather than introducing a new
table; auto-detected runs are tagged via
``GraphCurationRun.config_json["source"] == "auto_detect"`` so admin UI
can split user-triggered vs sync-driven curation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Iterable, Sequence

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.repositories.base import AsyncBaseRepository
from aperag.domains.knowledge_graph.db.models import (
    GraphCurationRun,
    GraphCurationRunStatus,
    GraphCurationSuggestion,
    GraphCurationSuggestionStatus,
)
from aperag.graph_curation.candidate_generation import (
    CandidatePair,
    build_candidate_pairs,
)
from aperag.graph_curation.dto import CurationEntity as LegacyEntity
from aperag.indexing.graph import EntityWithLineage, LineageGraphStore
from aperag.utils.utils import utc_now
from aperag.vectorstore.base import VectorStoreConnector
from aperag.vectorstore.dto import QueryRequest
from aperag.vectorstore.filters import Eq

logger = logging.getLogger(__name__)


# ``config_json["source"]`` discriminator + ``GraphCurationSuggestion.reason``
# tag — both are how auto-detected rows are filtered out from
# user-triggered curation runs in queries / supersede / admin UI.
AUTO_DETECT_SOURCE: str = "auto_detect"
AUTO_DETECT_REASON: str = "auto_detected"


# Mirrors the user-triggered tunables in
# :mod:`aperag.graph_curation.service` so sync-driven and user-driven
# detection share the same scoring frontier — divergence here would let
# one path silently surface candidates the other rejects.
DEFAULT_VECTOR_TOP_K: int = 4
DEFAULT_VECTOR_SCORE_THRESHOLD: float = 0.72
DEFAULT_MAX_PAIRS_PER_ENTITY: int = 6
DEFAULT_MAX_TOTAL_PAIRS: int = 240
DEFAULT_MAX_SUGGESTIONS: int = 32


# Embedder API: callers pass any object with a sync ``embed_query``
# (matches :class:`EmbeddingService`); we offload the call to a worker
# thread so the event loop stays free.
EmbedQueryFn = Callable[[str], list[float]]


class MergeCandidateDetector(AsyncBaseRepository):
    """Detect and persist merge-candidate suggestions for one collection.

    Per-collection bound — caller wires the four dependencies (lineage
    store, vector connector, embedder, optional LLM) to the collection
    being synced before constructing the detector. The constructor does
    no I/O.

    The detector is *write-only* with respect to lineage data — it
    reads entities through ``store.get_entity`` but never writes lineage
    rows back, preserving §K.12 invariant #1 (L1 graph data not polluted
    by L2 derivatives).
    """

    def __init__(
        self,
        *,
        store: LineageGraphStore,
        vector_connector: VectorStoreConnector,
        embedder: Any,
        collection_id: str,
        user_id: str,
        # Optional LLM judge tier — reserved for follow-up, not invoked
        # by the v1 PR. When wired, accepts an ``LLMCall`` async callable
        # and runs the existing ``_adjudicate_pairs`` flow.
        llm: Callable[[str], Awaitable[str]] | None = None,
        vector_top_k: int = DEFAULT_VECTOR_TOP_K,
        vector_score_threshold: float = DEFAULT_VECTOR_SCORE_THRESHOLD,
        max_pairs_per_entity: int = DEFAULT_MAX_PAIRS_PER_ENTITY,
        max_total_pairs: int = DEFAULT_MAX_TOTAL_PAIRS,
        max_suggestions: int = DEFAULT_MAX_SUGGESTIONS,
        session: AsyncSession | None = None,
    ) -> None:
        super().__init__(session)
        self._store = store
        self._vector_connector = vector_connector
        self._embedder = embedder
        self._collection_id = collection_id
        self._user_id = user_id
        self._llm = llm
        self._vector_top_k = vector_top_k
        self._vector_score_threshold = vector_score_threshold
        self._max_pairs_per_entity = max_pairs_per_entity
        self._max_total_pairs = max_total_pairs
        self._max_suggestions = max_suggestions

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    async def detect_for_sync(
        self,
        *,
        sync_run_id: str,
        affected_entity_names: Sequence[str],
    ) -> int:
        """Run candidate detection over ``affected_entity_names`` and
        persist the result as ``PENDING`` ``GraphCurationSuggestion`` rows.

        Returns the number of suggestions written. Always supersedes
        prior auto-detected ``PENDING`` rows for this collection so
        reviewers never see stale duplicates from earlier syncs.

        Empty ``affected_entity_names`` short-circuits to ``0`` (no run
        row, no DB write) — there is nothing to detect.

        Cross-document full-collection sweep is *not* implemented here:
        the user-triggered ``GraphCurationService.start_run`` path
        owns full sweeps. A sync-driven detector that reaches every
        entity in the collection would need a ``list_entities`` Protocol
        method on :class:`LineageGraphStore`, which is intentionally
        deferred (no caller needs it today; adding it would broaden the
        Protocol surface for one consumer — earayu2 simple-stable
        directive #1, 不无限扩范围).
        """
        if not affected_entity_names:
            return 0

        suggestions = await self._detect_suggestion_payloads(affected_entity_names)
        capped = suggestions[: self._max_suggestions]

        await self._persist_run_and_suggestions(
            sync_run_id=sync_run_id,
            suggestions=capped,
            stats={
                "affected_entities": len(affected_entity_names),
                "raw_pair_suggestions": len(suggestions),
                "suggestions": len(capped),
            },
        )
        return len(capped)

    # ------------------------------------------------------------------
    # detection (pure compute except for store / vector / embedder I/O —
    # no DB writes, fully testable with mock dependencies)
    # ------------------------------------------------------------------

    async def _detect_suggestion_payloads(
        self,
        affected_entity_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        affected_entities = await self._fetch_entities(affected_entity_names)
        if not affected_entities:
            return []

        entities_by_name: dict[str, EntityWithLineage] = {e.name: e for e in affected_entities}
        vector_neighbors: dict[str, list[tuple[str, float]]] = {}

        for entity in affected_entities:
            neighbours = await self._vector_neighbours_for(entity)
            if not neighbours:
                continue
            vector_neighbors[entity.name] = neighbours
            extra_names = [name for name, _ in neighbours if name not in entities_by_name]
            for fetched in await self._fetch_entities(extra_names):
                entities_by_name.setdefault(fetched.name, fetched)

        legacy_entities = [self._to_legacy_entity(e) for e in entities_by_name.values()]
        pairs = build_candidate_pairs(
            entities=legacy_entities,
            vector_neighbors=vector_neighbors,
            max_pairs_per_entity=self._max_pairs_per_entity,
            max_total_pairs=self._max_total_pairs,
        )
        return self._pairs_to_suggestion_rows(pairs, entities_by_name)

    async def _fetch_entities(self, names: Iterable[str]) -> list[EntityWithLineage]:
        out: list[EntityWithLineage] = []
        for name in names:
            entity = await self._store.get_entity(name)
            if entity is not None:
                out.append(entity)
        return out

    async def _vector_neighbours_for(self, entity: EntityWithLineage) -> list[tuple[str, float]]:
        text = self._description_text_for_scoring(entity)
        if not text:
            return []
        try:
            embedding = await asyncio.to_thread(self._embedder.embed_query, text)
        except Exception:
            logger.warning("merge_candidate_detector: embed failed for %r", entity.name, exc_info=True)
            return []
        # +1 to reserve a slot for the entity itself, which the search
        # will return when its own vector is in the index.
        request = QueryRequest(
            embedding=embedding,
            top_k=self._vector_top_k + 1,
            flt=Eq("indexer", "graph_entity"),
            score_threshold=self._vector_score_threshold,
        )
        try:
            hits = await asyncio.to_thread(self._vector_connector.search, request)
        except Exception:
            logger.warning(
                "merge_candidate_detector: vector search failed for %r",
                entity.name,
                exc_info=True,
            )
            return []
        out: list[tuple[str, float]] = []
        for hit in hits:
            name = hit.payload.get("entity_name") or hit.payload.get("name")
            if not name or name == entity.name:
                continue
            out.append((str(name), float(hit.score)))
            if len(out) >= self._vector_top_k:
                break
        return out

    @staticmethod
    def _description_text_for_scoring(entity: EntityWithLineage) -> str:
        # Prefer ``compacted_description`` once task #1 lands; fall back
        # to the raw concatenated parts so scoring still works on
        # entities the compactor hasn't touched yet (or compactor was
        # below threshold).
        compacted = getattr(entity, "compacted_description", None)
        if compacted:
            return str(compacted)
        return "\n\n".join(p.text for p in entity.description_parts if p.text)

    def _to_legacy_entity(self, entity: EntityWithLineage) -> LegacyEntity:
        # ``build_candidate_pairs`` predates Wave 7 and consumes the
        # legacy ``Entity`` DTO; we adapt rather than fork the scoring
        # algorithm (simple-stable directive #3, 简单稳定优于复杂).
        # ``name`` doubles as ``entity_id`` because the lineage shape
        # uses name as the natural key (see
        # ``LineageGraphStore.get_entity(entity_name)``).
        chunk_ids: list[str] = []
        for member in entity.source_lineage:
            chunk_ids.extend(member.chunk_ids)
        return LegacyEntity(
            entity_id=entity.name,
            collection_id=self._collection_id,
            name=entity.name,
            type=entity.entity_type,
            description=self._description_text_for_scoring(entity),
            source_chunk_ids=tuple(chunk_ids),
        )

    def _pairs_to_suggestion_rows(
        self,
        pairs: list[CandidatePair],
        entities_by_name: dict[str, EntityWithLineage],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pair in pairs:
            left = entities_by_name.get(pair.left_id)
            right = entities_by_name.get(pair.right_id)
            if left is None or right is None:
                continue
            # Deterministic alphabetical-first canonical pick — same
            # ``(a, b)`` pair across syncs always points at the same
            # surviving entity (architect ratify #3, msg=6ab89fbb).
            ordered = sorted([pair.left_id, pair.right_id])
            target = ordered[0]
            rows.append(
                {
                    "entity_ids": ordered,
                    "entity_snapshots": [self._snapshot(entities_by_name[name]) for name in ordered],
                    "target_entity_id": target,
                    "confidence_score": float(pair.score),
                    "reason": AUTO_DETECT_REASON,
                    "evidence": dict(pair.signals),
                }
            )
        rows.sort(
            key=lambda row: (
                -float(row["confidence_score"]),
                row["target_entity_id"],
            )
        )
        return rows

    @staticmethod
    def _snapshot(entity: EntityWithLineage) -> dict[str, Any]:
        return {
            "entity_id": entity.name,
            "entity_name": entity.name,
            "entity_type": entity.entity_type,
            "description": MergeCandidateDetector._description_text_for_scoring(entity),
            "source_chunk_count": sum(len(member.chunk_ids) for member in entity.source_lineage),
        }

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    async def _persist_run_and_suggestions(
        self,
        *,
        sync_run_id: str,
        suggestions: list[dict[str, Any]],
        stats: dict[str, int],
    ) -> None:
        async def _op(session: AsyncSession) -> None:
            now = utc_now()
            # Always supersede prior auto-detected ``PENDING`` rows for
            # this collection — even when the new run yields zero
            # suggestions — so reviewers don't see stale duplicates from
            # an earlier sync that detected something this sync no
            # longer agrees with (huangheng CR plan, msg=ee1a2c78).
            await session.execute(
                update(GraphCurationSuggestion)
                .where(
                    GraphCurationSuggestion.collection_id == self._collection_id,
                    GraphCurationSuggestion.status == GraphCurationSuggestionStatus.PENDING,
                    GraphCurationSuggestion.reason == AUTO_DETECT_REASON,
                )
                .values(
                    status=GraphCurationSuggestionStatus.SUPERSEDED,
                    resolution_note=f"superseded_by_sync_run:{sync_run_id}",
                    gmt_updated=now,
                    gmt_operated=now,
                )
            )
            run = GraphCurationRun(
                user_id=self._user_id,
                collection_id=self._collection_id,
                status=GraphCurationRunStatus.COMPLETED,
                config_json={
                    "source": AUTO_DETECT_SOURCE,
                    "sync_run_id": sync_run_id,
                },
                stats=stats,
                gmt_started=now,
                gmt_finished=now,
            )
            session.add(run)
            await session.flush()
            for row in suggestions:
                session.add(
                    GraphCurationSuggestion(
                        run_id=run.id,
                        user_id=self._user_id,
                        collection_id=self._collection_id,
                        status=GraphCurationSuggestionStatus.PENDING,
                        entity_ids=row["entity_ids"],
                        entity_snapshots=row["entity_snapshots"],
                        target_entity_id=row["target_entity_id"],
                        confidence_score=row["confidence_score"],
                        reason=row["reason"],
                        evidence=row["evidence"],
                    )
                )

        await self.execute_with_transaction(_op)


__all__ = [
    "MergeCandidateDetector",
    "EmbedQueryFn",
    "AUTO_DETECT_SOURCE",
    "AUTO_DETECT_REASON",
    "DEFAULT_VECTOR_TOP_K",
    "DEFAULT_VECTOR_SCORE_THRESHOLD",
    "DEFAULT_MAX_PAIRS_PER_ENTITY",
    "DEFAULT_MAX_TOTAL_PAIRS",
    "DEFAULT_MAX_SUGGESTIONS",
]
