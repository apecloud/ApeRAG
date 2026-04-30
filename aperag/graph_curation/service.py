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

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter, defaultdict
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.db.repositories.base import AsyncBaseRepository
from aperag.domains.knowledge_base.db.models import Collection
from aperag.domains.knowledge_graph.db.models import (
    GraphCurationRun,
    GraphCurationRunStatus,
    GraphCurationSuggestion,
    GraphCurationSuggestionStatus,
)
from aperag.exceptions import CollectionNotFoundException
from aperag.graph_curation.candidate_generation import (
    CandidatePair,
    build_candidate_pairs,
    entity_snapshot,
)
from aperag.graph_curation.dto import CurationEntity as Entity
from aperag.graph_curation.prompts import render_merge_adjudication_prompt
from aperag.indexing.graph import LineageGraphStore
from aperag.indexing.llm import LLMCall
from aperag.utils.utils import utc_now
from aperag.vectorstore.base import VectorStoreConnector
from aperag.vectorstore.dto import QueryRequest
from aperag.vectorstore.filters import Eq

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTITIES = 400
DEFAULT_MAX_PAIRS_PER_ENTITY = 6
DEFAULT_MAX_CANDIDATE_PAIRS = 240
DEFAULT_MAX_SUGGESTIONS = 32
DEFAULT_VECTOR_TOP_K = 4
# Threshold is on the normalized [0, 1] similarity scale per task #61
# P0-B (``aperag.vectorstore.base.normalize_score``). 0.72 was tuned on
# cosine-distance embeddings, where 0.72 ≈ "strongly similar". The
# normalize layer makes this number directly comparable across adapters
# but the *intent* is still cosine-grade strictness; collections that
# choose ``euclid`` or ``dot`` distance may want to override this default
# (see follow-up: metric-aware defaults, Lesson #12 v7.3).
DEFAULT_VECTOR_SCORE_THRESHOLD = 0.72
DEFAULT_LLM_CONCURRENCY = 4


class MergeJudgement(BaseModel):
    same_entity: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    recommended_target_entity_id: Optional[str] = None


class GraphCurationService(AsyncBaseRepository):
    def __init__(self, session: AsyncSession = None):
        super().__init__(session)
        from aperag.domains.knowledge_base.service.collection_service import collection_service

        self.collection_service = collection_service
        self.db_ops = async_db_ops if session is None else AsyncDatabaseOps(session)

    async def start_run(self, user_id: str, collection_id: str) -> dict[str, Any]:
        await self._get_and_validate_collection(user_id, collection_id)

        async def _op(session: AsyncSession):
            active_stmt = (
                select(GraphCurationRun)
                .where(
                    GraphCurationRun.collection_id == collection_id,
                    GraphCurationRun.status.in_([GraphCurationRunStatus.PENDING, GraphCurationRunStatus.RUNNING]),
                )
                .order_by(GraphCurationRun.gmt_created.desc())
                .limit(1)
            )
            active = (await session.execute(active_stmt)).scalars().first()
            if active is not None:
                return active, False

            run = GraphCurationRun(
                user_id=user_id,
                collection_id=collection_id,
                status=GraphCurationRunStatus.PENDING,
                config_json=self._default_run_config(),
            )
            session.add(run)
            await session.flush()
            await session.refresh(run)
            return run, True

        run, created = await self.execute_with_transaction(_op)

        if created:
            # task #31 Phase A1 (per spec § 3.1.1 + ziang msg=92321bcc +
            # Bryce msg=4c23f87e BLOCKER 1): API process MUST NOT
            # execute the merge sweep itself — this used to be a
            # ``asyncio.create_task(asyncio.to_thread(generate_graph_curation_run_task, ...))``
            # fire-and-forget which violated task #17 hard-cut +
            # task-system-invariants § 2.3 ("API doesn't own heavy
            # execution"). Post-A1 it's a thin enqueue onto the
            # independent ``q:graph_curation_run`` Redis queue; the
            # actual sweep runs in the indexing-worker process via
            # ``run_graph_curation_run_worker``.
            from aperag.indexing.runtime import get_runtime

            runtime = get_runtime()
            if runtime is None or runtime.queue is None:
                # No runtime / queue installed (test environment, sync-
                # only mode, or pre-startup boot sequence). Mark the
                # run failed so the caller doesn't see a "started"
                # response that never executes — better fail-loud than
                # silent "PENDING forever" (per service.py existing
                # ``_mark_run_failed`` discipline).
                logger.error(
                    "graph curation: indexing runtime / queue not installed; cannot enqueue run %s for collection %s",
                    run.id,
                    collection_id,
                )
                await self._mark_run_failed(run.id, "enqueue_failed: runtime not installed")
                raise RuntimeError("Failed to schedule graph curation run: runtime not installed")

            try:
                await runtime.queue.push_graph_curation_run(
                    payload={
                        "run_id": str(run.id),
                        "collection_id": str(collection_id),
                    }
                )
            except Exception as exc:
                logger.exception(
                    "graph curation: failed to enqueue run %s for collection %s",
                    run.id,
                    collection_id,
                )
                await self._mark_run_failed(run.id, f"enqueue_failed: {exc}")
                raise RuntimeError("Failed to schedule graph curation run") from exc

        return {
            "run": self._run_to_dict(run),
            "started": created,
            "message": "Graph curation run started" if created else "Graph curation run already in progress",
        }

    async def get_latest(self, user_id: str, collection_id: str) -> dict[str, Any]:
        await self._get_and_validate_collection(user_id, collection_id)

        async def _query(session: AsyncSession):
            run_stmt = (
                select(GraphCurationRun)
                .where(GraphCurationRun.collection_id == collection_id)
                .order_by(GraphCurationRun.gmt_created.desc())
                .limit(1)
            )
            run = (await session.execute(run_stmt)).scalars().first()
            if run is None:
                return None, []

            suggestion_stmt = (
                select(GraphCurationSuggestion)
                .where(GraphCurationSuggestion.run_id == run.id)
                .order_by(
                    GraphCurationSuggestion.confidence_score.desc(),
                    GraphCurationSuggestion.gmt_created.asc(),
                )
            )
            suggestions = (await session.execute(suggestion_stmt)).scalars().all()
            return run, suggestions

        run, suggestions = await self._execute_query(_query)
        return {
            "run": self._run_to_dict(run) if run is not None else None,
            "suggestions": [self._suggestion_to_dict(item) for item in suggestions],
        }

    async def handle_action(
        self,
        user_id: str,
        collection_id: str,
        suggestion_id: str,
        *,
        action: str,
    ) -> dict[str, Any]:
        collection = await self._get_and_validate_collection(user_id, collection_id)
        action_normalized = (action or "").strip().lower()
        if action_normalized not in {"accept", "reject", "dismiss"}:
            raise ValueError("action must be one of: accept, reject, dismiss")

        async def _load(session: AsyncSession):
            stmt = select(GraphCurationSuggestion).where(
                GraphCurationSuggestion.id == suggestion_id,
                GraphCurationSuggestion.collection_id == collection_id,
            )
            return (await session.execute(stmt)).scalars().first()

        suggestion = await self._execute_query(_load)
        if suggestion is None:
            raise KeyError(f"Suggestion {suggestion_id!r} not found")
        if suggestion.status != GraphCurationSuggestionStatus.PENDING:
            raise ValueError(f"Suggestion {suggestion_id!r} is already {suggestion.status}")

        if action_normalized == "reject":
            await self._mark_suggestion_status(
                suggestion_id=suggestion_id,
                status=GraphCurationSuggestionStatus.REJECTED,
                operated_by=user_id,
                resolution_note="rejected_by_user",
            )
            return {
                "status": "success",
                "message": f"Suggestion {suggestion_id} has been rejected",
                "suggestion_id": suggestion_id,
                "action": "reject",
                "suggestion_status": GraphCurationSuggestionStatus.REJECTED.value,
                "merge_result": None,
            }

        if action_normalized == "dismiss":
            await self._mark_suggestion_status(
                suggestion_id=suggestion_id,
                status=GraphCurationSuggestionStatus.DISMISSED,
                operated_by=user_id,
                resolution_note="dismissed_by_user",
            )
            return {
                "status": "success",
                "message": f"Suggestion {suggestion_id} has been dismissed",
                "suggestion_id": suggestion_id,
                "action": "dismiss",
                "suggestion_status": GraphCurationSuggestionStatus.DISMISSED.value,
                "merge_result": None,
            }

        # Wave 7 W7-10 cutover: route the curation accept-merge through
        # the new ``LineageEntityMerger`` (W7-6, PR #1758). Same merger
        # the W7-8 cutover wired into ``GraphService.merge_entities``
        # for ``POST /graphs/nodes/merge`` (PR #1762) — both surfaces
        # converge on a single merge path so user-merge-from-curation
        # vs user-merge-from-graph-view never diverge.
        from aperag.graph_curation.alias_map import AliasCycleError
        from aperag.graph_curation.lineage_merge import build_lineage_entity_merger_for

        merger = build_lineage_entity_merger_for(collection)
        entity_ids = list(suggestion.entity_ids or [])
        target_entity_id = suggestion.target_entity_id
        source_entity_ids = [entity_id for entity_id in entity_ids if entity_id != target_entity_id]
        try:
            merge_result = await merger.merge_entities(
                target_name=target_entity_id,
                source_names=source_entity_ids,
                merged_by=user_id,
            )
        except AliasCycleError as exc:
            raise ValueError(str(exc)) from exc

        await self._accept_and_supersede(
            collection_id=collection_id,
            suggestion_id=suggestion_id,
            entity_ids=entity_ids,
            operated_by=user_id,
        )

        # Backward-compat response shape (mirrors
        # ``domains/knowledge_graph/service.py:merge_entities`` cutover
        # in W7-8). ``edges_redirected`` / ``edges_collapsed`` are 0
        # by design — alias redirect happens at indexer write-time via
        # the decorator, not as a per-merge count we can surface.
        #
        # ⚠️ DEPRECATED (task #31 A3, spec § 3.1.5 — Lesson #14 multi-
        # iteration cleanup family): the ``merge_result`` carries
        # ``compacted_description`` / ``unified_description`` only on
        # the legacy ``LineageEntityMerger.merge_entities`` path —
        # post-Wave-5 the description-free apply variant
        # (``merge_entities_apply_description_free``) returns ``None``
        # / ``""`` for both fields. This sync ``handle_action()`` API
        # is preserved for back-compat with existing callers that
        # consume ``merge_result.description`` in the response;
        # follow-up cleanup once all consumers migrate to the async
        # accept-apply path will drop this field from the response
        # shape entirely. The boundary test
        # ``tests/boundaries/test_graph_curation_description_free.py``
        # explicitly allowlists ``merge_result.compacted_description``
        # via the ``NON_ENTITY_BASE_NAMES`` mechanism (it is a
        # ``LineageMergeResult`` field, not an ``entity`` description
        # read).
        merge_description = merge_result.compacted_description or merge_result.unified_description
        chunk_ids: set[str] = set()
        target_after = await merger._store.get_entity(merge_result.final_target)  # noqa: SLF001
        if target_after is not None:
            for member in target_after.source_lineage or ():
                for cid in getattr(member, "chunk_ids", ()) or ():
                    if cid:
                        chunk_ids.add(str(cid))

        return {
            "status": "success",
            "message": f"Suggestion {suggestion_id} has been accepted and merge completed",
            "suggestion_id": suggestion_id,
            "action": "accept",
            "suggestion_status": GraphCurationSuggestionStatus.ACCEPTED.value,
            "merge_result": {
                "target_entity_id": merge_result.final_target,
                "merged_source_ids": list(merge_result.merged_source_ids),
                "description": merge_description,
                "source_chunk_ids": sorted(chunk_ids),
                "edges_redirected": 0,
                "edges_collapsed": 0,
            },
        }

    async def generate_run(
        self,
        *,
        run_id: str,
        collection: Collection,
        store: LineageGraphStore,
        vector_connector: VectorStoreConnector,
        embedder: Any,
        llm: LLMCall,
    ) -> None:
        """Wave 7 W7-10 cutover: drive the user-triggered candidate
        sweep over the four new injected dependencies (per architect
        Q3 ratify msg=838d57c3) instead of the legacy
        ``GraphIndexService`` bundle.

        ``store`` — :class:`LineageGraphStore` per-collection-bound
        (Wave 4 lineage table backing).
        ``vector_connector`` — shared :class:`VectorStoreConnector`
        (Qdrant / pgvector) carrying the Wave 7 W7-3
        ``indexer="graph_entity"`` payload.
        ``embedder`` — any object exposing ``embed_query(text) ->
        list[float]`` (matches :class:`EmbeddingService`).
        ``llm`` — async ``(prompt) -> str`` adjudication callable.
        """
        collection_id = str(collection.id)
        await self._mark_run_running(run_id)

        try:
            entities = await self._enumerate_curation_entities(
                store=store,
                collection_id=collection_id,
                limit=DEFAULT_MAX_ENTITIES,
            )
            entities_by_id = {entity.entity_id: entity for entity in entities}
            vector_neighbors = await self._fetch_shadow_neighbors(
                vector_connector=vector_connector,
                embedder=embedder,
                entities=entities,
                top_k_per_entity=DEFAULT_VECTOR_TOP_K,
                score_threshold=DEFAULT_VECTOR_SCORE_THRESHOLD,
            )

            candidate_pairs = build_candidate_pairs(
                entities=entities,
                vector_neighbors=vector_neighbors,
                max_pairs_per_entity=DEFAULT_MAX_PAIRS_PER_ENTITY,
                max_total_pairs=DEFAULT_MAX_CANDIDATE_PAIRS,
            )
            judgements = await self._adjudicate_pairs(candidate_pairs, entities_by_id, llm)
            suggestions = self._aggregate_positive_judgements(
                entities_by_id=entities_by_id,
                adjudications=judgements,
            )[:DEFAULT_MAX_SUGGESTIONS]

            stats = {
                "analyzed_entities": len(entities),
                "candidate_pairs": len(candidate_pairs),
                "positive_pairs": len(judgements),
                "suggestions": len(suggestions),
            }
            await self._finish_run(
                run_id=run_id,
                user_id=str(collection.user),
                collection_id=collection_id,
                suggestions=suggestions,
                stats=stats,
            )
        except Exception as exc:
            logger.exception("graph curation run %s failed", run_id)
            await self._mark_run_failed(run_id, str(exc))
            raise

    async def expire_pending_for_collection(self, collection_id: str, *, reason: str) -> None:
        now = utc_now()

        async def _op(session: AsyncSession):
            await session.execute(
                update(GraphCurationSuggestion)
                .where(
                    GraphCurationSuggestion.collection_id == collection_id,
                    GraphCurationSuggestion.status == GraphCurationSuggestionStatus.PENDING,
                )
                .values(
                    status=GraphCurationSuggestionStatus.EXPIRED,
                    resolution_note=reason,
                    gmt_updated=now,
                    gmt_operated=now,
                )
            )

        await self.execute_with_transaction(_op)

    async def expire_pending_for_entities(
        self,
        collection_id: str,
        entity_ids: Sequence[str],
        *,
        reason: str,
    ) -> None:
        if not entity_ids:
            return
        entity_id_set = set(entity_ids)
        now = utc_now()

        async def _op(session: AsyncSession):
            stmt = select(GraphCurationSuggestion).where(
                GraphCurationSuggestion.collection_id == collection_id,
                GraphCurationSuggestion.status == GraphCurationSuggestionStatus.PENDING,
            )
            suggestions = (await session.execute(stmt)).scalars().all()
            for suggestion in suggestions:
                if entity_id_set & set(suggestion.entity_ids or []):
                    suggestion.status = GraphCurationSuggestionStatus.EXPIRED
                    suggestion.resolution_note = reason
                    suggestion.gmt_updated = now
                    suggestion.gmt_operated = now

        await self.execute_with_transaction(_op)

    async def purge_collection(self, collection_id: str) -> None:
        async def _op(session: AsyncSession):
            await session.execute(
                delete(GraphCurationSuggestion).where(GraphCurationSuggestion.collection_id == collection_id)
            )
            await session.execute(delete(GraphCurationRun).where(GraphCurationRun.collection_id == collection_id))

        await self.execute_with_transaction(_op)

    async def _get_and_validate_collection(self, user_id: str, collection_id: str) -> Collection:
        try:
            view_collection = await self.collection_service.get_collection(user_id, collection_id)
        except Exception:
            raise CollectionNotFoundException(collection_id)

        if not view_collection.config or not view_collection.config.enable_knowledge_graph:
            raise ValueError(f"Knowledge graph is not enabled for collection {collection_id}")

        db_collection = await self.collection_service.db_ops.query_collection(user_id, collection_id)
        if not db_collection:
            raise CollectionNotFoundException(collection_id)

        return db_collection

    @staticmethod
    def _default_run_config() -> dict[str, Any]:
        return {
            "max_entities": DEFAULT_MAX_ENTITIES,
            "max_pairs_per_entity": DEFAULT_MAX_PAIRS_PER_ENTITY,
            "max_candidate_pairs": DEFAULT_MAX_CANDIDATE_PAIRS,
            "max_suggestions": DEFAULT_MAX_SUGGESTIONS,
            "vector_top_k": DEFAULT_VECTOR_TOP_K,
            "vector_score_threshold": DEFAULT_VECTOR_SCORE_THRESHOLD,
            "llm_concurrency": DEFAULT_LLM_CONCURRENCY,
        }

    @staticmethod
    def _run_to_dict(run: GraphCurationRun) -> dict[str, Any]:
        return {
            "id": run.id,
            "collection_id": run.collection_id,
            "status": run.status,
            "stats": run.stats or {},
            "error_message": run.error_message,
            "created": run.gmt_created.isoformat() if run.gmt_created else None,
            "updated": run.gmt_updated.isoformat() if run.gmt_updated else None,
            "started": run.gmt_started.isoformat() if run.gmt_started else None,
            "finished": run.gmt_finished.isoformat() if run.gmt_finished else None,
        }

    @staticmethod
    def _suggestion_to_dict(suggestion: GraphCurationSuggestion) -> dict[str, Any]:
        entities = list(suggestion.entity_snapshots or [])
        evidence_refs = list(suggestion.evidence_refs or [])
        target_entity = GraphCurationService._target_entity_projection(
            entities=entities,
            target_entity_id=str(suggestion.target_entity_id),
        )
        return {
            "id": suggestion.id,
            "run_id": suggestion.run_id,
            "suggestion_batch_id": suggestion.run_id,
            "collection_id": suggestion.collection_id,
            "status": suggestion.status,
            "entity_ids": list(suggestion.entity_ids or []),
            "entities": entities,
            "target_entity_id": suggestion.target_entity_id,
            "suggested_target_entity": target_entity,
            "confidence_score": float(suggestion.confidence_score),
            "reason": suggestion.reason,
            "merge_reason": suggestion.reason,
            "evidence": suggestion.evidence or {},
            "evidence_refs": evidence_refs,
            "resolution_note": suggestion.resolution_note,
            "created": suggestion.gmt_created.isoformat() if suggestion.gmt_created else None,
            "updated": suggestion.gmt_updated.isoformat() if suggestion.gmt_updated else None,
            "operated_at": suggestion.gmt_operated.isoformat() if suggestion.gmt_operated else None,
        }

    @staticmethod
    def _target_entity_projection(
        *,
        entities: Sequence[dict[str, Any]],
        target_entity_id: str,
    ) -> dict[str, str]:
        for entity in entities:
            entity_id = str(entity.get("entity_id") or "")
            entity_name = str(entity.get("entity_name") or entity_id)
            if target_entity_id in {entity_id, entity_name}:
                return {
                    "entity_name": entity_name or target_entity_id,
                    "entity_type": str(entity.get("entity_type") or ""),
                }
        return {
            "entity_name": target_entity_id,
            "entity_type": "",
        }

    async def _mark_run_running(self, run_id: str) -> None:
        now = utc_now()

        async def _op(session: AsyncSession):
            await session.execute(
                update(GraphCurationRun)
                .where(GraphCurationRun.id == run_id)
                .values(
                    status=GraphCurationRunStatus.RUNNING,
                    gmt_updated=now,
                    gmt_started=now,
                    error_message=None,
                )
            )

        await self.execute_with_transaction(_op)

    async def _mark_run_failed(self, run_id: str, error_message: str) -> None:
        now = utc_now()

        async def _op(session: AsyncSession):
            await session.execute(
                update(GraphCurationRun)
                .where(GraphCurationRun.id == run_id)
                .values(
                    status=GraphCurationRunStatus.FAILED,
                    error_message=error_message[:2000],
                    gmt_updated=now,
                    gmt_finished=now,
                )
            )

        await self.execute_with_transaction(_op)

    async def _mark_suggestion_status(
        self,
        *,
        suggestion_id: str,
        status: GraphCurationSuggestionStatus,
        operated_by: str,
        resolution_note: str,
    ) -> None:
        now = utc_now()

        async def _op(session: AsyncSession):
            await session.execute(
                update(GraphCurationSuggestion)
                .where(GraphCurationSuggestion.id == suggestion_id)
                .values(
                    status=status,
                    operated_by=operated_by,
                    resolution_note=resolution_note,
                    gmt_updated=now,
                    gmt_operated=now,
                )
            )

        await self.execute_with_transaction(_op)

    async def _accept_and_supersede(
        self,
        *,
        collection_id: str,
        suggestion_id: str,
        entity_ids: Sequence[str],
        operated_by: str,
    ) -> None:
        now = utc_now()
        entity_id_set = set(entity_ids)

        async def _op(session: AsyncSession):
            stmt = select(GraphCurationSuggestion).where(
                GraphCurationSuggestion.collection_id == collection_id,
                GraphCurationSuggestion.status == GraphCurationSuggestionStatus.PENDING,
            )
            pending = (await session.execute(stmt)).scalars().all()
            for suggestion in pending:
                if suggestion.id == suggestion_id:
                    suggestion.status = GraphCurationSuggestionStatus.ACCEPTED
                    suggestion.operated_by = operated_by
                    suggestion.resolution_note = "accepted_by_user"
                    suggestion.gmt_updated = now
                    suggestion.gmt_operated = now
                    continue
                if entity_id_set & set(suggestion.entity_ids or []):
                    suggestion.status = GraphCurationSuggestionStatus.SUPERSEDED
                    suggestion.operated_by = operated_by
                    suggestion.resolution_note = f"superseded_by:{suggestion_id}"
                    suggestion.gmt_updated = now
                    suggestion.gmt_operated = now

        await self.execute_with_transaction(_op)

    async def _finish_run(
        self,
        *,
        run_id: str,
        user_id: str,
        collection_id: str,
        suggestions: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> None:
        now = utc_now()

        async def _op(session: AsyncSession):
            await session.execute(
                update(GraphCurationSuggestion)
                .where(
                    GraphCurationSuggestion.collection_id == collection_id,
                    GraphCurationSuggestion.status == GraphCurationSuggestionStatus.PENDING,
                    GraphCurationSuggestion.run_id != run_id,
                )
                .values(
                    status=GraphCurationSuggestionStatus.SUPERSEDED,
                    resolution_note=f"superseded_by_run:{run_id}",
                    gmt_updated=now,
                    gmt_operated=now,
                )
            )
            await session.execute(delete(GraphCurationSuggestion).where(GraphCurationSuggestion.run_id == run_id))
            for suggestion in suggestions:
                session.add(
                    GraphCurationSuggestion(
                        run_id=run_id,
                        user_id=user_id,
                        collection_id=collection_id,
                        status=GraphCurationSuggestionStatus.PENDING,
                        entity_ids=suggestion["entity_ids"],
                        entity_snapshots=suggestion["entity_snapshots"],
                        target_entity_id=suggestion["target_entity_id"],
                        confidence_score=suggestion["confidence_score"],
                        reason=suggestion["reason"],
                        evidence=suggestion["evidence"],
                        evidence_refs=suggestion.get("evidence_refs"),
                    )
                )
            await session.execute(
                update(GraphCurationRun)
                .where(GraphCurationRun.id == run_id)
                .values(
                    status=GraphCurationRunStatus.COMPLETED,
                    stats=stats,
                    gmt_updated=now,
                    gmt_finished=now,
                    error_message=None,
                )
            )

        await self.execute_with_transaction(_op)

    async def _adjudicate_pairs(
        self,
        pairs: Sequence[CandidatePair],
        entities_by_id: dict[str, Entity],
        llm: LLMCall,
    ) -> list[tuple[CandidatePair, MergeJudgement]]:
        if not pairs:
            return []

        semaphore = asyncio.Semaphore(DEFAULT_LLM_CONCURRENCY)

        async def _one(pair: CandidatePair):
            left = entities_by_id[pair.left_id]
            right = entities_by_id[pair.right_id]
            prompt = render_merge_adjudication_prompt(
                left=entity_snapshot(left),
                right=entity_snapshot(right),
                signals=pair.signals,
            )
            try:
                async with semaphore:
                    raw = await llm(prompt)
                judgement = MergeJudgement.model_validate(self._extract_json_object(raw))
                if judgement.same_entity and judgement.recommended_target_entity_id not in {
                    pair.left_id,
                    pair.right_id,
                }:
                    judgement.recommended_target_entity_id = None
                return pair, judgement
            except Exception:
                logger.exception("graph curation: pair adjudication failed for %s/%s", pair.left_id, pair.right_id)
                return None

        results = await asyncio.gather(*(_one(pair) for pair in pairs))
        return [result for result in results if result and result[1].same_entity]

    def _aggregate_positive_judgements(
        self,
        *,
        entities_by_id: dict[str, Entity],
        adjudications: Sequence[tuple[CandidatePair, MergeJudgement]],
    ) -> list[dict[str, Any]]:
        if not adjudications:
            return []

        parent = {entity_id: entity_id for entity_id in entities_by_id}

        def _find(entity_id: str) -> str:
            while parent[entity_id] != entity_id:
                parent[entity_id] = parent[parent[entity_id]]
                entity_id = parent[entity_id]
            return entity_id

        def _union(left_id: str, right_id: str) -> None:
            root_left = _find(left_id)
            root_right = _find(right_id)
            if root_left != root_right:
                parent[root_right] = root_left

        for pair, _judgement in adjudications:
            _union(pair.left_id, pair.right_id)

        components: dict[str, set[str]] = defaultdict(set)
        for entity_id in entities_by_id:
            components[_find(entity_id)].add(entity_id)

        by_component: dict[frozenset[str], list[tuple[CandidatePair, MergeJudgement]]] = defaultdict(list)
        for pair, judgement in adjudications:
            component_key = frozenset(components[_find(pair.left_id)])
            if len(component_key) >= 2:
                by_component[component_key].append((pair, judgement))

        suggestions: list[dict[str, Any]] = []
        for component_ids, component_pairs in by_component.items():
            ordered_ids = sorted(component_ids)
            target_entity_id = self._choose_target_entity(ordered_ids, component_pairs, entities_by_id)
            snapshots = [entity_snapshot(entities_by_id[entity_id]) for entity_id in ordered_ids]
            confidence = round(
                sum(judgement.confidence for _pair, judgement in component_pairs) / len(component_pairs),
                3,
            )
            reasons = []
            for _pair, judgement in component_pairs:
                reason = (judgement.reason or "").strip()
                if reason and reason not in reasons:
                    reasons.append(reason)
            suggestions.append(
                {
                    "entity_ids": ordered_ids,
                    "entity_snapshots": snapshots,
                    "target_entity_id": target_entity_id,
                    "confidence_score": confidence,
                    "reason": " ; ".join(reasons[:3]) or "LLM judged these entities to be the same entity.",
                    "evidence": {
                        "pair_count": len(component_pairs),
                        "pairs": [
                            {
                                "left_id": pair.left_id,
                                "right_id": pair.right_id,
                                "pair_score": pair.score,
                                "signals": pair.signals,
                                "confidence": judgement.confidence,
                                "recommended_target_entity_id": judgement.recommended_target_entity_id,
                                "reason": judgement.reason,
                            }
                            for pair, judgement in component_pairs
                        ],
                    },
                }
            )

        suggestions.sort(
            key=lambda item: (
                -float(item["confidence_score"]),
                len(item["entity_ids"]),
                item["target_entity_id"],
            )
        )
        return suggestions

    @staticmethod
    def _choose_target_entity(
        entity_ids: Sequence[str],
        component_pairs: Sequence[tuple[CandidatePair, MergeJudgement]],
        entities_by_id: dict[str, Entity],
    ) -> str:
        vote_counter: Counter[str] = Counter(
            judgement.recommended_target_entity_id
            for _pair, judgement in component_pairs
            if judgement.recommended_target_entity_id in set(entity_ids)
        )
        ranked = sorted(
            entity_ids,
            key=lambda entity_id: (
                -vote_counter.get(entity_id, 0),
                -len(entities_by_id[entity_id].source_chunk_ids or ()),
                entity_id,
            ),
        )
        return ranked[0]

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, Any]:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM did not return a JSON object")
        return json.loads(raw_text[start : end + 1])

    # ------------------------------------------------------------------
    # Wave 7 W7-10: helper methods replacing the legacy
    # ``GraphIndexService.list_entities_for_curation`` /
    # ``GraphIndexService.find_entity_shadow_neighbors`` calls.
    # ------------------------------------------------------------------

    @staticmethod
    async def _enumerate_curation_entities(
        *,
        store: LineageGraphStore,
        collection_id: str,
        limit: int,
    ) -> list[Entity]:
        """Page through ``store.list_entities`` until the requested
        ``limit`` is hit (or the collection is exhausted), and adapt to
        :class:`CurationEntity`.

        Replaces legacy ``list_entities_for_curation`` (per architect
        Q1 ratify msg=838d57c3 — ``list_entities`` is the new primary
        full-collection enumeration entry point).
        """
        if limit <= 0:
            return []
        page_size = min(1000, limit)
        out: list[Entity] = []
        offset = 0
        while len(out) < limit:
            remaining = limit - len(out)
            batch_size = min(page_size, remaining)
            batch = await store.list_entities(limit=batch_size, offset=offset)
            if not batch:
                break
            for lineage_entity in batch:
                out.append(Entity.from_lineage(lineage_entity, collection_id=collection_id))
            if len(batch) < batch_size:
                break
            offset += len(batch)
        return out

    @staticmethod
    async def _fetch_shadow_neighbors(
        *,
        vector_connector: VectorStoreConnector,
        embedder: Any,
        entities: Sequence[Entity],
        top_k_per_entity: int,
        score_threshold: float,
    ) -> dict[str, list[tuple[str, float]]]:
        """Vector-recall nearest neighbours by entity description, scoped
        to ``indexer="graph_entity"`` Wave 7 W7-3 vector points.

        Replaces legacy ``find_entity_shadow_neighbors`` (which filtered
        by legacy ``entity_id`` payload field; Wave 7 vector points
        carry the 3-field payload ``{indexer, entity_name, entity_type}``
        per spec §K.12.5 lock).
        """
        if not entities or top_k_per_entity <= 0:
            return {}

        flt = Eq("indexer", "graph_entity")
        out: dict[str, list[tuple[str, float]]] = {}
        for entity in entities:
            # Wave 5 description-NULL invariant (task #31 A3, spec § 3.1.5):
            # ``CurationEntity.description`` is always ``""`` post Wave 5
            # (see ``CurationEntity.from_lineage``); the legacy
            # ``description or name`` fallback short-circuits to ``name``
            # uniformly, so read the name directly. Mirrors the
            # ``MergeCandidateDetector._embedding_query_text`` shape but
            # this consumer pre-dates the helper and only needs a stable
            # text input — name is sufficient.
            text = entity.name
            if not text:
                continue
            try:
                embedding = await asyncio.to_thread(embedder.embed_query, text)
            except Exception:  # noqa: BLE001 — embedder flake non-fatal
                logger.warning(
                    "graph_curation: embed failed for entity=%r (skipping shadow neighbours)",
                    entity.entity_id,
                    exc_info=True,
                )
                continue
            request = QueryRequest(
                embedding=embedding,
                top_k=top_k_per_entity + 1,
                flt=flt,
                score_threshold=score_threshold,
            )
            try:
                hits = await asyncio.to_thread(vector_connector.search, request)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "graph_curation: vector search failed for entity=%r (skipping shadow neighbours)",
                    entity.entity_id,
                    exc_info=True,
                )
                continue
            neighbours: list[tuple[str, float]] = []
            for hit in hits:
                neighbour_name = (hit.payload or {}).get("entity_name")
                if not neighbour_name or neighbour_name == entity.entity_id:
                    continue
                neighbours.append((str(neighbour_name), float(hit.score)))
                if len(neighbours) >= top_k_per_entity:
                    break
            if neighbours:
                out[entity.entity_id] = neighbours
        return out


graph_curation_service = GraphCurationService()

__all__ = ["GraphCurationService", "graph_curation_service"]
