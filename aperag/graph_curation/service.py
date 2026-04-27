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

from aperag.db.models import (
    Collection,
    GraphCurationRun,
    GraphCurationRunStatus,
    GraphCurationSuggestion,
    GraphCurationSuggestionStatus,
)
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.db.repositories.base import AsyncBaseRepository
from aperag.domains.knowledge_graph.graphindex.dto import Entity
from aperag.domains.knowledge_graph.graphindex.service import GraphIndexService, LLMCall
from aperag.exceptions import CollectionNotFoundException
from aperag.graph_curation.candidate_generation import (
    CandidatePair,
    build_candidate_pairs,
    entity_snapshot,
)
from aperag.graph_curation.prompts import render_merge_adjudication_prompt
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTITIES = 400
DEFAULT_MAX_PAIRS_PER_ENTITY = 6
DEFAULT_MAX_CANDIDATE_PAIRS = 240
DEFAULT_MAX_SUGGESTIONS = 32
DEFAULT_VECTOR_TOP_K = 4
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
        from aperag.service.collection_service import collection_service

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
            from config.celery_tasks import generate_graph_curation_run_task

            try:
                generate_graph_curation_run_task.delay(run.id, collection_id)
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
        if action_normalized not in {"accept", "reject"}:
            raise ValueError("action must be one of: accept, reject")

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
                "suggestion_id": suggestion_id,
                "action": "reject",
                "suggestion_status": GraphCurationSuggestionStatus.REJECTED.value,
                "merge_result": None,
            }

        from aperag.domains.knowledge_graph.graphindex.integration import make_service_for_collection

        graph_service = make_service_for_collection(collection)
        entity_ids = list(suggestion.entity_ids or [])
        target_entity_id = suggestion.target_entity_id
        source_entity_ids = [entity_id for entity_id in entity_ids if entity_id != target_entity_id]
        merge_result = await graph_service.merge_entities(
            collection_id=collection_id,
            target_entity_id=target_entity_id,
            source_entity_ids=source_entity_ids,
        )

        await self._accept_and_supersede(
            collection_id=collection_id,
            suggestion_id=suggestion_id,
            entity_ids=entity_ids,
            operated_by=user_id,
        )

        return {
            "status": "success",
            "suggestion_id": suggestion_id,
            "action": "accept",
            "suggestion_status": GraphCurationSuggestionStatus.ACCEPTED.value,
            "merge_result": {
                "target_entity_id": merge_result.target_entity_id,
                "merged_source_ids": list(merge_result.merged_source_ids),
                "description": merge_result.description,
                "source_chunk_ids": list(merge_result.source_chunk_ids),
                "edges_redirected": merge_result.edges_redirected,
                "edges_collapsed": merge_result.edges_collapsed,
            },
        }

    async def generate_run(
        self,
        *,
        run_id: str,
        collection: Collection,
        graph_service: GraphIndexService,
        llm: LLMCall,
    ) -> None:
        collection_id = str(collection.id)
        await self._mark_run_running(run_id)

        try:
            entities = await graph_service.list_entities_for_curation(
                collection_id=collection_id,
                limit=DEFAULT_MAX_ENTITIES,
            )
            entities_by_id = {entity.entity_id: entity for entity in entities}
            vector_neighbors = await graph_service.find_entity_shadow_neighbors(
                collection_id=collection_id,
                entity_ids=list(entities_by_id.keys()),
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
        return {
            "id": suggestion.id,
            "run_id": suggestion.run_id,
            "collection_id": suggestion.collection_id,
            "status": suggestion.status,
            "entity_ids": list(suggestion.entity_ids or []),
            "entities": list(suggestion.entity_snapshots or []),
            "target_entity_id": suggestion.target_entity_id,
            "confidence_score": float(suggestion.confidence_score),
            "reason": suggestion.reason,
            "evidence": suggestion.evidence or {},
            "resolution_note": suggestion.resolution_note,
            "created": suggestion.gmt_created.isoformat() if suggestion.gmt_created else None,
            "updated": suggestion.gmt_updated.isoformat() if suggestion.gmt_updated else None,
            "operated_at": suggestion.gmt_operated.isoformat() if suggestion.gmt_operated else None,
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


graph_curation_service = GraphCurationService()

__all__ = ["GraphCurationService", "graph_curation_service"]
