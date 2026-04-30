# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

from aperag.domains.knowledge_graph.db.models import GraphCurationSuggestionStatus
from aperag.domains.knowledge_graph.schemas import SuggestionActionRequest
from aperag.graph_curation.candidate_generation import CandidatePair
from aperag.graph_curation.dto import CurationEntity as Entity
from aperag.graph_curation.service import GraphCurationService, MergeJudgement


def _entity(entity_id: str, name: str, *, chunk_ids: tuple[str, ...] = ()) -> Entity:
    return Entity(
        entity_id=entity_id,
        collection_id="c1",
        name=name,
        type="organization",
        description=f"description for {name}",
        source_chunk_ids=chunk_ids,
    )


def test_aggregate_positive_judgements_groups_pairs_and_picks_voted_target():
    service = GraphCurationService.__new__(GraphCurationService)
    entities_by_id = {
        "e1": _entity("e1", "墨香居", chunk_ids=("c1", "c2")),
        "e2": _entity("e2", "旧书店", chunk_ids=("c3",)),
        "e3": _entity("e3", "墨香旧书店", chunk_ids=("c4",)),
    }
    adjudications = [
        (
            CandidatePair(
                left_id="e1",
                right_id="e2",
                score=0.92,
                signals={"normalized_name_contains": True},
            ),
            MergeJudgement(
                same_entity=True,
                confidence=0.93,
                reason="名称和描述高度重合",
                recommended_target_entity_id="e1",
            ),
        ),
        (
            CandidatePair(
                left_id="e2",
                right_id="e3",
                score=0.87,
                signals={"vector_neighbor": True},
            ),
            MergeJudgement(
                same_entity=True,
                confidence=0.89,
                reason="上下文表明是同一家店",
                recommended_target_entity_id="e1",
            ),
        ),
    ]

    suggestions = service._aggregate_positive_judgements(
        entities_by_id=entities_by_id,
        adjudications=adjudications,
    )

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion["entity_ids"] == ["e1", "e2", "e3"]
    assert suggestion["target_entity_id"] == "e1"
    assert suggestion["confidence_score"] == 0.91
    assert suggestion["evidence"]["pair_count"] == 2
    assert {item["entity_id"] for item in suggestion["entity_snapshots"]} == {
        "e1",
        "e2",
        "e3",
    }


def test_extract_json_object_ignores_non_json_prefix_suffix():
    raw = 'analysis...\n```json\n{"same_entity": true, "confidence": 0.88, "reason": "match"}\n```\n'
    result = GraphCurationService._extract_json_object(raw)

    assert result == {
        "same_entity": True,
        "confidence": 0.88,
        "reason": "match",
    }


# Wave 3 T3.1 chunk 3 (per architect msg=3890c9d7 Item 4): the legacy
# ``test_start_run_marks_failed_when_enqueue_raises`` test was deleted
# alongside the Celery decorator on ``generate_graph_curation_run_task``.
# The new Pattern C dispatch wraps in
# ``asyncio.create_task(asyncio.to_thread(...))`` which never raises at
# schedule time, so the synchronous-failure assertion no longer maps to
# any reachable behaviour.


def test_suggestion_action_request_normalizes_case_insensitively():
    request = SuggestionActionRequest(action=" REJECT ")

    assert request.action == "reject"


def test_suggestion_to_dict_exposes_evidence_refs_and_new_status():
    created = datetime(2026, 4, 30, tzinfo=timezone.utc)
    suggestion = SimpleNamespace(
        id="gcs_1",
        run_id="gcr_1",
        collection_id="col1",
        status=GraphCurationSuggestionStatus.APPLY_PENDING,
        entity_ids=["e1", "e2"],
        entity_snapshots=[],
        target_entity_id="e1",
        confidence_score=Decimal("0.910"),
        reason="same entity",
        evidence={"pair_count": 1},
        evidence_refs=[
            {"document_id": "doc1", "chunk_id": "chunk1", "parse_version": "v1"},
        ],
        resolution_note=None,
        gmt_created=created,
        gmt_updated=created,
        gmt_operated=None,
    )

    out = GraphCurationService._suggestion_to_dict(suggestion)

    assert out["status"] == GraphCurationSuggestionStatus.APPLY_PENDING
    assert out["confidence_score"] == 0.91
    assert out["evidence_refs"] == [
        {"document_id": "doc1", "chunk_id": "chunk1", "parse_version": "v1"},
    ]
