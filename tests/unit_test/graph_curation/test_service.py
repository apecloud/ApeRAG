# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aperag.domains.knowledge_graph.graphindex.dto import Entity
from aperag.domains.knowledge_graph.schemas import SuggestionActionRequest
from aperag.graph_curation.candidate_generation import CandidatePair
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


@pytest.mark.asyncio
async def test_start_run_marks_failed_when_enqueue_raises(monkeypatch):
    service = GraphCurationService.__new__(GraphCurationService)
    service._get_and_validate_collection = AsyncMock(return_value=object())

    run = SimpleNamespace(
        id="gcr_run1",
        collection_id="col1",
        status="PENDING",
        stats={},
        error_message=None,
        gmt_created=None,
        gmt_updated=None,
        gmt_started=None,
        gmt_finished=None,
    )

    async def fake_execute_with_transaction(_operation):
        return run, True

    service.execute_with_transaction = fake_execute_with_transaction
    service._mark_run_failed = AsyncMock()

    class _FakeTask:
        @staticmethod
        def delay(_run_id, _collection_id):
            raise RuntimeError("broker unavailable")

    monkeypatch.setattr("config.celery_tasks.generate_graph_curation_run_task", _FakeTask)

    with pytest.raises(RuntimeError, match="Failed to schedule graph curation run"):
        await service.start_run("user1", "col1")

    service._mark_run_failed.assert_awaited_once()
    assert service._mark_run_failed.await_args.args[0] == "gcr_run1"
    assert "enqueue_failed:" in service._mark_run_failed.await_args.args[1]


def test_suggestion_action_request_normalizes_case_insensitively():
    request = SuggestionActionRequest(action=" REJECT ")

    assert request.action == "reject"
