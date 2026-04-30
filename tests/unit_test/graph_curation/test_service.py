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

import pytest

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
# The Pattern C dispatch wrapped in
# ``asyncio.create_task(asyncio.to_thread(...))`` never raises at
# schedule time, so the synchronous-failure assertion no longer mapped
# to any reachable behaviour.
#
# task #31 Phase A1 (PR #1938) reintroduces a real failure path: the
# enqueue is now ``await runtime.queue.push_graph_curation_run(...)``
# which CAN raise, and ``_mark_run_failed`` + raise must run in that
# branch (per spec § 3.1.1 + huangzhangshu testing CR msg=fe66bd72).
# The ``test_start_run_*`` family below pins the post-A1 behaviour.


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
        entity_snapshots=[
            {
                "entity_id": "e1",
                "entity_name": "墨香居",
                "entity_type": "ORGANIZATION",
                "description": "",
                "source_chunk_count": 1,
            },
            {
                "entity_id": "e2",
                "entity_name": "旧书店",
                "entity_type": "ORGANIZATION",
                "description": "",
                "source_chunk_count": 1,
            },
        ],
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
    assert out["suggestion_batch_id"] == "gcr_1"
    assert out["merge_reason"] == "same entity"
    assert out["suggested_target_entity"] == {
        "entity_name": "墨香居",
        "entity_type": "ORGANIZATION",
    }
    assert out["confidence_score"] == 0.91
    assert out["evidence_refs"] == [
        {"document_id": "doc1", "chunk_id": "chunk1", "parse_version": "v1"},
    ]


# ---------------------------------------------------------------------
# task #31 Phase A1 — ``GraphCurationService.start_run`` enqueue
# behaviour gate (per huangzhangshu msg=fe66bd72 CR gap).
# ---------------------------------------------------------------------
#
# The pre-A1 path was a fire-and-forget
# ``asyncio.create_task(asyncio.to_thread(generate_graph_curation_run_task, ...))``
# inside the API process. Post-A1 it's a thin
# ``runtime.queue.push_graph_curation_run(payload)`` enqueue onto the
# independent ``q:graph_curation_run`` queue family. These tests pin
# three properties:
#
# 1. **Payload shape** — what the worker reads when it pops must match
#    what the service writes. The worker's
#    :class:`GraphCurationRunDispatchPayload.from_dict` requires
#    ``run_id`` + ``collection_id`` as strings; we assert the service
#    actually produces that shape, not just any dict.
# 2. **No double enqueue** — when an active PENDING/RUNNING run exists,
#    ``start_run`` returns ``started=False`` and MUST NOT enqueue
#    again. Otherwise every duplicate API call would multiply Redis
#    payloads, leaving the worker doing redundant N×N sweeps.
# 3. **Enqueue failure** — if the queue rejects the push (Redis down,
#    etc.), the run row must be marked FAILED and a ``RuntimeError``
#    raised. Silent success would leave the row in PENDING forever.
#
# 4. **Runtime not installed** — fail-loud guard for test environments
#    or pre-startup boot. Same FAILED + raise discipline.


class _FakeQueue:
    """Capture-mode stub of :class:`aperag.indexing.orchestrator.WorkQueue`.

    Records the payloads pushed onto the graph_curation_run lane so
    tests can assert exact shape, and toggles ``raise_on_push`` to
    simulate Redis enqueue failure.
    """

    def __init__(self, *, raise_on_push: bool = False) -> None:
        self.pushed: list[dict] = []
        self.raise_on_push = raise_on_push

    async def push_graph_curation_run(self, *, payload: dict) -> None:
        if self.raise_on_push:
            raise RuntimeError("simulated redis push failure")
        # Match production semantics — store a copy so callers can't
        # mutate the captured payload after-the-fact.
        self.pushed.append(dict(payload))


def _install_runtime_with(queue) -> None:
    from aperag.indexing.runtime import IndexingRuntime, set_runtime

    set_runtime(IndexingRuntime(engine=None, queue=queue, workers={}))


def _clear_runtime() -> None:
    from aperag.indexing.runtime import set_runtime

    set_runtime(None)


class _FakeRun:
    """Minimal ``GraphCurationRun`` stand-in carrying ``id``."""

    def __init__(self, run_id: str) -> None:
        self.id = run_id


def _build_service_with_run(*, run_id: str, created: bool):
    """Construct a ``GraphCurationService`` with the heavy collaborators
    stubbed so we exercise only the post-transaction enqueue branch.
    """
    from aperag.graph_curation.service import GraphCurationService

    service = GraphCurationService.__new__(GraphCurationService)

    async def _validate(user_id, collection_id):
        return None  # validation succeeds

    async def _execute(_op):
        return _FakeRun(run_id), created

    captured_failures: list[tuple[str, str]] = []

    async def _mark_failed(run_id_arg: str, reason: str) -> None:
        captured_failures.append((run_id_arg, reason))

    def _run_to_dict(run):
        return {"id": run.id}

    service._get_and_validate_collection = _validate  # type: ignore[attr-defined]
    service.execute_with_transaction = _execute  # type: ignore[attr-defined]
    service._mark_run_failed = _mark_failed  # type: ignore[attr-defined]
    service._run_to_dict = _run_to_dict  # type: ignore[attr-defined]
    return service, captured_failures


@pytest.mark.asyncio
async def test_start_run_enqueues_canonical_payload_when_created():
    queue = _FakeQueue()
    _install_runtime_with(queue)
    try:
        service, failures = _build_service_with_run(run_id="run-abc", created=True)
        result = await service.start_run(user_id="u1", collection_id="c1")
    finally:
        _clear_runtime()

    assert result["started"] is True
    assert result["run"] == {"id": "run-abc"}
    assert failures == []
    # Payload shape is the API/worker contract — pin it precisely so
    # ``GraphCurationRunDispatchPayload.from_dict`` keeps working.
    assert queue.pushed == [{"run_id": "run-abc", "collection_id": "c1"}]
    # Both fields must be ``str`` so the worker's ``from_dict``
    # normalisation is a no-op (it expects strings).
    [pushed] = queue.pushed
    assert isinstance(pushed["run_id"], str)
    assert isinstance(pushed["collection_id"], str)


@pytest.mark.asyncio
async def test_start_run_does_not_enqueue_when_run_already_active():
    queue = _FakeQueue()
    _install_runtime_with(queue)
    try:
        service, failures = _build_service_with_run(run_id="run-existing", created=False)
        result = await service.start_run(user_id="u1", collection_id="c1")
    finally:
        _clear_runtime()

    assert result["started"] is False
    assert result["run"] == {"id": "run-existing"}
    assert failures == []
    assert queue.pushed == [], (
        "An active PENDING/RUNNING run was found, so start_run must NOT "
        "enqueue again — duplicate enqueues multiply worker N×N sweeps "
        "and waste LLM quota"
    )


@pytest.mark.asyncio
async def test_start_run_marks_run_failed_and_raises_when_enqueue_raises():
    queue = _FakeQueue(raise_on_push=True)
    _install_runtime_with(queue)
    try:
        service, failures = _build_service_with_run(run_id="run-doomed", created=True)
        with pytest.raises(RuntimeError, match="Failed to schedule graph curation run"):
            await service.start_run(user_id="u1", collection_id="c1")
    finally:
        _clear_runtime()

    # Run row must be marked FAILED with the reason carrying the original
    # exception — a silent failure would leave the row in PENDING forever.
    assert len(failures) == 1
    failed_run_id, reason = failures[0]
    assert failed_run_id == "run-doomed"
    assert "enqueue_failed" in reason
    assert "simulated redis push failure" in reason


@pytest.mark.asyncio
async def test_start_run_marks_run_failed_and_raises_when_runtime_not_installed():
    """Fail-loud guard for test environments / pre-startup boot.

    The pre-A1 path silently spawned an ``asyncio.create_task`` even
    when no runtime was installed; the post-A1 path needs a real queue
    to enqueue onto, so ``runtime is None`` must surface as a marked-
    FAILED row + raised RuntimeError rather than a "started=True" lie.
    """
    _clear_runtime()  # ensure no runtime is in place
    service, failures = _build_service_with_run(run_id="run-orphan", created=True)
    with pytest.raises(RuntimeError, match="runtime not installed"):
        await service.start_run(user_id="u1", collection_id="c1")

    assert len(failures) == 1
    failed_run_id, reason = failures[0]
    assert failed_run_id == "run-orphan"
    assert "runtime not installed" in reason


@pytest.mark.asyncio
async def test_start_run_marks_run_failed_when_runtime_has_no_queue():
    """Symmetric to ``runtime not installed`` — runtime is present but
    its ``queue`` slot is ``None`` (e.g. INLINE-mode test runtime).
    """
    from aperag.indexing.runtime import IndexingRuntime, set_runtime

    set_runtime(IndexingRuntime(engine=None, queue=None, workers={}))
    try:
        service, failures = _build_service_with_run(run_id="run-noq", created=True)
        with pytest.raises(RuntimeError, match="runtime not installed"):
            await service.start_run(user_id="u1", collection_id="c1")
    finally:
        _clear_runtime()

    assert len(failures) == 1
    failed_run_id, reason = failures[0]
    assert failed_run_id == "run-noq"
    assert "runtime not installed" in reason
