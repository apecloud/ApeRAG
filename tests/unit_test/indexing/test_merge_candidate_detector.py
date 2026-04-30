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

"""Unit tests for ``aperag.indexing.merge_candidate_detector`` — Wave 7
task #4.

Pins the contract that:

* Sync-driven detection writes ``GraphCurationSuggestion`` rows in the
  same shape the user-triggered curation flow already uses (so the
  existing UI / REST surface is zero-touch — task #9 / cuiwenbo).
* Auto-detected runs are tagged via ``GraphCurationRun.config_json
  ["source"] == "auto_detect"`` and the suggestions carry
  ``reason == "auto_detected"`` so admin UIs can split user vs sync
  triggered runs without a schema change.
* Canonical pick is the alphabetical-first entity name — same
  ``(entity_a, entity_b)`` pair across syncs always points at the same
  surviving entity (architect ratify #3, msg=6ab89fbb).
* Each sync supersedes the prior auto-detected ``PENDING`` rows for
  the collection so reviewers never see stale duplicates.
* Empty ``affected_entity_names`` short-circuits — no run row, no DB
  write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aperag.indexing.graph import (
    DescriptionPart,
    EntityWithLineage,
    LineageMember,
)
from aperag.indexing.merge_candidate_detector import (
    AUTO_DETECT_REASON,
    AUTO_DETECT_SOURCE,
    MergeCandidateDetector,
)
from aperag.vectorstore.dto import SearchHit
from aperag.vectorstore.filters import Eq

# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------


def _entity(
    name: str,
    *,
    entity_type: str = "organization",
    description: str = "",
    chunk_ids: tuple[str, ...] = ("c0",),
    document_id: str = "doc1",
) -> EntityWithLineage:
    return EntityWithLineage(
        name=name,
        entity_type=entity_type,
        source_lineage=(
            LineageMember(
                document_id=document_id,
                parse_version="v1",
                tenant_scope_key="tenant-1",
                chunk_ids=chunk_ids,
            ),
        ),
        description_parts=(
            DescriptionPart(
                document_id=document_id,
                parse_version="v1",
                text=description or f"description of {name}",
            ),
        ),
    )


class _FakeStore:
    def __init__(self, entities: dict[str, EntityWithLineage]) -> None:
        self._entities = entities

    async def get_entity(self, entity_name: str) -> EntityWithLineage | None:
        return self._entities.get(entity_name)


class _FakeEmbedder:
    """Returns a unique deterministic vector per text so tests can match
    on ``(text, vector)`` pairs."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._cache: dict[str, list[float]] = {}

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        if text not in self._cache:
            self._cache[text] = [float(len(self._cache) + 1), 0.0, 0.0]
        return self._cache[text]


@dataclass
class _RecordedSearch:
    request: Any  # QueryRequest captured for filter / threshold assertion


class _FakeVectorConnector:
    def __init__(self, hits_by_query: dict[tuple[float, ...], list[SearchHit]]) -> None:
        self._hits = hits_by_query
        self.searches: list[_RecordedSearch] = []

    def search(self, request) -> list[SearchHit]:
        self.searches.append(_RecordedSearch(request=request))
        return list(self._hits.get(tuple(request.embedding), []))


class _RecordingPersistence:
    """Capture what the detector would persist without standing up SQLA.

    The real persistence path uses ``execute_with_transaction``; we
    monkeypatch that on a per-detector basis to record the ``run`` /
    ``suggestions`` payload the persistence ``_op`` would have written.
    """

    def __init__(self) -> None:
        self.runs: list[Any] = []
        self.suggestions: list[Any] = []
        self.supersede_filters: list[dict[str, Any]] = []
        self.flush_count = 0

    def install(self, detector: MergeCandidateDetector) -> None:
        recorder = self

        async def _fake_execute(operation):
            await operation(_FakeSession(recorder))

        detector.execute_with_transaction = _fake_execute  # type: ignore[method-assign]


class _FakeSession:
    def __init__(self, recorder: _RecordingPersistence) -> None:
        self._recorder = recorder
        self._next_run_id = "gcr_test_0001"

    async def execute(self, stmt) -> Any:
        # Best-effort capture of the supersede UPDATE: store the WHERE
        # criteria + the values dict so tests can assert on filter shape.
        try:
            criteria = [str(c) for c in stmt._where_criteria]  # type: ignore[attr-defined]
        except Exception:
            criteria = []
        try:
            values = dict(stmt._values)  # type: ignore[attr-defined]
        except Exception:
            values = {}
        self._recorder.supersede_filters.append({"criteria": criteria, "values": values})
        return None

    def add(self, instance) -> None:
        # Discriminate by class name to avoid importing internals.
        cls_name = type(instance).__name__
        if cls_name == "GraphCurationRun":
            instance.id = self._next_run_id
            self._recorder.runs.append(instance)
        elif cls_name == "GraphCurationSuggestion":
            self._recorder.suggestions.append(instance)
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected ORM type: {cls_name}")

    async def flush(self) -> None:
        self._recorder.flush_count += 1


def _make_detector(
    *,
    entities: dict[str, EntityWithLineage],
    hits_by_query: dict[tuple[float, ...], list[SearchHit]] | None = None,
    embedder: _FakeEmbedder | None = None,
    user_id: str = "user-1",
    collection_id: str = "col-1",
    vector_top_k: int = 4,
    vector_score_threshold: float = 0.72,
) -> tuple[MergeCandidateDetector, _RecordingPersistence, _FakeVectorConnector, _FakeEmbedder]:
    embedder = embedder or _FakeEmbedder()
    connector = _FakeVectorConnector(hits_by_query or {})
    detector = MergeCandidateDetector(
        store=_FakeStore(entities),
        vector_connector=connector,
        embedder=embedder,
        collection_id=collection_id,
        user_id=user_id,
        vector_top_k=vector_top_k,
        vector_score_threshold=vector_score_threshold,
    )
    recorder = _RecordingPersistence()
    recorder.install(detector)
    return detector, recorder, connector, embedder


# ---------------------------------------------------------------------
# Empty / no-op cases
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_affected_short_circuits_with_no_db_write():
    detector, recorder, connector, embedder = _make_detector(entities={})

    written = await detector.detect_for_sync(sync_run_id="sync-0", affected_entity_names=[])

    assert written == 0
    assert recorder.runs == []
    assert recorder.suggestions == []
    assert recorder.supersede_filters == []  # short-circuited before DB
    assert embedder.calls == []
    assert connector.searches == []


@pytest.mark.asyncio
async def test_affected_entity_gced_before_run_is_skipped():
    """Caller hands us names; if the entity was GC'd between sync and
    detect, ``store.get_entity`` returns ``None`` and we treat it as not
    affected. We still create a run row so the audit trail records the
    detector ran (with stats = 0)."""
    detector, recorder, *_ = _make_detector(entities={})  # store empty

    written = await detector.detect_for_sync(sync_run_id="sync-1", affected_entity_names=["GoneEntity"])

    assert written == 0
    # Detector still recorded the run for audit (stats reflect the GC).
    assert len(recorder.runs) == 1
    assert recorder.runs[0].stats["affected_entities"] == 1
    assert recorder.runs[0].stats["suggestions"] == 0
    assert recorder.suggestions == []


# ---------------------------------------------------------------------
# Run / suggestion shape pinning
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_and_suggestions_carry_auto_detect_tags():
    a = _entity("OpenAI", description="AI lab founded in 2015 in San Francisco")
    b = _entity("OpenAI Inc.", description="AI lab founded in 2015 in San Francisco")
    detector, recorder, *_ = _make_detector(entities={a.name: a, b.name: b})

    written = await detector.detect_for_sync(
        sync_run_id="sync-42",
        affected_entity_names=[a.name, b.name],
    )

    assert written == 1, "lexical near-duplicate should yield exactly one suggestion"
    assert len(recorder.runs) == 1
    run = recorder.runs[0]
    assert run.config_json == {
        "source": AUTO_DETECT_SOURCE,
        "sync_run_id": "sync-42",
    }
    assert run.collection_id == "col-1"
    assert run.user_id == "user-1"
    # Run is created in COMPLETED state because the detection completes
    # synchronously (no PENDING/RUNNING phase to expose).
    assert run.status.value == "COMPLETED"
    assert run.stats["affected_entities"] == 2
    assert run.stats["suggestions"] == 1

    suggestion = recorder.suggestions[0]
    assert suggestion.reason == AUTO_DETECT_REASON
    assert suggestion.collection_id == "col-1"
    assert suggestion.user_id == "user-1"
    assert suggestion.run_id == run.id
    assert suggestion.status.value == "PENDING"


# ---------------------------------------------------------------------
# Canonical pick is alphabetical-first
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_target_entity_id_is_alphabetical_first():
    """Architect ratify #3: deterministic, reproducible canonical pick.
    ``min(name_a, name_b)`` so the same pair across syncs always points
    at the same surviving entity.

    Wave 5 description-NULL invariant (task #31 A3): pre-Wave-5 this
    test surfaced the candidate pair via ``description_token_overlap``
    on identical descriptions. That signal is gone post Wave 5
    (descriptions are NULL); this test now drives the pair via the
    vector-neighbour signal, which is the canonical Wave-5+
    description-free recall surface.
    """
    # Construct two entities where alphabetical order ≠ insertion order.
    z = _entity("Zephyr Corp")
    a = _entity("Aeolus Inc")
    embedder = _FakeEmbedder()
    z_text = MergeCandidateDetector._embedding_query_text(z)
    a_text = MergeCandidateDetector._embedding_query_text(a)
    embedder.embed_query(z_text)  # warm cache
    embedder.embed_query(a_text)  # warm cache
    z_vec = tuple(embedder._cache[z_text])
    a_vec = tuple(embedder._cache[a_text])
    detector, recorder, *_ = _make_detector(
        entities={z.name: z, a.name: a},
        hits_by_query={
            z_vec: [
                SearchHit(
                    id="self_z",
                    score=1.0,
                    payload={"entity_name": z.name, "indexer": "graph_entity"},
                ),
                SearchHit(
                    id="nbr_a",
                    score=0.91,
                    payload={"entity_name": a.name, "indexer": "graph_entity"},
                ),
            ],
            a_vec: [
                SearchHit(
                    id="self_a",
                    score=1.0,
                    payload={"entity_name": a.name, "indexer": "graph_entity"},
                ),
                SearchHit(
                    id="nbr_z",
                    score=0.91,
                    payload={"entity_name": z.name, "indexer": "graph_entity"},
                ),
            ],
        },
        embedder=embedder,
    )

    await detector.detect_for_sync(
        sync_run_id="sync-7",
        affected_entity_names=[z.name, a.name],  # insertion order: Z first
    )

    suggestion = recorder.suggestions[0]
    assert suggestion.target_entity_id == "Aeolus Inc"  # alphabetical-first
    assert suggestion.entity_ids == ["Aeolus Inc", "Zephyr Corp"]
    # Snapshot ordering also follows alphabetical-first.
    assert [snap["entity_name"] for snap in suggestion.entity_snapshots] == [
        "Aeolus Inc",
        "Zephyr Corp",
    ]


# ---------------------------------------------------------------------
# Vector ANN integration
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_neighbour_pulls_in_unsynced_existing_entity():
    """The affected entity ``A`` has a vector ANN neighbour ``B`` that
    was NOT in the sync batch — detector still scores ``(A, B)`` and
    surfaces the candidate."""
    affected = _entity("OpenAI", description="An AI research lab")
    neighbour = _entity("OpenAI Foundation", description="An AI research lab")
    embedder = _FakeEmbedder()
    # Wave 5 description-NULL invariant (task #31 A3): embedding query
    # is built from ``entity.name + entity_type`` — descriptions are
    # NULL post Wave 5, so the embedding query text mirrors how the
    # graph_vectors worker writes the entity vector.
    affected_text = MergeCandidateDetector._embedding_query_text(affected)
    embedder.embed_query(affected_text)  # warm cache so we know the vector
    affected_vec = tuple(embedder._cache[affected_text])
    detector, recorder, connector, _ = _make_detector(
        entities={affected.name: affected, neighbour.name: neighbour},
        hits_by_query={
            affected_vec: [
                # First hit is self — must be filtered.
                SearchHit(
                    id="self",
                    score=1.0,
                    payload={"entity_name": affected.name, "indexer": "graph_entity"},
                ),
                SearchHit(
                    id="nbr",
                    score=0.91,
                    payload={"entity_name": neighbour.name, "indexer": "graph_entity"},
                ),
            ],
        },
        embedder=embedder,
    )

    written = await detector.detect_for_sync(
        sync_run_id="sync-vec",
        affected_entity_names=[affected.name],
    )

    assert written == 1
    suggestion = recorder.suggestions[0]
    assert sorted(suggestion.entity_ids) == sorted([affected.name, neighbour.name])
    # Evidence carries the vector signal so a reviewer can see *why*
    # the pair was surfaced.
    assert suggestion.evidence.get("vector_neighbor") is True
    assert suggestion.evidence.get("vector_score") == pytest.approx(0.91, rel=1e-3)


@pytest.mark.asyncio
async def test_vector_search_uses_graph_entity_filter_and_threshold():
    """§K.12 invariant #5 — vector search must filter by
    ``Eq("indexer", "graph_entity")`` so the ANN never bleeds into
    chunk / summary points sharing the physical collection."""
    a = _entity("Acme")
    embedder = _FakeEmbedder()
    # Wave 5 description-NULL invariant (task #31 A3): embedding query
    # text is ``"<name> (<entity_type>)"`` — see ``_embedding_query_text``.
    embedder.embed_query("Acme (organization)")  # warm
    detector, _, connector, _ = _make_detector(
        entities={a.name: a},
        hits_by_query={},
        embedder=embedder,
        vector_score_threshold=0.5,
        vector_top_k=3,
    )
    await detector.detect_for_sync(
        sync_run_id="sync-flt",
        affected_entity_names=[a.name],
    )

    assert len(connector.searches) == 1
    request = connector.searches[0].request
    assert request.flt == Eq("indexer", "graph_entity")
    assert request.score_threshold == 0.5
    assert request.top_k == 4  # vector_top_k + 1 (slot for self)


# ---------------------------------------------------------------------
# Supersede semantics
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersede_runs_even_when_no_suggestions():
    """Reviewers must never see a stale auto-detected PENDING from a
    prior sync that this sync no longer agrees with — so the supersede
    UPDATE runs unconditionally before any new suggestion insert."""
    detector, recorder, *_ = _make_detector(entities={})  # store empty

    await detector.detect_for_sync(
        sync_run_id="sync-empty",
        affected_entity_names=["MissingEntity"],
    )

    assert recorder.suggestions == []
    # Exactly one supersede UPDATE was recorded.
    assert len(recorder.supersede_filters) == 1


# ---------------------------------------------------------------------
# Wave 5 description-NULL invariant (task #31 A3, spec § 3.1.5)
# ---------------------------------------------------------------------


def test_embedding_query_text_uses_entity_name_and_type() -> None:
    """``_embedding_query_text`` must build the query from
    ``entity.name + entity_type`` — Wave 5 description-NULL invariant
    (task #31 A3, spec § 3.1.5 item 5).

    The legacy ``_description_text_for_scoring`` helper read
    ``compacted_description`` / ``description_parts``; Wave 5 graph
    extractor no longer emits either, so the legacy path always
    short-circuited to ``""`` and produced zero candidates after the
    cut-over. The replacement query mirrors how the graph_vectors
    worker writes the entity vector in the first place — name + type.
    """
    e = _entity("OpenAI", entity_type="organization", description="legacy residue")
    text = MergeCandidateDetector._embedding_query_text(e)
    assert text == "OpenAI (organization)", (
        "Embedding query text must be `<name> (<entity_type>)` so the "
        "vector recall side and the graph_vectors write side embed the "
        "same input shape (Wave 5 task #5 / #7 split)."
    )


def test_embedding_query_text_falls_back_to_name_when_type_missing() -> None:
    """Entities with no ``entity_type`` (legacy rows / loose extractions)
    still produce a stable embedding query — bare ``entity.name``."""
    e = _entity("Acme")
    object.__setattr__(e, "entity_type", "")
    assert MergeCandidateDetector._embedding_query_text(e) == "Acme"


def test_embedding_query_text_does_not_read_description_parts() -> None:
    """Wave 5 description-NULL invariant: even if a stale row carries
    ``description_parts`` text, ``_embedding_query_text`` must not read
    it. Reviewer recall would otherwise vary between freshly extracted
    rows (parts empty) and pre-Wave-5 residue rows (parts populated).
    """
    # Construct an entity whose description_parts looks "interesting"
    # — the helper must still produce only the name+type string.
    e = _entity("Stale", entity_type="legacy", description="sensitive residue text")
    assert MergeCandidateDetector._embedding_query_text(e) == "Stale (legacy)"
    assert "sensitive residue" not in MergeCandidateDetector._embedding_query_text(e)


def test_snapshot_does_not_include_description() -> None:
    """``MergeCandidateDetector._snapshot`` is the payload persisted
    to ``GraphCurationSuggestion.entity_snapshots``. Wave 5 description-
    NULL invariant + spec § 3.1.5 item 6: snapshot must not surface
    ``description`` — reviewers operate on name + type +
    source_chunk_count instead.
    """
    e = _entity("OpenAI", entity_type="organization", description="legacy residue", chunk_ids=("c1", "c2"))
    snapshot = MergeCandidateDetector._snapshot(e)
    assert "description" not in snapshot, (
        "Snapshot must not carry a `description` key — Wave 5 description-"
        "NULL invariant. Allowed shape: entity_id / entity_name / "
        "entity_type / source_chunk_count."
    )
    assert snapshot == {
        "entity_id": "OpenAI",
        "entity_name": "OpenAI",
        "entity_type": "organization",
        "source_chunk_count": 2,
    }
