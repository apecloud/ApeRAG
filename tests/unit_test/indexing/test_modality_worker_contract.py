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

"""Cross-modality ``ModalityWorker.derive() + sync()`` contract supplement.

Per @earayu2 / @不穷 msg=981ae30d (Wave 3 standby task): the per-modality
acceptance tests in ``test_t1_2_graph.py`` / ``test_t1_3_vector_fulltext.py``
/ ``test_t1_4_summary_vision.py`` already cover happy-path derive +
single double-call idempotency + missing-artifact no-op for each
modality individually. This file supplements with **cross-modality
contract tests** that exercise invariants the spec promises across all
4 simple modalities (vector / fulltext / summary / vision) — the graph
modality has its own §D.3 lineage semantic exhaustively covered in
``test_t1_2_graph.py`` and is intentionally excluded from the parametric
sweeps below.

Tests cover:

1. **§C.7 reschedule contract**: ``derive()`` of summary + vision with a
   missing upstream source returns ``DeriveResult(derived_artifact_path
   ="")`` (the empty-string signal that the orchestrator interprets as
   "derive incomplete, leave PENDING for next reconciler cycle"). Vector
   + fulltext are pass-through and always return the input source_path,
   so the empty-path semantic is cover by the per-modality
   "no-op on missing chunks" tests.
2. **N-call replay convergence**: 5 consecutive ``sync()`` calls produce
   a backend state byte-equivalent to a single sync (extends the
   existing "double call" tests one rung higher to lock the invariant
   under arbitrary retry storms).
3. **Cross-document parse_version isolation**: ``sync()`` of doc-A's
   ``(doc_a, parse_version)`` slot must NOT touch doc-B's backend state.
   Locks the §D.1 DELETE-by-(doc, parse_version) scope.
4. **All-5-modality enum discriminator**: each ``ModalityWorker``
   subclass binds the ``modality`` class attr to the corresponding
   ``Modality`` enum value (the orchestrator uses this to route
   payloads).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from aperag.indexing import (
    DeriveResult,
    FulltextModality,
    GraphModalityWorker,
    InMemoryEntityLock,
    InMemoryFulltextBackend,
    InMemoryLineageGraphStore,
    InMemoryObjectStore,
    InMemorySummaryBackend,
    InMemoryVectorBackend,
    InMemoryVisionBackend,
    Modality,
    SummaryModality,
    VectorModality,
    VisionModality,
    parse_document,
    write_atomic,
)

SAMPLE_MARKDOWN = (
    "# Project Beacon\n\n"
    "Beacon is a sample document used by the indexing simulator tests.\n\n"
    "## Architecture\n\n"
    "Beacon has a controller and a worker pool that share a Redis queue.\n\n"
    "## Operations\n\n"
    "Operators use the dashboard to inspect queue depth and worker utilisation.\n"
)

# Distinct content for the cross-document isolation tests below — using
# the same body for both docs would collide on the parser's content-
# hashed ``chunk_id``, which is a fixture limitation rather than a
# production-code invariant violation.
SAMPLE_MARKDOWN_DOC_B = (
    "# Project Sentinel\n\n"
    "Sentinel is a separate sample document used to lock cross-document "
    "isolation invariants in the indexing simulator tests.\n\n"
    "## Topology\n\n"
    "Sentinel has a coordinator and an executor fleet behind a "
    "service mesh.\n\n"
    "## Monitoring\n\n"
    "Reviewers use the dashboard to inspect throughput and latency.\n"
)

SAMPLE_VISION_RECORDS = json.dumps(
    [
        {"image_id": "img-1", "alt_text": "controller diagram", "page_idx": 1},
        {"image_id": "img-2", "alt_text": "worker pool topology", "page_idx": 2},
    ]
).encode("utf-8")


def _seed_parser(
    store,
    *,
    document_id: str = "doc-beacon",
    collection_id: str = "col-1",
    body: str = SAMPLE_MARKDOWN,
):
    return parse_document(
        store=store,
        collection_id=collection_id,
        document_id=document_id,
        source_bytes=body.encode("utf-8"),
    )


def _seed_vision_source(store, *, document_id: str, collection_id: str = "col-1") -> str:
    """Place the synthetic vision-records JSON under the canonical
    ``collections/<cid>/documents/<did>/images.json`` path that
    ``vision.derive`` discovers.
    """
    path = f"collections/{collection_id}/documents/{document_id}/images.json"
    write_atomic(store, path, SAMPLE_VISION_RECORDS)
    return path


# ---------------------------------------------------------------------
# (1) §C.7 reschedule contract — summary + vision
# ---------------------------------------------------------------------


def test_summary_derive_returns_empty_path_when_markdown_missing():
    """Summary derive returns ``DeriveResult(derived_artifact_path="")``
    when the upstream parser markdown isn't yet present, signalling
    the orchestrator to leave the row PENDING for the next reconciler
    cycle (§C.7 read contract: "derive 还没完成 → reschedule, 不报错").
    """

    store = InMemoryObjectStore()
    modality = SummaryModality(backend=InMemorySummaryBackend(), store=store)

    result = asyncio.run(
        modality.derive(
            document_id="doc-missing",
            parse_version="0123456789abcdef",
            source_path="collections/col-1/documents/doc-missing/derived/parse_0123456789abcdef/markdown.md",
        )
    )
    assert isinstance(result, DeriveResult)
    assert result.derived_artifact_path == "", (
        "summary.derive must return empty path when source markdown is missing — this is the reschedule signal per §C.7"
    )


def test_vision_derive_returns_empty_path_when_image_records_missing():
    """Vision derive returns ``DeriveResult(derived_artifact_path="")``
    when the synthetic image-records JSON isn't yet present at the
    expected source path. Same §C.7 reschedule contract as summary.
    """

    store = InMemoryObjectStore()
    modality = VisionModality(backend=InMemoryVisionBackend(), store=store)

    result = asyncio.run(
        modality.derive(
            document_id="doc-no-images",
            parse_version="fedcba9876543210",
            source_path="collections/col-1/documents/doc-no-images/images.json",
        )
    )
    assert isinstance(result, DeriveResult)
    assert result.derived_artifact_path == "", (
        "vision.derive must return empty path when source image records are missing (§C.7 reschedule signal)"
    )


# ---------------------------------------------------------------------
# (2) N-call replay convergence — vector / fulltext / summary / vision
# ---------------------------------------------------------------------


# Each builder returns ``(modality, backend, inspector(document_id,
# parse_version) -> list[dict])``. The inspector is a closure over the
# backend so the parametric tests don't have to know whether the
# backend uses ``points_for_document`` (vector / summary / vision) or
# ``documents_for_document`` (fulltext).


def _build_vector_modality(store):
    backend = InMemoryVectorBackend()
    return (
        VectorModality(backend=backend, store=store),
        backend,
        lambda doc_id, pv: backend.points_for_document(doc_id, pv),
    )


def _build_fulltext_modality(store):
    backend = InMemoryFulltextBackend()
    return (
        FulltextModality(backend=backend, store=store),
        backend,
        lambda doc_id, pv: backend.documents_for_document(doc_id, pv),
    )


def _build_summary_modality(store):
    backend = InMemorySummaryBackend()
    return (
        SummaryModality(backend=backend, store=store),
        backend,
        lambda doc_id, pv: backend.points_for_document(doc_id, pv),
    )


def _build_vision_modality(store):
    backend = InMemoryVisionBackend()
    return (
        VisionModality(backend=backend, store=store),
        backend,
        lambda doc_id, pv: backend.points_for_document(doc_id, pv),
    )


_REPLAY_BUILDERS = (
    pytest.param(_build_vector_modality, "chunks", id="vector"),
    pytest.param(_build_fulltext_modality, "chunks", id="fulltext"),
    pytest.param(_build_summary_modality, "summary_derive", id="summary"),
    pytest.param(_build_vision_modality, "vision_derive", id="vision"),
)


@pytest.mark.parametrize("builder, derive_kind", _REPLAY_BUILDERS)
def test_sync_is_replay_convergent_after_5_calls(builder, derive_kind):
    """5 consecutive ``sync()`` calls must produce a backend state
    byte-equivalent to a single sync (§D.4 idempotency under retry
    storms — extends the existing 2-call tests one rung higher).
    """

    store = InMemoryObjectStore()
    modality, backend, inspect = builder(store)
    parsed = _seed_parser(store)

    if derive_kind == "chunks":
        derived_path = parsed.chunks_path
    elif derive_kind == "summary_derive":
        derived = asyncio.run(
            modality.derive(
                document_id="doc-beacon",
                parse_version=parsed.parse_version,
                source_path=parsed.markdown_path,
            )
        )
        derived_path = derived.derived_artifact_path
    elif derive_kind == "vision_derive":
        source_path = _seed_vision_source(store, document_id="doc-beacon")
        derived = asyncio.run(
            modality.derive(
                document_id="doc-beacon",
                parse_version=parsed.parse_version,
                source_path=source_path,
            )
        )
        derived_path = derived.derived_artifact_path
    else:  # pragma: no cover — guard against derive_kind typo
        raise AssertionError(f"unknown derive_kind {derive_kind!r}")

    # First sync establishes the baseline state.
    asyncio.run(
        modality.sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=derived_path,
        )
    )
    baseline = inspect("doc-beacon", parsed.parse_version)
    assert baseline, "first sync must populate the backend for a non-empty document"

    # 4 more replays — backend must converge to the same state every
    # time. Use ``range(4)`` (not ``range(5)``) since baseline already
    # counts as one sync.
    for _ in range(4):
        asyncio.run(
            modality.sync(
                document_id="doc-beacon",
                parse_version=parsed.parse_version,
                derived_artifact_path=derived_path,
            )
        )
        replay = inspect("doc-beacon", parsed.parse_version)
        assert replay == baseline, "sync replay must be byte-equivalent to baseline (§D.4)"


# ---------------------------------------------------------------------
# (3) Cross-document parse_version isolation — vector / fulltext /
# summary / vision
# ---------------------------------------------------------------------


@pytest.mark.parametrize("builder, derive_kind", _REPLAY_BUILDERS)
def test_sync_does_not_touch_other_documents(builder, derive_kind):
    """``sync()`` of doc-A must not delete or modify backend state for
    doc-B. Locks the §D.1 DELETE scope: ``WHERE document_id=A AND
    parse_version=V`` only.
    """

    store = InMemoryObjectStore()
    modality, backend, inspect = builder(store)

    # Seed two independent documents with DISTINCT bodies so the
    # parser's content-hashed ``chunk_id`` differs between the two
    # docs (otherwise the InMemoryVectorBackend's chunk_id-keyed dict
    # would have the second doc's upsert overwrite the first's points,
    # which is a fixture limitation not a production-code invariant
    # violation).
    parsed_a = _seed_parser(store, document_id="doc-A", body=SAMPLE_MARKDOWN)
    parsed_b = _seed_parser(store, document_id="doc-B", body=SAMPLE_MARKDOWN_DOC_B)

    if derive_kind == "chunks":
        path_a = parsed_a.chunks_path
        path_b = parsed_b.chunks_path
    elif derive_kind == "summary_derive":
        derived_a = asyncio.run(
            modality.derive(
                document_id="doc-A",
                parse_version=parsed_a.parse_version,
                source_path=parsed_a.markdown_path,
            )
        )
        derived_b = asyncio.run(
            modality.derive(
                document_id="doc-B",
                parse_version=parsed_b.parse_version,
                source_path=parsed_b.markdown_path,
            )
        )
        path_a = derived_a.derived_artifact_path
        path_b = derived_b.derived_artifact_path
    elif derive_kind == "vision_derive":
        src_a = _seed_vision_source(store, document_id="doc-A")
        src_b = _seed_vision_source(store, document_id="doc-B")
        derived_a = asyncio.run(
            modality.derive(
                document_id="doc-A",
                parse_version=parsed_a.parse_version,
                source_path=src_a,
            )
        )
        derived_b = asyncio.run(
            modality.derive(
                document_id="doc-B",
                parse_version=parsed_b.parse_version,
                source_path=src_b,
            )
        )
        path_a = derived_a.derived_artifact_path
        path_b = derived_b.derived_artifact_path
    else:  # pragma: no cover
        raise AssertionError(f"unknown derive_kind {derive_kind!r}")

    asyncio.run(
        modality.sync(
            document_id="doc-A",
            parse_version=parsed_a.parse_version,
            derived_artifact_path=path_a,
        )
    )
    asyncio.run(
        modality.sync(
            document_id="doc-B",
            parse_version=parsed_b.parse_version,
            derived_artifact_path=path_b,
        )
    )
    state_b_after_b_sync = inspect("doc-B", parsed_b.parse_version)
    assert state_b_after_b_sync, "sync of doc-B must populate doc-B's slot"

    # Re-sync doc-A — doc-B's state must remain byte-identical.
    asyncio.run(
        modality.sync(
            document_id="doc-A",
            parse_version=parsed_a.parse_version,
            derived_artifact_path=path_a,
        )
    )
    state_b_after_a_resync = inspect("doc-B", parsed_b.parse_version)

    assert state_b_after_a_resync == state_b_after_b_sync, (
        "sync of doc-A must not perturb doc-B's backend state — §D.1 DELETE "
        "scope is (document_id, parse_version), not the whole backend"
    )


# ---------------------------------------------------------------------
# (4) All-5-modality enum discriminator
# ---------------------------------------------------------------------


def _instantiate_graph_modality(store):
    """Build a minimal ``GraphModalityWorker`` for enum-attribute checks
    without executing any sync — the lineage semantic is exhaustively
    covered in ``test_t1_2_graph.py`` and intentionally not re-tested
    here. We only need an instance so we can read ``.modality``.

    The constructor requires ``extractor`` / ``collection_id`` /
    ``tenant_scope_key`` (per §H.2 tenant scoping); pass minimal
    placeholders since none of them are exercised by the
    enum-attribute assertion.
    """

    def _noop_extractor(*_args, **_kwargs):  # pragma: no cover — never called
        return ([], [])

    return GraphModalityWorker(
        store=InMemoryLineageGraphStore(),
        extractor=_noop_extractor,
        entity_lock=InMemoryEntityLock(),
        object_store=store,
        collection_id="col-test",
        tenant_scope_key="user:test",
    )


def test_all_five_modality_workers_bind_modality_enum_correctly():
    """Each ``ModalityWorker`` subclass binds the class-level
    ``modality`` attribute to the matching ``Modality`` enum value.
    The orchestrator uses this attribute to route dispatched payloads
    to the right worker — a misbind would silently mis-route work to
    the wrong backend.
    """

    store = InMemoryObjectStore()
    workers = (
        VectorModality(backend=InMemoryVectorBackend(), store=store),
        FulltextModality(backend=InMemoryFulltextBackend(), store=store),
        SummaryModality(backend=InMemorySummaryBackend(), store=store),
        VisionModality(backend=InMemoryVisionBackend(), store=store),
        _instantiate_graph_modality(store),
    )
    expected_enums = (
        Modality.VECTOR,
        Modality.FULLTEXT,
        Modality.SUMMARY,
        Modality.VISION,
        Modality.GRAPH,
    )
    for worker, expected in zip(workers, expected_enums):
        assert worker.modality is expected, (
            f"{type(worker).__name__}.modality must bind to {expected!r}, got {worker.modality!r}"
        )
