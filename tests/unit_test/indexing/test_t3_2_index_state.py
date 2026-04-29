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

"""T3.2 acceptance tests — SearchResultMetadata §G.5 extension.

Two coverage groups, mapping to the architect-locked acceptance gate
for the search-side metadata lane (msg=268f9022):

1. **Pydantic schema** — :class:`SearchResultMetadata` accepts the
   §G.5 ``parse_version`` / ``index_modality`` /
   ``index_state_per_modality`` fields, and :meth:`from_raw` extracts
   them from the upstream raw indexer metadata payload while
   sanitising any malformed values. Backward compatibility: the
   D10.h-locked existing fields (``modality`` content-shape,
   ``chunk_id`` / ``section_path`` / ``heading_anchor``) keep their
   semantics unchanged.

2. **DB read helper** — :func:`query_index_state_for_documents`
   batch-translates :class:`DocumentIndex` rows into the
   ``{doc_id: {modality: state}}`` shape the search pipeline hands
   to clients. Covers ACTIVE+serving / FAILED / INDEXING (PENDING /
   ACTIVE-but-not-serving cutover transit) / NOT_ENABLED resolution,
   per-modality independence, and an empty-input fast path.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, create_engine, insert
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.domains.retrieval.schemas import (
    IndexerModality,
    IndexStateValue,
    SearchResultMetadata,
)
from aperag.indexing.index_state import (
    PUBLIC_MODALITY_VALUES,
    _state_for_row,
    query_index_state_for_documents,
)
from aperag.indexing.models import DocumentIndex, IndexStatus, Modality

# ---------------------------------------------------------------------
# Group 1: SearchResultMetadata schema + from_raw
# ---------------------------------------------------------------------


def test_metadata_accepts_parse_version_and_index_state_fields():
    md = SearchResultMetadata(
        parse_version="abcd1234deadbeef",
        index_modality="vector",
        index_state_per_modality={
            "vector": "ACTIVE",
            "fulltext": "INDEXING",
            "graph": "FAILED",
            "summary": "NOT_ENABLED",
            "vision": "ACTIVE",
        },
    )
    assert md.parse_version == "abcd1234deadbeef"
    assert md.index_modality == "vector"
    assert md.index_state_per_modality is not None
    assert md.index_state_per_modality["fulltext"] == "INDEXING"


def test_metadata_extra_forbid_still_holds_post_g5_extension():
    """The §D10.h-locked ``extra='forbid'`` config must still reject
    unknown fields — the §G.5 additions widen the allowlist by exactly
    three entries; a typo / future shadow field must fail loudly."""
    with pytest.raises(ValueError):
        SearchResultMetadata(unexpected_field="boom")  # type: ignore[call-arg]


def test_metadata_index_modality_rejects_unknown_value():
    """The ``index_modality`` field is a Literal of the 5 indexer
    modalities; passing an unknown string fails Pydantic validation."""
    with pytest.raises(ValueError):
        SearchResultMetadata(index_modality="not_a_modality")  # type: ignore[arg-type]


def test_metadata_index_state_value_rejects_unknown_value():
    with pytest.raises(ValueError):
        SearchResultMetadata(index_state_per_modality={"vector": "WEIRD"})  # type: ignore[dict-item]


def test_from_raw_extracts_g5_fields_from_upstream_metadata():
    raw = {
        "source": "doc.pdf",
        "document_id": "doc-1",
        "chunk_id": "doc-1:0",
        "parse_version": "v1",
        "index_modality": "graph",
        "index_state_per_modality": {
            "vector": "ACTIVE",
            "fulltext": "FAILED",
            "graph": "ACTIVE",
        },
    }
    md = SearchResultMetadata.from_raw(raw)
    assert md is not None
    assert md.parse_version == "v1"
    assert md.index_modality == "graph"
    assert md.index_state_per_modality == {
        "vector": "ACTIVE",
        "fulltext": "FAILED",
        "graph": "ACTIVE",
    }


def test_from_raw_falls_back_to_legacy_indexer_key_for_index_modality():
    """Legacy upstream pipelines tagged the indexer modality under
    the ``indexer`` key; the §G.5 surface accepts both for backward
    compat with vector / fulltext / graph indexers that haven't been
    rewired yet."""
    raw = {"document_id": "doc-1", "indexer": "fulltext"}
    md = SearchResultMetadata.from_raw(raw)
    assert md is not None
    assert md.index_modality == "fulltext"


def test_from_raw_drops_malformed_index_state_entries_silently():
    """Sanitise upstream payload — keys / values that don't match the
    locked enum are dropped rather than surfaced. Prevents an upstream
    bug from leaking unknown values to clients."""
    raw = {
        "document_id": "doc-1",
        "index_state_per_modality": {
            "vector": "ACTIVE",
            "fulltext": "GIBBERISH",
            12345: "ACTIVE",
            "summary": 42,
        },
    }
    md = SearchResultMetadata.from_raw(raw)
    assert md is not None
    assert md.index_state_per_modality == {"vector": "ACTIVE"}


def test_from_raw_preserves_d10h_locked_fields_unchanged():
    """§G.5 amendments must not perturb the D10.h locks on
    ``chunk_id`` / ``section_path`` / ``heading_anchor``."""
    raw = {
        "chunk_id": "doc-1:0",
        "section_path": "1/2",
        "heading_anchor": "intro",
        "modality": "image",
        "indexer": "vision",
    }
    md = SearchResultMetadata.from_raw(raw)
    assert md is not None
    assert md.chunk_id == "doc-1:0"
    assert md.section_path == "1/2"
    assert md.heading_anchor == "intro"
    # D10.h content modality stays as-is; new index_modality also
    # populated from the legacy ``indexer`` fallback.
    assert md.modality == "image"
    assert md.index_modality == "vision"


def test_from_raw_returns_none_when_metadata_empty_or_missing():
    assert SearchResultMetadata.from_raw(None) is None
    assert SearchResultMetadata.from_raw({}) is None


# ---------------------------------------------------------------------
# Group 2: query_index_state_for_documents helper
# ---------------------------------------------------------------------


@pytest.fixture
def engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DocumentIndex.metadata.create_all(eng, tables=[DocumentIndex.__table__])
    return eng


def _seed(
    engine: Engine,
    *,
    collection_id: str,
    document_id: str,
    modality: Modality,
    status: IndexStatus,
    is_serving: bool = False,
    parse_version: str = "v1",
) -> None:
    with Session(engine) as session, session.begin():
        session.execute(
            insert(DocumentIndex).values(
                document_id=document_id,
                parse_version=parse_version,
                modality=modality.value,
                status=status.value,
                tenant_scope_key="user:test",
                source_path="collections/c/documents/d/derived/parse_v/chunks.jsonl",
                collection_id=collection_id,
                is_serving=is_serving,
            )
        )


def test_state_for_row_translates_status_serving_pair_to_g5_enum():
    """Pin :data:`IndexStateValue` translation contract."""
    assert _state_for_row(IndexStatus.ACTIVE.value, True) == "ACTIVE"
    assert _state_for_row(IndexStatus.ACTIVE.value, False) == "INDEXING"
    assert _state_for_row(IndexStatus.PENDING.value, False) == "INDEXING"
    assert _state_for_row(IndexStatus.RUNNING.value, False) == "INDEXING"
    assert _state_for_row(IndexStatus.FAILED.value, False) == "FAILED"


def test_query_returns_empty_for_empty_input(engine):
    assert query_index_state_for_documents(engine=engine, collection_id="c", document_ids=[]) == {}


def test_query_returns_dense_not_enabled_when_no_rows(engine):
    """A document with no DocumentIndex rows shows every modality as
    ``NOT_ENABLED``. Dense shape — clients always see all 5 keys."""
    result = query_index_state_for_documents(engine=engine, collection_id="col-1", document_ids=["doc-untouched"])
    assert "doc-untouched" in result
    assert set(result["doc-untouched"].keys()) == set(PUBLIC_MODALITY_VALUES)
    assert all(v == "NOT_ENABLED" for v in result["doc-untouched"].values())


def test_query_translates_active_serving_row_to_active(engine):
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-1",
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )
    result = query_index_state_for_documents(engine=engine, collection_id="col-1", document_ids=["doc-1"])
    assert result["doc-1"]["vector"] == "ACTIVE"
    # Other modalities default to NOT_ENABLED for this doc.
    assert result["doc-1"]["fulltext"] == "NOT_ENABLED"


def test_query_treats_active_but_not_serving_as_indexing(engine):
    """§F.3 cutover transit window — row reached ACTIVE but the
    cutover TX hasn't promoted is_serving=TRUE yet (or is in progress
    on a different worker). §F.4 says clients should treat as
    INDEXING."""
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-1",
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=False,
    )
    result = query_index_state_for_documents(engine=engine, collection_id="col-1", document_ids=["doc-1"])
    assert result["doc-1"]["vector"] == "INDEXING"


def test_query_translates_pending_and_running_to_indexing(engine):
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-pending",
        modality=Modality.VECTOR,
        status=IndexStatus.PENDING,
    )
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-running",
        modality=Modality.VECTOR,
        status=IndexStatus.RUNNING,
    )
    result = query_index_state_for_documents(
        engine=engine,
        collection_id="col-1",
        document_ids=["doc-pending", "doc-running"],
    )
    assert result["doc-pending"]["vector"] == "INDEXING"
    assert result["doc-running"]["vector"] == "INDEXING"


def test_query_translates_failed_to_failed(engine):
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-bad",
        modality=Modality.GRAPH,
        status=IndexStatus.FAILED,
    )
    result = query_index_state_for_documents(engine=engine, collection_id="col-1", document_ids=["doc-bad"])
    assert result["doc-bad"]["graph"] == "FAILED"


def test_query_filters_by_collection_id(engine):
    """The helper filters by collection_id — passing the wrong
    collection returns NOT_ENABLED for every modality even though
    the document exists under a different collection. Mirrors the
    multi-tenant boundary at §H.3 (Collection.user owns the tenant
    scope; queries always include collection_id in the WHERE clause).
    """
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-in-col-1",
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )

    same_col = query_index_state_for_documents(engine=engine, collection_id="col-1", document_ids=["doc-in-col-1"])
    other_col = query_index_state_for_documents(engine=engine, collection_id="col-2", document_ids=["doc-in-col-1"])
    assert same_col["doc-in-col-1"]["vector"] == "ACTIVE"
    # Wrong collection_id → not visible → all modalities NOT_ENABLED.
    assert other_col["doc-in-col-1"]["vector"] == "NOT_ENABLED"


def test_query_serving_row_wins_over_pending_sibling(engine):
    """§F.3 cutover model: a (doc, modality) may have an old PENDING /
    superseded row coexisting with the new ACTIVE+serving row. The
    helper returns the ACTIVE state so clients see the user-relevant
    answer."""
    # Old superseded parse_version with PENDING status
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-1",
        modality=Modality.VECTOR,
        status=IndexStatus.PENDING,
        is_serving=False,
        parse_version="v_old",
    )
    # New parse_version: ACTIVE + serving
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-1",
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=True,
        parse_version="v_new",
    )
    result = query_index_state_for_documents(engine=engine, collection_id="col-1", document_ids=["doc-1"])
    assert result["doc-1"]["vector"] == "ACTIVE"


def test_query_per_modality_independence_under_partial_failures(engine):
    """A doc with vector ACTIVE + fulltext FAILED + graph PENDING +
    summary not enqueued + vision running shows the §F.4 per-modality
    independent visibility shape clients depend on."""
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-1",
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-1",
        modality=Modality.FULLTEXT,
        status=IndexStatus.FAILED,
    )
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-1",
        modality=Modality.GRAPH,
        status=IndexStatus.PENDING,
    )
    _seed(
        engine,
        collection_id="col-1",
        document_id="doc-1",
        modality=Modality.VISION,
        status=IndexStatus.RUNNING,
    )
    # SUMMARY: no row at all → NOT_ENABLED in result.
    # GRAPH_FACTS / GRAPH_VECTORS: new modalities added in #indexing优化
    # task #5 — no rows seeded here either, so they appear NOT_ENABLED.
    # Once the call sites switch from the legacy ``graph`` to the new
    # split modalities, callers will hide the legacy ``graph`` field
    # behind the §4.5 dual-scenario compatibility logic; this test
    # only pins the per-modality independence shape, not the
    # compatibility mapping.
    result = query_index_state_for_documents(engine=engine, collection_id="col-1", document_ids=["doc-1"])
    assert result["doc-1"] == {
        "vector": "ACTIVE",
        "fulltext": "FAILED",
        "graph": "INDEXING",
        "graph_facts": "NOT_ENABLED",
        "graph_vectors": "NOT_ENABLED",
        "summary": "NOT_ENABLED",
        "vision": "INDEXING",
    }


# ---------------------------------------------------------------------
# Group 3: type alias parity + Literal export
# ---------------------------------------------------------------------


def test_indexer_modality_and_index_state_value_aliases_are_exported():
    """Pin both type aliases as importable from the public surface so
    callers (search pipeline, MCP tools) don't have to re-derive
    them."""
    # These are typing aliases (not classes) — just ensure import +
    # truthiness as a smoke check against a future rename / deletion.
    assert IndexerModality is not None
    assert IndexStateValue is not None
