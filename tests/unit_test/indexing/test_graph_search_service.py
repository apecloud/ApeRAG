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

"""Unit tests for ``aperag.indexing.graph_search_service`` — Wave 7
task #5 + Wave 8 W8-1 task #12.

Pins the vector-recall contract:

* ``search_entities`` embeds the query, ANN-searches the
  ``graph_entity`` indexer slice (filter + threshold pinned), then
  fetches matching ``EntityWithLineage`` rows via per-name
  ``get_entity`` (asyncio.gather). De-dups payload names so an aliased
  entity returned twice doesn't double-fetch.
* ``search_relations`` (Wave 8 W8-1 upgrade) embeds the query,
  ANN-searches the ``graph_relation`` indexer slice, parses each hit's
  ``entity_name="src->tgt"`` + ``entity_type=relation_type`` payload
  per the task #3 writer, then reverse-looks-up
  ``RelationWithLineage`` via ``store.get_relation``. Hits whose
  payload doesn't parse cleanly are skipped silently. Replaces the
  Wave 7 conservative 1-hop expansion path now that task #3 is
  writing relation vectors.
* ``get_subgraph`` is a thin pass-through to
  ``expand_neighbors_n_hops`` for MCP / retrieval callers.
* ``compose_context`` renders byte-for-byte the same graph-RAG context
  block ``aperag/domains/retrieval/pipeline.py:_render_graph_context_text``
  produces today (so task #8's swap is zero-functional-change), and
  prefers ``compacted_description`` when present.
* Empty / failure paths short-circuit cleanly (no spurious recall, no
  exception bubbling out of the read path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aperag.domains.retrieval.pipeline import _render_graph_context_text
from aperag.indexing.graph import (
    DescriptionPart,
    EntityWithLineage,
    LineageMember,
    RelationWithLineage,
)
from aperag.indexing.graph_search_service import (
    GRAPH_ENTITY_INDEXER,
    GraphSearchService,
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


def _relation(
    source: str,
    target: str,
    *,
    relation_type: str = "founded",
    description: str = "",
) -> RelationWithLineage:
    return RelationWithLineage(
        source=source,
        target=target,
        relation_type=relation_type,
        evidence_lineage=(
            LineageMember(
                document_id="doc1",
                parse_version="v1",
                tenant_scope_key="tenant-1",
                chunk_ids=("c0",),
            ),
        ),
        description_parts=(
            DescriptionPart(
                document_id="doc1",
                parse_version="v1",
                text=description or f"{source} {relation_type} {target}",
            ),
        ),
    )


class _FakeStore:
    def __init__(
        self,
        entities: dict[str, EntityWithLineage] | None = None,
        expansions: dict[tuple[str, ...], tuple[list[EntityWithLineage], list[RelationWithLineage]]] | None = None,
        relations: dict[tuple[str, str, str], RelationWithLineage] | None = None,
        keyword_matches: list[EntityWithLineage] | None = None,
    ) -> None:
        self._entities = entities or {}
        self._expansions = expansions or {}
        self._relations = relations or {}
        self._keyword_matches = keyword_matches
        self.get_entity_calls: list[str] = []
        self.get_relation_calls: list[tuple[str, str, str]] = []
        self.query_entities_by_keyword_calls: list[tuple[str, int]] = []

    async def get_entity(self, entity_name: str) -> EntityWithLineage | None:
        self.get_entity_calls.append(entity_name)
        return self._entities.get(entity_name)

    async def get_relation(self, source: str, target: str, type: str) -> RelationWithLineage | None:
        self.get_relation_calls.append((source, target, type))
        return self._relations.get((source, target, type))

    async def query_entities_by_keyword(self, *, query: str, top_k: int) -> list[EntityWithLineage]:
        self.query_entities_by_keyword_calls.append((query, top_k))
        if self._keyword_matches is not None:
            return self._keyword_matches[:top_k]
        needle = query.lower()
        return [entity for name, entity in sorted(self._entities.items()) if needle in name.lower()][:top_k]

    async def expand_neighbors_n_hops(
        self,
        *,
        entity_names: list[str],
        hops: int = 1,
    ) -> tuple[list[EntityWithLineage], list[RelationWithLineage]]:
        key = tuple(sorted(entity_names))
        return self._expansions.get(key, ([], []))


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text)), 0.0, 0.0]


class _FailingEmbedder:
    def embed_query(self, text: str) -> list[float]:  # pragma: no cover - exercised via test
        raise RuntimeError("embedder offline")


@dataclass
class _RecordedSearch:
    request: Any


class _FakeVectorConnector:
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._hits = hits or []
        self.searches: list[_RecordedSearch] = []

    def search(self, request) -> list[SearchHit]:
        self.searches.append(_RecordedSearch(request=request))
        return list(self._hits)


class _FailingVectorConnector:
    def __init__(self) -> None:
        self.searches: list[_RecordedSearch] = []

    def search(self, request):
        self.searches.append(_RecordedSearch(request=request))
        raise RuntimeError("vector store down")


def _make_service(
    *,
    entities: dict[str, EntityWithLineage] | None = None,
    expansions: dict[tuple[str, ...], tuple[list[EntityWithLineage], list[RelationWithLineage]]] | None = None,
    relations: dict[tuple[str, str, str], RelationWithLineage] | None = None,
    keyword_matches: list[EntityWithLineage] | None = None,
    hits: list[SearchHit] | None = None,
    embedder: Any | None = None,
    connector: Any | None = None,
    top_k: int = 10,
    score_threshold: float = 0.0,
) -> tuple[GraphSearchService, _FakeStore, _FakeVectorConnector | Any, _FakeEmbedder | Any]:
    store = _FakeStore(
        entities=entities,
        expansions=expansions,
        relations=relations,
        keyword_matches=keyword_matches,
    )
    connector = connector if connector is not None else _FakeVectorConnector(hits=hits)
    embedder = embedder if embedder is not None else _FakeEmbedder()
    service = GraphSearchService(
        store=store,
        vector_connector=connector,
        embedder=embedder,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    return service, store, connector, embedder


# ---------------------------------------------------------------------
# search_entities
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_entities_empty_query_returns_empty():
    service, store, connector, embedder = _make_service(entities={})
    assert await service.search_entities("") == []
    assert await service.search_entities("   ") == []
    assert connector.searches == []
    assert embedder.calls == []
    assert store.get_entity_calls == []


@pytest.mark.asyncio
async def test_search_entities_zero_topk_returns_empty():
    service, _, connector, embedder = _make_service(entities={})
    assert await service.search_entities("query", top_k=0) == []
    assert connector.searches == []
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_search_entities_uses_graph_entity_filter_and_threshold():
    service, _, connector, _ = _make_service(
        entities={},
        hits=[],
        score_threshold=0.55,
        top_k=7,
    )
    await service.search_entities("query")
    assert len(connector.searches) == 1
    request = connector.searches[0].request
    assert request.flt == Eq("indexer", GRAPH_ENTITY_INDEXER)
    assert request.score_threshold == 0.55
    assert request.top_k == 7


@pytest.mark.asyncio
async def test_search_entities_returns_entities_in_hit_order():
    a = _entity("Alpha")
    b = _entity("Beta")
    c = _entity("Gamma")
    service, store, _, _ = _make_service(
        entities={"Alpha": a, "Beta": b, "Gamma": c},
        hits=[
            SearchHit(id="1", score=0.9, payload={"entity_name": "Beta"}),
            SearchHit(id="2", score=0.8, payload={"entity_name": "Alpha"}),
            SearchHit(id="3", score=0.7, payload={"entity_name": "Gamma"}),
        ],
    )
    result = await service.search_entities("query")
    assert [e.name for e in result] == ["Beta", "Alpha", "Gamma"]
    # Each name fetched exactly once.
    assert store.get_entity_calls[0] == "query"
    assert sorted(store.get_entity_calls[1:]) == ["Alpha", "Beta", "Gamma"]


@pytest.mark.asyncio
async def test_search_entities_dedupes_repeated_payload_names():
    a = _entity("Alpha")
    service, store, _, _ = _make_service(
        entities={"Alpha": a},
        hits=[
            SearchHit(id="1", score=0.9, payload={"entity_name": "Alpha"}),
            SearchHit(id="2", score=0.85, payload={"entity_name": "Alpha"}),  # alias point
        ],
    )
    result = await service.search_entities("query")
    assert len(result) == 1
    assert store.get_entity_calls == ["query", "Alpha"]


@pytest.mark.asyncio
async def test_search_entities_skips_hits_with_no_name_payload():
    a = _entity("Alpha")
    service, _, _, _ = _make_service(
        entities={"Alpha": a},
        hits=[
            SearchHit(id="ghost", score=0.99, payload={}),  # no name → skipped
            SearchHit(id="1", score=0.9, payload={"entity_name": "Alpha"}),
        ],
    )
    result = await service.search_entities("query")
    assert [e.name for e in result] == ["Alpha"]


@pytest.mark.asyncio
async def test_search_entities_drops_gced_entity():
    """Vector store hit for an entity that was GC'd between sync and
    search → ``store.get_entity`` returns None → entity dropped from
    result, no exception."""
    service, _, _, _ = _make_service(
        entities={},  # store empty: every get_entity returns None
        hits=[SearchHit(id="1", score=0.9, payload={"entity_name": "Ghost"})],
    )
    assert await service.search_entities("query") == []


@pytest.mark.asyncio
async def test_search_entities_swallows_embedder_failure():
    service, store, connector, _ = _make_service(entities={}, embedder=_FailingEmbedder())
    assert await service.search_entities("query") == []
    assert connector.searches == []
    assert store.get_entity_calls == ["query"]


@pytest.mark.asyncio
async def test_search_entities_swallows_vector_store_failure():
    service, store, connector, embedder = _make_service(entities={}, connector=_FailingVectorConnector())
    assert await service.search_entities("query") == []
    # Embedder still called; the failure happened on vector search.
    assert embedder.calls == ["query"]
    assert len(connector.searches) == 1
    assert store.get_entity_calls == ["query"]


@pytest.mark.asyncio
async def test_search_entities_exact_match_works_when_embedder_fails():
    exact = _entity("Alice")
    service, store, connector, _ = _make_service(
        entities={"Alice": exact},
        embedder=_FailingEmbedder(),
    )
    result = await service.search_entities("Alice")
    assert [entity.name for entity in result] == ["Alice"]
    assert connector.searches == []
    assert store.get_entity_calls == ["Alice"]


@pytest.mark.asyncio
async def test_search_entities_keyword_fallback_works_when_vector_store_fails():
    alpha = _entity("Alpha")
    service, store, connector, _ = _make_service(
        entities={},
        keyword_matches=[alpha],
        connector=_FailingVectorConnector(),
    )
    result = await service.search_entities("alp")
    assert [entity.name for entity in result] == ["Alpha"]
    assert len(connector.searches) == 1
    assert store.query_entities_by_keyword_calls == [("alp", 10)]


# ---------------------------------------------------------------------
# search_relations (Wave 8 W8-1 vector recall path)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_relations_empty_query_returns_empty():
    service, _, connector, embedder = _make_service(entities={})
    assert await service.search_relations("") == []
    assert await service.search_relations("   ") == []
    assert connector.searches == []
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_search_relations_zero_topk_returns_empty():
    service, _, connector, embedder = _make_service(entities={})
    assert await service.search_relations("query", top_k=0) == []
    assert connector.searches == []
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_search_relations_uses_graph_relation_filter():
    """Wave 8 W8-1: filter pinned to ``Eq("indexer", "graph_relation")``
    so the ANN never bleeds into entity / chunk vectors sharing the
    physical collection."""
    service, _, connector, _ = _make_service(entities={}, hits=[], score_threshold=0.42, top_k=5)
    await service.search_relations("query")
    assert len(connector.searches) == 1
    request = connector.searches[0].request
    from aperag.indexing.graph_search_service import GRAPH_RELATION_INDEXER

    assert request.flt == Eq("indexer", GRAPH_RELATION_INDEXER)
    assert request.score_threshold == 0.42
    assert request.top_k == 5


@pytest.mark.asyncio
async def test_search_relations_parses_payload_and_resolves_via_get_relation():
    """Hit payload carries ``entity_name="src->tgt"`` +
    ``entity_type=relation_type`` (per task #3 writer
    ``aperag/indexing/graph.py:1631``); we split, reverse-lookup via
    ``store.get_relation`` and return the full ``RelationWithLineage``
    so :meth:`compose_context` keeps its byte-parity rendering."""
    rel = _relation("Alpha", "Beta", relation_type="founded")
    service, store, _, _ = _make_service(
        relations={("Alpha", "Beta", "founded"): rel},
        hits=[
            SearchHit(
                id="1",
                score=0.9,
                payload={"entity_name": "Alpha->Beta", "entity_type": "founded"},
            ),
        ],
    )
    relations = await service.search_relations("query")
    assert relations == [rel]
    assert store.get_relation_calls == [("Alpha", "Beta", "founded")]


@pytest.mark.asyncio
async def test_search_relations_preserves_hit_order_and_dedupes():
    rel_ab = _relation("Alpha", "Beta", relation_type="founded")
    rel_bc = _relation("Beta", "Gamma", relation_type="acquired")
    service, store, _, _ = _make_service(
        relations={
            ("Alpha", "Beta", "founded"): rel_ab,
            ("Beta", "Gamma", "acquired"): rel_bc,
        },
        hits=[
            SearchHit(id="1", score=0.9, payload={"entity_name": "Beta->Gamma", "entity_type": "acquired"}),
            SearchHit(id="2", score=0.8, payload={"entity_name": "Alpha->Beta", "entity_type": "founded"}),
            # Duplicate of the first hit (e.g. alias-redirect side-effect)
            # — must dedupe so we don't double-fetch the same edge.
            SearchHit(id="3", score=0.7, payload={"entity_name": "Beta->Gamma", "entity_type": "acquired"}),
        ],
    )
    relations = await service.search_relations("query")
    assert [(r.source, r.target) for r in relations] == [
        ("Beta", "Gamma"),
        ("Alpha", "Beta"),
    ]
    assert store.get_relation_calls == [("Beta", "Gamma", "acquired"), ("Alpha", "Beta", "founded")]


@pytest.mark.asyncio
async def test_search_relations_skips_payload_missing_arrow_or_type():
    """A hit whose payload doesn't parse cleanly is dropped silently —
    better to skip than to surface an edge we can't reconstruct."""
    rel_ab = _relation("Alpha", "Beta", relation_type="founded")
    service, store, _, _ = _make_service(
        relations={("Alpha", "Beta", "founded"): rel_ab},
        hits=[
            SearchHit(id="ghost1", score=1.0, payload={}),  # no payload at all
            SearchHit(
                id="ghost2", score=0.99, payload={"entity_name": "AlphaBeta", "entity_type": "founded"}
            ),  # missing arrow
            SearchHit(id="ghost3", score=0.98, payload={"entity_name": "Alpha->Beta"}),  # missing entity_type
            SearchHit(
                id="ghost4", score=0.97, payload={"entity_name": "->Beta", "entity_type": "founded"}
            ),  # empty source
            SearchHit(
                id="ghost5", score=0.96, payload={"entity_name": "Alpha->", "entity_type": "founded"}
            ),  # empty target
            SearchHit(id="real", score=0.9, payload={"entity_name": "Alpha->Beta", "entity_type": "founded"}),
        ],
    )
    relations = await service.search_relations("query")
    assert [(r.source, r.target) for r in relations] == [("Alpha", "Beta")]
    # Only the real hit hit the store.
    assert store.get_relation_calls == [("Alpha", "Beta", "founded")]


@pytest.mark.asyncio
async def test_search_relations_drops_edge_gced_from_store():
    """Vector hit for an edge that was deleted between sync and search
    → store.get_relation returns None → drop from result, no exception."""
    service, _, _, _ = _make_service(
        relations={},  # store empty: every get_relation returns None
        hits=[
            SearchHit(id="1", score=0.9, payload={"entity_name": "Alpha->Beta", "entity_type": "founded"}),
        ],
    )
    assert await service.search_relations("query") == []


@pytest.mark.asyncio
async def test_search_relations_swallows_embedder_failure():
    service, store, connector, _ = _make_service(entities={}, embedder=_FailingEmbedder())
    assert await service.search_relations("query") == []
    assert connector.searches == []
    assert store.get_relation_calls == []


@pytest.mark.asyncio
async def test_search_relations_swallows_vector_store_failure():
    service, store, connector, embedder = _make_service(entities={}, connector=_FailingVectorConnector())
    assert await service.search_relations("query") == []
    assert embedder.calls == ["query"]
    assert len(connector.searches) == 1
    assert store.get_relation_calls == []


# ---------------------------------------------------------------------
# get_subgraph
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_subgraph_passthrough_to_store():
    a = _entity("Alpha")
    b = _entity("Beta")
    rel = _relation("Alpha", "Beta")
    service, _, _, _ = _make_service(
        entities={},
        expansions={("Alpha",): ([a, b], [rel])},
    )
    entities, relations = await service.get_subgraph(["Alpha"], hops=1)
    assert [e.name for e in entities] == ["Alpha", "Beta"]
    assert relations == [rel]


@pytest.mark.asyncio
async def test_get_subgraph_empty_seeds_returns_empty():
    service, store, _, _ = _make_service(entities={})
    entities, relations = await service.get_subgraph([])
    assert entities == [] and relations == []
    # Must NOT call the store with an empty seed list — backends are
    # not obligated to short-circuit and the trip would be wasted.
    assert store.get_entity_calls == []


# ---------------------------------------------------------------------
# compose_context — must match retrieval-pipeline render byte-for-byte
# ---------------------------------------------------------------------


def test_compose_context_matches_retrieval_pipeline_render_byte_for_byte():
    """The retrieval pipeline (task #8 wiring) must drop in
    ``GraphSearchService.compose_context`` without behaviour change.
    Pin the equivalence so any future render tweak forces a deliberate
    edit on both sides."""
    a = _entity("OpenAI", entity_type="organization", description="AI research lab")
    b = _entity("ChatGPT", entity_type="product", description="LLM-based chatbot")
    rel = _relation("OpenAI", "ChatGPT", relation_type="produces", description="builds and operates")

    composed = GraphSearchService.compose_context([a, b], [rel])
    legacy = _render_graph_context_text([a, b], [rel])
    assert composed == legacy


def test_compose_context_empty_returns_empty_string():
    assert GraphSearchService.compose_context([], []) == ""


def test_compose_context_prefers_compacted_description_when_present():
    a = _entity("OpenAI", description="raw chunk fragment 1")
    object.__setattr__(a, "compacted_description", "OpenAI is an AI research lab.")

    text = GraphSearchService.compose_context([a], [])
    assert "OpenAI is an AI research lab." in text
    assert "raw chunk fragment 1" not in text


def test_compose_context_handles_missing_description_with_fallback_marker():
    bare = EntityWithLineage(
        name="Empty",
        entity_type="organization",
        source_lineage=(),
        description_parts=(),
    )
    text = GraphSearchService.compose_context([bare], [])
    assert "(no description)" in text
