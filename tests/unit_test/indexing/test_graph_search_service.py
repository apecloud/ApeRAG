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
task #5.

Pins the Wave 7 vector-recall contract:

* ``search_entities`` embeds the query, ANN-searches the
  ``graph_entity`` indexer slice (filter + threshold pinned), then
  fetches matching ``EntityWithLineage`` rows via per-name
  ``get_entity`` (asyncio.gather). De-dups payload names so an aliased
  entity returned twice doesn't double-fetch.
* ``search_relations`` derives relations as the 1-hop expansion of the
  vector-recalled entities — vector store carries no per-relation
  vectors in Wave 7.
* ``get_subgraph`` is a thin pass-through to
  ``expand_neighbors_n_hops`` for MCP / retrieval callers.
* ``compose_context`` renders byte-for-byte the same LightRAG-style
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
    ) -> None:
        self._entities = entities or {}
        self._expansions = expansions or {}
        self.get_entity_calls: list[str] = []

    async def get_entity(self, entity_name: str) -> EntityWithLineage | None:
        self.get_entity_calls.append(entity_name)
        return self._entities.get(entity_name)

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
    hits: list[SearchHit] | None = None,
    embedder: Any | None = None,
    connector: Any | None = None,
    top_k: int = 10,
    score_threshold: float = 0.0,
) -> tuple[GraphSearchService, _FakeStore, _FakeVectorConnector | Any, _FakeEmbedder | Any]:
    store = _FakeStore(entities=entities, expansions=expansions)
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
    assert sorted(store.get_entity_calls) == ["Alpha", "Beta", "Gamma"]


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
    assert store.get_entity_calls == ["Alpha"]


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
    assert store.get_entity_calls == []


@pytest.mark.asyncio
async def test_search_entities_swallows_vector_store_failure():
    service, store, connector, embedder = _make_service(entities={}, connector=_FailingVectorConnector())
    assert await service.search_entities("query") == []
    # Embedder still called; the failure happened on vector search.
    assert embedder.calls == ["query"]
    assert len(connector.searches) == 1
    assert store.get_entity_calls == []


# ---------------------------------------------------------------------
# search_relations
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_relations_empty_when_no_entities_match():
    service, _, _, _ = _make_service(entities={}, hits=[])
    assert await service.search_relations("query") == []


@pytest.mark.asyncio
async def test_search_relations_returns_one_hop_expansion_of_entity_results():
    a = _entity("Alpha")
    b = _entity("Beta")
    rel = _relation("Alpha", "Beta")
    service, _, _, _ = _make_service(
        entities={"Alpha": a, "Beta": b},
        hits=[
            SearchHit(id="1", score=0.9, payload={"entity_name": "Alpha"}),
            SearchHit(id="2", score=0.8, payload={"entity_name": "Beta"}),
        ],
        expansions={("Alpha", "Beta"): ([a, b], [rel])},
    )
    relations = await service.search_relations("query")
    assert relations == [rel]


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
