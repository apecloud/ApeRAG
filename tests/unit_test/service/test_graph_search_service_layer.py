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

"""Wave 7 §K.12.6 — :class:`GraphService` graph-search method tests (task #7).

Covers the three new domain-service methods that back the internal-only
REST endpoints under ``/collections/{id}/graphs/entities/...`` and
``/graphs/subgraph/...``. Mocks the per-collection factory so the test
exercises the projection + flow without touching Qdrant / Postgres.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aperag.domains.knowledge_graph.schemas import (
    GraphEntitiesSearchResponse,
    GraphRelationView,
    GraphSearchEntity,
    GraphSubgraphExpandResponse,
)
from aperag.domains.knowledge_graph.service import (
    GraphService,
    _entity_to_search_view,
    _relation_to_view,
)
from aperag.indexing.graph import (
    DescriptionPart,
    EntityWithLineage,
    LineageMember,
    RelationWithLineage,
)


def _make_entity(
    *,
    name: str = "墨香居",
    entity_type: str = "organization",
    parts: list[tuple[str, str, str]] | None = None,
    chunks: list[tuple[str, str, list[str]]] | None = None,
    compacted: str | None = None,
) -> EntityWithLineage:
    parts = parts or [("doc1", "v1", "raw description")]
    chunks = chunks or [("doc1", "v1", ["chunk-a", "chunk-b"])]
    return EntityWithLineage(
        name=name,
        entity_type=entity_type,
        source_lineage=tuple(
            LineageMember(
                document_id=doc,
                parse_version=ver,
                tenant_scope_key="user:test",
                chunk_ids=tuple(cs),
            )
            for doc, ver, cs in chunks
        ),
        description_parts=tuple(
            DescriptionPart(document_id=doc, parse_version=ver, text=text) for doc, ver, text in parts
        ),
        compacted_description=compacted,
    )


def _make_relation(
    *,
    source: str = "李明华",
    target: str = "墨香居",
    relation_type: str = "owns",
    parts: list[tuple[str, str, str]] | None = None,
    compacted: str | None = None,
) -> RelationWithLineage:
    parts = parts or [("doc1", "v1", "李明华经营墨香居")]
    return RelationWithLineage(
        source=source,
        target=target,
        relation_type=relation_type,
        evidence_lineage=(
            LineageMember(document_id="doc1", parse_version="v1", tenant_scope_key="user:test", chunk_ids=("c1",)),
        ),
        description_parts=tuple(
            DescriptionPart(document_id=doc, parse_version=ver, text=text) for doc, ver, text in parts
        ),
        compacted_description=compacted,
    )


# --- _entity_to_search_view / _relation_to_view ---------------------


def test_entity_view_prefers_compacted_description():
    e = _make_entity(parts=[("d", "v", "raw")], compacted="compacted text")
    view = _entity_to_search_view(e)
    assert isinstance(view, GraphSearchEntity)
    assert view.description == "compacted text"
    assert view.entity_type == "organization"


def test_entity_view_falls_back_to_joined_parts():
    e = _make_entity(
        parts=[("doc1", "v1", "first"), ("doc2", "v2", "second")],
        compacted=None,
    )
    view = _entity_to_search_view(e)
    assert "first" in view.description and "second" in view.description


def test_entity_view_source_chunk_count_dedupes_across_lineage():
    e = _make_entity(
        chunks=[
            ("doc1", "v1", ["a", "b"]),
            ("doc1", "v2", ["b", "c"]),  # 'b' appears twice — must dedup
            ("doc2", "v1", ["d"]),
        ]
    )
    view = _entity_to_search_view(e)
    assert view.source_chunk_count == 4  # {a,b,c,d}


def test_relation_view_uses_render_description_rule():
    r = _make_relation(parts=[("d", "v", "primary")], compacted="compacted")
    view = _relation_to_view(r)
    assert isinstance(view, GraphRelationView)
    assert view.source == "李明华"
    assert view.target == "墨香居"
    assert view.description == "compacted"


# --- search_entities ------------------------------------------------


@pytest.mark.asyncio
async def test_search_entities_projects_graph_search_service_results():
    e1 = _make_entity(name="A", parts=[("d1", "v1", "alpha")])
    e2 = _make_entity(name="B", parts=[("d2", "v1", "beta")], compacted="B compact")

    fake_search_service = MagicMock()
    fake_search_service.search_entities = AsyncMock(return_value=[e1, e2])

    svc = GraphService()
    db_collection = SimpleNamespace(id="col1")
    with (
        patch.object(svc, "_get_and_validate_collection", new=AsyncMock(return_value=db_collection)),
        patch(
            "aperag.indexing.graph_search_service.build_graph_search_service_for",
            return_value=fake_search_service,
        ),
    ):
        result = await svc.search_entities("u", "col1", query="hello", top_k=5)

    assert isinstance(result, GraphEntitiesSearchResponse)
    fake_search_service.search_entities.assert_awaited_once_with(query="hello", top_k=5)
    assert [e.name for e in result.entities] == ["A", "B"]
    assert result.entities[1].description == "B compact"


# --- expand_subgraph ------------------------------------------------


@pytest.mark.asyncio
async def test_expand_subgraph_projects_entities_and_relations():
    e_a = _make_entity(name="A")
    e_b = _make_entity(name="B")
    r_ab = _make_relation(source="A", target="B")

    fake_search_service = MagicMock()
    fake_search_service.get_subgraph = AsyncMock(return_value=([e_a, e_b], [r_ab]))

    svc = GraphService()
    db_collection = SimpleNamespace(id="col1")
    with (
        patch.object(svc, "_get_and_validate_collection", new=AsyncMock(return_value=db_collection)),
        patch(
            "aperag.indexing.graph_search_service.build_graph_search_service_for",
            return_value=fake_search_service,
        ),
    ):
        result = await svc.expand_subgraph("u", "col1", entity_names=["A"], hops=2)

    assert isinstance(result, GraphSubgraphExpandResponse)
    fake_search_service.get_subgraph.assert_awaited_once_with(entity_names=["A"], hops=2)
    assert [e.name for e in result.entities] == ["A", "B"]
    assert len(result.relations) == 1
    assert result.relations[0].source == "A" and result.relations[0].target == "B"


# --- get_entity_detail ----------------------------------------------


@pytest.mark.asyncio
async def test_get_entity_detail_returns_view_when_found():
    entity = _make_entity(name="X")

    fake_store = MagicMock()
    fake_store.get_entity = AsyncMock(return_value=entity)

    svc = GraphService()
    db_collection = SimpleNamespace(id="col1")
    with (
        patch.object(svc, "_get_and_validate_collection", new=AsyncMock(return_value=db_collection)),
        patch(
            "aperag.indexing.worker_factory._resolve_graph_backend_type",
            return_value="postgres",
        ),
        patch(
            "aperag.indexing.worker_factory._build_lineage_graph_store",
            return_value=fake_store,
        ),
    ):
        result = await svc.get_entity_detail("u", "col1", entity_name="X")

    assert isinstance(result, GraphSearchEntity)
    assert result.name == "X"
    fake_store.get_entity.assert_awaited_once_with("X")


@pytest.mark.asyncio
async def test_get_entity_detail_returns_none_when_missing():
    fake_store = MagicMock()
    fake_store.get_entity = AsyncMock(return_value=None)

    svc = GraphService()
    db_collection = SimpleNamespace(id="col1")
    with (
        patch.object(svc, "_get_and_validate_collection", new=AsyncMock(return_value=db_collection)),
        patch(
            "aperag.indexing.worker_factory._resolve_graph_backend_type",
            return_value="postgres",
        ),
        patch(
            "aperag.indexing.worker_factory._build_lineage_graph_store",
            return_value=fake_store,
        ),
    ):
        result = await svc.get_entity_detail("u", "col1", entity_name="missing")

    assert result is None


# --- byte-parity sanity (compose_context render rule reuse) ---------


def test_entity_view_description_matches_compose_context_for_compacted():
    """The MCP entity view description rule MUST match
    :func:`GraphSearchService.compose_context` so an MCP caller chaining
    ``query_graph_entities -> expand_graph_subgraph -> compose_context``
    sees identical text per entity (no double-render drift).
    """
    from aperag.indexing.graph_search_service import GraphSearchService

    e = _make_entity(parts=[("d", "v", "raw")], compacted="compacted via Compactor")
    view = _entity_to_search_view(e)
    composed = GraphSearchService.compose_context([e], [])
    assert view.description in composed
