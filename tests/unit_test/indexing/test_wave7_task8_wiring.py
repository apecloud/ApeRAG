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

"""Wave 7 §K.12.8 task #8 — three wiring integration tests.

Per architect msg=da89237c / huangheng msg=55974e1e the task #8 PR
must land three wiring points at the same time. These tests pin each
in place so a future refactor cannot silently regress any of them:

1. ``worker_factory._build_lineage_graph_store`` now returns the
   alias-redirect decorator (PR #1758 inseparability gate alive).
2. ``retrieval/pipeline._graph_search`` now composes
   ``GraphSearchService.search_entities`` + ``get_subgraph`` +
   ``compose_context`` — vector recall is alive (PR #1756 wired).
3. ``GraphService.merge_entities`` (the route layer for
   ``POST /graphs/nodes/merge``) now delegates to
   ``LineageEntityMerger`` (PR #1758 user-driven merge alive).

Each test mocks at the boundary between the wiring layer and the
heavy I/O underneath (vector store / DB / LLM) so it exercises the
glue, not the production backends.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aperag.indexing.alias_redirect_store import LineageGraphStoreWithAliasRedirect

# ---------------------------------------------------------------------
# Wiring 1 — worker_factory returns the alias-redirect decorator
# ---------------------------------------------------------------------


def test_build_lineage_graph_store_returns_alias_redirect_decorator():
    """The Wave 7 invariant #9 gate: every indexer / retrieval read of
    the lineage store must be wrapped so the alias map is consulted on
    upsert. Without this swap, the alias rows the
    :class:`LineageEntityMerger` writes are silently ignored on the
    next indexer sync — re-creating the merged-away entities and
    breaking the user's curation intent.
    """
    from aperag.indexing import worker_factory

    fake_inner_store = MagicMock(name="inner_store")

    with patch.object(
        worker_factory,
        "_build_lineage_graph_store_inner",
        return_value=fake_inner_store,
    ) as build_inner:
        collection = SimpleNamespace(id="col_w7t8")
        store = worker_factory._build_lineage_graph_store(
            backend_type="postgres",
            collection=collection,
        )

    build_inner.assert_called_once_with(backend_type="postgres", collection=collection)
    # The returned object is the decorator, not the raw inner store.
    assert isinstance(store, LineageGraphStoreWithAliasRedirect)
    # And the decorator wraps the same inner store the factory built.
    assert store._inner is fake_inner_store
    assert store._collection_id == "col_w7t8"


def test_build_lineage_graph_store_inner_returns_raw_backend():
    """Symmetric to the above: callers that need the raw inner store
    (the merger writes canonical names directly and must NOT be
    intercepted) get an undecorated backend adapter from
    ``_build_lineage_graph_store_inner``.
    """
    from aperag.indexing import worker_factory

    # Patch the postgres engine singleton so this test does not need a
    # live database — the inner factory will still construct the
    # adapter object.
    with patch.object(worker_factory, "_postgres_async_engine_singleton", return_value=MagicMock()):
        collection = SimpleNamespace(id="col_w7t8")
        store = worker_factory._build_lineage_graph_store_inner(
            backend_type="postgres",
            collection=collection,
        )

    # The raw adapter is NOT the decorator.
    assert not isinstance(store, LineageGraphStoreWithAliasRedirect)


# ---------------------------------------------------------------------
# Wiring 2 — retrieval/pipeline._graph_search uses GraphSearchService
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_search_pipeline_composes_search_entities_and_get_subgraph():
    """Wave 7 §K.12.5 / §K.12.8 task #8 cutover: the retrieval pipeline
    must call ``search_entities`` (vector recall) + ``get_subgraph``
    (1-hop traversal) + ``compose_context`` (graph-RAG render),
    not the Wave 6 keyword-only ``query_entities_by_keyword`` path.
    """
    from aperag.domains.retrieval.pipeline import SearchPipelineService

    fake_service = MagicMock(name="graph_search_service")
    fake_entities = [SimpleNamespace(name="A"), SimpleNamespace(name="B")]
    fake_subgraph = ([SimpleNamespace(name="A")], [SimpleNamespace(source="A", target="B")])
    fake_service.search_entities = AsyncMock(return_value=fake_entities)
    fake_service.get_subgraph = AsyncMock(return_value=fake_subgraph)

    collection = SimpleNamespace(id="col_w7t8", config='{"enable_knowledge_graph": true}')

    with (
        patch(
            "aperag.indexing.graph_search_service.build_graph_search_service_for",
            return_value=fake_service,
        ),
        patch(
            "aperag.indexing.graph_search_service.GraphSearchService.compose_context",
            return_value="-----Entities (KG)-----\n- [entity] A — d",
        ) as compose_mock,
    ):
        pipeline = SearchPipelineService.__new__(SearchPipelineService)
        results = await pipeline._graph_search(collection, "what is A?", top_k=5)

    fake_service.search_entities.assert_awaited_once_with(query="what is A?", top_k=5)
    fake_service.get_subgraph.assert_awaited_once_with(entity_names=["A", "B"], hops=1)
    compose_mock.assert_called_once_with(fake_subgraph[0], fake_subgraph[1])

    assert len(results) == 1
    assert results[0].metadata == {"recall_type": "graph_search"}
    assert "Entities (KG)" in results[0].text


@pytest.mark.asyncio
async def test_graph_search_pipeline_returns_empty_when_kg_disabled():
    """``enable_knowledge_graph=false`` short-circuits — the cutover
    must preserve the Wave 6 gate so collections without graph indexing
    return ``[]`` instead of attempting a no-op vector recall.
    """
    from aperag.domains.retrieval.pipeline import SearchPipelineService

    collection = SimpleNamespace(id="col_w7t8", config='{"enable_knowledge_graph": false}')
    pipeline = SearchPipelineService.__new__(SearchPipelineService)
    results = await pipeline._graph_search(collection, "anything", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_graph_search_pipeline_swallows_factory_failures():
    """A collection whose vector connector / embedder is misconfigured
    should degrade to an empty result rather than raise into the
    retrieval pipeline."""
    from aperag.domains.retrieval.pipeline import SearchPipelineService

    collection = SimpleNamespace(id="col_w7t8", config='{"enable_knowledge_graph": true}')

    with patch(
        "aperag.indexing.graph_search_service.build_graph_search_service_for",
        side_effect=RuntimeError("embedder not configured"),
    ):
        pipeline = SearchPipelineService.__new__(SearchPipelineService)
        results = await pipeline._graph_search(collection, "x", top_k=5)

    assert results == []


# ---------------------------------------------------------------------
# Wiring 3 — graph_service.merge_entities → LineageEntityMerger
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_delegates_to_lineage_entity_merger():
    """The route layer for ``POST /graphs/nodes/merge`` must run the
    new merger, not the legacy ``GraphIndexService.merge_entities``.
    The response shape stays backward-compat (frontend / OpenAPI regen
    is a no-op), but the underlying engine is the Wave 7 lineage merger.
    """
    from aperag.domains.knowledge_graph.service import GraphService
    from aperag.graph_curation.lineage_merge import LineageMergeResult

    fake_merger = MagicMock(name="lineage_entity_merger")
    fake_merger.merge_entities = AsyncMock(
        return_value=LineageMergeResult(
            final_target="A",
            merged_source_ids=["B", "C"],
            unified_description="LLM-merged description",
            compacted_description="compacted text",
        )
    )

    svc = GraphService()
    db_collection = SimpleNamespace(id="col_w7t8")

    with (
        patch.object(svc, "_get_and_validate_collection", new=AsyncMock(return_value=db_collection)),
        patch(
            "aperag.graph_curation.lineage_merge.build_lineage_entity_merger_for",
            return_value=fake_merger,
        ),
        patch.object(svc, "_collect_source_chunk_ids", new=AsyncMock(return_value=["chunk-a", "chunk-b"])),
        # Don't actually fire the suggestion-expire side-effect.
        patch("aperag.graph_curation.graph_curation_service.expire_pending_for_entities", new=AsyncMock()),
    ):
        result = await svc.merge_entities(
            "user-1",
            "col_w7t8",
            target_entity_id="A",
            source_entity_ids=["B", "C"],
        )

    fake_merger.merge_entities.assert_awaited_once_with(
        target_name="A",
        source_names=["B", "C"],
        merged_by="user-1",
    )

    # Backward-compat response shape — these field names are the
    # frontend contract and must not drift on cutover.
    assert result == {
        "target_entity_id": "A",
        "merged_source_ids": ["B", "C"],
        "description": "compacted text",
        "source_chunk_ids": ["chunk-a", "chunk-b"],
        "edges_redirected": 0,
        "edges_collapsed": 0,
    }


@pytest.mark.asyncio
async def test_merge_entities_falls_back_to_unified_when_no_compacted():
    """When the merger returns no ``compacted_description`` (small
    merge under the compactor threshold), the response surfaces the
    unified description directly so the UI panel still has text.
    """
    from aperag.domains.knowledge_graph.service import GraphService
    from aperag.graph_curation.lineage_merge import LineageMergeResult

    fake_merger = MagicMock()
    fake_merger.merge_entities = AsyncMock(
        return_value=LineageMergeResult(
            final_target="A",
            merged_source_ids=["B"],
            unified_description="just the unified text",
            compacted_description=None,
        )
    )

    svc = GraphService()
    db_collection = SimpleNamespace(id="col_w7t8")
    with (
        patch.object(svc, "_get_and_validate_collection", new=AsyncMock(return_value=db_collection)),
        patch(
            "aperag.graph_curation.lineage_merge.build_lineage_entity_merger_for",
            return_value=fake_merger,
        ),
        patch.object(svc, "_collect_source_chunk_ids", new=AsyncMock(return_value=[])),
        patch("aperag.graph_curation.graph_curation_service.expire_pending_for_entities", new=AsyncMock()),
    ):
        result = await svc.merge_entities(
            "user-1",
            "col_w7t8",
            target_entity_id="A",
            source_entity_ids=["B"],
        )

    assert result["description"] == "just the unified text"


@pytest.mark.asyncio
async def test_merge_entities_alias_cycle_surfaces_as_value_error():
    """A merge that would form an alias cycle must surface as a
    ``ValueError`` so the FastAPI route returns 400 (preserving the
    existing ``ValueError`` → 400 mapping in ``merge_nodes_view``).
    """
    from aperag.domains.knowledge_graph.service import GraphService
    from aperag.graph_curation.alias_map import AliasCycleError

    fake_merger = MagicMock()
    fake_merger.merge_entities = AsyncMock(side_effect=AliasCycleError("cycle: A → B → A"))

    svc = GraphService()
    db_collection = SimpleNamespace(id="col_w7t8")

    with (
        patch.object(svc, "_get_and_validate_collection", new=AsyncMock(return_value=db_collection)),
        patch(
            "aperag.graph_curation.lineage_merge.build_lineage_entity_merger_for",
            return_value=fake_merger,
        ),
    ):
        with pytest.raises(ValueError, match="cycle"):
            await svc.merge_entities(
                "user-1",
                "col_w7t8",
                target_entity_id="A",
                source_entity_ids=["B"],
            )
