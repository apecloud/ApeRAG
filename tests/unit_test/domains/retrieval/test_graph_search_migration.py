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

"""Wave 6 #33 chunk 3 — retrieval pipeline graph-search migration.

Pin the chunk 3 contract: ``SearchPipelineService._graph_search``
routes through the new :class:`LineageGraphStore` Protocol
(``query_entities_by_keyword`` + ``expand_neighbors_n_hops``) and
no longer imports the legacy
``aperag.domains.knowledge_graph.graphindex`` package.

Per architect Option C ruling msg=6fccd9ab the legacy graphindex
package itself stays alive (UI / curation flows still consume it),
but retrieval/pipeline.py has a grep-zero contract — verified by
:func:`test_retrieval_pipeline_no_graphindex_import` below.
"""

from __future__ import annotations

import asyncio
import pathlib

from aperag.domains.retrieval.pipeline import (
    _render_graph_context_text,
)
from aperag.indexing.graph import (
    DescriptionPart,
    EntityWithLineage,
    LineageMember,
    RelationWithLineage,
)


def test_retrieval_pipeline_no_graphindex_import():
    """Chunk 3 grep-zero invariant: ``retrieval/pipeline.py`` must not
    import the legacy ``aperag.domains.knowledge_graph.graphindex``
    package any longer. The migrated path uses ``LineageGraphStore``
    instead. Docstring references that mention the legacy package by
    name (without ``import`` / ``from``) are allowed — they document
    the migration."""
    pipeline_path = pathlib.Path(__file__).resolve().parents[4] / "aperag" / "domains" / "retrieval" / "pipeline.py"
    body = pipeline_path.read_text(encoding="utf-8")
    # Two real-import shapes the grep-zero contract forbids.
    assert "from aperag.domains.knowledge_graph.graphindex" not in body
    assert "import aperag.domains.knowledge_graph.graphindex" not in body


def test_render_graph_context_empty_inputs_returns_empty_string():
    assert _render_graph_context_text([], []) == ""


def test_render_graph_context_renders_lightrag_style_sections():
    members = (
        LineageMember(
            document_id="doc-1",
            parse_version="v1",
            tenant_scope_key="user:t",
            chunk_ids=("chunk-1",),
        ),
    )
    parts_a = (DescriptionPart(document_id="doc-1", parse_version="v1", text="Founder of company X"),)
    parts_b = (DescriptionPart(document_id="doc-1", parse_version="v1", text="Released product Y in 2024"),)
    entities = [
        EntityWithLineage(
            name="Alice",
            entity_type="person",
            source_lineage=members,
            description_parts=parts_a,
        ),
        EntityWithLineage(
            name="Acme",
            entity_type="organization",
            source_lineage=members,
            description_parts=parts_b,
        ),
    ]
    rel_parts = (DescriptionPart(document_id="doc-1", parse_version="v1", text="Alice founded Acme"),)
    relations = [
        RelationWithLineage(
            source="Alice",
            target="Acme",
            relation_type="founded",
            evidence_lineage=members,
            description_parts=rel_parts,
        )
    ]
    text = _render_graph_context_text(entities, relations)

    assert "-----Entities (KG)-----" in text
    assert "- [person] Alice — Founder of company X" in text
    assert "- [organization] Acme — Released product Y in 2024" in text
    assert "-----Relationships (KG)-----" in text
    assert "- Alice → Acme: Alice founded Acme" in text


def test_render_graph_context_joins_multiple_description_parts():
    members = (
        LineageMember(
            document_id="doc-1",
            parse_version="v1",
            tenant_scope_key="user:t",
            chunk_ids=("c1",),
        ),
    )
    parts = (
        DescriptionPart(document_id="doc-1", parse_version="v1", text="frag-1"),
        DescriptionPart(document_id="doc-1", parse_version="v1", text="frag-2"),
        DescriptionPart(document_id="doc-1", parse_version="v1", text="   "),  # whitespace skipped
    )
    entities = [
        EntityWithLineage(name="X", entity_type="thing", source_lineage=members, description_parts=parts),
    ]
    text = _render_graph_context_text(entities, [])
    assert "- [thing] X — frag-1 | frag-2" in text


def test_render_graph_context_falls_back_to_no_description_marker():
    members = (
        LineageMember(
            document_id="doc-1",
            parse_version="v1",
            tenant_scope_key="user:t",
            chunk_ids=("c1",),
        ),
    )
    entities = [
        EntityWithLineage(
            name="Y",
            entity_type="",
            source_lineage=members,
            description_parts=(),
        ),
    ]
    text = _render_graph_context_text(entities, [])
    # Empty type defaults to "entity" label per implementation.
    assert "- [entity] Y — (no description)" in text


def test_graph_search_returns_empty_when_no_anchors(monkeypatch):
    """If vector recall returns no anchor entities, _graph_search
    returns an empty list — no subgraph call, no rendered context.

    Wave 7 §K.12.8 cutover: the pipeline now talks to
    :class:`GraphSearchService`, not directly to the lineage store
    keyword path. The "no anchors → []" contract still holds.
    """
    from aperag.domains.retrieval.pipeline import SearchPipelineService

    class _StubService:
        def __init__(self) -> None:
            self.subgraph_called = False

        async def search_entities(self, *, query, top_k):
            return []

        async def get_subgraph(self, *, entity_names, hops=1):
            self.subgraph_called = True
            return ([], [])

    stub_service = _StubService()
    monkeypatch.setattr(
        "aperag.indexing.graph_search_service.build_graph_search_service_for",
        lambda _collection: stub_service,
    )

    class _Collection:
        id = "col-1"
        user = "u"
        config = '{"enable_knowledge_graph": true}'

    svc = SearchPipelineService()
    results = asyncio.run(svc._graph_search(_Collection(), "anything", 5))
    assert results == []
    assert stub_service.subgraph_called is False, "get_subgraph should not be called when no anchors"


def test_graph_search_composes_text_from_anchors_and_neighbors(monkeypatch):
    """Wave 7 §K.12.8 cutover: the pipeline composes
    ``GraphSearchService.search_entities`` + ``get_subgraph`` +
    ``compose_context``. The byte-parity test in PR #1756 keeps the
    rendered text identical to the Wave 6 keyword-only path so the
    assertions on body content stay unchanged.
    """
    members = (
        LineageMember(
            document_id="doc-1",
            parse_version="v1",
            tenant_scope_key="user:t",
            chunk_ids=("c1",),
        ),
    )
    alice = EntityWithLineage(
        name="Alice",
        entity_type="person",
        source_lineage=members,
        description_parts=(DescriptionPart(document_id="doc-1", parse_version="v1", text="Lead engineer"),),
    )
    bob = EntityWithLineage(
        name="Bob",
        entity_type="person",
        source_lineage=members,
        description_parts=(DescriptionPart(document_id="doc-1", parse_version="v1", text="CFO"),),
    )
    alice_bob = RelationWithLineage(
        source="Alice",
        target="Bob",
        relation_type="reports_to",
        evidence_lineage=members,
        description_parts=(DescriptionPart(document_id="doc-1", parse_version="v1", text="Alice reports to Bob"),),
    )

    class _StubService:
        def __init__(self) -> None:
            self.last_query: str | None = None
            self.last_seeds: list[str] | None = None

        async def search_entities(self, *, query, top_k):
            self.last_query = query
            return [alice]

        async def get_subgraph(self, *, entity_names, hops=1):
            self.last_seeds = entity_names
            return ([alice, bob], [alice_bob])

    stub_service = _StubService()
    monkeypatch.setattr(
        "aperag.indexing.graph_search_service.build_graph_search_service_for",
        lambda _collection: stub_service,
    )

    from aperag.domains.retrieval.pipeline import SearchPipelineService

    class _Collection:
        id = "col-1"
        user = "u"
        config = '{"enable_knowledge_graph": true}'

    svc = SearchPipelineService()
    results = asyncio.run(svc._graph_search(_Collection(), "Alice", 5))
    assert len(results) == 1
    doc = results[0]
    assert doc.metadata == {"recall_type": "graph_search"}
    assert "Alice" in doc.text
    assert "Bob" in doc.text
    assert "reports_to" not in doc.text  # type label not in body, only the description
    assert "Alice → Bob" in doc.text
    assert stub_service.last_query == "Alice"
    assert stub_service.last_seeds == ["Alice"]


def test_graph_search_returns_empty_when_kg_disabled(monkeypatch):
    """If ``enable_knowledge_graph`` is False the path short-circuits
    before touching the lineage store."""
    from aperag.domains.retrieval.pipeline import SearchPipelineService

    called = {"build": False}

    def _should_not_be_called(_collection):
        called["build"] = True
        raise AssertionError("graph store builder must not be reached when KG disabled")

    monkeypatch.setattr(
        "aperag.domains.retrieval.pipeline._build_lineage_graph_store_for",
        _should_not_be_called,
    )

    class _Collection:
        id = "col-1"
        user = "u"
        config = '{"enable_knowledge_graph": false}'

    svc = SearchPipelineService()
    results = asyncio.run(svc._graph_search(_Collection(), "anything", 5))
    assert results == []
    assert called["build"] is False


def test_build_lineage_graph_store_for_imports_only_on_demand(monkeypatch):
    """``_build_lineage_graph_store_for`` keeps the indexing import
    local — calling it must reach worker_factory's resolver helpers
    without importing them at module level. We exercise the function
    via a stubbed resolver to assert the contract."""
    from aperag.domains.retrieval import pipeline as pipeline_module

    seen: list[str] = []

    def _stub_resolve(_collection):
        seen.append("resolve")
        return "postgres"

    def _stub_build(*, backend_type, collection):
        seen.append(f"build:{backend_type}")
        return object()

    monkeypatch.setattr(
        "aperag.indexing.worker_factory._resolve_graph_backend_type",
        _stub_resolve,
    )
    monkeypatch.setattr(
        "aperag.indexing.worker_factory._build_lineage_graph_store",
        _stub_build,
    )

    class _Collection:
        id = "col-1"
        config = '{"graph_backend_type": "postgres"}'

    pipeline_module._build_lineage_graph_store_for(_Collection())
    assert seen == ["resolve", "build:postgres"]
