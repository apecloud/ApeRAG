from types import SimpleNamespace

import pytest

from aperag.graph.lightrag import utils_graph
from aperag.graph.lightrag.base import QueryParam
from aperag.graph.lightrag.kg.pg_ops_sync_vector_storage import PGOpsSyncVectorStorage
from aperag.graph.lightrag.lightrag import LightRAG
from aperag.graph.lightrag.namespace import NameSpace
from aperag.graph.lightrag.operate import (
    _find_most_related_edges_from_entities,
    get_high_degree_nodes,
    merge_nodes_and_edges,
)
from aperag.graph.lightrag.prompt import GRAPH_FIELD_SEP
from aperag.graph.lightrag.utils import compute_mdhash_id
from aperag.graph.lightrag_manager import _process_document_async


class _FakeRag:
    def __init__(self):
        self.calls = []

    async def adelete_by_doc_id(self, doc_id: str):
        self.calls.append(("delete", doc_id))

    async def ainsert_and_chunk_document(self, documents, doc_ids, file_paths):
        self.calls.append(("insert", list(doc_ids)))
        return {
            "results": [
                {
                    "doc_id": doc_ids[0],
                    "chunks_data": {
                        "chunk-1": {
                            "content": documents[0],
                            "full_doc_id": doc_ids[0],
                            "file_path": file_paths[0],
                        }
                    },
                    "chunk_count": 1,
                }
            ]
        }

    async def aprocess_graph_indexing(self, chunks, collection_id=None):
        self.calls.append(("graph", sorted(chunks.keys()), collection_id))
        return {"entities_extracted": 2, "relations_extracted": 1, "groups_processed": 1}

    async def finalize_storages(self):
        self.calls.append(("finalize",))


class _FailingDeleteRag(_FakeRag):
    async def adelete_by_doc_id(self, doc_id: str):
        self.calls.append(("delete", doc_id))
        raise RuntimeError("delete failed")


class _FailingGraphIndexRag(_FakeRag):
    async def aprocess_graph_indexing(self, chunks, collection_id=None):
        self.calls.append(("graph", sorted(chunks.keys()), collection_id))
        raise RuntimeError("graph rebuild failed")


@pytest.mark.asyncio
async def test_process_document_async_deletes_existing_state_before_rebuild(monkeypatch):
    fake_rag = _FakeRag()

    async def _fake_create_lightrag_instance(collection):
        return fake_rag

    monkeypatch.setattr(
        "aperag.graph.lightrag_manager.create_lightrag_instance",
        _fake_create_lightrag_instance,
    )

    collection = SimpleNamespace(id="collection-1")
    result = await _process_document_async(collection, "new content", "doc-1", "/tmp/doc.txt")

    assert result["status"] == "success"
    assert result["chunks_created"] == 1
    assert fake_rag.calls[:3] == [
        ("delete", "doc-1"),
        ("insert", ["doc-1"]),
        ("graph", ["chunk-1"], "collection-1"),
    ]
    assert fake_rag.calls[-1] == ("finalize",)


@pytest.mark.asyncio
async def test_process_document_async_stops_when_delete_preflight_fails_and_finalizes(monkeypatch):
    fake_rag = _FailingDeleteRag()

    async def _fake_create_lightrag_instance(collection):
        return fake_rag

    monkeypatch.setattr(
        "aperag.graph.lightrag_manager.create_lightrag_instance",
        _fake_create_lightrag_instance,
    )

    collection = SimpleNamespace(id="collection-1")
    with pytest.raises(RuntimeError, match="delete failed"):
        await _process_document_async(collection, "new content", "doc-1", "/tmp/doc.txt")

    assert fake_rag.calls == [
        ("delete", "doc-1"),
        ("finalize",),
    ]


@pytest.mark.asyncio
async def test_process_document_async_propagates_graph_rebuild_failure_and_finalizes(monkeypatch):
    fake_rag = _FailingGraphIndexRag()

    async def _fake_create_lightrag_instance(collection):
        return fake_rag

    monkeypatch.setattr(
        "aperag.graph.lightrag_manager.create_lightrag_instance",
        _fake_create_lightrag_instance,
    )

    collection = SimpleNamespace(id="collection-1")
    with pytest.raises(RuntimeError, match="graph rebuild failed"):
        await _process_document_async(collection, "new content", "doc-1", "/tmp/doc.txt")

    assert fake_rag.calls[:3] == [
        ("delete", "doc-1"),
        ("insert", ["doc-1"]),
        ("graph", ["chunk-1"], "collection-1"),
    ]
    assert fake_rag.calls[-1] == ("finalize",)


@pytest.mark.asyncio
async def test_pg_vector_upsert_reuses_existing_vector_for_metadata_only_update(monkeypatch):
    captured = {}

    async def _embedding_func(_batch):
        raise AssertionError("embedding_func should not be called for metadata-only updates")

    def _capture_upsert(_workspace, vector_data):
        captured["vector_data"] = vector_data

    monkeypatch.setattr(
        "aperag.db.ops.db_ops.upsert_lightrag_vdb_entity",
        _capture_upsert,
    )

    storage = PGOpsSyncVectorStorage(
        namespace=NameSpace.VECTOR_STORE_ENTITIES,
        workspace="workspace-1",
        embedding_func=_embedding_func,
    )

    await storage.upsert(
        {
            "entity-1": {
                "entity_name": "Alpha",
                "content": "alpha content",
                "content_vector": [0.1, 0.2, 0.3],
                "chunk_ids": ["chunk-a", "chunk-b"],
                "file_path": "/tmp/a.txt",
            }
        }
    )

    assert captured["vector_data"]["entity-1"]["content_vector"] == [0.1, 0.2, 0.3]
    assert captured["vector_data"]["entity-1"]["chunk_ids"] == ["chunk-a", "chunk-b"]


@pytest.mark.asyncio
async def test_pg_vector_upsert_normalizes_chunk_ids_from_legacy_source_id(monkeypatch):
    captured = {}

    def _capture_upsert(_workspace, vector_data):
        captured["vector_data"] = vector_data

    monkeypatch.setattr(
        "aperag.db.ops.db_ops.upsert_lightrag_vdb_relation",
        _capture_upsert,
    )

    storage = PGOpsSyncVectorStorage(
        namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
        workspace="workspace-1",
        embedding_func=lambda _batch: None,
    )

    await storage.upsert(
        {
            "rel-1": {
                "src_id": "Alpha",
                "tgt_id": "Beta",
                "content": "Alpha\tBeta\nworks_with\nrelation desc",
                "content_vector": [0.1, 0.2, 0.3],
                "source_id": f"chunk-b{GRAPH_FIELD_SEP}chunk-a{GRAPH_FIELD_SEP}chunk-b",
                "file_path": "/tmp/rel.txt",
            }
        }
    )

    assert captured["vector_data"]["rel-1"]["chunk_ids"] == ["chunk-a", "chunk-b"]


@pytest.mark.asyncio
async def test_merge_nodes_and_edges_upserts_explicit_chunk_ids_to_vector_storage():
    graph_storage = _FakeGraphStorage()
    entities_vdb = _FakeEntityVectorStorage()
    relationships_vdb = _FakeRelationVectorStorage()

    chunk_results = [
        (
            {
                "Alpha": [
                    {
                        "entity_name": "Alpha",
                        "entity_type": "ORG",
                        "description": "alpha desc",
                        "source_id": f"chunk-b{GRAPH_FIELD_SEP}chunk-a",
                        "chunk_ids": ["chunk-b", "chunk-a"],
                        "file_path": "/tmp/alpha.txt",
                    }
                ]
            },
            {
                ("Alpha", "Beta"): [
                    {
                        "src_id": "Alpha",
                        "tgt_id": "Beta",
                        "description": "relation desc",
                        "keywords": "works_with",
                        "weight": 1.0,
                        "source_id": f"chunk-b{GRAPH_FIELD_SEP}chunk-a",
                        "chunk_ids": ["chunk-b", "chunk-a"],
                        "file_path": "/tmp/alpha.txt",
                    }
                ]
            },
        )
    ]

    result = await merge_nodes_and_edges(
        chunk_results=chunk_results,
        component=["Alpha", "Beta"],
        workspace="workspace-1",
        knowledge_graph_inst=graph_storage,
        entity_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        llm_model_func=lambda *_args, **_kwargs: None,
        tokenizer=_FakeTokenizer(),
        llm_model_max_token_size=2048,
        summary_to_max_tokens=256,
        language="zh-CN",
        force_llm_summary_on_merge=99,
        lightrag_logger=_FakeLogger(),
    )

    entity_id = compute_mdhash_id("Alpha", prefix="ent-", workspace="workspace-1")
    rel_id = compute_mdhash_id("AlphaBeta", prefix="rel-", workspace="workspace-1")

    assert result == {"entity_count": 1, "relation_count": 1}
    assert entities_vdb.upserts[0][entity_id]["chunk_ids"] == ["chunk-a", "chunk-b"]
    assert relationships_vdb.upserts[0][rel_id]["chunk_ids"] == ["chunk-a", "chunk-b"]


class _FakeGraphStorage:
    def __init__(self, nodes=None, edges_with_data_by_node=None):
        self.nodes = dict(nodes or {})
        self.edges_with_data_by_node = {
            node_id: list(edges) for node_id, edges in (edges_with_data_by_node or {}).items()
        }
        self.calls = []
        self.upserted_nodes = []
        self.upserted_edges = []
        self.deleted_nodes = []
        self.removed_nodes_batches = []
        self.removed_edges_batches = []

    async def get_nodes_batch(self, node_ids):
        self.calls.append(("get_nodes_batch", tuple(node_ids)))
        return {node_id: self.nodes[node_id] for node_id in node_ids if node_id in self.nodes}

    async def get_nodes_edges_with_data_batch(self, node_ids):
        self.calls.append(("get_nodes_edges_with_data_batch", tuple(node_ids)))
        return {node_id: list(self.edges_with_data_by_node.get(node_id, [])) for node_id in node_ids}

    async def get_incident_edges_with_data_batch(self, node_ids):
        self.calls.append(("get_incident_edges_with_data_batch", tuple(node_ids)))
        return await self.get_nodes_edges_with_data_batch(node_ids)

    async def get_nodes_edges_batch(self, node_ids):
        self.calls.append(("get_nodes_edges_batch", tuple(node_ids)))
        return {
            node_id: [(source, target) for source, target, _edge_data in self.edges_with_data_by_node.get(node_id, [])]
            for node_id in node_ids
        }

    async def get_edges_batch(self, edge_pairs):
        self.calls.append(("get_edges_batch", tuple((edge["src"], edge["tgt"]) for edge in edge_pairs)))
        all_edges = {}
        for edges in self.edges_with_data_by_node.values():
            for edge_source, edge_target, edge_data in edges:
                all_edges[(edge_source, edge_target)] = dict(edge_data)
        return {
            (edge["src"], edge["tgt"]): all_edges[(edge["src"], edge["tgt"])]
            for edge in edge_pairs
            if (edge["src"], edge["tgt"]) in all_edges
        }

    async def upsert_node(self, node_id, node_data):
        self.calls.append(("upsert_node", node_id))
        self.nodes[node_id] = dict(node_data)
        self.upserted_nodes.append((node_id, dict(node_data)))

    async def upsert_edge(self, source, target, edge_data):
        self.calls.append(("upsert_edge", source, target))
        self.upserted_edges.append((source, target, dict(edge_data)))

    async def delete_node(self, node_id):
        self.calls.append(("delete_node", node_id))
        self.deleted_nodes.append(node_id)
        self.nodes.pop(node_id, None)

    async def remove_nodes(self, node_ids):
        self.calls.append(("remove_nodes", tuple(node_ids)))
        self.removed_nodes_batches.append(list(node_ids))
        for node_id in node_ids:
            self.nodes.pop(node_id, None)

    async def remove_edges(self, edges):
        self.calls.append(("remove_edges", tuple(edges)))
        self.removed_edges_batches.append(list(edges))

    async def get_node(self, node_id):
        self.calls.append(("get_node", node_id))
        return self.nodes.get(node_id)

    async def get_edge(self, source, target):
        self.calls.append(("get_edge", source, target))
        for edges in self.edges_with_data_by_node.values():
            for edge_source, edge_target, edge_data in edges:
                if (edge_source, edge_target) == (source, target):
                    return dict(edge_data)
        return None

    async def has_node(self, _node_id):
        raise AssertionError("legacy has_node() should not be used")

    async def has_edge(self, _source, _target):
        raise AssertionError("legacy has_edge() should not be used")

    async def get_node_edges(self, _node_id):
        raise AssertionError("legacy get_node_edges() should not be used")

    async def get_top_degree_nodes(self, _limit):
        return None

    async def get_node_ids(self, limit=None):
        self.calls.append(("get_node_ids", limit))
        return list(self.nodes.keys())[:limit] if limit is not None else list(self.nodes.keys())

    async def node_degrees_batch(self, node_ids):
        self.calls.append(("node_degrees_batch", tuple(node_ids)))
        degrees = {node_id: 0 for node_id in node_ids}
        for edges in self.edges_with_data_by_node.values():
            for edge_source, edge_target, _edge_data in edges:
                if edge_source in degrees:
                    degrees[edge_source] += 1
                if edge_target in degrees:
                    degrees[edge_target] += 1
        return degrees


class _FakeEntityVectorStorage:
    def __init__(self):
        self.workspace = "workspace-1"
        self.deleted = []
        self.deleted_entities = []
        self.upserts = []

    async def delete(self, ids):
        self.deleted.append(list(ids))

    async def delete_entity(self, entity_name):
        self.deleted_entities.append(entity_name)

    async def upsert(self, data):
        self.upserts.append(data)

    async def get_by_id(self, entity_id):
        return {"id": entity_id}


class _FakeRelationVectorStorage:
    def __init__(self):
        self.workspace = "workspace-1"
        self.deleted = []
        self.upserts = []

    async def delete(self, ids):
        self.deleted.append(list(ids))

    async def upsert(self, data):
        self.upserts.append(data)

    async def get_by_id(self, relation_id):
        return {"id": relation_id}


class _FakeAsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_aedit_entity_rename_uses_batch_graph_primitives():
    graph_storage = _FakeGraphStorage(
        nodes={
            "old": {
                "entity_id": "old",
                "entity_type": "PERSON",
                "description": "old desc",
                "source_id": "chunk-1",
            },
            "neighbor": {
                "entity_id": "neighbor",
                "entity_type": "ORG",
                "description": "neighbor desc",
                "source_id": "chunk-2",
            },
        },
        edges_with_data_by_node={
            "old": [
                ("old", "neighbor", {"description": "works at", "keywords": "job", "source_id": "chunk-1"}),
            ],
        },
    )
    entities_vdb = _FakeEntityVectorStorage()
    relationships_vdb = _FakeRelationVectorStorage()

    result = await utils_graph.aedit_entity(
        graph_storage,
        entities_vdb,
        relationships_vdb,
        "old",
        {"entity_name": "new", "description": "new desc"},
        graph_db_lock=_FakeAsyncLock(),
    )

    assert ("get_nodes_batch", ("old", "new")) in graph_storage.calls
    assert ("get_incident_edges_with_data_batch", ("old",)) in graph_storage.calls
    assert ("delete_node", "old") in graph_storage.calls
    assert any(source == "new" and target == "neighbor" for source, target, _ in graph_storage.upserted_edges)
    assert entities_vdb.deleted
    assert relationships_vdb.deleted
    assert result["entity_name"] == "new"


@pytest.mark.asyncio
async def test_amerge_entities_routes_through_canonical_lightrag_merge(monkeypatch):
    graph_storage = _FakeGraphStorage(
        nodes={
            "A": {"entity_id": "A", "entity_type": "PERSON", "description": "desc A", "source_id": "chunk-a"},
            "B": {"entity_id": "B", "entity_type": "PERSON", "description": "desc B", "source_id": "chunk-b"},
        },
    )
    entities_vdb = _FakeEntityVectorStorage()
    relationships_vdb = _FakeRelationVectorStorage()
    captured = {}

    async def _fake_amerge_nodes(self, entity_ids, target_entity_data=None):
        captured["self"] = self
        captured["entity_ids"] = entity_ids
        captured["target_entity_data"] = target_entity_data
        return {
            "status": "success",
            "message": "canonical merge invoked",
            "target_entity_data": target_entity_data,
            "source_entities": entity_ids,
        }

    monkeypatch.setattr(LightRAG, "amerge_nodes", _fake_amerge_nodes)

    result = await utils_graph.amerge_entities(
        graph_storage,
        entities_vdb,
        relationships_vdb,
        ["A", "B"],
        "merged",
        graph_db_lock=_FakeAsyncLock(),
    )

    assert result["status"] == "success"
    assert captured["entity_ids"] == ["A", "B"]
    assert captured["target_entity_data"] == {"entity_name": "merged"}
    assert captured["self"].workspace == "workspace-1"
    assert captured["self"].chunk_entity_relation_graph is graph_storage
    assert captured["self"].entities_vdb is entities_vdb
    assert captured["self"].relationships_vdb is relationships_vdb
    assert graph_storage.calls == []
    assert entities_vdb.deleted == []
    assert entities_vdb.deleted_entities == []
    assert entities_vdb.upserts == []
    assert relationships_vdb.deleted == []
    assert relationships_vdb.upserts == []


@pytest.mark.asyncio
async def test_amerge_entities_rejects_custom_merge_strategy():
    graph_storage = _FakeGraphStorage(
        nodes={
            "A": {"entity_id": "A", "entity_type": "PERSON", "description": "desc A", "source_id": "chunk-a"},
            "B": {"entity_id": "B", "entity_type": "PERSON", "description": "desc B", "source_id": "chunk-b"},
        }
    )
    entities_vdb = _FakeEntityVectorStorage()
    relationships_vdb = _FakeRelationVectorStorage()

    with pytest.raises(ValueError, match="Custom merge_strategy is no longer supported"):
        await utils_graph.amerge_entities(
            graph_storage,
            entities_vdb,
            relationships_vdb,
            ["A", "B"],
            "merged",
            merge_strategy={"description": "keep_last"},
            graph_db_lock=_FakeAsyncLock(),
        )

    assert graph_storage.calls == []
    assert entities_vdb.deleted == []
    assert entities_vdb.deleted_entities == []
    assert entities_vdb.upserts == []
    assert relationships_vdb.deleted == []
    assert relationships_vdb.upserts == []


@pytest.mark.asyncio
async def test_amerge_entities_propagates_canonical_merge_failures(monkeypatch):
    graph_storage = _FakeGraphStorage(
        nodes={
            "A": {"entity_id": "A", "entity_type": "PERSON", "description": "desc A", "source_id": "chunk-a"},
            "B": {"entity_id": "B", "entity_type": "PERSON", "description": "desc B", "source_id": "chunk-b"},
        }
    )
    entities_vdb = _FakeEntityVectorStorage()
    relationships_vdb = _FakeRelationVectorStorage()

    async def _failing_amerge_nodes(self, entity_ids, target_entity_data=None):
        raise RuntimeError("canonical merge failed")

    monkeypatch.setattr(LightRAG, "amerge_nodes", _failing_amerge_nodes)

    with pytest.raises(RuntimeError, match="canonical merge failed"):
        await utils_graph.amerge_entities(
            graph_storage,
            entities_vdb,
            relationships_vdb,
            ["A", "B"],
            "merged",
            graph_db_lock=_FakeAsyncLock(),
        )

    assert graph_storage.calls == []
    assert entities_vdb.deleted == []
    assert entities_vdb.deleted_entities == []
    assert entities_vdb.upserts == []
    assert relationships_vdb.deleted == []
    assert relationships_vdb.upserts == []


@pytest.mark.asyncio
async def test_amerge_nodes_upserts_explicit_chunk_ids_to_vector_storage():
    rag = LightRAG.__new__(LightRAG)
    rag.workspace = "workspace-1"
    rag.lightrag_logger = _FakeLogger()
    rag.chunk_entity_relation_graph = _FakeGraphStorage(
        nodes={
            "A": {"entity_id": "A", "entity_type": "PERSON", "description": "desc A", "source_id": "chunk-b"},
            "B": {"entity_id": "B", "entity_type": "PERSON", "description": "desc B", "source_id": "chunk-a"},
            "X": {"entity_id": "X", "entity_type": "ORG", "description": "desc X", "source_id": "chunk-x"},
            "Y": {"entity_id": "Y", "entity_type": "ORG", "description": "desc Y", "source_id": "chunk-y"},
        },
        edges_with_data_by_node={
            "A": [("A", "X", {"description": "edge ax", "keywords": "k1", "source_id": "chunk-b", "weight": 1})],
            "B": [("B", "Y", {"description": "edge by", "keywords": "k2", "source_id": "chunk-a", "weight": 2})],
        },
    )
    rag.entities_vdb = _FakeEntityVectorStorage()
    rag.relationships_vdb = _FakeRelationVectorStorage()

    result = await LightRAG.amerge_nodes(
        rag,
        ["A", "B"],
        {"entity_name": "merged"},
    )

    merged_entity_id = compute_mdhash_id("merged", prefix="ent-", workspace=rag.workspace)

    assert result["status"] == "success"
    assert rag.entities_vdb.upserts[0][merged_entity_id]["chunk_ids"] == ["chunk-a", "chunk-b"]
    assert all(
        relation_data["chunk_ids"]
        for upsert_batch in rag.relationships_vdb.upserts
        for relation_data in upsert_batch.values()
    )


class _FakeHighDegreeGraphStorage:
    async def get_top_degree_nodes(self, limit):
        assert limit == 2
        return (
            {
                "entity-1": {
                    "entity_id": "entity-1",
                    "entity_type": "ORG",
                    "description": "node one",
                    "source_id": "chunk-1",
                    "degree": 7,
                },
                "entity-2": {
                    "entity_id": "entity-2",
                    "entity_type": "PERSON",
                    "description": "node two",
                    "source_id": "chunk-2",
                    "degree": 5,
                },
            },
            42,
        )

    async def get_all_labels(self):
        raise AssertionError("get_all_labels() should not be used when top-degree primitive is available")


@pytest.mark.asyncio
async def test_get_high_degree_nodes_prefers_storage_top_degree_primitive():
    selected_nodes, total_nodes = await get_high_degree_nodes(_FakeHighDegreeGraphStorage(), max_analyze_nodes=2)

    assert total_nodes == 42
    assert set(selected_nodes.nodes_by_id.keys()) == {"entity-1", "entity-2"}
    assert selected_nodes.nodes_by_id["entity-1"].degree == 7


class _FakeExportGraphStorage:
    def __init__(self):
        self.calls = []

    async def get_node_ids(self, limit=None):
        self.calls.append(("get_node_ids", limit))
        return ["entity-1", "entity-2"][:limit] if limit is not None else ["entity-1", "entity-2"]

    async def get_all_labels(self):
        raise AssertionError("get_all_labels() should not be used when get_node_ids() is available")

    async def get_nodes_batch(self, node_ids):
        self.calls.append(("get_nodes_batch", tuple(node_ids)))
        return {
            "entity-1": {
                "entity_id": "entity-1",
                "entity_name": "Alpha",
                "entity_type": "ORG",
                "description": "alpha desc",
                "source_id": "chunk-1",
            },
            "entity-2": {
                "entity_id": "entity-2",
                "entity_name": "Beta",
                "entity_type": "PERSON",
                "description": "beta desc",
                "source_id": "chunk-2",
            },
        }

    async def get_incident_edges_with_data_batch(self, node_ids):
        self.calls.append(("get_incident_edges_with_data_batch", tuple(node_ids)))
        return {
            "entity-1": [
                (
                    "entity-1",
                    "entity-2",
                    {"description": "rel", "keywords": "kw", "weight": 3.0, "source_id": "chunk-1"},
                )
            ],
            "entity-2": [
                (
                    "entity-1",
                    "entity-2",
                    {"description": "rel", "keywords": "kw", "weight": 3.0, "source_id": "chunk-1"},
                )
            ],
        }


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        return None

    def debug(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def log_entity_merge(self, *_args, **_kwargs):
        return None

    def log_relation_merge(self, *_args, **_kwargs):
        return None


class _FakeTokenizer:
    def encode(self, text):
        return list(text or "")


@pytest.mark.asyncio
async def test_export_for_kg_eval_prefers_node_ids_and_incident_edges_primitive():
    rag = LightRAG.__new__(LightRAG)
    rag.workspace = "workspace-1"
    rag.lightrag_logger = _FakeLogger()
    rag.chunk_entity_relation_graph = _FakeExportGraphStorage()

    result = await LightRAG.export_for_kg_eval(rag, sample_size=2, include_source_texts=False)

    assert result["entities"][0]["entity_name"] == "Alpha"
    assert result["relationships"][0]["source_entity_name"] == "Alpha"
    assert ("get_node_ids", 2) in rag.chunk_entity_relation_graph.calls
    assert ("get_incident_edges_with_data_batch", ("entity-1", "entity-2")) in rag.chunk_entity_relation_graph.calls


class _FakeExportGraphStorageWithoutNodeIds:
    async def get_node_ids(self, limit=None):
        return None

    async def get_all_labels(self):
        raise AssertionError("get_all_labels() should no longer be used as silent fallback for export")


@pytest.mark.asyncio
async def test_export_for_kg_eval_raises_when_bounded_node_id_sampling_is_missing():
    rag = LightRAG.__new__(LightRAG)
    rag.workspace = "workspace-1"
    rag.lightrag_logger = _FakeLogger()
    rag.chunk_entity_relation_graph = _FakeExportGraphStorageWithoutNodeIds()

    with pytest.raises(NotImplementedError, match="bounded node-id sampling"):
        await LightRAG.export_for_kg_eval(rag, sample_size=2, include_source_texts=False)


class _FakeOperateGraphStorage:
    def __init__(self):
        self.calls = []

    async def get_incident_edges_with_data_batch(self, node_ids):
        self.calls.append(("get_incident_edges_with_data_batch", tuple(node_ids)))
        return {
            "entity-1": [
                (
                    "entity-1",
                    "entity-2",
                    {"description": "edge desc", "keywords": "kw", "weight": 2.0, "source_id": "chunk-1"},
                )
            ]
        }

    async def edge_degrees_batch(self, edge_pairs):
        self.calls.append(("edge_degrees_batch", tuple(edge_pairs)))
        return {("entity-1", "entity-2"): 9}

    async def get_nodes_edges_batch(self, _node_ids):
        raise AssertionError("legacy get_nodes_edges_batch() should not be used")

    async def get_edges_batch(self, _pairs):
        raise AssertionError("legacy get_edges_batch() should not be used")


@pytest.mark.asyncio
async def test_find_most_related_edges_from_entities_prefers_incident_edge_primitive():
    graph_storage = _FakeOperateGraphStorage()
    node_datas = [{"entity_name": "entity-1"}]

    result = await _find_most_related_edges_from_entities(
        node_datas,
        QueryParam(max_token_for_global_context=1000),
        graph_storage,
        _FakeTokenizer(),
    )

    assert result[0]["src_tgt"] == ("entity-1", "entity-2")
    assert result[0]["rank"] == 9
    assert ("get_incident_edges_with_data_batch", ("entity-1",)) in graph_storage.calls


class _FakeDeleteTextChunks:
    async def get_by_doc_id(self, _doc_id):
        return {
            "chunk-1": {
                "full_doc_id": "doc-1",
                "content": "chunk content",
            }
        }

    async def delete(self, _ids):
        return None


class _FakeDeleteVectorStorage:
    async def get_by_chunk_ids(self, _chunk_ids):
        return {}

    async def upsert(self, _data):
        return None

    async def delete(self, _ids):
        return None

    async def delete_entity(self, _entity_name):
        return None


class _FakeDeleteGraphStorageWithoutSourcePrimitive:
    async def get_nodes_by_source_ids(self, _chunk_ids):
        return None

    async def get_edges_by_source_ids(self, _chunk_ids):
        return None


@pytest.mark.asyncio
async def test_adelete_by_doc_id_raises_when_bounded_source_lookup_is_missing():
    rag = LightRAG.__new__(LightRAG)
    rag.workspace = "workspace-1"
    rag.lightrag_logger = _FakeLogger()
    rag.text_chunks = _FakeDeleteTextChunks()
    rag.entities_vdb = _FakeDeleteVectorStorage()
    rag.relationships_vdb = _FakeDeleteVectorStorage()
    rag.chunk_entity_relation_graph = _FakeDeleteGraphStorageWithoutSourcePrimitive()

    with pytest.raises(NotImplementedError, match="bounded source-id graph lookups"):
        await LightRAG.adelete_by_doc_id(rag, "doc-1")
