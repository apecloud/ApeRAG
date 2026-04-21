import pytest

from aperag.graph.lightrag.lightrag import LightRAG
from aperag.graph.lightrag.prompt import GRAPH_FIELD_SEP
from aperag.graph.lightrag.utils import compute_mdhash_id


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        return None

    def debug(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class _FakeTokenizer:
    def encode(self, text):
        return list(text or "")


class _FakeChunkStorage:
    def __init__(self, initial_records=None):
        self.records = {key: dict(value) for key, value in (initial_records or {}).items()}
        self.upsert_calls = []
        self.deleted_ids = []

    async def upsert(self, data):
        snapshot = {key: dict(value) for key, value in data.items()}
        self.upsert_calls.append(snapshot)
        self.records.update(snapshot)

    async def delete(self, ids):
        self.deleted_ids.append(list(ids))
        for record_id in ids:
            self.records.pop(record_id, None)

    async def get_by_doc_id(self, doc_id):
        return {
            record_id: dict(value)
            for record_id, value in self.records.items()
            if value.get("full_doc_id") == doc_id
        }


class _FakeEntityVectorStorage:
    def __init__(self, initial_records=None):
        self.records = {key: dict(value) for key, value in (initial_records or {}).items()}
        self.upsert_calls = []
        self.deleted_entities = []

    async def get_by_chunk_ids(self, chunk_ids):
        target_chunk_ids = set(chunk_ids)
        return {
            record_id: dict(value)
            for record_id, value in self.records.items()
            if set(value.get("chunk_ids", [])).intersection(target_chunk_ids)
        }

    async def upsert(self, data):
        snapshot = {key: dict(value) for key, value in data.items()}
        self.upsert_calls.append(snapshot)
        self.records.update(snapshot)

    async def delete_entity(self, entity_name):
        self.deleted_entities.append(entity_name)
        for record_id, value in list(self.records.items()):
            if value.get("entity_name") == entity_name:
                self.records.pop(record_id, None)


class _FakeRelationshipVectorStorage:
    def __init__(self, initial_records=None):
        self.records = {key: dict(value) for key, value in (initial_records or {}).items()}
        self.upsert_calls = []
        self.deleted_ids = []

    async def get_by_chunk_ids(self, chunk_ids):
        target_chunk_ids = set(chunk_ids)
        return {
            record_id: dict(value)
            for record_id, value in self.records.items()
            if set(value.get("chunk_ids", [])).intersection(target_chunk_ids)
        }

    async def upsert(self, data):
        snapshot = {key: dict(value) for key, value in data.items()}
        self.upsert_calls.append(snapshot)
        self.records.update(snapshot)

    async def delete(self, ids):
        self.deleted_ids.append(list(ids))
        for record_id in ids:
            self.records.pop(record_id, None)


class _FakeGraphStorage:
    def __init__(self, nodes=None, edges=None):
        self.nodes = {key: dict(value) for key, value in (nodes or {}).items()}
        self.edges = {key: dict(value) for key, value in (edges or {}).items()}
        self.upserted_nodes = []
        self.upserted_edges = []
        self.removed_nodes = []
        self.removed_edges = []

    async def get_nodes_by_source_ids(self, chunk_ids):
        target_chunk_ids = set(chunk_ids)
        return {
            node_id: dict(node_data)
            for node_id, node_data in self.nodes.items()
            if set(node_data.get("source_id", "").split(GRAPH_FIELD_SEP)).intersection(target_chunk_ids)
        }

    async def get_edges_by_source_ids(self, chunk_ids):
        target_chunk_ids = set(chunk_ids)
        return {
            edge_pair: dict(edge_data)
            for edge_pair, edge_data in self.edges.items()
            if set(edge_data.get("source_id", "").split(GRAPH_FIELD_SEP)).intersection(target_chunk_ids)
        }

    async def upsert_node(self, node_id, node_data):
        snapshot = dict(node_data)
        self.upserted_nodes.append((node_id, snapshot))
        self.nodes[node_id] = snapshot

    async def upsert_edge(self, src, tgt, edge_data):
        snapshot = dict(edge_data)
        self.upserted_edges.append(((src, tgt), snapshot))
        self.edges[(src, tgt)] = snapshot

    async def remove_nodes(self, node_ids):
        self.removed_nodes.append(list(node_ids))
        for node_id in node_ids:
            self.nodes.pop(node_id, None)

    async def remove_edges(self, edge_pairs):
        self.removed_edges.append(list(edge_pairs))
        for edge_pair in edge_pairs:
            self.edges.pop(tuple(edge_pair), None)


def _build_rag_for_chunk_tests():
    rag = LightRAG.__new__(LightRAG)
    rag.workspace = "workspace-1"
    rag.lightrag_logger = _FakeLogger()
    rag.tokenizer = _FakeTokenizer()
    rag.chunk_overlap_token_size = 0
    rag.chunk_token_size = 64
    rag.chunking_func = lambda *_args, **_kwargs: [
        {
            "tokens": 1,
            "content": "same chunk content",
            "chunk_order_index": 0,
        }
    ]
    rag.chunks_vdb = _FakeChunkStorage()
    rag.text_chunks = _FakeChunkStorage()
    return rag


@pytest.mark.asyncio
async def test_ainsert_and_chunk_document_uses_document_scoped_chunk_ids_for_duplicate_content():
    rag = _build_rag_for_chunk_tests()

    result = await LightRAG.ainsert_and_chunk_document(
        rag,
        documents=["doc A", "doc B"],
        doc_ids=["doc-a", "doc-b"],
        file_paths=["/tmp/a.txt", "/tmp/b.txt"],
    )

    chunk_id_a = result["results"][0]["chunks"][0]
    chunk_id_b = result["results"][1]["chunks"][0]

    assert chunk_id_a != chunk_id_b
    assert result["results"][0]["chunks_data"][chunk_id_a]["full_doc_id"] == "doc-a"
    assert result["results"][1]["chunks_data"][chunk_id_b]["full_doc_id"] == "doc-b"


@pytest.mark.asyncio
async def test_ainsert_and_chunk_document_keeps_chunk_ids_stable_for_same_document_instance():
    rag = _build_rag_for_chunk_tests()

    first = await LightRAG.ainsert_and_chunk_document(
        rag,
        documents=["doc A"],
        doc_ids=["doc-a"],
        file_paths=["/tmp/a.txt"],
    )
    second = await LightRAG.ainsert_and_chunk_document(
        rag,
        documents=["doc A"],
        doc_ids=["doc-a"],
        file_paths=["/tmp/a.txt"],
    )

    assert first["results"][0]["chunks"] == second["results"][0]["chunks"]


@pytest.mark.asyncio
async def test_adelete_by_doc_id_updates_shared_refs_and_deletes_exclusive_refs():
    rag = LightRAG.__new__(LightRAG)
    rag.workspace = "workspace-1"
    rag.lightrag_logger = _FakeLogger()
    rag.text_chunks = _FakeChunkStorage(
        {
            "chunk-a": {"full_doc_id": "doc-1", "content": "alpha"},
            "chunk-b": {"full_doc_id": "doc-2", "content": "beta"},
        }
    )
    rag.chunks_vdb = _FakeChunkStorage(
        {
            "chunk-a": {"full_doc_id": "doc-1", "content": "alpha"},
            "chunk-b": {"full_doc_id": "doc-2", "content": "beta"},
        }
    )
    rag.entities_vdb = _FakeEntityVectorStorage(
        {
            "ent-shared": {
                "entity_name": "SharedEntity",
                "chunk_ids": ["chunk-a", "chunk-b"],
            },
            "ent-exclusive": {
                "entity_name": "ExclusiveEntity",
                "chunk_ids": ["chunk-a"],
            },
        }
    )
    rag.relationships_vdb = _FakeRelationshipVectorStorage(
        {
            "rel-shared": {
                "src_id": "node-a",
                "tgt_id": "node-b",
                "chunk_ids": ["chunk-a", "chunk-b"],
            },
            "rel-exclusive": {
                "src_id": "node-x",
                "tgt_id": "node-y",
                "chunk_ids": ["chunk-a"],
            },
        }
    )
    rag.chunk_entity_relation_graph = _FakeGraphStorage(
        nodes={
            "SharedNode": {"source_id": f"chunk-a{GRAPH_FIELD_SEP}chunk-b"},
            "ExclusiveNode": {"source_id": "chunk-a"},
        },
        edges={
            ("node-a", "node-b"): {"source_id": f"chunk-a{GRAPH_FIELD_SEP}chunk-b"},
            ("node-x", "node-y"): {"source_id": "chunk-a"},
        },
    )

    await LightRAG.adelete_by_doc_id(rag, "doc-1")

    assert rag.entities_vdb.upsert_calls == [
        {
            "ent-shared": {
                "entity_name": "SharedEntity",
                "chunk_ids": ["chunk-b"],
            }
        }
    ]
    assert rag.entities_vdb.deleted_entities == ["ExclusiveEntity"]

    assert rag.relationships_vdb.upsert_calls == [
        {
            "rel-shared": {
                "src_id": "node-a",
                "tgt_id": "node-b",
                "chunk_ids": ["chunk-b"],
            }
        }
    ]
    deleted_relation_ids = set(rag.relationships_vdb.deleted_ids[0])
    assert deleted_relation_ids == {
        compute_mdhash_id("node-xnode-y", prefix="rel-", workspace=rag.workspace),
        compute_mdhash_id("node-ynode-x", prefix="rel-", workspace=rag.workspace),
    }

    assert rag.chunk_entity_relation_graph.upserted_nodes == [
        ("SharedNode", {"source_id": "chunk-b"})
    ]
    assert rag.chunk_entity_relation_graph.removed_nodes == [["ExclusiveNode"]]
    assert rag.chunk_entity_relation_graph.upserted_edges == [
        (("node-a", "node-b"), {"source_id": "chunk-b"})
    ]
    assert rag.chunk_entity_relation_graph.removed_edges == [[("node-x", "node-y")]]

    assert rag.chunks_vdb.deleted_ids == [["chunk-a"]]
    assert rag.text_chunks.deleted_ids == [["chunk-a"]]
    assert await rag.text_chunks.get_by_doc_id("doc-1") == {}
