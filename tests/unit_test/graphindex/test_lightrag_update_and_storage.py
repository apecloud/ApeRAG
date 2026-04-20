from types import SimpleNamespace

import pytest

from aperag.graph.lightrag.kg.pg_ops_sync_vector_storage import PGOpsSyncVectorStorage
from aperag.graph.lightrag.namespace import NameSpace
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
