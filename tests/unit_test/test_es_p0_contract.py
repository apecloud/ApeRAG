import json
import logging
from types import SimpleNamespace

import pytest

from aperag.db.models import CollectionStatus, DocumentIndexType
from aperag.domains.retrieval.pipeline import SearchPipelineService
from aperag.query.query import DocumentWithScore
from aperag.service.document_service import DocumentService
from aperag.tasks.collection import CollectionTask
from aperag.tasks.document import DocumentIndexTask


def _collection_config(enable_fulltext=True):
    return json.dumps(
        {
            "source": "system",
            "enable_vector": True,
            "enable_fulltext": enable_fulltext,
            "enable_knowledge_graph": False,
            "enable_summary": False,
            "enable_vision": False,
        }
    )


def test_document_service_respects_enable_fulltext():
    service = DocumentService()

    enabled = service._get_index_types_for_collection(json.loads(_collection_config(True)))
    disabled = service._get_index_types_for_collection(json.loads(_collection_config(False)))

    assert DocumentIndexType.FULLTEXT in enabled
    assert DocumentIndexType.FULLTEXT not in disabled


def test_collection_task_skips_fulltext_init_when_disabled(monkeypatch):
    collection = SimpleNamespace(id="col-1", config=_collection_config(False), status=CollectionStatus.ACTIVE)
    called = []

    monkeypatch.setattr("aperag.tasks.collection.db_ops.query_collection_by_id", lambda *_args, **_kwargs: collection)
    monkeypatch.setattr("aperag.tasks.collection.db_ops.update_collection", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(CollectionTask, "_initialize_vector_databases", lambda self, *_args, **_kwargs: None)
    monkeypatch.setattr(
        CollectionTask, "_initialize_fulltext_index", lambda self, collection_id: called.append(collection_id)
    )

    result = CollectionTask().initialize_collection("col-1", 1)

    assert result.success is True
    assert called == []


def test_document_index_task_skips_fulltext_create_when_disabled(monkeypatch):
    collection = SimpleNamespace(id="col-1", config=_collection_config(False))
    parsed = SimpleNamespace(content="content", doc_parts=[], file_path="/tmp/doc.txt")

    monkeypatch.setattr(
        "aperag.tasks.utils.get_document_and_collection",
        lambda *_args, **_kwargs: (SimpleNamespace(id="doc-1"), collection),
    )
    monkeypatch.setattr(
        "aperag.domains.indexing.fulltext_index.fulltext_indexer.create_index",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("fulltext create_index should not be called")),
    )

    result = DocumentIndexTask().create_index("doc-1", DocumentIndexType.FULLTEXT.value, parsed)

    assert result.success is True
    assert result.data["message"] == "Fulltext indexing disabled"


@pytest.mark.asyncio
async def test_fulltext_search_uses_fulltext_helper_and_query_fallback(monkeypatch):
    captured = {}

    async def fake_extract_keywords(*_args, **_kwargs):
        return []

    async def fake_search_document(index_name, collection_id, keywords, topk, chat_id=None):
        captured["index_name"] = index_name
        captured["collection_id"] = collection_id
        captured["keywords"] = keywords
        captured["topk"] = topk
        captured["chat_id"] = chat_id
        return [DocumentWithScore(text="doc", score=1.0, metadata={})]

    monkeypatch.setattr("aperag.domains.retrieval.pipeline.extract_keywords", fake_extract_keywords)
    monkeypatch.setattr("aperag.domains.retrieval.pipeline.generate_fulltext_index_name", lambda cid: f"ft-{cid}")
    monkeypatch.setattr("aperag.domains.indexing.fulltext_index.fulltext_indexer.search_document", fake_search_document)

    service = SearchPipelineService()
    collection = SimpleNamespace(id="col-1", config=_collection_config(True))

    docs = await service._fulltext_search(collection, "中文问题", 2, None, "user-1", chat_id="chat-1")

    assert captured == {
        "index_name": "ft-col-1",
        "collection_id": "col-1",
        "keywords": ["中文问题"],
        "topk": 6,
        "chat_id": "chat-1",
    }
    assert docs[0].metadata["recall_type"] == "fulltext_search"


@pytest.mark.asyncio
async def test_fulltext_search_logs_explicit_degrade_on_backend_failure(monkeypatch, caplog):
    from aperag.domains.indexing.fulltext_index import FulltextSearchDegradedError

    async def fake_extract_keywords(*_args, **_kwargs):
        return ["x"]

    async def fake_search_document(*_args, **_kwargs):
        raise FulltextSearchDegradedError("boom")

    monkeypatch.setattr("aperag.domains.retrieval.pipeline.extract_keywords", fake_extract_keywords)
    monkeypatch.setattr("aperag.domains.indexing.fulltext_index.fulltext_indexer.search_document", fake_search_document)

    service = SearchPipelineService()
    collection = SimpleNamespace(id="col-1", config=_collection_config(True))

    with caplog.at_level(logging.WARNING):
        docs = await service._fulltext_search(collection, "q", 1, ["x"], "user-1", chat_id=None)

    assert docs == []
    assert "Fulltext search degraded for collection col-1" in caplog.text


@pytest.mark.asyncio
async def test_fulltext_index_search_uses_dual_read_chat_filter(monkeypatch):
    captured = {}

    class FakeAsyncIndices:
        async def exists(self, index):
            return SimpleNamespace(body=True)

    class FakeAsyncEs:
        def __init__(self):
            self.indices = FakeAsyncIndices()

        async def search(self, index, query, sort, size, routing):
            captured["index"] = index
            captured["query"] = query
            captured["sort"] = sort
            captured["size"] = size
            captured["routing"] = routing
            return SimpleNamespace(body={"hits": {"hits": []}})

    from aperag.domains.indexing.fulltext_index import FulltextIndexer

    indexer = object.__new__(FulltextIndexer)
    indexer.async_es = FakeAsyncEs()

    docs = await FulltextIndexer.search_document(indexer, "ft-col-1", "col-1", ["hello"], topk=3, chat_id="chat-1")

    assert docs == []
    assert captured["query"]["bool"]["filter"] == [
        {"term": {"collection_id": "col-1"}},
        {
            "bool": {
                "should": [
                    {"term": {"chat_id": "chat-1"}},
                    {"term": {"metadata.chat_id": "chat-1"}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]
    assert captured["routing"] == "col-1"


def test_create_index_mapping_exposes_explicit_filter_fields(monkeypatch):
    captured = {}

    class FakeIndices:
        def exists(self, index):
            return SimpleNamespace(body=False)

        def exists_alias(self, name):
            return SimpleNamespace(body=False)

        def get_alias(self, name):
            return SimpleNamespace(body={})

        def put_alias(self, index, name):
            captured["alias"] = (index, name)

        def create(self, index, body):
            captured["index"] = index
            captured["body"] = body

    class FakeElasticsearch:
        def __init__(self, *_args, **_kwargs):
            self.indices = FakeIndices()

    monkeypatch.setattr("aperag.domains.indexing.fulltext_index.Elasticsearch", FakeElasticsearch)

    from aperag.domains.indexing.fulltext_index import create_index

    create_index("ft-col-1")

    props = captured["body"]["mappings"]["properties"]
    assert captured["alias"] == ("aperag-fulltext-v1", "ft-col-1")
    assert props["collection_id"]["type"] == "keyword"
    assert props["document_id"]["type"] == "keyword"
    assert props["chunk_id"]["type"] == "keyword"
    assert props["chat_id"]["type"] == "keyword"
