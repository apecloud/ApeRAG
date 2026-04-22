from types import SimpleNamespace

import pytest

from aperag.tasks.collection import CollectionTask
from aperag.utils.utils import (
    generate_fulltext_index_alias,
    generate_fulltext_index_name,
    generate_fulltext_physical_index_name,
    generate_legacy_fulltext_index_name,
)


def test_fulltext_name_helpers_use_shared_alias_and_legacy_names():
    assert generate_fulltext_index_alias() == "aperag-fulltext"
    assert generate_fulltext_index_name("col-1") == "aperag-fulltext"
    assert generate_fulltext_physical_index_name() == "aperag-fulltext-v1"
    assert generate_fulltext_physical_index_name("v2") == "aperag-fulltext-v2"
    assert generate_fulltext_physical_index_name(3) == "aperag-fulltext-v3"
    assert generate_legacy_fulltext_index_name("col-1") == "col-1"


def test_create_index_ensures_alias_for_shared_index(monkeypatch):
    captured = {"created": [], "aliases": []}

    class FakeIndices:
        def exists(self, index):
            return SimpleNamespace(body=index == "aperag-fulltext-v1")

        def exists_alias(self, name):
            return SimpleNamespace(body=False)

        def get_alias(self, name):
            return SimpleNamespace(body={})

        def put_alias(self, index, name):
            captured["aliases"].append((index, name))

        def create(self, index, body):
            captured["created"].append((index, body))

    class FakeElasticsearch:
        def __init__(self, *_args, **_kwargs):
            self.indices = FakeIndices()

    monkeypatch.setattr("aperag.index.fulltext_index.Elasticsearch", FakeElasticsearch)

    from aperag.index.fulltext_index import create_index

    create_index("aperag-fulltext")

    assert captured["created"] == []
    assert captured["aliases"] == [("aperag-fulltext-v1", "aperag-fulltext")]


def test_create_index_materializes_explicit_shard_and_replica_settings(monkeypatch):
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

    monkeypatch.setattr("aperag.index.fulltext_index.Elasticsearch", FakeElasticsearch)
    monkeypatch.setattr("aperag.index.fulltext_index.settings.es_fulltext_number_of_shards", 3)
    monkeypatch.setattr("aperag.index.fulltext_index.settings.es_fulltext_number_of_replicas", 1)

    from aperag.index.fulltext_index import create_index

    create_index("aperag-fulltext")

    assert captured["index"] == "aperag-fulltext-v1"
    assert captured["alias"] == ("aperag-fulltext-v1", "aperag-fulltext")
    assert captured["body"]["settings"] == {"number_of_shards": 3, "number_of_replicas": 1}


def test_create_index_preserves_existing_alias_target(monkeypatch):
    captured = {"created": [], "aliases": [], "updated": []}

    class FakeIndices:
        def exists(self, index):
            return SimpleNamespace(body=index == "aperag-fulltext-v2")

        def exists_alias(self, name):
            return SimpleNamespace(body=True)

        def get_alias(self, name):
            return SimpleNamespace(body={"aperag-fulltext-v2": {}})

        def put_alias(self, index, name):
            captured["aliases"].append((index, name))

        def create(self, index, body):
            captured["created"].append((index, body))

        def update_aliases(self, body):
            captured["updated"].append(body)

    class FakeElasticsearch:
        def __init__(self, *_args, **_kwargs):
            self.indices = FakeIndices()

    monkeypatch.setattr("aperag.index.fulltext_index.Elasticsearch", FakeElasticsearch)

    from aperag.index.fulltext_index import create_index

    result = create_index("aperag-fulltext")

    assert result["physical_index"] == "aperag-fulltext-v2"
    assert captured["created"] == []
    assert captured["aliases"] == []
    assert captured["updated"] == []


@pytest.mark.asyncio
async def test_fulltext_search_filters_on_collection_and_chat_id():
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

    from aperag.index.fulltext_index import FulltextIndexer

    indexer = object.__new__(FulltextIndexer)
    indexer.async_es = FakeAsyncEs()

    docs = await FulltextIndexer.search_document(indexer, "aperag-fulltext", "col-1", ["hello"], topk=3, chat_id="chat-1")

    assert docs == []
    assert captured["routing"] == "col-1"
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


def test_collection_task_deletes_shared_docs_and_legacy_index(monkeypatch):
    calls = {"shared": [], "legacy": []}

    monkeypatch.setattr(
        "aperag.tasks.collection.delete_collection_documents",
        lambda collection_id, index=None: calls["shared"].append((collection_id, index)) or 3,
    )
    monkeypatch.setattr("aperag.tasks.collection.delete_index", lambda index: calls["legacy"].append(index))

    CollectionTask()._delete_fulltext_index("col-1")

    assert calls["shared"] == [("col-1", "aperag-fulltext")]
    assert calls["legacy"] == ["col-1"]


def test_build_legacy_reindex_body_promotes_contract_fields():
    from aperag.index.fulltext_index import build_legacy_reindex_body

    body = build_legacy_reindex_body("col-1", "col-1")

    assert body["source"]["index"] == "col-1"
    assert body["dest"]["index"] == "aperag-fulltext-v1"
    assert body["script"]["params"]["collection_id"] == "col-1"
    assert "ctx._source.collection_id" in body["script"]["source"]
    assert "ctx._routing = params.collection_id" in body["script"]["source"]
    assert "ctx._source.chat_id" in body["script"]["source"]
