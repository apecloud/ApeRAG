from __future__ import annotations

from types import SimpleNamespace

import pytest

from aperag.domains.retrieval import pipeline as pipeline_module
from aperag.domains.retrieval.pipeline import SearchPipelineService


@pytest.mark.asyncio
async def test_vector_search_defaults_optional_mcp_params(monkeypatch):
    captured: dict[str, object] = {}

    class _Embedding:
        def embed_query(self, query: str) -> list[float]:
            assert query == "充电联盟"
            return [0.1, 0.2]

    class _ContextManager:
        def __init__(self, *args, **kwargs):
            pass

        def query(self, query, *, score_threshold, topk, vector, index_types, chat_id):
            captured.update(
                {
                    "score_threshold": score_threshold,
                    "topk": topk,
                    "vector": vector,
                    "index_types": index_types,
                    "chat_id": chat_id,
                }
            )
            return []

    monkeypatch.setattr(pipeline_module, "get_collection_embedding_service_sync", lambda collection: (_Embedding(), 2))
    monkeypatch.setattr(pipeline_module, "build_vector_db_context", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline_module, "ContextManager", _ContextManager)

    result = await SearchPipelineService()._vector_search(
        collection=SimpleNamespace(id="col-1", config={}),
        query="充电联盟",
        top_k=None,
        similarity_threshold=None,
    )

    assert result == []
    assert captured == {
        "score_threshold": 0.5,
        "topk": 5,
        "vector": [0.1, 0.2],
        "index_types": ["vector"],
        "chat_id": None,
    }


@pytest.mark.asyncio
async def test_fulltext_search_includes_raw_query_when_keywords_are_extracted(monkeypatch):
    captured: dict[str, object] = {}

    async def _extract_keywords(query: str, ctx: dict) -> list[str]:
        assert query == "充电联盟"
        return ["充电", "联盟"]

    class _ExistsResponse:
        body = True

    class _SearchResponse:
        body = {"hits": {"hits": []}}

    class _AsyncElasticsearch:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def indices(self):
            return self

        async def exists(self, index):
            return _ExistsResponse()

        async def search(self, *, index, query, sort, size, routing):
            captured.update({"query": query, "size": size, "routing": routing})
            return _SearchResponse()

        async def close(self):
            return None

    monkeypatch.setattr("aperag.indexing.keyword_extract.extract_keywords", _extract_keywords)
    monkeypatch.setattr("elasticsearch.AsyncElasticsearch", _AsyncElasticsearch)

    result = await SearchPipelineService()._fulltext_search(
        collection=SimpleNamespace(id="col-1", config="{}"),
        query="充电联盟",
        top_k=None,
        keywords=None,
        user_id="user-1",
    )

    assert result == []
    should = captured["query"]["bool"]["should"]
    match_terms = [clause["match"][field] for clause in should for field in clause["match"]]
    assert "充电联盟" in match_terms
    assert captured["query"]["bool"]["minimum_should_match"] == 1
    assert captured["size"] == 15
    assert captured["routing"] == "col-1"
