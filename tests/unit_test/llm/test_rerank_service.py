import httpx
import litellm
import pytest

from aperag.llm.rerank.rerank_service import RerankService
from aperag.platform.query.query import DocumentWithScore


@pytest.mark.asyncio
async def test_async_rerank_preserves_provider_scores_and_duplicate_documents(monkeypatch):
    async def fake_arerank(**kwargs):
        assert kwargs["query"] == "duplicate"
        assert kwargs["documents"] == ["same text", "same text"]
        return {
            "results": [
                {"index": 1, "relevance_score": 0.92},
                {"index": 0, "relevance_score": 0.41},
            ]
        }

    monkeypatch.setattr(litellm, "arerank", fake_arerank)

    service = RerankService(
        rerank_provider="jina_ai",
        rerank_model="test-reranker",
        rerank_service_url="https://example.com",
        rerank_service_api_key="test-key",
    )
    docs = [
        DocumentWithScore(text="same text", score=0.1, metadata={"id": "first"}),
        DocumentWithScore(text="same text", score=0.2, metadata={"id": "second"}),
    ]

    detailed = await service.async_rerank_with_details("duplicate", docs)

    assert [item.original_index for item in detailed] == [1, 0]
    assert [item.relevance_score for item in detailed] == pytest.approx([0.92, 0.41])
    assert [item.document.metadata["id"] for item in detailed] == ["second", "first"]

    reranked = await service.async_rerank("duplicate", docs)

    assert [doc.score for doc in reranked] == pytest.approx([0.92, 0.41])
    assert [doc.metadata["id"] for doc in reranked] == ["second", "first"]


@pytest.mark.asyncio
async def test_async_rerank_accepts_score_alias(monkeypatch):
    async def fake_arerank(**kwargs):
        return {"results": [{"index": 0, "score": 0.73}]}

    monkeypatch.setattr(litellm, "arerank", fake_arerank)

    service = RerankService(
        rerank_provider="jina_ai",
        rerank_model="test-reranker",
        rerank_service_url="https://example.com",
        rerank_service_api_key="test-key",
    )

    detailed = await service.async_rerank_with_details(
        "query",
        [DocumentWithScore(text="hello world", score=0.1, metadata={"id": "doc-1"})],
    )

    assert len(detailed) == 1
    assert detailed[0].original_index == 0
    assert detailed[0].relevance_score == pytest.approx(0.73)


@pytest.mark.parametrize(
    ("configured_base_url", "expected_url"),
    [
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        ),
        (
            "https://proxy.example.com/dashscope/compatible-mode/v1",
            "https://proxy.example.com/dashscope/api/v1/services/rerank/text-rerank/text-rerank",
        ),
        (
            "https://proxy.example.com/dashscope/api/v1/services/rerank/text-rerank/text-rerank",
            "https://proxy.example.com/dashscope/api/v1/services/rerank/text-rerank/text-rerank",
        ),
    ],
)
def test_resolve_alibabacloud_rerank_url(configured_base_url, expected_url):
    service = RerankService(
        rerank_provider="alibabacloud",
        rerank_model="gte-rerank-v2",
        rerank_service_url=configured_base_url,
        rerank_service_api_key="test-key",
    )

    assert service._resolve_alibabacloud_rerank_url() == expected_url


@pytest.mark.asyncio
async def test_call_alibabacloud_rerank_api_uses_resolved_base_url(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output": {"results": [{"index": 1, "relevance_score": 0.88}]}}

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 60.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    service = RerankService(
        rerank_provider="alibabacloud",
        rerank_model="gte-rerank-v2",
        rerank_service_url="https://proxy.example.com/dashscope/compatible-mode/v1",
        rerank_service_api_key="test-key",
    )

    response = await service._call_alibabacloud_rerank_api("capital of France", ["Paris", "Tokyo"])

    assert captured["url"] == "https://proxy.example.com/dashscope/api/v1/services/rerank/text-rerank/text-rerank"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "gte-rerank-v2"
    assert captured["json"]["parameters"]["top_n"] == 2
    assert response == {"results": [{"index": 1, "relevance_score": 0.88}]}
