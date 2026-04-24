from aperag.domains.model_platform.api.llm_routes import _build_rerank_response_items
from aperag.llm.rerank.rerank_service import RerankResult
from aperag.query.query import DocumentWithScore


def test_build_rerank_response_items_preserves_original_indices_for_duplicate_texts():
    reranked_documents = [
        RerankResult(
            original_index=1,
            relevance_score=0.92,
            document=DocumentWithScore(text="same text", score=0.92, metadata={"id": "second"}),
        ),
        RerankResult(
            original_index=0,
            relevance_score=0.41,
            document=DocumentWithScore(text="same text", score=0.41, metadata={"id": "first"}),
        ),
    ]

    response_items = _build_rerank_response_items(reranked_documents, return_documents=True)

    assert [item.index for item in response_items] == [1, 0]
    assert [item.relevance_score for item in response_items] == [0.92, 0.41]
    assert [item.document["metadata"]["id"] for item in response_items] == ["second", "first"]
