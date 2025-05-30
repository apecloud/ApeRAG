import os

import pytest

from aperag.embed.embedding_service import EmbeddingService
from aperag.llm.litellm_track import register_llm_track

PROVIDERS = [
    {
        "test_name": "alibabacloud",
        "dialect": "openai",
        "model": "text-embedding-v3",
        "api_base_env": "QIANWEN_API_BASE",
        "api_key_env": "QIANWEN_API_KEY",
    },
    {
        "test_name": "siliconflow",
        "dialect": "openai",
        "model": "BAAI/bge-m3",
        "api_base_env": "SILICONFLOW_API_BASE",
        "api_key_env": "SILICONFLOW_API_KEY",
    },
]

BASE_TEXTS = [
    "hello world",
    "openai embeddings test",
    "this is the third sentence",
    "lorem ipsum dolor sit amet",
    "another short line",
    "the quick brown fox jumps over the lazy dog",
    "embedding consistency check",
    "duplicate input duplicate input",
]


register_llm_track()


@pytest.fixture(params=PROVIDERS)
def embedding_service(request):
    config = request.param
    api_base = os.getenv(config["api_base_env"])
    api_key = os.getenv(config["api_key_env"])

    if not api_base or not api_key:
        pytest.skip(f"Missing env vars: {config['api_base_env']}, {config['api_key_env']}")

    def _build_service(batch_size: int):
        return EmbeddingService(
            embedding_provider=config["dialect"],
            embedding_model=config["model"],
            embedding_service_url=api_base,
            embedding_service_api_key=api_key,
            embedding_max_chunks_in_batch=batch_size,
        )

    return _build_service


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_length_and_order_are_preserved(embedding_service, batch_size):
    svc = embedding_service(batch_size)
    texts = BASE_TEXTS[:]
    vectors = svc.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert all(isinstance(v, list) for v in vectors)
    # assert vectors[0] == svc.embed_query(texts[0])
    assert_vectors_equal(vectors[0], svc.embed_query(texts[0]))


def test_identical_text_returns_identical_vector(embedding_service):
    svc = embedding_service(8)
    text = "duplicate input duplicate input"
    vec1 = svc.embed_query(text)
    vec2 = svc.embed_documents([text])[0]
    assert_vectors_equal(vec1, vec2)


def assert_vectors_equal(vec1, vec2, tolerance=1e-3):
    assert len(vec1) == len(vec2), "Vector dimension mismatch"

    max_diff = max(abs(a - b) for a, b in zip(vec1, vec2))
    assert max_diff < tolerance, f"Max difference {max_diff:.2e} exceeds tolerance {tolerance:.0e}"

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a**2 for a in vec1) ** 0.5
    norm2 = sum(b**2 for b in vec2) ** 0.5
    cosine_sim = dot_product / (norm1 * norm2)
    assert cosine_sim > 0.999, f"Cosine similarity {cosine_sim:.4f} below threshold"
