# Pytest test-suite for EmbeddingService.
# All tests are skipped automatically if the environment variable
# OPENAI_API_KEY is not defined, so the suite can run in CI environments
# that do not have secrets configured.

import os
import sys
import random
import pytest

# Allow "python -m pytest" from project root
sys.path.append(os.path.dirname(__file__))

from embedding_service import EmbeddingService


BACKEND = "openai"
MODEL = "text-embedding-3-small"
API_URL = "https://api.gptsapi.net/v1"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Global marker: each test that uses @SKIP will run only if the key exists
SKIP = pytest.mark.skipif(
    not OPENAI_KEY,
    reason="Environment variable OPENAI_API_KEY is not set",
)

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


def build_service(batch_size: int) -> EmbeddingService:
    """Utility helper to build an EmbeddingService with a given batch size."""
    return EmbeddingService(
        embedding_backend=BACKEND,
        embedding_model=MODEL,
        embedding_service_url=API_URL,
        embedding_service_api_key=OPENAI_KEY,
        embedding_max_chunks_in_batch=batch_size,
    )


@SKIP
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_length_and_order_are_preserved(batch_size):
    """The number of vectors equals the number of texts and order is untouched."""
    svc = build_service(batch_size)
    texts = BASE_TEXTS[:]
    vectors = svc.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert all(isinstance(v, list) for v in vectors)
    # embed_query should return the same vector as the first element
    assert vectors[0] == svc.embed_query(texts[0])


@SKIP
def test_identical_text_returns_identical_vector():
    """Calling the service twice with the same input returns identical vectors."""
    svc = build_service(8)
    text = "duplicate input duplicate input"
    vec1 = svc.embed_query(text)
    vec2 = svc.embed_documents([text])[0]
    assert vec1 == vec2
