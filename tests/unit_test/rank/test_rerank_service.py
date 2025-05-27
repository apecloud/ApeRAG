import os

import pytest

from aperag.query.query import DocumentWithScore
from aperag.rank.reranker import rerank

# ---------- Constants and Skip Markers ----------
API_BASE   = os.getenv("RERANK_SERVICE_URL")
API_KEY    = os.getenv("RERANK_SERVICE_TOKEN_API_KEY")
BACKEND    = os.getenv("RERANK_BACKEND")
MODEL_NAME = os.getenv("RERANK_SERVICE_MODEL")

SKIP = pytest.mark.skipif(
    not API_BASE or not API_KEY,
    reason="Environment variable RERANK_SERVICE_URL or RERANK_SERVICE_TOKEN_API_KEY is not set",
)


# ---------- Basic Test Data ----------
QUERY = "hello world"

DOC_MATCH        = "hello world"                       # Highly relevant to query
DOC_PARTIAL      = "hello there, beautiful world!"     # Moderately relevant
DOC_UNRELATED    = "lorem ipsum dolor sit amet"        # Mostly irrelevant

BASE_RESULTS = [
    DocumentWithScore(text=DOC_UNRELATED, score=0.1),
    DocumentWithScore(text=DOC_MATCH,     score=0.1),
    DocumentWithScore(text=DOC_PARTIAL,   score=0.1),
]


# ---------- Test Cases ----------

@pytest.mark.asyncio
@SKIP
async def test_rerank_length_contents_and_type_are_preserved():
    """
    1. Returned results count matches input after reranking
    2. Each element remains DocumentWithScore type
    3. No documents are lost or added (set equality)
    """
    ranked = await rerank(QUERY, BASE_RESULTS)

    # Count check
    assert len(ranked) == len(BASE_RESULTS)

    # Type check
    assert all(isinstance(d, DocumentWithScore) for d in ranked)

    # Set equality (comparing text fields only)
    assert {d.text for d in ranked} == {d.text for d in BASE_RESULTS}


@pytest.mark.asyncio
@SKIP
async def test_rerank_expected_top1_is_most_relevant():
    """
    Assuming the service is robust enough, for query 'hello world',
    the exactly matching document should rank first.
    Remove this assertion if your actual model doesn't guarantee this.
    """
    ranked = await rerank(QUERY, BASE_RESULTS)
    assert ranked[0].text == DOC_MATCH


@pytest.mark.asyncio
@SKIP
async def test_rerank_is_deterministic_for_same_inputs():
    """
    When called twice with same query + document set,
    the ranking results should be consistent (same index sequence).
    """
    r1 = await rerank(QUERY, BASE_RESULTS)
    r2 = await rerank(QUERY, BASE_RESULTS)

    order1 = [d.text for d in r1]
    order2 = [d.text for d in r2]

    assert order1 == order2