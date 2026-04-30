# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the cross-adapter score-normalization helpers.

Pinned by task #61 P0-B. The contract:

* :func:`normalize_score` always returns a float in ``[0, 1]`` with
  higher = better.
* The transform is monotone non-decreasing per metric, so top-k ordering
  is preserved compared to the raw score.
* :func:`denormalize_threshold_to_native` is a true inverse, modulo the
  clamp endpoints — round-tripping a normalized threshold back to the
  raw scale and forward through normalize_score reproduces the input
  to within float-precision drift.
"""

from __future__ import annotations

import math

import pytest

from aperag.vectorstore.base import (
    UnsupportedFilterError,
    denormalize_threshold_to_native,
    normalize_score,
)

# ---------------------------------------------------------------------------
# normalize_score range invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric,raw",
    [
        ("cosine", 0.0),
        ("cosine", 0.5),
        ("cosine", 1.0),
        ("cosine", -0.3),  # drift below 0 should clamp
        ("cosine", 1.2),  # drift above 1 should clamp
        ("euclid", 0.0),
        ("euclid", -0.5),
        ("euclid", -10.0),
        ("euclid", -1e6),
        ("dot", 0.0),
        ("dot", 5.0),
        ("dot", -5.0),
        ("dot", 1e3),
        ("dot", -1e3),
    ],
)
def test_normalize_score_in_unit_interval(metric, raw):
    s = normalize_score(metric, raw)
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_normalize_score_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown distance metric"):
        normalize_score("hamming", 0.5)


# ---------------------------------------------------------------------------
# monotone ordering: more-similar must produce a higher (or equal) score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric,better,worse",
    [
        ("cosine", 0.95, 0.20),
        ("euclid", -0.10, -2.50),  # smaller L2 distance ⇒ larger negated raw
        ("dot", 8.0, 0.5),
        ("dot", -0.5, -3.0),
    ],
)
def test_normalize_score_preserves_ordering(metric, better, worse):
    sb = normalize_score(metric, better)
    sw = normalize_score(metric, worse)
    assert sb > sw, f"{metric}: normalize({better})={sb} should exceed normalize({worse})={sw}"


# ---------------------------------------------------------------------------
# denormalize_threshold_to_native is the inverse on the open interval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["cosine", "euclid", "dot"])
@pytest.mark.parametrize("normalized", [0.05, 0.25, 0.50, 0.72, 0.95])
def test_denormalize_then_normalize_roundtrip(metric, normalized):
    raw = denormalize_threshold_to_native(metric, normalized)
    back = normalize_score(metric, raw)
    assert math.isclose(back, normalized, rel_tol=1e-9, abs_tol=1e-9)


def test_denormalize_endpoints():
    # n = 0 → all rows pass; we encode that as -inf so callers know to skip
    # the SQL/Qdrant pushdown clause entirely.
    assert denormalize_threshold_to_native("cosine", 0.0) == pytest.approx(0.0)
    assert denormalize_threshold_to_native("euclid", 0.0) == -math.inf
    assert denormalize_threshold_to_native("dot", 0.0) == -math.inf

    # n = 1 → only an exact match passes.
    assert denormalize_threshold_to_native("cosine", 1.0) == pytest.approx(1.0)
    assert denormalize_threshold_to_native("euclid", 1.0) == pytest.approx(0.0)
    assert denormalize_threshold_to_native("dot", 1.0) == math.inf


def test_denormalize_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown distance metric"):
        denormalize_threshold_to_native("hamming", 0.5)


# ---------------------------------------------------------------------------
# UnsupportedFilterError typing
# ---------------------------------------------------------------------------


def test_unsupported_filter_error_subclasses_typeerror():
    """Backwards-compat: callers that ``except TypeError`` must keep working
    after the cross-adapter rename."""
    err = UnsupportedFilterError("nope")
    assert isinstance(err, TypeError)


# ---------------------------------------------------------------------------
# Backend-native raw-direction regression: pinned by Weston msg=86e05a8e
# ---------------------------------------------------------------------------
#
# The shared ``normalize_score`` helper assumes "higher-is-better raw" input,
# which matches Qdrant's native convention for cosine + dot but NOT for
# euclid — Qdrant returns positive L2 distance (lower = better). The Qdrant
# adapter therefore negates ``p.score`` for euclid before calling the helper.
# These tests pin the end-to-end behaviour against a real Qdrant ``:memory:``
# client so a future refactor that drops the boundary conversion fails fast.


def _qdrant_local_search_scores(distance: str, queries_and_points):
    """Run a tiny Qdrant ``:memory:`` round-trip and return the connector's
    normalized scores. Returns a dict keyed by the input id so tests can
    pin per-point scores without depending on top-k ordering.
    """
    pytest.importorskip("qdrant_client")

    import uuid as _uuid

    from aperag.vectorstore.dto import VectorPoint
    from aperag.vectorstore.qdrant_connector import QdrantVectorStoreConnector

    ctx = {
        "url": ":memory:",
        "collection": f"unit_eu_{_uuid.uuid4().hex[:8]}",
        "vector_size": 4,
        "distance": distance,
        "multitenant": False,
    }
    conn = QdrantVectorStoreConnector(ctx)
    points = [
        VectorPoint(id=str(_uuid.uuid5(_uuid.NAMESPACE_URL, name)), vector=vec, payload={"name": name})
        for name, vec in queries_and_points
    ]
    conn.upsert(points)
    return points, conn


def test_qdrant_euclid_normalized_scores_strictly_decreasing_with_distance():
    """Pinned by Weston msg=86e05a8e: Qdrant returns positive L2 distance
    natively (smaller = better). The connector must negate at the boundary
    so the shared helper produces *strictly decreasing* normalized scores
    as L2 grows."""
    pytest.importorskip("qdrant_client")
    from aperag.vectorstore.dto import QueryRequest

    points, conn = _qdrant_local_search_scores(
        "Euclid",
        [
            ("near", [0.0, 0.0, 0.0, 0.0]),  # L2=0 from the query
            ("mid", [1.0, 0.0, 0.0, 0.0]),  # L2=1
            ("far", [3.0, 0.0, 0.0, 0.0]),  # L2=3
        ],
    )
    hits = conn.search(QueryRequest(embedding=[0.0, 0.0, 0.0, 0.0], top_k=5, score_threshold=0.0))
    by_name = {h.payload["name"]: h.score for h in hits}
    assert "near" in by_name and "mid" in by_name and "far" in by_name
    # All three in [0, 1].
    for s in by_name.values():
        assert 0.0 <= s <= 1.0
    # Strictly decreasing: near closer than mid closer than far.
    assert by_name["near"] > by_name["mid"] > by_name["far"], (
        f"Qdrant euclid normalized scores must decrease with L2; got {by_name}"
    )


def test_qdrant_euclid_score_threshold_filters_far_keeps_near():
    """Pinned by Weston msg=86e05a8e: ``score_threshold=0.9`` on Qdrant
    euclid must pass through the boundary conversion so a tight threshold
    keeps the closest point and drops the far one — not silent-return-empty
    nor return-all."""
    pytest.importorskip("qdrant_client")
    from aperag.vectorstore.dto import QueryRequest

    _, conn = _qdrant_local_search_scores(
        "Euclid",
        [
            ("near", [0.0, 0.0, 0.0, 0.0]),
            ("mid", [1.0, 0.0, 0.0, 0.0]),
            ("far", [3.0, 0.0, 0.0, 0.0]),
        ],
    )
    hits = conn.search(QueryRequest(embedding=[0.0, 0.0, 0.0, 0.0], top_k=5, score_threshold=0.9))
    names = {h.payload["name"] for h in hits}
    assert "near" in names, "tight threshold must keep the L2=0 point"
    assert "far" not in names, "tight threshold must drop the L2=3 point"


def test_qdrant_dot_normalized_scores_strictly_increasing_with_inner_product():
    """Sanity-check the other Qdrant convention — dot product. Native
    Qdrant dot is "higher = better" so the helper convention matches and
    no boundary negation is needed; this test pins that no future refactor
    accidentally negates dot too."""
    pytest.importorskip("qdrant_client")
    from aperag.vectorstore.dto import QueryRequest

    _, conn = _qdrant_local_search_scores(
        "Dot",
        [
            ("low", [0.1, 0.0, 0.0, 0.0]),
            ("mid", [0.5, 0.0, 0.0, 0.0]),
            ("hi", [1.0, 0.0, 0.0, 0.0]),
        ],
    )
    hits = conn.search(QueryRequest(embedding=[1.0, 0.0, 0.0, 0.0], top_k=5, score_threshold=0.0))
    by_name = {h.payload["name"]: h.score for h in hits}
    assert "low" in by_name and "mid" in by_name and "hi" in by_name
    for s in by_name.values():
        assert 0.0 <= s <= 1.0
    assert by_name["hi"] > by_name["mid"] > by_name["low"], (
        f"Qdrant dot normalized scores must increase with inner product; got {by_name}"
    )


def test_qdrant_cosine_normalized_scores_strictly_increasing_with_similarity():
    """Pin Qdrant cosine — already exercised by the compat fixture but a
    unit-level pin keeps the contract obvious next to euclid + dot."""
    pytest.importorskip("qdrant_client")
    from aperag.vectorstore.dto import QueryRequest

    _, conn = _qdrant_local_search_scores(
        "Cosine",
        [
            ("orth", [0.0, 1.0, 0.0, 0.0]),  # orthogonal to query
            ("part", [1.0, 1.0, 0.0, 0.0]),  # ~45°
            ("same", [1.0, 0.0, 0.0, 0.0]),  # parallel to query
        ],
    )
    hits = conn.search(QueryRequest(embedding=[1.0, 0.0, 0.0, 0.0], top_k=5, score_threshold=0.0))
    by_name = {h.payload["name"]: h.score for h in hits}
    for s in by_name.values():
        assert 0.0 <= s <= 1.0
    assert by_name["same"] > by_name["part"] > by_name["orth"], (
        f"Qdrant cosine normalized scores must increase with similarity; got {by_name}"
    )
