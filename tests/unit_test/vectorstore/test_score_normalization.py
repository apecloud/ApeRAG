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
