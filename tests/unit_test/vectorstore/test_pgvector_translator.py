# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the pgvector SQL filter translator and vector-literal encoder.

Zero database contact: we exercise the string-building helpers only. The
end-to-end integration test lives in ``test_pgvector_end_to_end.py`` and
is gated on ``APERAG_TEST_PGVECTOR_URL``.
"""

from __future__ import annotations

import pytest

from aperag.vectorstore.dto import VectorShape
from aperag.vectorstore.filters import Eq, In, IsEmpty, Not, all_of, any_of
from aperag.vectorstore.pgvector_connector import (
    _table_name,
    _translate_filter,
    _vector_literal,
)

# ---------------------------------------------------------------------------
# table naming
# ---------------------------------------------------------------------------


def test_table_name_stable_across_distance_casing():
    """``VectorShape`` normalizes to lowercase so the physical table name
    is deterministic regardless of whether the caller wrote ``Cosine``,
    ``COSINE``, or ``cosine``."""
    assert _table_name(VectorShape(size=1024, distance="Cosine")) == "aperag_vectors_1024_cosine"
    assert _table_name(VectorShape(size=1024, distance="cosine")) == "aperag_vectors_1024_cosine"


def test_table_name_rejects_non_positive_size():
    with pytest.raises(ValueError):
        VectorShape(size=0, distance="cosine")


def test_table_name_rejects_unsupported_distance():
    with pytest.raises(ValueError, match="distance must be one of"):
        VectorShape(size=1024, distance="jaccard")


# ---------------------------------------------------------------------------
# translator: leaves
# ---------------------------------------------------------------------------


def test_translate_none_is_noop():
    sql, params = _translate_filter(None)
    assert sql == ""
    assert params == {}


def test_translate_eq_uses_param_binding_not_value_interpolation():
    """Values are parameterised; only keys are interpolated (and keys are
    hard-coded business vocabulary: ``indexer``, ``chat_id``, …). This
    guards against SQL injection via payload values."""
    sql, params = _translate_filter(Eq(key="chat_id", value="c42"))
    assert sql == "payload->>'chat_id' = :f0"
    assert params == {"f0": "c42"}


def test_translate_eq_coerces_booleans_to_json_casing():
    """JSON stringifies ``True`` as ``"true"``; pgvector's ``->>`` returns
    the same. Using ``str(True)`` naively would yield ``"True"`` and never
    match — hence the explicit coercion in ``_scalar_to_text``."""
    sql, params = _translate_filter(Eq(key="flag", value=True))
    assert params == {"f0": "true"}


def test_translate_in_generates_unique_parameter_names_per_value():
    sql, params = _translate_filter(In(key="indexer", values=("vector", "vision")))
    assert sql == "payload->>'indexer' IN (:f0, :f1)"
    assert params == {"f0": "vector", "f1": "vision"}


def test_translate_in_empty_values_raises():
    with pytest.raises(ValueError, match="empty values"):
        _translate_filter(In(key="indexer", values=()))


def test_translate_is_empty_matches_missing_or_json_null():
    sql, _ = _translate_filter(IsEmpty(key="indexer"))
    assert "NOT (payload ? 'indexer')" in sql
    assert "'null'::jsonb" in sql


# ---------------------------------------------------------------------------
# translator: combinators
# ---------------------------------------------------------------------------


def test_translate_and_of_eq_eq_emits_and_join():
    sql, params = _translate_filter(all_of(Eq(key="a", value="1"), Eq(key="b", value="2")))
    # Outer parens + AND join; exact whitespace insensitive.
    normalized = " ".join(sql.split())
    assert normalized == "(payload->>'a' = :f0 AND payload->>'b' = :f1)"
    assert params == {"f0": "1", "f1": "2"}


def test_translate_or_uses_or_join():
    sql, _ = _translate_filter(any_of(Eq(key="a", value="1"), Eq(key="b", value="2")))
    assert " OR " in sql


def test_translate_not_wraps_inner_sql():
    sql, _ = _translate_filter(Not(inner=Eq(key="a", value="1")))
    assert sql.startswith("NOT (")


def test_translate_context_manager_shape():
    """The production shape from ``ContextManager._create_combined_filter``:
    ``all_of(any_of(In(indexer,...), IsEmpty(indexer)), Eq(chat_id,...))``.
    Asserting on this structure guards against silent regressions when
    the translator gets refactored."""
    flt = all_of(
        any_of(
            In(key="indexer", values=("vector", "vision")),
            IsEmpty(key="indexer"),
        ),
        Eq(key="chat_id", value="c42"),
    )
    sql, params = _translate_filter(flt)
    # There are three IN values → f0/f1, one IsEmpty (no bind), one Eq → f2.
    assert params == {"f0": "vector", "f1": "vision", "f2": "c42"}
    # Outer AND, inner OR.
    assert sql.count(" AND ") == 1
    assert " OR " in sql


def test_translate_rejects_unknown_node_loudly():
    """Pinned by task #61 P0-A: pgvector translator must raise the same
    cross-adapter ``UnsupportedFilterError`` (subclass of ``TypeError`` so
    the existing ``except TypeError`` callers keep working)."""
    from aperag.vectorstore.base import UnsupportedFilterError

    class Bogus:
        pass

    with pytest.raises(UnsupportedFilterError, match="Unsupported VectorFilter node"):
        _translate_filter(Bogus())  # type: ignore[arg-type]

    # Backwards-compat: TypeError parent still catches it.
    with pytest.raises(TypeError):
        _translate_filter(Bogus())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# vector literal
# ---------------------------------------------------------------------------


def test_vector_literal_lossless_roundtrip():
    """pgvector wants ``'[1.0, 2.0]'::vector``; the round-trip must be
    lossless for typical floats (we use ``repr`` on purpose)."""
    lit = _vector_literal([1.0, 0.0, -0.5])
    assert lit == "[1.0,0.0,-0.5]"


def test_vector_literal_handles_int_input():
    """Callers may hand us ints when the embedding model has integer
    output (unusual but not unheard of); we cast to float uniformly."""
    lit = _vector_literal([1, 2, 3])
    assert lit == "[1.0,2.0,3.0]"
