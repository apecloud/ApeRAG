# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the Qdrant translator of the ``VectorFilter`` DSL.

Every DSL node must map to a well-formed ``qdrant_client.models.Filter``
tree. We assert on structural equality of the produced Filter (not on its
string representation) so translator refactors don't drift silently.
"""

from __future__ import annotations

import pytest
from qdrant_client import models as rest

from aperag.vectorstore import qdrant_connector as qc
from aperag.vectorstore.filters import (
    Eq,
    In,
    IsEmpty,
    Not,
    all_of,
    any_of,
)

# ---------------------------------------------------------------------------
# null / pass-through / unknown
# ---------------------------------------------------------------------------


def test_translate_none_returns_none():
    assert qc._translate_filter(None) is None


def test_normalize_passes_through_raw_qdrant_filter():
    """The migration script hand-rolls ``rest.Filter``; we can't break that
    code path, so ``_normalize_filter_input`` short-circuits and returns the
    Filter unchanged."""
    raw = rest.Filter(must=[rest.FieldCondition(key="k", match=rest.MatchValue(value="v"))])
    assert qc._normalize_filter_input(raw) is raw


def test_normalize_unknown_type_raises_unsupported_filter_error():
    """Any value that is neither DSL nor Qdrant Filter must be refused
    fail-loud — never silently dropped (which would degrade into an
    unfiltered full-collection scan, a correctness bug, not a graceful
    degradation). Pinned by task #61 P0-A."""
    from aperag.vectorstore.base import UnsupportedFilterError

    with pytest.raises(UnsupportedFilterError, match="unsupported filter input"):
        qc._normalize_filter_input({"garbage": True})

    # Backwards-compat: callers that ``except TypeError`` keep working.
    with pytest.raises(TypeError):
        qc._normalize_filter_input(object())


def test_translate_or_with_zero_translatable_parts_raises():
    """Pinned by task #61 P1-V3: an Or filter that ends up with zero
    translatable parts (after the construction guard is bypassed) MUST
    raise rather than produce a vacuous ``Filter(should=[])`` that
    Qdrant would treat as "match everything".

    ``Or.__post_init__`` already rejects empty ``parts`` at construction;
    we exercise the translator-level defense-in-depth path by mutating
    a frozen dataclass via ``object.__setattr__`` to simulate a
    downstream caller that ended up with empty parts.
    """
    from aperag.vectorstore.base import UnsupportedFilterError
    from aperag.vectorstore.filters import Eq, Or

    or_node = Or(parts=(Eq(key="k", value="v"),))
    object.__setattr__(or_node, "parts", ())

    with pytest.raises(UnsupportedFilterError, match="zero translatable parts"):
        qc._translate_filter(or_node)


# ---------------------------------------------------------------------------
# leaf nodes
# ---------------------------------------------------------------------------


def test_translate_eq():
    got = qc._translate_filter(Eq(key="chat_id", value="c42"))
    assert isinstance(got, rest.Filter)
    assert got.must == [rest.FieldCondition(key="chat_id", match=rest.MatchValue(value="c42"))]


def test_translate_in():
    got = qc._translate_filter(In(key="indexer", values=("vector", "vision")))
    assert got.must == [rest.FieldCondition(key="indexer", match=rest.MatchAny(any=["vector", "vision"]))]


def test_translate_in_empty_values_raises():
    """Empty ``In`` would match nothing, which is almost always a bug; we
    refuse it loudly rather than produce a "silent zero-results" query."""
    with pytest.raises(ValueError, match="empty values"):
        qc._translate_filter(In(key="indexer", values=()))


def test_translate_is_empty():
    got = qc._translate_filter(IsEmpty(key="indexer"))
    assert got.must == [rest.IsEmptyCondition(is_empty=rest.PayloadField(key="indexer"))]


# ---------------------------------------------------------------------------
# combinators
# ---------------------------------------------------------------------------


def test_translate_and_nests_filters_under_must():
    """Qdrant allows Filter inside ``must`` as a sub-Filter; we rely on that
    so each DSL child is translated to a Filter and attached directly."""
    flt = all_of(Eq(key="k1", value="v"), Eq(key="k2", value="w"))
    got = qc._translate_filter(flt)
    assert isinstance(got, rest.Filter)
    assert len(got.must) == 2
    for sub in got.must:
        assert isinstance(sub, rest.Filter)


def test_translate_or_uses_should():
    flt = any_of(Eq(key="k1", value="v"), IsEmpty(key="k1"))
    got = qc._translate_filter(flt)
    assert isinstance(got, rest.Filter)
    assert len(got.should) == 2
    assert got.must in (None, [])


def test_translate_not_uses_must_not():
    got = qc._translate_filter(Not(inner=Eq(key="k", value="v")))
    assert isinstance(got, rest.Filter)
    assert len(got.must_not) == 1


def test_translate_real_context_manager_shape():
    """End-to-end shape check mirroring what ``ContextManager`` produces
    when both ``index_types`` and ``chat_id`` are set. Any regression here
    breaks retrieval filtering, so we keep a structural snapshot."""
    flt = all_of(
        any_of(
            In(key="indexer", values=("vector", "vision")),
            IsEmpty(key="indexer"),
        ),
        Eq(key="chat_id", value="c42"),
    )
    got = qc._translate_filter(flt)
    assert isinstance(got, rest.Filter)
    # Two top-level AND parts: the OR-block and the chat_id Eq.
    assert len(got.must) == 2
    or_block, eq_block = got.must
    assert isinstance(or_block, rest.Filter)
    assert len(or_block.should) == 2
    assert isinstance(eq_block, rest.Filter)
    assert eq_block.must == [rest.FieldCondition(key="chat_id", match=rest.MatchValue(value="c42"))]
