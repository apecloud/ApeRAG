# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the ``VectorFilter`` DSL itself — translator-agnostic.

These tests pin down the *semantics* of the DSL (construction rules,
short-circuit helpers, structural equality). Backend translation is tested
separately in ``test_qdrant_filter_translation.py``.
"""

from __future__ import annotations

import pytest

from aperag.vectorstore.filters import (
    And,
    Eq,
    In,
    IsEmpty,
    Not,
    Or,
    _in,
    all_of,
    any_of,
)

# ---------------------------------------------------------------------------
# leaf nodes
# ---------------------------------------------------------------------------


def test_eq_is_frozen_and_hashable():
    """Frozen dataclasses are a hard requirement: callers cache filter
    structures in sets/dicts for deduplication."""
    a = Eq(key="indexer", value="vector")
    b = Eq(key="indexer", value="vector")
    assert a == b
    assert {a, b} == {a}
    with pytest.raises(Exception):
        a.value = "other"  # frozen -> FrozenInstanceError


def test_in_builder_normalizes_to_tuple():
    """``_in`` is the ergonomic builder; the dataclass itself keeps a tuple
    (immutable + hashable). A list passed to ``In`` directly would fail
    equality with another constructed from the same source list, which is
    why we have the helper."""
    assert _in("chat_id", ["x", "y"]) == In(key="chat_id", values=("x", "y"))


def test_is_empty_holds_only_key():
    assert IsEmpty(key="indexer").key == "indexer"


# ---------------------------------------------------------------------------
# combinators
# ---------------------------------------------------------------------------


def test_and_rejects_empty_parts():
    """Empty And is a logic bug (matches-everything is not what callers
    want). Helpers ``all_of()`` / ``any_of()`` are the right path for
    ergonomic short-circuit."""
    with pytest.raises(ValueError, match="at least one part"):
        And(parts=())


def test_or_rejects_empty_parts():
    with pytest.raises(ValueError, match="at least one part"):
        Or(parts=())


def test_not_wraps_inner():
    assert Not(inner=Eq(key="k", value="v")).inner == Eq(key="k", value="v")


# ---------------------------------------------------------------------------
# all_of / any_of short-circuit
# ---------------------------------------------------------------------------


def test_all_of_none_input_returns_none():
    """Callers build filters conditionally; passing a missing part as None
    must not become an empty-And bug."""
    assert all_of(None, None) is None
    assert all_of() is None


def test_all_of_single_part_returns_part():
    """Degenerate And/Or is meaningless noise; we return the inner node so
    translators don't need to special-case single-part combinators."""
    e = Eq(key="k", value="v")
    assert all_of(e, None) is e
    assert all_of(None, e) is e


def test_all_of_multiple_parts_wraps_in_and():
    a, b = Eq(key="k1", value="v"), Eq(key="k2", value="w")
    got = all_of(a, b)
    assert isinstance(got, And)
    assert got.parts == (a, b)


def test_any_of_mirrors_all_of_semantics():
    a, b = Eq(key="k1", value="v"), Eq(key="k2", value="w")
    assert any_of() is None
    assert any_of(None) is None
    assert any_of(a) is a
    assert isinstance(any_of(a, b), Or)


def test_combinators_nest_cleanly():
    """Deep nesting (the realistic shape when ContextManager produces
    combined filters) composes without surprises."""
    combined = all_of(
        Eq(key="chat_id", value="c42"),
        any_of(
            In(key="indexer", values=("vector", "vision")),
            IsEmpty(key="indexer"),
        ),
    )
    assert isinstance(combined, And)
    assert len(combined.parts) == 2
    inner_or = combined.parts[1]
    assert isinstance(inner_or, Or)
    assert len(inner_or.parts) == 2
