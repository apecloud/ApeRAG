# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for ``ContextManager._create_combined_filter`` producing DSL.

These tests guard the boundary between "business intent" (index_types /
chat_id parameters the service layer passes in) and "backend-neutral filter
shape" (what gets handed to the connector). A regression here silently
changes retrieval semantics, so we assert on exact tree shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from aperag.domains.retrieval.context.context import ContextManager
from aperag.vectorstore.filters import And, Eq, In, IsEmpty, Or


def _cm() -> ContextManager:
    """Build a ContextManager bypassing __init__ — we don't need a real
    adaptor / embedding for pure filter-construction tests, and wiring one
    up would drag Qdrant into a unit test."""
    cm = ContextManager.__new__(ContextManager)
    cm.collection_name = "col_unit_test"
    cm.embedding_model = MagicMock()
    cm.vectordb_type = "qdrant"
    cm.adaptor = MagicMock()
    return cm


# ---------------------------------------------------------------------------
# no filters => None
# ---------------------------------------------------------------------------


def test_no_constraints_returns_none():
    """Connector contract: None means "no filter"; preserves the
    short-circuit path in the Qdrant search call."""
    assert _cm()._create_combined_filter(None, None) is None
    assert _cm()._create_combined_filter([], None) is None


# ---------------------------------------------------------------------------
# index_types only: Or over (In, IsEmpty) to preserve pre-migration points
# ---------------------------------------------------------------------------


def test_index_types_only_emits_or_with_is_empty_backward_compat():
    """The ``IsEmpty`` branch is the only thing keeping pre-``indexer``
    data searchable — dropping it silently halves recall on legacy points.
    Test pins the shape so future cleanups don't remove it by accident."""
    flt = _cm()._create_combined_filter(["vector", "vision"], None)
    assert isinstance(flt, Or)
    assert len(flt.parts) == 2
    in_part, empty_part = flt.parts
    assert in_part == In(key="indexer", values=("vector", "vision"))
    assert empty_part == IsEmpty(key="indexer")


# ---------------------------------------------------------------------------
# chat_id only: single Eq
# ---------------------------------------------------------------------------


def test_chat_id_only_emits_single_eq():
    flt = _cm()._create_combined_filter(None, "c42")
    assert flt == Eq(key="chat_id", value="c42")


# ---------------------------------------------------------------------------
# both: AND of (Or-block, Eq)
# ---------------------------------------------------------------------------


def test_both_index_types_and_chat_id_build_and_of_or_and_eq():
    """The production shape. Verifies the combinator arithmetic matches
    the legacy semantics (``Filter(must=[chat_id, Filter(should=...)])``):
    chat_id is AND-ed at the outer level, index_types is OR-ed internally.
    """
    flt = _cm()._create_combined_filter(["vector"], "c42")
    assert isinstance(flt, And)
    assert len(flt.parts) == 2
    or_block, eq_block = flt.parts
    assert isinstance(or_block, Or)
    # (In(vector), IsEmpty) — the backward-compat branch survives.
    assert len(or_block.parts) == 2
    assert eq_block == Eq(key="chat_id", value="c42")


def test_vectordb_type_does_not_influence_filter_shape():
    """The whole point of moving to DSL is that filter shape is backend-
    agnostic. Even when the (now diagnostic-only) ``vectordb_type`` string
    is weird, the DSL output must not change."""
    cm = _cm()
    cm.vectordb_type = "unknown-but-valid"
    flt = cm._create_combined_filter(["vector"], "c42")
    cm.vectordb_type = "qdrant"
    flt2 = cm._create_combined_filter(["vector"], "c42")
    assert flt == flt2
