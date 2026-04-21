# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the Qdrant connector's pure-logic helpers.

We deliberately avoid spinning up a Qdrant process here — only the
configuration/filter-synthesis logic is covered. End-to-end behavior is
exercised manually against staging during the migration window.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from qdrant_client import models as rest

from aperag.vectorstore.qdrant_connector import (
    TENANT_PAYLOAD_KEY,
    _coerce_distance,
    _hnsw_config,
    _merge_tenant_filter,
    _optimizers_config,
    _quantization_config,
    global_collection_name,
)

# ---------------------------------------------------------------------------
# global_collection_name
# ---------------------------------------------------------------------------


def test_global_collection_name_is_stable_and_lowercased():
    assert global_collection_name(1024, "Cosine") == "aperag_vectors_1024_cosine"
    assert global_collection_name(1536, "cosine") == "aperag_vectors_1536_cosine"
    # Case-insensitive on the distance component so callers don't have to care.
    assert global_collection_name(768, "DOT") == "aperag_vectors_768_dot"


def test_global_collection_name_accepts_float_size():
    # callers sometimes float-ify the dim by accident — we must still produce a
    # valid collection name without decimals.
    assert global_collection_name(1024.0, "Cosine") == "aperag_vectors_1024_cosine"


# ---------------------------------------------------------------------------
# _coerce_distance
# ---------------------------------------------------------------------------


def test_coerce_distance_accepts_strings_and_enum():
    assert _coerce_distance("Cosine") is rest.Distance.COSINE
    assert _coerce_distance("cosine") is rest.Distance.COSINE
    assert _coerce_distance(rest.Distance.DOT) is rest.Distance.DOT


def test_coerce_distance_rejects_garbage():
    with pytest.raises(ValueError):
        _coerce_distance("banana")


# ---------------------------------------------------------------------------
# quantization / hnsw / optimizer config builders
# ---------------------------------------------------------------------------


def test_quantization_config_disabled_returns_none():
    assert _quantization_config({"quantization_enabled": False}) is None


def test_quantization_config_int8_defaults():
    cfg = _quantization_config({"quantization_enabled": True})
    assert isinstance(cfg, rest.ScalarQuantization)
    assert cfg.scalar.type is rest.ScalarType.INT8
    # defaults: quantile=0.99, always_ram=True
    assert cfg.scalar.quantile == pytest.approx(0.99)
    assert cfg.scalar.always_ram is True


def test_quantization_config_binary():
    cfg = _quantization_config(
        {"quantization_enabled": True, "quantization_type": "binary", "quantization_always_ram": False}
    )
    assert isinstance(cfg, rest.BinaryQuantization)
    assert cfg.binary.always_ram is False


def test_quantization_config_unknown_type_raises():
    with pytest.raises(ValueError):
        _quantization_config({"quantization_enabled": True, "quantization_type": "product"})


def test_hnsw_config_defaults_on_disk():
    cfg = _hnsw_config({})
    assert cfg.m == 16
    assert cfg.ef_construct == 100
    assert cfg.on_disk is True


def test_optimizer_config_defaults():
    cfg = _optimizers_config({})
    assert cfg.default_segment_number == 2
    assert cfg.memmap_threshold == 20480


# ---------------------------------------------------------------------------
# _merge_tenant_filter
# ---------------------------------------------------------------------------


def _tenant_cond(value: str) -> rest.FieldCondition:
    return rest.FieldCondition(key=TENANT_PAYLOAD_KEY, match=rest.MatchValue(value=value))


def test_merge_tenant_filter_without_user_filter():
    out = _merge_tenant_filter(None, "colabc")
    assert isinstance(out, rest.Filter)
    assert out.must == [_tenant_cond("colabc")]


def test_merge_tenant_filter_without_tenant_is_noop():
    # If tenant_id is falsy we must not alter the user's filter.
    existing = rest.Filter(must=[rest.FieldCondition(key="foo", match=rest.MatchValue(value="bar"))])
    assert _merge_tenant_filter(existing, None) is existing
    assert _merge_tenant_filter(None, None) is None


def test_merge_tenant_filter_preserves_should_and_must_not():
    user = rest.Filter(
        must=[rest.FieldCondition(key="chat_id", match=rest.MatchValue(value="cX"))],
        should=[rest.FieldCondition(key="indexer", match=rest.MatchValue(value="vector"))],
    )
    out = _merge_tenant_filter(user, "colabc")

    assert isinstance(out, rest.Filter)
    # tenant clause appended to must
    assert len(out.must) == 2
    assert out.must[-1] == _tenant_cond("colabc")
    # should clause untouched
    assert out.should == user.should


def test_merge_tenant_filter_handles_unknown_filter_type():
    # If something non-Filter slips through we drop it (logging a warning) and
    # return only the tenant guard; we must never try to stuff arbitrary objects
    # into rest.Filter.must since pydantic will reject them at the API boundary.
    foreign: Dict[str, Any] = {"weird": True}
    out = _merge_tenant_filter(foreign, "colabc")
    assert isinstance(out, rest.Filter)
    assert out.must == [_tenant_cond("colabc")]
