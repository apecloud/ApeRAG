# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Behavioral tests for ``aperag.config.build_vector_db_context``.

These tests guard the contract that every call site depends on:

* ``collection`` always ends up in ctx as the tenant id (even when overridden
  by VECTOR_DB_CONTEXT).
* All optimization flags (multitenant / quantization / HNSW / segments /
  mmap / payload on_disk) appear in ctx with their settings-level defaults so
  downstream ``QdrantVectorStoreConnector`` never has to fall back to hard-
  coded defaults.
* Per-call ctx overrides win over settings-level defaults (important for the
  migration script which explicitly disables multitenant for source reads).
"""

from __future__ import annotations

import json

import pytest

from aperag.config import build_vector_db_context, settings


@pytest.fixture
def clean_vector_db_context(monkeypatch):
    """Reset VECTOR_DB_CONTEXT to a known minimal shape for each test."""
    monkeypatch.setattr(
        settings,
        "vector_db_context",
        json.dumps({"url": "http://example", "port": 6333, "distance": "Cosine"}),
    )


def test_build_context_injects_collection_and_vector_size(clean_vector_db_context):
    ctx = build_vector_db_context("coltest", vector_size=1024)
    assert ctx["collection"] == "coltest"
    assert ctx["vector_size"] == 1024


def test_build_context_propagates_all_optimization_flags(clean_vector_db_context, monkeypatch):
    # explicit settings values to avoid drift if defaults change
    monkeypatch.setattr(settings, "qdrant_multitenant", True)
    monkeypatch.setattr(settings, "qdrant_quantization_enabled", True)
    monkeypatch.setattr(settings, "qdrant_quantization_type", "int8")
    monkeypatch.setattr(settings, "qdrant_quantization_quantile", 0.98)
    monkeypatch.setattr(settings, "qdrant_quantization_always_ram", False)
    monkeypatch.setattr(settings, "qdrant_hnsw_on_disk", True)
    monkeypatch.setattr(settings, "qdrant_default_segment_number", 3)
    monkeypatch.setattr(settings, "qdrant_mmap_threshold_kb", 10000)
    monkeypatch.setattr(settings, "qdrant_vectors_on_disk", True)
    monkeypatch.setattr(settings, "qdrant_on_disk_payload", True)

    ctx = build_vector_db_context("col1", vector_size=1024)

    assert ctx["multitenant"] is True
    assert ctx["quantization_enabled"] is True
    assert ctx["quantization_type"] == "int8"
    assert ctx["quantization_quantile"] == pytest.approx(0.98)
    assert ctx["quantization_always_ram"] is False
    assert ctx["hnsw_on_disk"] is True
    assert ctx["default_segment_number"] == 3
    assert ctx["mmap_threshold_kb"] == 10000
    assert ctx["vectors_on_disk"] is True
    assert ctx["on_disk_payload"] is True


def test_build_context_populates_pgvector_knobs_even_for_qdrant_deploys(clean_vector_db_context):
    """pgvector settings are merged into ctx regardless of which backend
    the deployment is configured for. Each connector reads only the keys
    it understands, so having both Qdrant and pgvector knobs present at
    the same time is cheap and makes ``VECTOR_DB_TYPE=pgvector`` a
    one-flag flip."""
    ctx = build_vector_db_context("col1", vector_size=1024)
    # Either explicit PGVECTOR_DATABASE_URL wins, or we fall back to the
    # main database_url — never empty.
    assert ctx["pgvector_database_url"]
    assert isinstance(ctx["pgvector_hnsw_m"], int)
    assert isinstance(ctx["pgvector_hnsw_ef_construction"], int)
    assert isinstance(ctx["pgvector_hnsw_ef_search"], int)


def test_build_context_respects_ctx_override(monkeypatch):
    # If the operator pins a flag inside VECTOR_DB_CONTEXT JSON, it must win
    # over the settings-level default. This is how the migration script tells
    # the connector "read me in legacy mode even though the global default is
    # multitenant".
    monkeypatch.setattr(
        settings,
        "vector_db_context",
        json.dumps({"url": "http://example", "port": 6333, "multitenant": False, "distance": "Cosine"}),
    )
    monkeypatch.setattr(settings, "qdrant_multitenant", True)

    ctx = build_vector_db_context("col1", vector_size=1024)
    assert ctx["multitenant"] is False  # ctx override wins
