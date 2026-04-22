# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the backend-neutral DTOs + ``flatten_node_payload`` helper.

These tests pin the *invariants* of the DTO module so that any refactor
(add a field, split a type, rename) doesn't silently change observable
behavior.
"""

from __future__ import annotations

import json

import pytest

from aperag.vectorstore.dto import (
    QueryRequest,
    SearchHit,
    TenantRef,
    VectorPoint,
    VectorShape,
    flatten_node_payload,
)

# ---------------------------------------------------------------------------
# VectorShape
# ---------------------------------------------------------------------------


def test_vector_shape_normalizes_distance_case():
    assert VectorShape(size=1024, distance="Cosine").canonical == "cosine"
    assert VectorShape(size=1024, distance="COSINE").canonical == "cosine"
    assert VectorShape(size=1024, distance="dot").canonical == "dot"


def test_vector_shape_accepts_euclidian_typo():
    """``euclidian`` is a common mis-spelling we've seen in configs; accept
    and normalize rather than force users to find the typo."""
    assert VectorShape(size=1024, distance="euclidian").canonical == "euclid"


def test_vector_shape_rejects_invalid_size():
    with pytest.raises(ValueError, match="positive int"):
        VectorShape(size=0, distance="cosine")
    with pytest.raises(ValueError, match="positive int"):
        VectorShape(size=-1, distance="cosine")


def test_vector_shape_is_hashable_and_equal():
    a = VectorShape(size=1024, distance="Cosine")
    b = VectorShape(size=1024, distance="cosine")
    assert a == b
    assert {a, b} == {a}


# ---------------------------------------------------------------------------
# TenantRef
# ---------------------------------------------------------------------------


def test_tenant_ref_rejects_empty_string():
    with pytest.raises(ValueError, match="non-empty"):
        TenantRef(id="")


# ---------------------------------------------------------------------------
# VectorPoint
# ---------------------------------------------------------------------------


def test_vector_point_rejects_non_list_vector():
    """The vector must be ``list[float]``. Numpy arrays or tuples are
    quietly coerced everywhere downstream — we reject at construction so
    the error surfaces close to the bug."""
    with pytest.raises(TypeError):
        VectorPoint(id="x", vector="not a list", payload={})  # type: ignore[arg-type]


def test_vector_point_requires_non_empty_id():
    with pytest.raises(ValueError, match="non-empty"):
        VectorPoint(id="", vector=[0.0], payload={})


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------


def test_query_request_rejects_empty_embedding():
    with pytest.raises(ValueError, match="non-empty"):
        QueryRequest(embedding=[], top_k=5)


def test_query_request_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="positive int"):
        QueryRequest(embedding=[0.1], top_k=0)


def test_query_request_hints_default_is_independent_per_instance():
    """Regression guard: a mutable default (``{}``) would be shared
    across instances. ``field(default_factory=dict)`` prevents that."""
    a = QueryRequest(embedding=[0.1], top_k=1)
    b = QueryRequest(embedding=[0.1], top_k=1)
    # Immutability: frozen dataclass hints dict is shared by reference
    # only when both constructors got the same default-factory output,
    # which they shouldn't (different calls).
    assert a.hints is not b.hints or a.hints == {}


# ---------------------------------------------------------------------------
# SearchHit
# ---------------------------------------------------------------------------


def test_search_hit_vector_optional():
    hit = SearchHit(id="x", score=0.8, payload={"k": "v"})
    assert hit.vector is None


# ---------------------------------------------------------------------------
# flatten_node_payload
# ---------------------------------------------------------------------------


def test_flatten_prefers_flat_top_level_fields():
    """New data writes ``{text, metadata}`` at the top level. Flattening
    is a no-op for that shape."""
    payload = {"text": "body", "metadata": {"source": "doc.md"}}
    flat = flatten_node_payload(payload)
    assert flat["text"] == "body"
    assert flat["metadata"] == {"source": "doc.md"}


def test_flatten_reads_legacy_node_content_blob():
    """Old data (LlamaIndex QdrantVectorStore writes) had only
    ``_node_content`` as a JSON string. Readers must continue to extract
    text/metadata from it without caller-side branching."""
    legacy = {"_node_content": json.dumps({"text": "hi", "metadata": {"source": "old.md", "chunk_id": 1}})}
    flat = flatten_node_payload(legacy)
    assert flat["text"] == "hi"
    assert flat["metadata"]["source"] == "old.md"
    assert flat["metadata"]["chunk_id"] == 1


def test_flatten_prefers_flat_over_legacy_when_both_present():
    """Mixed data (a record was updated after the refactor but the old
    ``_node_content`` is still there) should use the new flat fields as
    the source of truth — they're what the latest writer intended."""
    payload = {
        "text": "new",
        "metadata": {"source": "new.md"},
        "_node_content": json.dumps({"text": "old", "metadata": {"source": "old.md"}}),
    }
    flat = flatten_node_payload(payload)
    assert flat["text"] == "new"
    assert flat["metadata"]["source"] == "new.md"


def test_flatten_extracts_source_from_relationships_when_metadata_missing():
    """Source-derivation fallback: old LlamaIndex writes encoded the
    parent document's source under ``relationships['1'].metadata.source``.
    Preserved so doc-preview pages still show a filename for legacy data."""
    legacy = {
        "_node_content": json.dumps(
            {
                "text": "chunk",
                "metadata": {},  # note: metadata has no 'source'
                "relationships": {"1": {"metadata": {"source": "/tmp/foo/bar.md"}}},
            }
        )
    }
    flat = flatten_node_payload(legacy)
    assert flat["metadata"].get("source") == "bar.md"


def test_flatten_handles_non_dict_payload():
    """Defensive: ``None`` / list / scalar inputs don't crash the
    flattener. Returns empty."""
    assert flatten_node_payload(None).get("text") is None  # type: ignore[arg-type]
    assert flatten_node_payload([]).get("metadata") == {}  # type: ignore[arg-type]


def test_flatten_handles_malformed_node_content_gracefully():
    """If ``_node_content`` isn't valid JSON (shouldn't happen in
    practice, but...) we fall back to the raw payload rather than raise."""
    payload = {"_node_content": "{ not valid json", "text": "fallback"}
    flat = flatten_node_payload(payload)
    assert flat["text"] == "fallback"
