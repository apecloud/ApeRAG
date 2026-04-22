# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the LlamaIndex → VectorPoint adapter.

This module is the thin boundary between "everything that speaks
LlamaIndex" (indexers, embedders) and "everything that speaks
VectorPoint" (backend connectors). A regression here silently drops
metadata on the floor.
"""

from __future__ import annotations

import pytest
from llama_index.core.schema import TextNode

from aperag.vectorstore.dto import VectorPoint
from aperag.vectorstore.llama_index_adapter import (
    node_to_vector_point,
    nodes_to_vector_points,
)


def _node(text: str = "hello", metadata: dict | None = None, embedding: list | None = None) -> TextNode:
    n = TextNode(text=text, metadata=metadata or {})
    if embedding is not None:
        n.embedding = embedding
    return n


def test_node_without_embedding_is_rejected():
    """Writing an embedding-less node was a silent-corruption case in the
    LlamaIndex path (``store.add`` would insert NULL-vector rows). We
    surface it as a loud error instead."""
    with pytest.raises(ValueError, match="no embedding"):
        node_to_vector_point(_node())


def test_node_to_vector_point_captures_text_and_metadata_flat():
    node = _node(
        text="body",
        metadata={"source": "doc.md", "chunk_id": 3},
        embedding=[0.1, 0.2, 0.3],
    )
    vp = node_to_vector_point(node)
    assert isinstance(vp, VectorPoint)
    assert vp.vector == [0.1, 0.2, 0.3]
    # Flat shape: ``{text, metadata}``, NOT ``{_node_content: json_str}``.
    # The connector is free to add its own ``_node_content`` later if it
    # really wants to, but by default we emit the backend-neutral form.
    assert vp.payload == {
        "text": "body",
        "metadata": {"source": "doc.md", "chunk_id": 3},
    }


def test_node_to_vector_point_stamps_tenant_when_missing():
    """Defense-in-depth: upstream callers sometimes forget to stamp
    ``metadata.collection_id``. The adapter fills it in from ``tenant_id``
    so downstream multitenancy filters still work. Mirrors the old
    ``embedding_utils`` behavior exactly."""
    node = _node(text="t", metadata={"source": "x"}, embedding=[0.1])
    vp = node_to_vector_point(node, tenant_id="col_xyz")
    assert vp.payload["metadata"]["collection_id"] == "col_xyz"


def test_node_to_vector_point_respects_existing_tenant_id():
    node = _node(
        text="t",
        metadata={"source": "x", "collection_id": "col_original"},
        embedding=[0.1],
    )
    vp = node_to_vector_point(node, tenant_id="col_other")
    # Don't silently rewrite; callers that pre-set it chose their value.
    assert vp.payload["metadata"]["collection_id"] == "col_original"


def test_nodes_to_vector_points_preserves_order():
    nodes = [_node(text=str(i), embedding=[float(i)]) for i in range(5)]
    out = nodes_to_vector_points(nodes, tenant_id="col_a")
    assert [p.payload["text"] for p in out] == ["0", "1", "2", "3", "4"]
