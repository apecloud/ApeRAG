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
    # No filterable keys are present in metadata here, so the payload
    # stays at exactly {text, metadata}.
    assert vp.payload == {
        "text": "body",
        "metadata": {"source": "doc.md", "chunk_id": 3},
    }


def test_node_to_vector_point_lifts_filterable_keys_to_top_level():
    """The filter DSL addresses flat payload keys (``indexer`` / ``chat_id`` /
    ``collection_id``). If we only stored them inside ``metadata``, every
    ``Eq("indexer", ...)`` and ``Eq("chat_id", ...)`` filter on real writer
    output would silently miss. Mirror those keys at the top level — and
    only those — so the existing translator contract keeps working."""
    node = _node(
        text="body",
        metadata={
            "source": "doc.md",
            "indexer": "vision",
            "chat_id": "chat_abc",
            "collection_id": "col_xyz",
        },
        embedding=[0.1, 0.2, 0.3],
    )
    vp = node_to_vector_point(node)
    # Top-level mirror for filter translator.
    assert vp.payload["indexer"] == "vision"
    assert vp.payload["chat_id"] == "chat_abc"
    assert vp.payload["collection_id"] == "col_xyz"
    # Nested copy preserved for reader code.
    assert vp.payload["metadata"]["indexer"] == "vision"
    assert vp.payload["metadata"]["chat_id"] == "chat_abc"
    assert vp.payload["metadata"]["collection_id"] == "col_xyz"
    # Other metadata keys are NOT flattened — we only lift the ones the
    # filter DSL actually addresses.
    assert "source" not in vp.payload
    assert vp.payload["metadata"]["source"] == "doc.md"


def test_node_to_vector_point_omits_absent_filterable_keys_from_top_level():
    """We deliberately leave the top-level key missing (instead of writing
    ``None``) when a filterable key isn't set on the node. That way
    ``IsEmpty(indexer)`` behaves the same on adapter-produced payloads as
    it did on hand-written flat seeds."""
    node = _node(
        text="body",
        metadata={"source": "doc.md"},
        embedding=[0.1],
    )
    vp = node_to_vector_point(node)
    assert "indexer" not in vp.payload
    assert "chat_id" not in vp.payload
    assert "collection_id" not in vp.payload


def test_node_to_vector_point_lifts_collection_id_stamped_by_tenant():
    """When the caller relies on ``tenant_id`` to stamp ``collection_id``,
    the lifted top-level key must reflect the stamped value, not stay
    empty. Otherwise multitenant search filters would still miss new data
    even though the metadata is correct."""
    node = _node(text="t", metadata={"source": "x"}, embedding=[0.1])
    vp = node_to_vector_point(node, tenant_id="col_xyz")
    assert vp.payload["metadata"]["collection_id"] == "col_xyz"
    assert vp.payload["collection_id"] == "col_xyz"


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
