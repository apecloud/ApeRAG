# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Chunker unit tests.

We pass ``str.split`` as the tokenizer so tests are hermetic (no
tiktoken, no LLM), and assert the mechanical contract — order, overlap,
empty-input handling — rather than token-exact slicing."""

from __future__ import annotations

from aperag.domains.knowledge_graph.graphindex.engine.chunking import chunk_document


def _split(s: str) -> list[str]:
    """Whitespace tokenizer for tests."""
    return s.split()


def test_empty_content_returns_empty_list():
    """Image-only / whitespace-only documents must not crash; the
    caller treats an empty list as "nothing to index"."""
    assert (
        chunk_document(
            collection_id="c",
            doc_id="d",
            content="",
            tokenize=_split,
        )
        == []
    )
    assert (
        chunk_document(
            collection_id="c",
            doc_id="d",
            content="   \n\t  ",
            tokenize=_split,
        )
        == []
    )


def test_short_content_produces_single_chunk():
    chunks = chunk_document(
        collection_id="c",
        doc_id="d",
        content="one two three",
        chunk_token_size=10,
        chunk_overlap_token_size=2,
        tokenize=_split,
    )
    assert len(chunks) == 1
    assert chunks[0].order_in_doc == 0
    assert chunks[0].doc_id == "d"


def test_long_content_chunks_with_stable_ordering():
    """The order field must start at 0 and increment monotonically;
    downstream indexer relies on this to reconstruct reading order."""
    tokens = [f"t{i}" for i in range(25)]
    content = " ".join(tokens)
    chunks = chunk_document(
        collection_id="c",
        doc_id="d",
        content=content,
        chunk_token_size=10,
        chunk_overlap_token_size=3,
        tokenize=_split,
    )
    assert len(chunks) >= 2
    orders = [c.order_in_doc for c in chunks]
    assert orders == list(range(len(chunks)))
    assert all(c.collection_id == "c" for c in chunks)
    assert all(c.doc_id == "d" for c in chunks)


def test_chunk_ids_are_unique():
    """Every chunk gets a fresh UUID; repeated ingest of the same text
    must produce different ids so existing rows are not silently
    overwritten before explicit delete."""
    chunks_a = chunk_document(collection_id="c", doc_id="d", content="x y z", tokenize=_split)
    chunks_b = chunk_document(collection_id="c", doc_id="d", content="x y z", tokenize=_split)
    assert chunks_a[0].chunk_id != chunks_b[0].chunk_id


def test_rejects_overlap_not_smaller_than_window():
    """Otherwise the sliding-window loop would never advance."""
    import pytest

    with pytest.raises(ValueError, match="overlap"):
        chunk_document(
            collection_id="c",
            doc_id="d",
            content="some text",
            chunk_token_size=5,
            chunk_overlap_token_size=5,
            tokenize=_split,
        )
