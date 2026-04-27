# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for graphindex DTOs.

The DTO layer is pure validation code; these tests exist to pin the
invariants that downstream code (and future contributors) rely on."""

from __future__ import annotations

import pytest

from aperag.domains.knowledge_graph.graphindex.dto import (
    DESCRIPTION_SEPARATOR,
    Chunk,
    Entity,
    GraphContext,
    KnowledgeGraph,
    MergeEntitiesResult,
    Relation,
)


class TestChunk:
    def test_rejects_empty_chunk_id(self):
        with pytest.raises(ValueError, match="chunk_id"):
            Chunk(
                chunk_id="",
                doc_id="d",
                collection_id="c",
                order_in_doc=0,
                text="hello",
            )

    def test_rejects_negative_order(self):
        with pytest.raises(ValueError, match="order_in_doc"):
            Chunk(
                chunk_id="c1",
                doc_id="d",
                collection_id="c",
                order_in_doc=-1,
                text="hello",
            )

    def test_is_hashable_and_structurally_equal(self):
        a = Chunk(chunk_id="c1", doc_id="d", collection_id="c", order_in_doc=0, text="t")
        b = Chunk(chunk_id="c1", doc_id="d", collection_id="c", order_in_doc=0, text="t")
        assert a == b
        assert {a, b} == {a}


class TestEntity:
    def test_rejects_empty_id(self):
        with pytest.raises(ValueError):
            Entity(entity_id="", collection_id="c", name="n", type="person", description="")

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            Entity(entity_id="e1", collection_id="c", name="", type="person", description="")

    def test_source_chunks_coerced_to_tuple(self):
        """Mutability is a footgun on frozen DTOs; the constructor
        coerces any sequence to a tuple so equality / hash stays stable
        regardless of what the caller passed in."""
        e = Entity(
            entity_id="e1",
            collection_id="c",
            name="n",
            type="person",
            description="",
            source_chunk_ids=["c1", "c2"],
        )
        assert e.source_chunk_ids == ("c1", "c2")
        # Pass the SAME data again as a tuple; equality must match.
        e2 = Entity(
            entity_id="e1",
            collection_id="c",
            name="n",
            type="person",
            description="",
            source_chunk_ids=("c1", "c2"),
        )
        assert e == e2


class TestRelation:
    def test_rejects_self_loop(self):
        """Self-loops are meaningless in a document-derived knowledge
        graph and usually indicate a prompt / parsing bug upstream. We
        refuse them at DTO construction so bugs surface loud."""
        with pytest.raises(ValueError, match="self-loop"):
            Relation(
                collection_id="c",
                source_id="e1",
                target_id="e1",
                description="bad",
            )

    def test_rejects_empty_endpoint(self):
        with pytest.raises(ValueError):
            Relation(collection_id="c", source_id="", target_id="e2", description="")


class TestCompoundResults:
    def test_graph_context_sequences_coerced(self):
        """Important: callers frequently build these with lists; the
        frozen DTO must normalize so equality / hashing works."""
        gc = GraphContext(text="x", entities=[], relations=[], chunks=[])
        assert isinstance(gc.entities, tuple)
        assert isinstance(gc.relations, tuple)
        assert isinstance(gc.chunks, tuple)

    def test_knowledge_graph_defaults(self):
        kg = KnowledgeGraph(nodes=[], edges=[])
        assert kg.is_truncated is False

    def test_merge_entities_result_carries_merge_payload(self):
        """The service layer reads ``description`` and
        ``source_chunk_ids`` off ``MergeEntitiesResult`` to decide
        whether to trigger LLM summarization. Pin the fields so a later
        refactor can't silently drop one."""
        r = MergeEntitiesResult(
            target_entity_id="t",
            merged_source_ids=("s1", "s2"),
            description="combined",
            source_chunk_ids=("c1", "c2"),
            edges_redirected=3,
            edges_collapsed=1,
        )
        assert r.target_entity_id == "t"
        assert r.description == "combined"
        assert r.source_chunk_ids == ("c1", "c2")

    def test_description_separator_is_double_newline(self):
        """Downstream code (service, storage, prompts) all rely on this
        separator being a double-newline. Pin it so a refactor can't
        change it by accident."""
        assert DESCRIPTION_SEPARATOR == "\n\n"
