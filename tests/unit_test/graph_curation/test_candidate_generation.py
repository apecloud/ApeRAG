# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from aperag.graph_curation.candidate_generation import build_candidate_pairs
from aperag.graph_curation.dto import CurationEntity as Entity


def _entity(
    entity_id: str,
    name: str,
    *,
    entity_type: str = "organization",
    description: str = "",
    chunk_ids: tuple[str, ...] = (),
) -> Entity:
    return Entity(
        entity_id=entity_id,
        collection_id="c1",
        name=name,
        type=entity_type,
        description=description,
        source_chunk_ids=chunk_ids,
    )


def test_build_candidate_pairs_prefers_same_type_pairs_with_real_signals():
    bookstore = _entity(
        "e1",
        "墨香居",
        description="这条老巷子里唯一的旧书店",
        chunk_ids=("chunk-a", "chunk-b"),
    )
    old_bookstore = _entity(
        "e2",
        "旧书店",
        description="经营各种书籍的小店",
        chunk_ids=("chunk-b",),
    )
    person = _entity(
        "e3",
        "李明华",
        entity_type="person",
        description="书店老板",
    )

    pairs = build_candidate_pairs(
        entities=[bookstore, old_bookstore, person],
        vector_neighbors={"e1": [("e2", 0.88)], "e3": [("e1", 0.99)]},
        max_pairs_per_entity=4,
        max_total_pairs=10,
    )

    assert len(pairs) == 1
    pair = pairs[0]
    assert (pair.left_id, pair.right_id) == ("e1", "e2")
    assert pair.signals["vector_neighbor"] is True
    assert pair.signals["vector_score"] == 0.88
    assert pair.signals["shared_chunk_count"] == 1
    assert pair.score > 0.0


def test_build_candidate_pairs_caps_pairs_per_entity():
    entities = [
        _entity("e1", "Acme"),
        _entity("e2", "Acme Inc"),
        _entity("e3", "Acme Labs"),
    ]

    pairs = build_candidate_pairs(
        entities=entities,
        vector_neighbors={},
        max_pairs_per_entity=1,
        max_total_pairs=10,
    )

    assert len(pairs) == 1
    pair = pairs[0]
    assert {pair.left_id, pair.right_id}.issubset({"e1", "e2", "e3"})
