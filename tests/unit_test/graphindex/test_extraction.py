# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Extraction layer tests with a mocked LLM.

The LLM is passed in as a callable, so tests can simulate every
failure mode (good JSON, malformed JSON, exception, forbidden entity
type, self-referential relation) without any real API contact."""

from __future__ import annotations

import json

import pytest

from aperag.domains.knowledge_graph.graphindex.config import GraphIndexConfig
from aperag.domains.knowledge_graph.graphindex.dto import Chunk
from aperag.domains.knowledge_graph.graphindex.engine.extraction import (
    extract_from_chunk,
    normalize_entity_id,
)


def _chunk(text: str = "hello world") -> Chunk:
    return Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        collection_id="col-1",
        order_in_doc=0,
        text=text,
    )


def _cfg(**overrides) -> GraphIndexConfig:
    base = dict(
        llm_max_retries=0,
        max_entities_per_chunk=10,
        max_relations_per_chunk=10,
        entity_types=("person", "organization"),
    )
    base.update(overrides)
    return GraphIndexConfig(**base)


def _mock_llm(response: str):
    async def _call(_prompt: str) -> str:
        return response

    return _call


def _raising_llm(exc: Exception):
    async def _call(_prompt: str) -> str:
        raise exc

    return _call


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extracts_clean_json_output():
    llm = _mock_llm(
        json.dumps(
            {
                "entities": [
                    {"name": "Alice", "type": "person", "description": "A researcher"},
                    {"name": "Acme", "type": "organization", "description": "A company"},
                ],
                "relations": [
                    {
                        "source": "Alice",
                        "target": "Acme",
                        "description": "Works at",
                        "weight": 8,
                    }
                ],
            }
        )
    )

    entities, relations = await extract_from_chunk(chunk=_chunk(), config=_cfg(), llm=llm)
    assert len(entities) == 2
    assert {e.name for e in entities} == {"Alice", "Acme"}
    assert all(e.source_chunk_ids == ("chunk-1",) for e in entities)

    assert len(relations) == 1
    assert relations[0].weight == 8.0
    assert relations[0].source_chunk_ids == ("chunk-1",)


@pytest.mark.asyncio
async def test_tolerates_markdown_fenced_json():
    """LLM occasionally wraps JSON in ```json fences despite the
    response_format flag. The parser must tolerate this without
    needing a custom stripping layer in every caller."""
    llm = _mock_llm(
        "```json\n"
        + json.dumps(
            {
                "entities": [{"name": "X", "type": "person", "description": "d"}],
                "relations": [],
            }
        )
        + "\n```"
    )
    entities, relations = await extract_from_chunk(chunk=_chunk(), config=_cfg(), llm=llm)
    assert len(entities) == 1
    assert relations == []


@pytest.mark.asyncio
async def test_empty_response_returns_empty_lists():
    """Boilerplate chunks (page footers, blank separators) may
    legitimately extract nothing."""
    llm = _mock_llm(json.dumps({"entities": [], "relations": []}))
    entities, relations = await extract_from_chunk(chunk=_chunk(), config=_cfg(), llm=llm)
    assert entities == []
    assert relations == []


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_forbidden_entity_type():
    """The prompt enumerates allowed types; anything else must be
    silently dropped so a prompt regression can't pollute the graph."""
    llm = _mock_llm(
        json.dumps(
            {
                "entities": [
                    {"name": "X", "type": "alien", "description": ""},
                    {"name": "Alice", "type": "person", "description": ""},
                ],
                "relations": [],
            }
        )
    )
    entities, _ = await extract_from_chunk(chunk=_chunk(), config=_cfg(), llm=llm)
    assert [e.name for e in entities] == ["Alice"]


@pytest.mark.asyncio
async def test_drops_self_referential_relations():
    """Self-loops are forbidden by the prompt but LLMs slip up."""
    llm = _mock_llm(
        json.dumps(
            {
                "entities": [{"name": "Alice", "type": "person", "description": ""}],
                "relations": [
                    {
                        "source": "Alice",
                        "target": "Alice",
                        "description": "self",
                        "weight": 5,
                    }
                ],
            }
        )
    )
    _, relations = await extract_from_chunk(chunk=_chunk(), config=_cfg(), llm=llm)
    assert relations == []


@pytest.mark.asyncio
async def test_drops_relations_with_unknown_endpoints():
    """A relation pointing at an entity we didn't actually capture is
    garbage — the prompt is explicit that both endpoints must be in
    the entity list."""
    llm = _mock_llm(
        json.dumps(
            {
                "entities": [{"name": "Alice", "type": "person", "description": ""}],
                "relations": [
                    {
                        "source": "Alice",
                        "target": "Ghost",
                        "description": "?",
                        "weight": 5,
                    }
                ],
            }
        )
    )
    _, relations = await extract_from_chunk(chunk=_chunk(), config=_cfg(), llm=llm)
    assert relations == []


@pytest.mark.asyncio
async def test_malformed_json_returns_empty_not_raise():
    """A single chunk's malformed output must not fail the whole
    document. Indexer relies on this."""
    llm = _mock_llm("not valid json at all { ")
    entities, relations = await extract_from_chunk(chunk=_chunk(), config=_cfg(), llm=llm)
    assert (entities, relations) == ([], [])


@pytest.mark.asyncio
async def test_llm_exception_propagates_when_retries_exhausted():
    """If the caller set ``llm_max_retries=0`` and the LLM raises, we
    DO propagate — this gives the indexer a clear signal to decide
    whether to fail the whole batch."""
    llm = _raising_llm(RuntimeError("network out"))
    with pytest.raises(RuntimeError, match="network out"):
        await extract_from_chunk(chunk=_chunk(), config=_cfg(), llm=llm)


@pytest.mark.asyncio
async def test_llm_caps_honoured():
    """A pathological LLM might return 500 entities; caps keep us safe."""
    many = {
        "entities": [{"name": f"E{i}", "type": "person", "description": ""} for i in range(100)],
        "relations": [],
    }
    llm = _mock_llm(json.dumps(many))
    entities, _ = await extract_from_chunk(chunk=_chunk(), config=_cfg(max_entities_per_chunk=5), llm=llm)
    assert len(entities) == 5


# ---------------------------------------------------------------------------
# normalize_entity_id stability
# ---------------------------------------------------------------------------


def test_entity_id_is_stable_across_casing_and_whitespace():
    """Same entity written two ways → same id, so upsert de-dupes."""
    assert normalize_entity_id("col-1", "Alice") == normalize_entity_id("col-1", "  alice  ")


def test_entity_id_collection_isolated():
    """Different collections with the same entity name must produce
    different ids to prevent cross-tenant bleed through."""
    assert normalize_entity_id("col-a", "Alice") != normalize_entity_id("col-b", "Alice")
