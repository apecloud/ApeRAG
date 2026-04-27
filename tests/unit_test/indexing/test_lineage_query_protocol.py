# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wave 6 #33 chunk 1 — Protocol surface for the LightRAG-style query
layer on :class:`aperag.indexing.graph.LineageGraphStore`.

Pin the Protocol method signatures so chunk 2 (cross-backend impl) can
be reviewed against a stable surface, and so callers (chunk 3 caller
migration) can rely on the methods existing even before the per-backend
implementations land.

Per §K.11.11 sub-spec the read API is:
* ``query_entities_by_keyword(query, top_k)`` — lexical recall
* ``query_entities_by_vector(embedding, top_k)`` — vector recall
* ``expand_neighbors_n_hops(entity_names, hops)`` — graph traversal
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from aperag.indexing.graph import (
    InMemoryLineageGraphStore,
    LineageGraphStore,
)


def test_lineage_query_protocol_exposes_three_read_methods():
    """The new query layer surface must declare exactly the three
    LightRAG-style read methods agreed in §K.11.11. Adding a method
    here implies a chunk 2 backend implementation contract."""

    members = set(LineageGraphStore.__dict__)
    assert "query_entities_by_keyword" in members
    assert "query_entities_by_vector" in members
    assert "expand_neighbors_n_hops" in members


def test_query_by_keyword_signature_takes_query_and_top_k():
    sig = inspect.signature(LineageGraphStore.query_entities_by_keyword)
    params = sig.parameters
    assert "query" in params
    assert "top_k" in params
    # kw-only by spec — the retrieval pipeline always names the args.
    assert params["query"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["top_k"].kind is inspect.Parameter.KEYWORD_ONLY


def test_query_by_vector_signature_takes_embedding_and_top_k():
    sig = inspect.signature(LineageGraphStore.query_entities_by_vector)
    params = sig.parameters
    assert "embedding" in params
    assert "top_k" in params
    assert params["embedding"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["top_k"].kind is inspect.Parameter.KEYWORD_ONLY


def test_expand_neighbors_signature_takes_entity_names_and_hops():
    sig = inspect.signature(LineageGraphStore.expand_neighbors_n_hops)
    params = sig.parameters
    assert "entity_names" in params
    assert "hops" in params
    assert params["entity_names"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["hops"].kind is inspect.Parameter.KEYWORD_ONLY
    # Default hops=1 is the simple-stable-directive choice — caller
    # must opt into multi-hop explicitly.
    assert params["hops"].default == 1


def test_in_memory_store_query_methods_raise_not_implemented_in_chunk_1():
    """Chunk 1 ships Protocol stubs only — the in-memory test store
    must surface a clear NotImplementedError so callers (chunk 3
    migration) can rely on the gate firing if they accidentally
    invoke the API before chunk 2 lands."""

    store = InMemoryLineageGraphStore()

    async def _run() -> None:
        with pytest.raises(NotImplementedError) as exc:
            await store.query_entities_by_keyword(query="anything", top_k=5)
        assert "chunk 2" in str(exc.value)

        with pytest.raises(NotImplementedError) as exc:
            await store.query_entities_by_vector(embedding=[0.1, 0.2, 0.3], top_k=5)
        assert "chunk 2" in str(exc.value)

        with pytest.raises(NotImplementedError) as exc:
            await store.expand_neighbors_n_hops(entity_names=["X"], hops=1)
        assert "chunk 2" in str(exc.value)

    asyncio.run(_run())
