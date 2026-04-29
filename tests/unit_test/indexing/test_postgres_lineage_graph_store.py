from __future__ import annotations

from types import SimpleNamespace

import pytest

from aperag.indexing.graph_storage.postgres import PostgresLineageGraphStore


class _FakeConnection:
    def __init__(self, results: list[list[SimpleNamespace]]) -> None:
        self._results = results

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def execute(self, _stmt):
        return self._results.pop(0)


class _FakeEngine:
    def __init__(self, results: list[list[SimpleNamespace]]) -> None:
        self._results = results

    def connect(self):
        return _FakeConnection(self._results)


def _entity_row(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        entity_type="person",
        source_lineage=[],
        description_parts=[],
        compacted_description=None,
    )


def _relation_row(source: str, target: str, relation_type: str = "knows") -> SimpleNamespace:
    return SimpleNamespace(
        source=source,
        target=target,
        relation_type=relation_type,
        evidence_lineage=[],
        description_parts=[],
        compacted_description=None,
    )


@pytest.mark.asyncio
async def test_postgres_expand_neighbors_initializes_frontier_for_split_relation_queries():
    store = PostgresLineageGraphStore(
        engine=_FakeEngine(
            [
                [_entity_row("Alice")],
                [_relation_row("Alice", "Bob")],
                [],
                [_entity_row("Bob")],
            ]
        ),
        collection_id="col-test",
    )

    entities, relations = await store.expand_neighbors_n_hops(entity_names=["Alice"], hops=1)

    assert {entity.name for entity in entities} == {"Alice", "Bob"}
    assert {(relation.source, relation.target, relation.relation_type) for relation in relations} == {
        ("Alice", "Bob", "knows")
    }
