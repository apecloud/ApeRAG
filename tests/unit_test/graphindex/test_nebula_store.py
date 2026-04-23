# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import pytest

from aperag.domains.knowledge_graph.graphindex.dto import Chunk
from aperag.domains.knowledge_graph.graphindex.storage.nebula import NebulaGraphStore, _escape


def test_ensure_space_retries_visibility_window_and_uses_create_edge(monkeypatch):
    store = NebulaGraphStore(hosts="127.0.0.1:9669", space_prefix="compat_test")

    execute_calls: list[tuple[str, str]] = []
    execute_multi_calls: list[tuple[str, list[str]]] = []
    outcomes = iter(
        [
            RuntimeError("Nebula USE failed: SpaceNotFound: SpaceName `compat_test_compat_test_demo`"),
            None,
        ]
    )

    class _Value:
        def __init__(self, value: str):
            self._value = value

        def is_string(self) -> bool:
            return True

        def as_string(self) -> str:
            return self._value

    class _ShowSpacesResult:
        def row_size(self) -> int:
            return 1

        def row_values(self, index: int):
            assert index == 0
            return [_Value("compat_test_compat_test_demo")]

    def fake_execute(space: str, stmt: str):
        execute_calls.append((space, stmt))
        if stmt == "SHOW SPACES":
            return _ShowSpacesResult()
        return None

    def fake_execute_multi(space: str, stmts: list[str]):
        execute_multi_calls.append((space, stmts))
        outcome = next(outcomes)
        if outcome is not None:
            raise outcome
        return None

    monkeypatch.setattr(store, "_execute", fake_execute)
    monkeypatch.setattr(store, "_execute_multi", fake_execute_multi)
    monkeypatch.setattr("aperag.domains.knowledge_graph.graphindex.storage.nebula.time.sleep", lambda _seconds: None)

    space = store._ensure_space("compat_test_demo")

    assert space == "compat_test_compat_test_demo"
    assert (
        "",
        "CREATE SPACE IF NOT EXISTS `compat_test_compat_test_demo` "
        "(vid_type=FIXED_STRING(128), partition_num=1, replica_factor=1)",
    ) in execute_calls
    assert ("", "SHOW SPACES") in execute_calls
    assert len(execute_multi_calls) == 2

    schema_stmts = execute_multi_calls[-1][1]
    assert any("CREATE EDGE IF NOT EXISTS `relates_to`" in stmt for stmt in schema_stmts)
    assert all("CREATE EDGE TYPE" not in stmt for stmt in schema_stmts)
    assert any("CREATE TAG INDEX IF NOT EXISTS `idx_entity_name`" in stmt for stmt in schema_stmts)


def test_escape_encodes_control_characters_for_ngql_strings():
    assert _escape('target entity\n\nsource "one"\\two') == 'target entity\\n\\nsource \\"one\\"\\\\two'


@pytest.mark.asyncio
async def test_upsert_chunks_retries_transient_schema_visibility_error(monkeypatch):
    store = NebulaGraphStore(hosts="127.0.0.1:9669", space_prefix="compat_test")
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        collection_id="compat_test_demo",
        order_in_doc=0,
        text="Alice met Bob at Acme Labs.",
    )

    execute_calls: list[tuple[str, str]] = []
    outcomes = iter(
        [
            RuntimeError("Nebula query failed: SemanticError: No schema found for `chunk'"),
            None,
        ]
    )

    monkeypatch.setattr(store, "_ensure_space", lambda _collection_id: "compat_test_compat_test_demo")

    def fake_execute(space: str, stmt: str):
        execute_calls.append((space, stmt))
        outcome = next(outcomes)
        if outcome is not None:
            raise outcome
        return None

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(store, "_execute", fake_execute)
    monkeypatch.setattr("aperag.domains.knowledge_graph.graphindex.storage.nebula.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr("aperag.domains.knowledge_graph.graphindex.storage.nebula.time.sleep", lambda _seconds: None)

    await store.upsert_chunks("compat_test_demo", [chunk])

    assert len(execute_calls) == 2
    assert all(space == "compat_test_compat_test_demo" for space, _stmt in execute_calls)
    assert all("INSERT VERTEX IF NOT EXISTS `chunk`" in stmt for _space, stmt in execute_calls)
