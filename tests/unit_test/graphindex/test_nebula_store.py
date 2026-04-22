# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from aperag.graphindex.storage.nebula import NebulaGraphStore


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

    def fake_execute(space: str, stmt: str):
        execute_calls.append((space, stmt))
        return None

    def fake_execute_multi(space: str, stmts: list[str]):
        execute_multi_calls.append((space, stmts))
        outcome = next(outcomes)
        if outcome is not None:
            raise outcome
        return None

    monkeypatch.setattr(store, "_execute", fake_execute)
    monkeypatch.setattr(store, "_execute_multi", fake_execute_multi)
    monkeypatch.setattr("aperag.graphindex.storage.nebula.time.sleep", lambda _seconds: None)

    space = store._ensure_space("compat_test_demo")

    assert space == "compat_test_compat_test_demo"
    assert execute_calls == [
        (
            "",
            "CREATE SPACE IF NOT EXISTS `compat_test_compat_test_demo` "
            "(vid_type=FIXED_STRING(128), partition_num=1, replica_factor=1)",
        )
    ]
    assert len(execute_multi_calls) == 2

    schema_stmts = execute_multi_calls[-1][1]
    assert any("CREATE EDGE IF NOT EXISTS `relates_to`" in stmt for stmt in schema_stmts)
    assert all("CREATE EDGE TYPE" not in stmt for stmt in schema_stmts)
