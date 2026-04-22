# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the process-level QdrantClient cache.

The cache is the only thing between "one HTTP/gRPC connection pool per
connector" and "one per endpoint"; without it, a read-heavy request path
pays a fresh TCP/TLS setup on every search.
"""

from __future__ import annotations

import threading

from aperag.vectorstore import qdrant_connector as qc


def setup_function(_fn) -> None:  # pytest-style per-test reset
    qc._reset_client_cache()


def teardown_function(_fn) -> None:
    qc._reset_client_cache()


def test_memory_url_is_never_cached():
    """``:memory:`` is used by tests for isolated per-test stores. Caching
    it would make the second test see state from the first, which is
    exactly the surprise mode we want to avoid."""
    a = qc._get_or_create_client(":memory:")
    b = qc._get_or_create_client(":memory:")
    assert a is not b


def test_same_endpoint_returns_same_client(monkeypatch):
    """Two connectors pointed at the same endpoint must share one client."""
    built = []

    def fake_ctor(*args, **kwargs):
        inst = object()
        built.append(inst)
        return inst

    import qdrant_client as _qc

    monkeypatch.setattr(_qc, "QdrantClient", fake_ctor)

    c1 = qc._get_or_create_client("http://qdrant.internal", port=6333)
    c2 = qc._get_or_create_client("http://qdrant.internal", port=6333)
    assert c1 is c2
    assert len(built) == 1, f"expected 1 client build, got {len(built)}"


def test_different_endpoints_build_different_clients(monkeypatch):
    """Cache key must include port/grpc_port/prefer_grpc/https/api_key so a
    staging + prod config in the same process doesn't alias."""
    built = []

    def fake_ctor(*args, **kwargs):
        inst = object()
        built.append(inst)
        return inst

    import qdrant_client as _qc

    monkeypatch.setattr(_qc, "QdrantClient", fake_ctor)

    a = qc._get_or_create_client("http://qdrant-staging", port=6333)
    b = qc._get_or_create_client("http://qdrant-prod", port=6333)
    c = qc._get_or_create_client("http://qdrant-staging", port=6333, prefer_grpc=True)
    d = qc._get_or_create_client("http://qdrant-staging", port=6333, api_key="secret")
    assert a is not b
    assert a is not c
    assert a is not d
    assert len({id(a), id(b), id(c), id(d)}) == 4


def test_concurrent_first_call_does_not_stampede(monkeypatch):
    """Under concurrent first access to a fresh endpoint, only one client
    should be built — the double-checked lock does its job."""
    built = []

    def fake_ctor(*args, **kwargs):
        inst = object()
        built.append(inst)
        return inst

    import qdrant_client as _qc

    monkeypatch.setattr(_qc, "QdrantClient", fake_ctor)

    clients = []

    def worker():
        clients.append(qc._get_or_create_client("http://qdrant-race", port=6333))

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(built) == 1, f"expected 1 client under race, got {len(built)}"
    # Every caller got the same instance.
    assert len({id(c) for c in clients}) == 1
