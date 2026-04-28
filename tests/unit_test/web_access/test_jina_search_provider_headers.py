# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression test for JinaSearchProvider header configuration.

The pre-fix provider sent ``X-Return-Format: json`` which Jina
``s.jina.ai`` interprets as a markdown-format hint with a valid API
key — the response body comes back as ``text/plain`` and
``await response.json()`` then explodes, the provider returns ``[]``
silently, and ``_search_with_jina_fallback`` cascades to DuckDuckGo
(also unstable from the same network).

Fix: rely on ``Accept: application/json`` (set in ``__init__``) and
drop the misnamed ``X-Return-Format`` header. Pin the contract here
so a future regression that re-adds the bad header is caught
immediately.
"""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest

from aperag.domains.web_access.search.providers.jina_search_provider import (
    JinaSearchProvider,
)


class _StubResponse:
    def __init__(self, status: int, payload: Any, content_type: str = "application/json"):
        self.status = status
        self._payload = payload
        self.content_type = content_type

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _StubSession:
    def __init__(self, response: _StubResponse):
        self._response = response
        self.captured_headers: dict[str, str] | None = None
        self.captured_url: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def get(self, url, headers):
        self.captured_url = url
        self.captured_headers = dict(headers)
        return self._response


@pytest.fixture
def stub_response_ok():
    return _StubResponse(
        status=200,
        payload={
            "data": [
                {
                    "url": "https://example.com/zhang-fei-niurou",
                    "title": "Zhang Fei beef intro",
                    "description": "Sichuan smoked beef product overview.",
                },
                {
                    "url": "https://example.com/sichuan-cuisine",
                    "title": "Sichuan local specialties",
                    "description": "Background on regional cured meats.",
                },
            ]
        },
    )


@pytest.mark.asyncio
async def test_jina_search_does_not_send_x_return_format(stub_response_ok, monkeypatch):
    """The misnamed ``X-Return-Format: json`` header must NEVER be
    on the outbound request — Jina interprets it (with a valid API
    key) as "respond in markdown" and the JSON parse below fails."""
    provider = JinaSearchProvider(config={"api_key": "test-key"})

    captured: dict[str, Any] = {}

    def _client_session_factory(timeout=None):
        session = _StubSession(stub_response_ok)
        captured["session"] = session
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", _client_session_factory)

    results = await provider.search(query="test", max_results=2, timeout=10, locale="zh-CN")

    assert len(results) == 2
    assert "session" in captured
    headers = captured["session"].captured_headers
    assert headers is not None

    # The fix's invariant: no X-Return-Format header.
    assert "X-Return-Format" not in headers
    # And Accept must still be application/json so Jina returns JSON.
    assert headers.get("Accept") == "application/json"
    # Authorization should be present when an API key is configured.
    assert headers.get("Authorization") == "Bearer test-key"


@pytest.mark.asyncio
async def test_jina_search_parses_valid_json_response(stub_response_ok, monkeypatch):
    """End-to-end on the happy path: Accept-driven JSON response is
    parsed into ``WebSearchResultItem`` entries with ranks 1-N."""
    provider = JinaSearchProvider(config={"api_key": "test-key"})

    def _client_session_factory(timeout=None):
        return _StubSession(stub_response_ok)

    monkeypatch.setattr(aiohttp, "ClientSession", _client_session_factory)

    results = await provider.search(query="test", max_results=5, timeout=10)

    assert [r.url for r in results] == [
        "https://example.com/zhang-fei-niurou",
        "https://example.com/sichuan-cuisine",
    ]
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert provider.last_search_meta["search_status"] == "ok"
    assert provider.last_search_meta["error_code"] is None


@pytest.mark.asyncio
async def test_jina_search_propagates_target_domain_header(stub_response_ok, monkeypatch):
    """``X-Target-Domain`` (the legitimate Jina header we still send
    when ``source`` is set) must remain on the outbound request."""
    provider = JinaSearchProvider(config={"api_key": "test-key"})

    captured: dict[str, Any] = {}

    def _client_session_factory(timeout=None):
        session = _StubSession(stub_response_ok)
        captured["session"] = session
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", _client_session_factory)

    await provider.search(query="aperag", max_results=3, timeout=10, source="https://example.com/path")

    headers = captured["session"].captured_headers
    assert headers is not None
    assert headers.get("X-Target-Domain") == "example.com"
    assert "X-Return-Format" not in headers
