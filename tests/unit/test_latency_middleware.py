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

"""Unit tests for LatencyLoggingMiddleware.

These tests wrap a minimal Starlette/ASGI app with the middleware and use
httpx's ASGITransport so no real network socket is needed.
"""

import logging

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from aperag.middleware.latency import _SKIP_PATHS, LatencyLoggingMiddleware


# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------

def _hello(request):
    return PlainTextResponse("hello world")


def _health(request):
    return PlainTextResponse("ok")


_routes = [
    Route("/hello", _hello),
    Route("/health", _health),
]

_app = Starlette(routes=_routes)
_app.add_middleware(LatencyLoggingMiddleware)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get(path: str):
    """Issue a GET to the test app and return the response."""
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://testserver") as client:
        return await client.get(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_x_response_time_header_present():
    """X-Response-Time header must be present on a normal response."""
    resp = await _get("/hello")
    assert resp.status_code == 200
    assert "x-response-time" in resp.headers


@pytest.mark.asyncio
async def test_x_response_time_header_format():
    """X-Response-Time value must look like '<N>ms'."""
    resp = await _get("/hello")
    value = resp.headers["x-response-time"]
    assert value.endswith("ms"), f"Expected '…ms', got {value!r}"
    ms_str = value[:-2]
    assert ms_str.isdigit(), f"Non-numeric part before 'ms': {ms_str!r}"
    assert int(ms_str) >= 0


@pytest.mark.asyncio
async def test_response_body_preserved():
    """The middleware must not alter the response body."""
    resp = await _get("/hello")
    assert resp.text == "hello world"


@pytest.mark.asyncio
async def test_normal_path_logged_at_info(caplog):
    """/hello should produce an INFO log line."""
    with caplog.at_level(logging.INFO, logger="aperag.middleware.latency"):
        await _get("/hello")

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_records, "Expected at least one INFO log record for /hello"
    # The message should contain the path
    assert any("/hello" in r.getMessage() for r in info_records)


@pytest.mark.asyncio
async def test_skip_path_logged_at_debug_not_info(caplog):
    """/health is in _SKIP_PATHS and must be logged at DEBUG, not INFO."""
    assert "/health" in _SKIP_PATHS, "/health should be a skip path"

    with caplog.at_level(logging.DEBUG, logger="aperag.middleware.latency"):
        await _get("/health")

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    info_records = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "/health" in r.getMessage()
    ]

    assert debug_records, "Expected at least one DEBUG log record for /health"
    assert not info_records, "Expected NO INFO log record for a skip path"


@pytest.mark.asyncio
async def test_log_line_contains_method_path_status_duration(caplog):
    """The INFO log line should contain method, path, status code, and duration."""
    with caplog.at_level(logging.INFO, logger="aperag.middleware.latency"):
        await _get("/hello")

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert messages, "No INFO log records found"
    msg = messages[0]
    assert "GET" in msg
    assert "/hello" in msg
    assert "200" in msg
    assert "ms" in msg
