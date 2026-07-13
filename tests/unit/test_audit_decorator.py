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

"""Unit tests for the @audit decorator.

We verify that:
- start_time and end_time are recorded and passed to audit_service.log_audit
- duration is always non-negative
- successful calls are recorded with status_code=200
- failed calls are recorded with status_code=500 and the exception is re-raised
- GET requests are skipped (audit decorator does not log GETs)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.testclient import TestClient

from aperag.utils.audit_decorator import audit


# ---------------------------------------------------------------------------
# Minimal fake Request helpers
# ---------------------------------------------------------------------------


def _make_request(method: str = "POST", path: str = "/api/v1/bots") -> Request:
    """Build a minimal Starlette Request object suitable for the decorator."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [],
        "state": {},
    }
    request = Request(scope)
    # The decorator reads user_id / username from request.state
    request.state.user_id = "user-42"
    request.state.username = "tester"
    return request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_patched_audit_service():
    """Return a mock for audit_service.log_audit that captures call args."""
    mock_log = AsyncMock()
    return mock_log


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_records_start_and_end_time():
    """start_time and end_time must be forwarded to audit_service.log_audit."""
    mock_log = _make_patched_audit_service()

    @audit(resource_type="bot", api_name="CreateBot")
    async def _view(request, **kwargs):
        return {"id": "bot-1"}

    request = _make_request("POST")

    with patch("aperag.utils.audit_decorator.audit_service.log_audit", mock_log):
        await _view(request=request)

    # asyncio.create_task wraps the coroutine; we need to flush the event loop.
    # In pytest-asyncio with asyncio_mode=auto the event loop runs between awaits,
    # but create_task schedules a new task.  We can inspect the mock directly since
    # create_task is called synchronously and the coroutine arg contains our args.
    assert mock_log.called or True  # create_task schedules it; see below

    # Instead, capture via direct call instead of create_task.  Patch create_task too.
    import asyncio

    captured_coros = []

    def _capture_task(coro):
        captured_coros.append(coro)
        # Return a real task so the event loop doesn't complain
        return asyncio.ensure_future(coro)

    with patch("asyncio.create_task", side_effect=_capture_task):
        await _view(request=request)

    # Wait for all captured coroutines
    import asyncio as _asyncio
    for coro in captured_coros:
        try:
            await coro
        except Exception:
            pass

    assert mock_log.called
    kwargs = mock_log.call_args.kwargs
    assert kwargs["start_time"] is not None
    assert kwargs["end_time"] is not None
    assert kwargs["end_time"] >= kwargs["start_time"]


@pytest.mark.asyncio
async def test_audit_duration_is_non_negative():
    """end_time - start_time must always be >= 0."""
    import asyncio

    mock_log = _make_patched_audit_service()
    captured_coros = []

    def _capture_task(coro):
        captured_coros.append(coro)
        return asyncio.ensure_future(coro)

    @audit(resource_type="bot", api_name="CreateBot")
    async def _view(request, **kwargs):
        return {"ok": True}

    request = _make_request("POST")

    with (
        patch("aperag.utils.audit_decorator.audit_service.log_audit", mock_log),
        patch("asyncio.create_task", side_effect=_capture_task),
    ):
        await _view(request=request)

    for coro in captured_coros:
        try:
            await coro
        except Exception:
            pass

    kwargs = mock_log.call_args.kwargs
    assert kwargs["end_time"] - kwargs["start_time"] >= 0


@pytest.mark.asyncio
async def test_audit_success_uses_status_200():
    """Successful calls must be audited with status_code=200."""
    import asyncio

    mock_log = _make_patched_audit_service()
    captured_coros = []

    def _capture_task(coro):
        captured_coros.append(coro)
        return asyncio.ensure_future(coro)

    @audit(resource_type="collection", api_name="CreateCollection")
    async def _view(request, **kwargs):
        return {"id": "col-1"}

    request = _make_request("POST", "/api/v1/collections")

    with (
        patch("aperag.utils.audit_decorator.audit_service.log_audit", mock_log),
        patch("asyncio.create_task", side_effect=_capture_task),
    ):
        result = await _view(request=request)

    for coro in captured_coros:
        try:
            await coro
        except Exception:
            pass

    assert result == {"id": "col-1"}
    kwargs = mock_log.call_args.kwargs
    assert kwargs["status_code"] == 200
    assert kwargs["error_message"] is None


@pytest.mark.asyncio
async def test_audit_failure_uses_status_500_and_reraises():
    """Failed calls must be audited with status_code=500, and the exception re-raised."""
    import asyncio

    mock_log = _make_patched_audit_service()
    captured_coros = []

    def _capture_task(coro):
        captured_coros.append(coro)
        return asyncio.ensure_future(coro)

    @audit(resource_type="bot", api_name="CreateBot")
    async def _view(request, **kwargs):
        raise ValueError("something went wrong")

    request = _make_request("POST")

    with (
        patch("aperag.utils.audit_decorator.audit_service.log_audit", mock_log),
        patch("asyncio.create_task", side_effect=_capture_task),
    ):
        with pytest.raises(ValueError, match="something went wrong"):
            await _view(request=request)

    for coro in captured_coros:
        try:
            await coro
        except Exception:
            pass

    kwargs = mock_log.call_args.kwargs
    assert kwargs["status_code"] == 500
    assert kwargs["error_message"] == "something went wrong"


@pytest.mark.asyncio
async def test_audit_skips_get_requests():
    """GET requests must be passed through without any audit log."""
    mock_log = _make_patched_audit_service()

    @audit(resource_type="bot", api_name="GetBot")
    async def _view(request, **kwargs):
        return {"id": "bot-1"}

    request = _make_request("GET")

    with patch("aperag.utils.audit_decorator.audit_service.log_audit", mock_log):
        result = await _view(request=request)

    assert result == {"id": "bot-1"}
    mock_log.assert_not_called()


@pytest.mark.asyncio
async def test_audit_api_name_defaults_to_function_name():
    """If api_name is omitted, the function name should be used."""
    import asyncio

    mock_log = _make_patched_audit_service()
    captured_coros = []

    def _capture_task(coro):
        captured_coros.append(coro)
        return asyncio.ensure_future(coro)

    @audit(resource_type="bot")
    async def create_bot_view(request, **kwargs):
        return {}

    request = _make_request("POST")

    with (
        patch("aperag.utils.audit_decorator.audit_service.log_audit", mock_log),
        patch("asyncio.create_task", side_effect=_capture_task),
    ):
        await create_bot_view(request=request)

    for coro in captured_coros:
        try:
            await coro
        except Exception:
            pass

    kwargs = mock_log.call_args.kwargs
    assert kwargs["api_name"] == "create_bot_view"
