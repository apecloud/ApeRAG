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

"""Unit tests for AuditService.

All DB interactions are mocked; these tests do *not* require a running
database.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aperag.db.models import AuditLog, AuditResource
from aperag.service.audit_service import AuditService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service() -> AuditService:
    return AuditService()


def _make_mock_session():
    """Return an async context-manager mock that acts like an AsyncSession."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def _ctx():
        yield session

    return _ctx(), session


# ---------------------------------------------------------------------------
# log_audit — duration_ms computation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_audit_duration_ms_computed_and_stored():
    """duration_ms must equal end_time - start_time and be saved on the row."""
    service = _make_service()

    start_time = 1_700_000_000_000  # milliseconds epoch
    end_time = start_time + 250
    expected_duration = 250

    saved_rows = []

    ctx, mock_session = _make_mock_session()

    def _capture_add(row):
        saved_rows.append(row)

    mock_session.add.side_effect = _capture_add

    with patch.object(service, "_make_session", return_value=ctx):
        await service.log_audit(
            user_id="u1",
            username="alice",
            resource_type=AuditResource.BOT,
            api_name="CreateBot",
            http_method="POST",
            path="/api/v1/bots",
            status_code=200,
            start_time=start_time,
            end_time=end_time,
        )

    assert len(saved_rows) == 1
    row: AuditLog = saved_rows[0]
    assert row.duration_ms == expected_duration
    assert row.start_time == start_time
    assert row.end_time == end_time


@pytest.mark.asyncio
async def test_log_audit_duration_ms_none_when_end_time_missing():
    """If end_time is None, duration_ms should remain None (not crash)."""
    service = _make_service()

    saved_rows = []
    ctx, mock_session = _make_mock_session()
    mock_session.add.side_effect = lambda row: saved_rows.append(row)

    with patch.object(service, "_make_session", return_value=ctx):
        await service.log_audit(
            user_id="u1",
            username="alice",
            resource_type=AuditResource.BOT,
            api_name="CreateBot",
            http_method="POST",
            path="/api/v1/bots",
            status_code=200,
            start_time=1_700_000_000_000,
            end_time=None,
        )

    assert len(saved_rows) == 1
    assert saved_rows[0].duration_ms is None


@pytest.mark.asyncio
async def test_log_audit_session_committed():
    """session.commit() must be called exactly once per log_audit call."""
    service = _make_service()

    ctx, mock_session = _make_mock_session()

    with patch.object(service, "_make_session", return_value=ctx):
        await service.log_audit(
            user_id="u1",
            username="alice",
            resource_type=AuditResource.COLLECTION,
            api_name="CreateCollection",
            http_method="POST",
            path="/api/v1/collections",
            status_code=200,
            start_time=1_700_000_000_000,
            end_time=1_700_000_000_100,
        )

    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_log_audit_does_not_raise_on_db_error():
    """A DB failure in log_audit must be swallowed (fire-and-forget semantics)."""
    service = _make_service()

    @asynccontextmanager
    async def _failing_ctx():
        raise RuntimeError("DB unavailable")
        yield  # noqa: unreachable

    with patch.object(service, "_make_session", return_value=_failing_ctx()):
        # Should not raise
        await service.log_audit(
            user_id="u1",
            username="alice",
            resource_type=AuditResource.BOT,
            api_name="CreateBot",
            http_method="POST",
            path="/api/v1/bots",
            status_code=200,
            start_time=1_700_000_000_000,
            end_time=1_700_000_000_200,
        )


# ---------------------------------------------------------------------------
# list_audit_logs — duration_ms back-fill
# ---------------------------------------------------------------------------


def _make_audit_log_row(duration_ms=None, start_time=None, end_time=None):
    """Create a minimal row-like object without touching the DB.

    We use SimpleNamespace rather than an uninitialised SQLAlchemy model
    instance because SA instrumentation requires _sa_instance_state to be
    present before column attributes can be set.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        duration_ms=duration_ms,
        start_time=start_time,
        end_time=end_time,
        resource_type=None,
        path=None,
        resource_id=None,
    )


@pytest.mark.asyncio
async def test_list_audit_logs_backfills_duration_ms_when_null():
    """Rows with duration_ms=NULL should have it filled from start/end_time."""
    service = _make_service()

    start = 1_700_000_000_000
    end = start + 500
    row = _make_audit_log_row(duration_ms=None, start_time=start, end_time=end)

    # Patch _make_session and paginate_query so no DB is needed
    ctx, mock_session = _make_mock_session()

    with (
        patch.object(service, "_make_session", return_value=ctx),
        patch(
            "aperag.utils.pagination.PaginationHelper.paginate_query",
            new=AsyncMock(return_value=([row], 1)),
        ),
        patch(
            "aperag.utils.pagination.PaginationHelper.build_response",
            return_value={"items": [row], "total": 1},
        ),
    ):
        result = await service.list_audit_logs()

    # After processing, row.duration_ms should be back-filled
    assert row.duration_ms == 500


@pytest.mark.asyncio
async def test_list_audit_logs_does_not_overwrite_existing_duration_ms():
    """Rows that already have duration_ms set must not be overwritten."""
    service = _make_service()

    start = 1_700_000_000_000
    end = start + 500
    row = _make_audit_log_row(duration_ms=42, start_time=start, end_time=end)

    ctx, mock_session = _make_mock_session()

    with (
        patch.object(service, "_make_session", return_value=ctx),
        patch(
            "aperag.utils.pagination.PaginationHelper.paginate_query",
            new=AsyncMock(return_value=([row], 1)),
        ),
        patch(
            "aperag.utils.pagination.PaginationHelper.build_response",
            return_value={"items": [row], "total": 1},
        ),
    ):
        await service.list_audit_logs()

    # Original value must be unchanged
    assert row.duration_ms == 42
