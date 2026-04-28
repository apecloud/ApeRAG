# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 10 §K.13 — bot_service is_system filter (default-deny enumeration).

Per architect own-up #17: hidden/system entities must enumerate the
full API surface and apply default-deny filtering. ``Bot.is_system=True``
rows must NOT surface in:

  * ``list_bots`` — would break ``Bot`` Pydantic schema's
    ``type: Literal["knowledge", "common", "agent"]`` validation.
  * ``get_bot`` — implementation detail, not user-addressable.
  * ``update_bot`` — would let a user mutate the regen pipeline's
    hidden bot.
  * ``delete_bot`` — would break the regen pipeline by removing its
    only entry point.

Filter lives at the ``db_ops.query_bot(s)`` layer (default
``exclude_system=True``) so the WHERE clause is narrow as bot count
grows, and so all four service paths share the same single guard.

Internal regen plumbing
(``get_or_create_summary_bot_for_user``) bypasses the db_ops layer
entirely with a direct ORM query, so the default-deny filter does
not block its lookups.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aperag.domains.conversation.service import bot_service as bot_service_module
from aperag.exceptions import ResourceNotFoundException


def _make_row(*, id_, type_, is_system, title="Bot"):
    return SimpleNamespace(
        id=id_,
        user="u",
        title=title,
        type=type_,
        description="d",
        status="ACTIVE",
        config="{}",
        is_system=is_system,
        gmt_created=None,
        gmt_updated=None,
        gmt_deleted=None,
    )


class _FakeDbOps:
    """Mimics the db_ops layer: ``query_bots`` / ``query_bot`` honour
    ``exclude_system`` like the real ``AsyncBotRepositoryMixin``.
    """

    def __init__(self, rows):
        self._rows = rows
        self.execute_with_transaction_calls: list[str] = []

    async def query_bots(self, _users, *, exclude_system: bool = True):
        if exclude_system:
            return [r for r in self._rows if not r.is_system]
        return list(self._rows)

    async def query_bot(self, _user, bot_id, *, exclude_system: bool = True):
        for row in self._rows:
            if row.id != bot_id:
                continue
            if exclude_system and row.is_system:
                return None
            return row
        return None

    async def execute_with_transaction(self, op):
        self.execute_with_transaction_calls.append("called")
        return None


@pytest.fixture
def fake_rows():
    return [
        _make_row(id_="bot_user", type_="agent", is_system=False, title="Default Agent Bot"),
        _make_row(id_="bot_summary", type_="summary", is_system=True, title="Summary Bot"),
    ]


# ---------------------------------------------------------------------
# list_bots — default-deny is_system
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_bots_excludes_system_bots(fake_rows, monkeypatch):
    svc = bot_service_module.BotService()
    svc.db_ops = _FakeDbOps(fake_rows)

    seen: list[str] = []

    async def _build(row):
        seen.append(row.id)
        # Return a dict the BotList Pydantic root will accept.
        return {
            "id": row.id,
            "title": row.title,
            "type": row.type,
            "description": row.description,
            "status": row.status,
            "config": json.loads(row.config),
        }

    monkeypatch.setattr(svc, "build_bot_response", _build)

    result = await svc.list_bots("u")

    assert seen == ["bot_user"]
    assert [item.id for item in result.items] == ["bot_user"]


# ---------------------------------------------------------------------
# get_bot — default-deny is_system
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_bot_returns_404_for_system_bot(fake_rows, monkeypatch):
    svc = bot_service_module.BotService()
    svc.db_ops = _FakeDbOps(fake_rows)

    async def _build(row):  # pragma: no cover — must not run
        raise AssertionError("system bot should not reach response builder")

    monkeypatch.setattr(svc, "build_bot_response", _build)

    with pytest.raises(ResourceNotFoundException):
        await svc.get_bot("u", "bot_summary")


@pytest.mark.asyncio
async def test_get_bot_returns_user_bot(fake_rows, monkeypatch):
    svc = bot_service_module.BotService()
    svc.db_ops = _FakeDbOps(fake_rows)

    async def _build(row):
        # ``get_bot`` returns whatever the response builder hands back,
        # so a SimpleNamespace is enough for this test (no Pydantic
        # round-trip needed).
        return SimpleNamespace(id=row.id, type=row.type)

    monkeypatch.setattr(svc, "build_bot_response", _build)

    result = await svc.get_bot("u", "bot_user")
    assert result.id == "bot_user"


# ---------------------------------------------------------------------
# update_bot — default-deny is_system
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_bot_returns_404_for_system_bot(fake_rows, monkeypatch):
    from aperag.domains.conversation.schemas import BotUpdate

    svc = bot_service_module.BotService()
    svc.db_ops = _FakeDbOps(fake_rows)

    async def _validate_collections(_user, _config):
        return None

    monkeypatch.setattr(svc, "validate_collections", _validate_collections)

    with pytest.raises(ResourceNotFoundException):
        await svc.update_bot("u", "bot_summary", BotUpdate(title="x"))
    # The atomic update must never fire — the ResourceNotFoundException
    # is the contract we surface to clients before any mutation.
    assert svc.db_ops.execute_with_transaction_calls == []


# ---------------------------------------------------------------------
# delete_bot — silent no-op for system bot
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_bot_silently_ignores_system_bot(fake_rows):
    """``delete_bot`` semantics are idempotent (return None on
    not-found), so a system bot is treated as not-existing — the
    user cannot remove it."""
    svc = bot_service_module.BotService()
    svc.db_ops = _FakeDbOps(fake_rows)

    result = await svc.delete_bot("u", "bot_summary")
    assert result is None
    assert svc.db_ops.execute_with_transaction_calls == []
