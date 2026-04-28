# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 10 §K.13 regression — ``chat_service.create_chat`` and
``TurnService.get_chat_and_bot`` must allow the per-user hidden
``BotType.SUMMARY`` bot through the trusted internal seam used by
``collection_regen_service`` Stage 1 Tier 1.

Without this flag, the regen flow surfaced as ``Bot not found: bot…``
404 toasts on the FE Regen button (see issue triaged via
msg=6e36e981 / msg=3d6024a2): the default-deny ``exclude_system`` filter
introduced by PR #1786 hid the summary bot from ``query_bot``, and the
pre-existing AGENT-only type guard rejected SUMMARY even before that
landed.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from aperag.domains.agent_runtime.schemas import CreateTurnRequest
from aperag.domains.agent_runtime.services import TurnService
from aperag.domains.conversation.db.models import BotType
from aperag.domains.conversation.service.chat_service import ChatService
from aperag.exceptions import ResourceNotFoundException, ValidationException


def _now():
    return datetime.now(timezone.utc)


class _FakeBotDbOps:
    """Mimics ``query_bot``'s ``exclude_system`` semantics: when the flag
    is True the system bot is hidden (returns None); when False the
    system bot is returned. ``create_chat`` is a pass-through that
    captures inputs."""

    def __init__(self, *, bot_type: BotType, is_system: bool):
        self._bot = SimpleNamespace(
            id="botsummary123",
            user="user-1",
            type=bot_type,
            status="ACTIVE",
            is_system=is_system,
        )
        self._chat = SimpleNamespace(
            id="chat-summary-1",
            title="Test Chat",
            bot_id="botsummary123",
            peer_type=None,
            peer_id=None,
            gmt_created=_now(),
            gmt_updated=_now(),
        )
        self.create_chat_calls = []

    async def query_bot(self, user, bot_id, *, exclude_system: bool = True):
        if exclude_system and self._bot.is_system:
            return None
        if user != "user-1" or bot_id != self._bot.id:
            return None
        return self._bot

    async def create_chat(self, *, user, bot_id, **kwargs):
        self.create_chat_calls.append((user, bot_id))
        return self._chat


@pytest.mark.asyncio
async def test_create_chat_default_rejects_summary_bot_with_404():
    """Default-deny path — user-facing API must keep returning 404."""
    db_ops = _FakeBotDbOps(bot_type=BotType.SUMMARY, is_system=True)
    service = ChatService()
    service.db_ops = db_ops

    with pytest.raises(ResourceNotFoundException):
        await service.create_chat("user-1", "botsummary123")
    assert db_ops.create_chat_calls == []


@pytest.mark.asyncio
async def test_create_chat_allow_system_accepts_summary_bot():
    """Trusted internal path — regen Stage 1 Tier 1 must succeed."""
    db_ops = _FakeBotDbOps(bot_type=BotType.SUMMARY, is_system=True)
    service = ChatService()
    service.db_ops = db_ops

    chat = await service.create_chat("user-1", "botsummary123", _allow_system_bot=True)
    assert chat.id == "chat-summary-1"
    assert chat.bot_id == "botsummary123"
    assert db_ops.create_chat_calls == [("user-1", "botsummary123")]


@pytest.mark.asyncio
async def test_create_chat_allow_system_still_rejects_unknown_type():
    """Even with the trusted flag, only AGENT and SUMMARY pass — a stray
    KNOWLEDGE/COMMON system bot must still be rejected so the seam is
    not a generic escape hatch."""
    db_ops = _FakeBotDbOps(bot_type=BotType.KNOWLEDGE, is_system=True)
    service = ChatService()
    service.db_ops = db_ops

    with pytest.raises(ValidationException):
        await service.create_chat("user-1", "botsummary123", _allow_system_bot=True)


class _FakeTurnDbOps(_FakeBotDbOps):
    async def query_chat_by_id(self, user, chat_id):
        if user == "user-1" and chat_id == self._chat.id:
            return self._chat
        return None


@pytest.mark.asyncio
async def test_get_chat_and_bot_default_rejects_summary_bot_with_404():
    db_ops = _FakeTurnDbOps(bot_type=BotType.SUMMARY, is_system=True)
    service = TurnService()
    service.db_ops = db_ops

    with pytest.raises(ResourceNotFoundException):
        await service.get_chat_and_bot("user-1", "chat-summary-1")


@pytest.mark.asyncio
async def test_get_chat_and_bot_allow_system_accepts_summary_bot():
    db_ops = _FakeTurnDbOps(bot_type=BotType.SUMMARY, is_system=True)
    service = TurnService()
    service.db_ops = db_ops

    chat, bot = await service.get_chat_and_bot("user-1", "chat-summary-1", _allow_system_bot=True)
    assert chat.id == "chat-summary-1"
    assert bot.type == BotType.SUMMARY


@pytest.mark.asyncio
async def test_create_or_get_turn_threads_allow_system_flag_through():
    """``_invoke_summary_agent`` calls ``create_or_get_turn`` with the
    flag — make sure the wrapper actually forwards it to
    ``get_chat_and_bot`` rather than silently dropping the kwarg."""
    db_ops = _FakeTurnDbOps(bot_type=BotType.SUMMARY, is_system=True)
    service = TurnService()
    service.db_ops = db_ops

    captured = {}
    original = service.get_chat_and_bot

    async def _spy(user, chat_id, *, _allow_system_bot=False):
        captured["flag"] = _allow_system_bot
        return await original(user, chat_id, _allow_system_bot=_allow_system_bot)

    service.get_chat_and_bot = _spy  # type: ignore[assignment]

    # We don't care about the rest of the turn-creation path failing
    # against the fake db — the only assertion that matters is that the
    # flag was propagated. ``query_agent_turn_by_idempotency`` will be
    # missing on the fake; catch any AttributeError from there.
    request = CreateTurnRequest(query="hello", collections=[])
    try:
        await service.create_or_get_turn("user-1", "chat-summary-1", request, _allow_system_bot=True)
    except AttributeError:
        pass

    assert captured["flag"] is True
