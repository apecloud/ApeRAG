# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Internal-source ``ChatPeerType.EVALUATION`` filter (per @earayu2
msg=41b411cd + Weston msg=387ce23d).

Evaluation runs spawn one ``Chat`` per case so the agent runtime has
a real conversation to drive, but the user did not start it — so the
row must not surface in the bot's chat list. The new
``ChatPeerType.EVALUATION`` enum value flags those rows; the
``list_chats`` query filters them out by default and accepts
``include_internal=True`` for internal trace lookups.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aperag.domains.conversation.db.models import BotType, ChatPeerType
from aperag.domains.conversation.service.chat_service import ChatService


def test_chat_peer_type_has_evaluation_value():
    assert ChatPeerType.EVALUATION.value == "evaluation"


class _FakeBotDbOps:
    def __init__(self):
        self.created_chat_calls = []
        self._bot = SimpleNamespace(
            id="bot-1",
            user="user-1",
            type=BotType.AGENT,
            status="ACTIVE",
            is_system=False,
        )

    async def query_bot(self, _user, _bot_id, *, exclude_system: bool = True):
        return self._bot

    async def create_chat(self, *, user, bot_id, peer_type=None, **_kwargs):
        from datetime import datetime, timezone

        self.created_chat_calls.append({"user": user, "bot_id": bot_id, "peer_type": peer_type})
        return SimpleNamespace(
            id="chat-1",
            title="t",
            bot_id=bot_id,
            peer_type=peer_type,
            peer_id=None,
            gmt_created=datetime.now(timezone.utc),
            gmt_updated=datetime.now(timezone.utc),
        )


@pytest.mark.asyncio
async def test_create_chat_default_peer_type_is_none():
    """Default user-driven path leaves ``peer_type`` unset; the repo
    layer fills in ``ChatPeerType.SYSTEM``."""
    db_ops = _FakeBotDbOps()
    svc = ChatService()
    svc.db_ops = db_ops
    await svc.create_chat("user-1", "bot-1")
    assert db_ops.created_chat_calls == [{"user": "user-1", "bot_id": "bot-1", "peer_type": None}]


@pytest.mark.asyncio
async def test_create_chat_threads_evaluation_peer_type_through():
    """Internal call sites (eval worker) explicitly pass EVALUATION so
    ``list_chats`` can hide the row from the bot's chat list."""
    db_ops = _FakeBotDbOps()
    svc = ChatService()
    svc.db_ops = db_ops
    await svc.create_chat("user-1", "bot-1", peer_type=ChatPeerType.EVALUATION)
    assert db_ops.created_chat_calls == [{"user": "user-1", "bot_id": "bot-1", "peer_type": ChatPeerType.EVALUATION}]
