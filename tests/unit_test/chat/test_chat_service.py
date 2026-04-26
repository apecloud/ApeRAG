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

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from aperag.domains.agent_runtime.db.models import AgentTurnStatus
from aperag.domains.agent_runtime.uimessage import (
    CitationData,
    DataCitationPart,
    SourceUrlPart,
    TextPart,
    UIMessage,
    UrlCitationLocation,
)
from aperag.domains.conversation.service.chat_service import ChatService


def _now():
    return datetime.now(timezone.utc)


class _FakeUIMessageStore:
    def __init__(self, messages=None):
        self.messages = dict(messages or {})

    async def read(self, turn_id):
        return self.messages.get(turn_id)


class _FakeChatDbOps:
    def __init__(self):
        self.chat = SimpleNamespace(
            id="chat-1",
            title="Test Chat",
            bot_id="bot-1",
            peer_type="system",
            peer_id=None,
            gmt_created=_now(),
            gmt_updated=_now(),
        )
        self.turn = SimpleNamespace(
            id="turn-1",
            chat_id="chat-1",
            user="user-1",
            input_text="What changed?",
            status=AgentTurnStatus.COMPLETED,
            error_message=None,
            timeline_cursor=2,
            gmt_created=_now(),
            gmt_started=_now(),
            gmt_finished=_now(),
            gmt_updated=_now(),
        )

    async def query_chat(self, user, bot_id, chat_id):
        if user == "user-1" and bot_id == "bot-1" and chat_id == "chat-1":
            return self.chat
        return None

    async def query_agent_turns(self, user, chat_id):
        if user == "user-1" and chat_id == "chat-1":
            return [self.turn]
        return []


def _build_persisted_message() -> UIMessage:
    return UIMessage(
        id="msg-turn-1",
        role="assistant",
        parts=[
            TextPart(text="Here is the answer."),
            SourceUrlPart(source_id="doc-a", url="https://example.com/doc-a", title="Doc A"),
            DataCitationPart(
                data=CitationData(
                    cited_text="Reference snippet",
                    location=UrlCitationLocation(url="https://example.com/doc-a", title="Doc A"),
                )
            ),
        ],
    )


@pytest.mark.asyncio
async def test_get_chat_returns_canonical_uimessage_history():
    """Phase 8 D8.6 (#80) chunk-2: ``ChatDetails.history`` reads
    canonical ``UIMessage`` parts directly from ``UIMessageStore`` —
    legacy ``AgentArtifact`` projection is gone.
    """

    service = ChatService(uimessage_store=_FakeUIMessageStore({"turn-1": _build_persisted_message()}))
    service.db_ops = _FakeChatDbOps()

    chat = await service.get_chat("user-1", "bot-1", "chat-1")

    assert chat.id == "chat-1"
    assert chat.history is not None
    assert len(chat.history) == 1

    snapshot = chat.history[0]
    assert snapshot.turn_id == "turn-1"
    assert snapshot.chat_id == "chat-1"
    assert snapshot.runtime_kind == "agent_runtime"
    assert snapshot.role == "assistant"
    assert snapshot.status == "COMPLETED"
    assert snapshot.input_text == "What changed?"
    assert snapshot.error_text is None
    assert snapshot.timeline_cursor == 2

    types = [getattr(part, "type", None) for part in snapshot.parts]
    assert types == ["text", "source-url", "data-citation"]

    assert isinstance(snapshot.parts[0], TextPart)
    assert snapshot.parts[0].text == "Here is the answer."

    assert isinstance(snapshot.parts[1], SourceUrlPart)
    assert snapshot.parts[1].source_id == "doc-a"
    assert snapshot.parts[1].url == "https://example.com/doc-a"

    assert isinstance(snapshot.parts[2], DataCitationPart)
    assert snapshot.parts[2].data.cited_text == "Reference snippet"
    assert snapshot.parts[2].data.location.url == "https://example.com/doc-a"


@pytest.mark.asyncio
async def test_get_chat_history_surfaces_error_text_for_failed_turn():
    """Phase 8 D8.6 (#80) chunk-2: a FAILED turn's ``error_text`` comes
    straight off the ``AgentTurn`` row (the legacy ``error_summary``
    artifact is gone). ``parts`` is empty when the runtime never
    persisted a UIMessage for the failed turn.
    """

    service = ChatService(uimessage_store=_FakeUIMessageStore())
    db_ops = _FakeChatDbOps()
    db_ops.turn = SimpleNamespace(
        id="turn-failed",
        chat_id="chat-1",
        user="user-1",
        input_text="Trigger failure",
        status=AgentTurnStatus.FAILED,
        error_message="upstream provider timeout",
        timeline_cursor=1,
        gmt_created=_now(),
        gmt_started=_now(),
        gmt_finished=_now(),
        gmt_updated=_now(),
    )
    service.db_ops = db_ops

    chat = await service.get_chat("user-1", "bot-1", "chat-1")
    assert chat.history is not None
    assert len(chat.history) == 1

    snapshot = chat.history[0]
    assert snapshot.status == "FAILED"
    assert snapshot.error_text == "upstream provider timeout"
    assert snapshot.parts == []


@pytest.mark.asyncio
async def test_get_chat_history_does_not_expose_legacy_chatmessage_shape():
    """Regression-guard: ``ChatDetails.history`` must not regress to the
    legacy ``list[list[ChatMessage]]`` shape (separate human / ai
    groups per turn). Pre-#92 callers that read ``history[i][j].role``
    will fail loudly; that is intentional — they need to migrate to
    the canonical snapshot shape.
    """

    service = ChatService(uimessage_store=_FakeUIMessageStore({"turn-1": _build_persisted_message()}))
    service.db_ops = _FakeChatDbOps()

    chat = await service.get_chat("user-1", "bot-1", "chat-1")

    serialised = [snapshot.model_dump(mode="json") for snapshot in chat.history or []]
    for entry in serialised:
        assert "turn_id" in entry
        assert "parts" in entry
        assert "role" in entry  # but role is a string, not a list of ChatMessages
        assert isinstance(entry["role"], str)
