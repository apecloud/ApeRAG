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
    DataCitationPart,
    SourceUrlPart,
    TextPart,
)
from aperag.domains.conversation.service.chat_service import ChatService


def _now():
    return datetime.now(timezone.utc)


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
            answer_artifact_id="artifact-answer",
            reference_bundle_artifact_id="artifact-refs",
            error_message=None,
            timeline_cursor=2,
            gmt_created=_now(),
            gmt_started=_now(),
            gmt_finished=_now(),
            gmt_updated=_now(),
        )
        self.answer_artifact = SimpleNamespace(
            id="artifact-answer",
            artifact_type="answer",
            summary="Here is the answer.",
            payload={"text": "Here is the answer."},
        )
        self.reference_artifact = SimpleNamespace(
            id="artifact-refs",
            artifact_type="reference_bundle",
            summary="1 references",
            payload={
                "items": [
                    {
                        "title": "Doc A",
                        "snippet": "Reference snippet",
                        "score": 0.9,
                        "source_type": "search_collection",
                        "source_id": "doc-a",
                        "uri": "https://example.com/doc-a",
                        "metadata": {"section": "intro"},
                    }
                ]
            },
        )

    async def query_chat(self, user, bot_id, chat_id):
        if user == "user-1" and bot_id == "bot-1" and chat_id == "chat-1":
            return self.chat
        return None

    async def query_agent_turns(self, user, chat_id):
        if user == "user-1" and chat_id == "chat-1":
            return [self.turn]
        return []

    async def query_agent_artifacts_by_turn(self, turn_id):
        if turn_id == "turn-1":
            return [self.answer_artifact, self.reference_artifact]
        return []


@pytest.mark.asyncio
async def test_get_chat_returns_canonical_uimessage_history():
    """Phase 8 D8.5-BE (#92): ``ChatDetails.history`` is now a list of
    canonical ``AgentTurnSnapshot`` envelopes (one per assistant turn),
    each carrying the same ``UIMessagePart`` shape the FE consumes from
    the live SSE stream. Replaces the legacy
    ``list[list[ChatMessage]]`` shape so historical and live turns
    render through a single canonical path.
    """

    service = ChatService()
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
    """A FAILED turn's error_text comes from ``error_summary`` artifact
    when present, falling back to ``turn.error_message`` (mirrors the
    snapshot endpoint contract from #90 D8.4d)."""

    service = ChatService()
    db_ops = _FakeChatDbOps()
    db_ops.turn = SimpleNamespace(
        id="turn-failed",
        chat_id="chat-1",
        user="user-1",
        input_text="Trigger failure",
        status=AgentTurnStatus.FAILED,
        answer_artifact_id=None,
        reference_bundle_artifact_id=None,
        error_message="upstream provider timeout",
        timeline_cursor=1,
        gmt_created=_now(),
        gmt_started=_now(),
        gmt_finished=_now(),
        gmt_updated=_now(),
    )
    db_ops.answer_artifact = None
    db_ops.reference_artifact = None

    async def _empty_artifacts(turn_id):
        return []

    db_ops.query_agent_artifacts_by_turn = _empty_artifacts
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

    service = ChatService()
    service.db_ops = _FakeChatDbOps()

    chat = await service.get_chat("user-1", "bot-1", "chat-1")

    serialised = [snapshot.model_dump(mode="json") for snapshot in chat.history or []]
    for entry in serialised:
        assert "turn_id" in entry
        assert "parts" in entry
        assert "role" in entry  # but role is a string, not a list of ChatMessages
        assert isinstance(entry["role"], str)
