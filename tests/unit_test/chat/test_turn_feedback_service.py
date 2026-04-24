from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from aperag.domains.conversation.service.turn_feedback_service import TurnFeedbackService


def _now():
    return datetime.now(timezone.utc)


class _FakeTurnFeedbackDbOps:
    def __init__(self):
        self.chat = SimpleNamespace(id="chat-1")
        self.turn = SimpleNamespace(id="turn-1", chat_id="chat-1", user="user-1")
        self.feedback = SimpleNamespace(
            chat_id="chat-1",
            turn_id="turn-1",
            type="good",
            tag=None,
            message=None,
            gmt_created=_now(),
            gmt_updated=_now(),
        )
        self.feedback_kwargs = None
        self.removed = None

    async def query_chat_by_id(self, user, chat_id):
        if user == "user-1" and chat_id == "chat-1":
            return self.chat
        return None

    async def query_turn_feedbacks(self, user, chat_id):
        if user == "user-1" and chat_id == "chat-1":
            return [self.feedback]
        return []

    async def query_agent_turn(self, user, chat_id, turn_id):
        if user == "user-1" and chat_id == "chat-1" and turn_id == "turn-1":
            return self.turn
        return None

    async def set_turn_feedback_state(self, **kwargs):
        self.feedback_kwargs = kwargs
        return SimpleNamespace(
            turn_id=kwargs["turn_id"],
            type=kwargs["feedback_type"],
            tag=kwargs["feedback_tag"],
            message=kwargs["feedback_message"],
            gmt_created=_now(),
            gmt_updated=_now(),
        )

    async def remove_turn_feedback(self, user, chat_id, turn_id):
        self.removed = {
            "user": user,
            "chat_id": chat_id,
            "turn_id": turn_id,
        }
        return True


@pytest.mark.asyncio
async def test_list_turn_feedbacks_returns_chat_scoped_feedback():
    service = TurnFeedbackService()
    service.db_ops = _FakeTurnFeedbackDbOps()

    result = await service.list_turn_feedbacks("user-1", "chat-1")

    assert len(result.items) == 1
    assert result.items[0].turn_id == "turn-1"
    assert result.items[0].type == "good"


@pytest.mark.asyncio
async def test_upsert_turn_feedback_uses_turn_identity():
    service = TurnFeedbackService()
    service.db_ops = _FakeTurnFeedbackDbOps()

    result = await service.upsert_turn_feedback(
        "user-1",
        "chat-1",
        "turn-1",
        SimpleNamespace(type="bad", tag="Fake", message="Wrong answer"),
    )

    assert result.turn_id == "turn-1"
    assert result.type == "bad"
    assert service.db_ops.feedback_kwargs == {
        "user": "user-1",
        "chat_id": "chat-1",
        "turn_id": "turn-1",
        "feedback_type": "bad",
        "feedback_tag": "Fake",
        "feedback_message": "Wrong answer",
    }


@pytest.mark.asyncio
async def test_delete_turn_feedback_deletes_by_turn_id():
    service = TurnFeedbackService()
    service.db_ops = _FakeTurnFeedbackDbOps()

    result = await service.delete_turn_feedback("user-1", "chat-1", "turn-1")

    assert result is True
    assert service.db_ops.removed == {
        "user": "user-1",
        "chat_id": "chat-1",
        "turn_id": "turn-1",
    }
