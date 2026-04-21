from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from aperag.db.models import AgentTurnStatus
from aperag.service.chat_service import ChatService


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
            gmt_created=_now(),
            gmt_finished=_now(),
        )
        self.answer_artifact = SimpleNamespace(
            id="artifact-answer",
            artifact_type="answer",
            payload={"text": "Here is the answer."},
        )
        self.reference_artifact = SimpleNamespace(
            id="artifact-refs",
            artifact_type="reference_bundle",
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
        self.feedback = SimpleNamespace(
            message_id="turn-1",
            type="good",
            tag=None,
            message=None,
        )
        self.feedback_kwargs = None

    async def query_chat(self, user, bot_id, chat_id):
        if user == "user-1" and bot_id == "bot-1" and chat_id == "chat-1":
            return self.chat
        return None

    async def query_agent_turns(self, user, chat_id):
        if user == "user-1" and chat_id == "chat-1":
            return [self.turn]
        return []

    async def query_chat_feedbacks(self, user, chat_id):
        if user == "user-1" and chat_id == "chat-1":
            return [self.feedback]
        return []

    async def query_agent_artifacts_by_turn(self, turn_id):
        if turn_id == "turn-1":
            return [self.answer_artifact, self.reference_artifact]
        return []

    async def query_agent_turn(self, user, chat_id, turn_id):
        if user == "user-1" and chat_id == "chat-1" and turn_id == "turn-1":
            return self.turn
        return None

    async def set_message_feedback_state(self, **kwargs):
        self.feedback_kwargs = kwargs
        return {"message_id": kwargs["message_id"], "type": kwargs["feedback_type"]}

    async def remove_message_feedback(self, user, chat_id, message_id):
        self.feedback_kwargs = {
            "action": "remove",
            "user": user,
            "chat_id": chat_id,
            "message_id": message_id,
        }
        return True


@pytest.mark.asyncio
async def test_get_chat_projects_v3_turn_history():
    service = ChatService()
    service.db_ops = _FakeChatDbOps()

    chat = await service.get_chat("user-1", "bot-1", "chat-1")

    assert chat.id == "chat-1"
    assert chat.history is not None
    assert len(chat.history) == 2

    user_group, ai_group = chat.history
    assert user_group[0].role == "human"
    assert user_group[0].data == "What changed?"

    assert ai_group[0].role == "ai"
    assert ai_group[0].id == "turn-1"
    assert ai_group[0].type == "message"
    assert ai_group[0].data == "Here is the answer."

    assert ai_group[1].type == "references"
    assert ai_group[1].references is not None
    assert len(ai_group[1].references) == 1
    assert ai_group[1].feedback is not None
    assert ai_group[1].feedback.type == "good"


@pytest.mark.asyncio
async def test_feedback_message_uses_v3_turn_and_artifacts():
    service = ChatService()
    service.db_ops = _FakeChatDbOps()

    result = await service.feedback_message(
        user="user-1",
        chat_id="chat-1",
        message_id="turn-1",
        feedback_type="good",
        feedback_tag=None,
        feedback_message=None,
    )

    assert result["action"] == "upserted"
    assert service.db_ops.feedback_kwargs["question"] == "What changed?"
    assert service.db_ops.feedback_kwargs["original_answer"] == "Here is the answer."
