import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aperag.domains.conversation.service.chat_title_service import ChatTitleService


class _EmptyHistory:
    def __init__(self, *args, **kwargs):
        pass

    @property
    async def messages(self):
        return []


def test_generate_title_uses_existing_chat_title_when_history_is_empty(monkeypatch):
    """If the chat has no Redis history yet, the service must short-circuit to
    the chat row's existing ``title`` without resolving the background-task
    ``ModelUse`` or invoking the LLM.

    The model-platform refactor replaced the legacy
    ``default_model_service.get_default_background_task_config`` call with a
    direct ``query_model_uses`` lookup against the conversation domain's
    ``db_ops``, so the early-return path now skips that lookup entirely.
    """

    invocation_mock = AsyncMock(side_effect=AssertionError("model invocation must not run for empty history"))

    service = ChatTitleService()
    service.db_ops = SimpleNamespace(
        query_bot=AsyncMock(return_value=SimpleNamespace(id="bot-1")),
        query_chat=AsyncMock(return_value=SimpleNamespace(id="chat-1", title="Existing Chat Title")),
        query_model_uses=AsyncMock(side_effect=AssertionError("model_uses must not be queried for empty history")),
    )

    monkeypatch.setattr(
        "aperag.domains.conversation.service.chat_title_service.RedisChatMessageHistory",
        _EmptyHistory,
    )
    monkeypatch.setattr(
        "aperag.domains.conversation.service.chat_title_service.model_invocation_service.chat",
        invocation_mock,
    )

    title = asyncio.run(service.generate_title("user-1", "bot-1", "chat-1"))

    assert title == "Existing Chat Title"
    invocation_mock.assert_not_awaited()
