import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aperag.service.chat_title_service import ChatTitleService


class _EmptyHistory:
    def __init__(self, *args, **kwargs):
        pass

    @property
    async def messages(self):
        return []


def test_generate_title_uses_existing_chat_title_when_history_is_empty(monkeypatch):
    service = ChatTitleService()
    service.db_ops = SimpleNamespace(
        query_bot=AsyncMock(return_value=SimpleNamespace(id="bot-1")),
        query_chat=AsyncMock(return_value=SimpleNamespace(id="chat-1", title="Existing Chat Title")),
    )

    default_model_mock = AsyncMock(return_value=("google/gemini-2.5-flash", "openrouter", "openrouter"))
    # Phase 5 step 5-S4e canonical (msg=940bd884): the service reaches
    # ``default_model_service`` via a direct cross-domain import from
    # ``aperag.domains.model_platform.service.default_model_service``,
    # so the patch target is the ``default_model_service`` symbol the
    # domain module re-binds at import time.
    monkeypatch.setattr(
        "aperag.domains.conversation.service.chat_title_service.default_model_service."
        "get_default_background_task_config",
        default_model_mock,
    )
    monkeypatch.setattr(
        "aperag.domains.conversation.service.chat_title_service.RedisChatMessageHistory", _EmptyHistory
    )

    title = asyncio.run(service.generate_title("user-1", "bot-1", "chat-1"))

    assert title == "Existing Chat Title"
    default_model_mock.assert_not_awaited()
