from types import SimpleNamespace

import pytest

from aperag.domains.conversation.service.chat_collection_service import ChatCollectionService


class _FakeUserDbOps:
    """Fake db_ops that always reports the user has no chat collection yet."""

    def __init__(self):
        self.queried_user_ids = []

    async def query_user_by_id(self, user_id):
        self.queried_user_ids.append(user_id)
        return SimpleNamespace(id=user_id, chat_collection_id=None)

    async def query_collection_by_id(self, collection_id):
        return None


@pytest.mark.asyncio
async def test_create_user_chat_collection_returns_none_when_no_embedding_model():
    """No provider configured → graceful skip (return None), no ValueError."""
    service = ChatCollectionService()
    service.db_ops = _FakeUserDbOps()

    async def _no_embedding(_user_id):
        return None

    service._get_default_embedding_model = _no_embedding

    result = await service.create_user_chat_collection("user-no-provider")

    assert result is None


@pytest.mark.asyncio
async def test_initialize_user_chat_collection_returns_none_when_no_embedding_model():
    """Registration flow tolerates missing provider — returns None."""
    service = ChatCollectionService()
    service.db_ops = _FakeUserDbOps()

    async def _no_embedding(_user_id):
        return None

    service._get_default_embedding_model = _no_embedding

    result = await service.initialize_user_chat_collection("user-no-provider")

    assert result is None


@pytest.mark.asyncio
async def test_initialize_user_chat_collection_returns_existing_when_present():
    """Existing chat collection takes precedence — graceful skip path is unreachable."""
    service = ChatCollectionService()

    existing_collection = SimpleNamespace(id="coll-existing", status="ACTIVE")

    class _ExistingChatOps:
        async def query_user_by_id(self, _user_id):
            return SimpleNamespace(id="user-1", chat_collection_id="coll-existing")

        async def query_collection_by_id(self, collection_id):
            assert collection_id == "coll-existing"
            return existing_collection

    service.db_ops = _ExistingChatOps()

    embedding_calls = []

    async def _no_embedding(user_id):
        embedding_calls.append(user_id)
        return None

    service._get_default_embedding_model = _no_embedding

    result = await service.initialize_user_chat_collection("user-1")

    assert result is existing_collection
    assert embedding_calls == [], "embedding lookup should be skipped when collection exists"
