import json
from types import SimpleNamespace

import pytest

import aperag.domains.conversation.service.chat_completion_service as completion_module
from aperag.domains.agent_runtime.db.models import AgentTurnStatus
from aperag.domains.agent_runtime.uimessage import (
    CitationData,
    DataCitationPart,
    TextPart,
    UIMessage,
    UrlCitationLocation,
)
from aperag.utils.constant import DOC_QA_REFERENCES


def _bot(*, bot_id="bot-1", model="mdl-chat"):
    return SimpleNamespace(
        id=bot_id,
        config=json.dumps(
            {
                "agent": {
                    "completion": {
                        "model_id": model,
                        "temperature": 0.1,
                    }
                }
            }
        ),
    )


def _turn(*, turn_id="turn-1", chat_id="chat-1", status=AgentTurnStatus.COMPLETED, error_message=None):
    return SimpleNamespace(
        id=turn_id,
        chat_id=chat_id,
        status=status,
        error_message=error_message,
    )


def _persisted_message(*, answer_text="done", references=None) -> UIMessage:
    """Phase 8 D8.6 (#80) chunk-2: the OpenAI-compat completion path
    now reads canonical ``UIMessage`` parts (``TextPart`` for the
    answer + ``DataCitationPart`` for each reference) directly from
    ``UIMessageStore`` instead of the legacy artifact rows.
    """

    references = references or [{"title": "Doc 1", "snippet": "hello", "url": "https://example.com/doc1"}]
    parts = [TextPart(text=answer_text)]
    for ref in references:
        parts.append(
            DataCitationPart(
                data=CitationData(
                    cited_text=ref.get("snippet", ""),
                    location=UrlCitationLocation(
                        url=ref.get("url", ""),
                        title=ref.get("title"),
                    ),
                )
            )
        )
    return UIMessage(id="msg-turn-1", role="assistant", parts=parts)


class _FakeUIMessageStore:
    def __init__(self, message: UIMessage | None):
        self._message = message

    async def read(self, _turn_id):
        return self._message


class _FakeEventService:
    def __init__(self, events=None):
        self._events = list(events or [])
        self.calls = 0

    async def get_events_after(self, _turn_id, after_sequence=0, limit=500):
        self.calls += 1
        if self._events:
            events, self._events = self._events, []
            return [event for event in events if event.sequence > after_sequence][:limit]
        return []


class _FakeTurnService:
    def __init__(self, *, chat, bot, turn, query_turns=None):
        self.chat = chat
        self.bot = bot
        self.turn = turn
        self.query_turns = list(query_turns or [turn])
        self.created_requests = []
        self.db_ops = SimpleNamespace(query_agent_turn=self._query_agent_turn)

    async def get_chat_and_bot(self, _user, _chat_id):
        return self.chat, self.bot

    async def create_or_get_turn(self, _user, _chat_id, request):
        self.created_requests.append(request)
        return self.chat, self.bot, self.turn, True

    async def _query_agent_turn(self, _user, _chat_id, _turn_id):
        if len(self.query_turns) > 1:
            return self.query_turns.pop(0)
        return self.query_turns[0]


class _FakeRuntimeManager:
    def __init__(self, *, turn_service, event_service, uimessage_store):
        self.turn_service = turn_service
        self.event_service = event_service
        self.uimessage_store = uimessage_store
        self.tasks = {}
        self.launch_calls = []
        self.cancel_calls = []
        self.claim_calls = []

    def launch_turn(self, **kwargs):
        self.launch_calls.append(kwargs)

    async def claim_turn(self, turn_id):
        self.claim_calls.append(turn_id)
        return "lease-owner-1"

    async def cancel_turn(self, turn_id):
        self.cancel_calls.append(turn_id)


class _FakeChatService:
    def __init__(self, *, created_chat_id="chat-ephemeral"):
        self.created_chat_id = created_chat_id
        self.created = []
        self.deleted = []

    async def create_chat(self, user, bot_id):
        self.created.append((user, bot_id))
        return SimpleNamespace(id=self.created_chat_id)

    async def delete_chat(self, user, bot_id, chat_id):
        self.deleted.append((user, bot_id, chat_id))


@pytest.mark.asyncio
async def test_openai_chat_completions_returns_openai_response_and_maps_overrides(monkeypatch):
    chat = SimpleNamespace(id="chat-1")
    bot = _bot()
    turn = _turn()
    fake_turn_service = _FakeTurnService(chat=chat, bot=bot, turn=turn)
    fake_runtime_manager = _FakeRuntimeManager(
        turn_service=fake_turn_service,
        event_service=_FakeEventService(),
        uimessage_store=_FakeUIMessageStore(_persisted_message(answer_text="final answer")),
    )

    monkeypatch.setattr(completion_module, "runtime_manager", fake_runtime_manager)

    service = completion_module.ChatCompletionService()
    stream, response = await service.openai_chat_completions(
        "user-1",
        {
            "model": "aperag",
            "messages": [{"role": "user", "content": "hello world"}],
            "temperature": 0.4,
            "stream": False,
        },
        {"bot_id": "bot-1", "chat_id": "chat-1"},
        {"Idempotency-Key": "idem-1"},
    )

    assert stream is None
    assert response["id"] == "turn-1"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert "final answer" in response["choices"][0]["message"]["content"]
    assert DOC_QA_REFERENCES in response["choices"][0]["message"]["content"]
    assert fake_runtime_manager.launch_calls
    assert fake_runtime_manager.claim_calls == ["turn-1"]
    created_request = fake_turn_service.created_requests[0]
    assert created_request.query == "hello world"
    assert created_request.client_idempotency_key == "idem-1"
    assert created_request.completion.model_id == "mdl-chat"
    assert created_request.completion.temperature == 0.4
    assert fake_runtime_manager.launch_calls[0]["lease_owner"] == "lease-owner-1"


@pytest.mark.asyncio
async def test_openai_chat_completions_creates_and_cleans_up_ephemeral_chat(monkeypatch):
    chat = SimpleNamespace(id="chat-ephemeral")
    bot = _bot()
    turn = _turn(chat_id="chat-ephemeral")
    fake_turn_service = _FakeTurnService(chat=chat, bot=bot, turn=turn)
    fake_runtime_manager = _FakeRuntimeManager(
        turn_service=fake_turn_service,
        event_service=_FakeEventService(),
        uimessage_store=_FakeUIMessageStore(_persisted_message(answer_text="ephemeral answer")),
    )
    fake_chat_service = _FakeChatService(created_chat_id="chat-ephemeral")

    monkeypatch.setattr(completion_module, "runtime_manager", fake_runtime_manager)
    monkeypatch.setattr(completion_module, "chat_service_global", fake_chat_service)

    service = completion_module.ChatCompletionService()
    _, response = await service.openai_chat_completions(
        "user-1",
        {"model": "aperag", "messages": [{"role": "user", "content": "temp"}], "stream": False},
        {"bot_id": "bot-1"},
        {},
    )

    assert response["choices"][0]["message"]["content"].startswith("ephemeral answer")
    assert fake_chat_service.created == [("user-1", "bot-1")]
    assert fake_chat_service.deleted == [("user-1", "bot-1", "chat-ephemeral")]


@pytest.mark.asyncio
async def test_openai_chat_completions_streams_sse_from_runtime_events(monkeypatch):
    chat = SimpleNamespace(id="chat-1")
    bot = _bot()
    running_turn = _turn(status=AgentTurnStatus.RUNNING)
    completed_turn = _turn(status=AgentTurnStatus.COMPLETED)
    fake_turn_service = _FakeTurnService(
        chat=chat,
        bot=bot,
        turn=running_turn,
        query_turns=[running_turn, completed_turn],
    )
    fake_event_service = _FakeEventService(
        events=[SimpleNamespace(sequence=1, type="text.delta", data={"delta": "hello"})]
    )
    fake_runtime_manager = _FakeRuntimeManager(
        turn_service=fake_turn_service,
        event_service=fake_event_service,
        uimessage_store=_FakeUIMessageStore(_persisted_message(answer_text="streamed answer")),
    )

    monkeypatch.setattr(completion_module, "runtime_manager", fake_runtime_manager)

    service = completion_module.ChatCompletionService()
    stream, response = await service.openai_chat_completions(
        "user-1",
        {"model": "aperag", "messages": [{"role": "user", "content": "stream me"}], "stream": True},
        {"bot_id": "bot-1", "chat_id": "chat-1"},
        {},
    )

    assert response is None
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    joined = "".join(chunks)
    assert '"object": "chat.completion.chunk"' in joined
    assert '"role": "assistant"' in joined
    assert '"content": "hello"' in joined
    assert '"finish_reason": "stop"' in joined
