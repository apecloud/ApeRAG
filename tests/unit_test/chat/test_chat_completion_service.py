import json
from types import SimpleNamespace

import pytest

import aperag.domains.conversation.service.chat_completion_service as completion_module
from aperag.domains.agent_runtime.db.models import AgentTurnStatus
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


def _artifacts(*, answer_text="done", references=None):
    """Phase 8 D8.4d (#90): the OpenAI-compat completion path now reads
    raw ``AgentArtifact`` rows directly from the DB instead of through
    ``get_turn_snapshot`` (which now returns canonical UIMessage parts
    for the FE)."""
    references = references or [{"title": "Doc 1", "snippet": "hello"}]
    return [
        SimpleNamespace(
            artifact_id="artifact-answer",
            artifact_type="answer",
            payload={"text": answer_text},
        ),
        SimpleNamespace(
            artifact_id="artifact-refs",
            artifact_type="reference_bundle",
            payload={"items": references},
        ),
    ]


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
    def __init__(self, *, chat, bot, turn, artifacts, query_turns=None):
        self.chat = chat
        self.bot = bot
        self.turn = turn
        self.artifacts = list(artifacts or [])
        self.query_turns = list(query_turns or [turn])
        self.created_requests = []
        self.db_ops = SimpleNamespace(
            query_agent_turn=self._query_agent_turn,
            query_agent_artifacts_by_turn=self._query_agent_artifacts_by_turn,
        )

    async def get_chat_and_bot(self, _user, _chat_id):
        return self.chat, self.bot

    async def create_or_get_turn(self, _user, _chat_id, request):
        self.created_requests.append(request)
        return self.chat, self.bot, self.turn, True

    async def _query_agent_turn(self, _user, _chat_id, _turn_id):
        if len(self.query_turns) > 1:
            return self.query_turns.pop(0)
        return self.query_turns[0]

    async def _query_agent_artifacts_by_turn(self, _turn_id):
        return list(self.artifacts)


class _FakeRuntimeManager:
    def __init__(self, *, turn_service, event_service):
        self.turn_service = turn_service
        self.event_service = event_service
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
    artifacts = _artifacts(answer_text="final answer")
    fake_turn_service = _FakeTurnService(chat=chat, bot=bot, turn=turn, artifacts=artifacts)
    fake_runtime_manager = _FakeRuntimeManager(turn_service=fake_turn_service, event_service=_FakeEventService())

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
    artifacts = _artifacts(answer_text="ephemeral answer")
    fake_turn_service = _FakeTurnService(chat=chat, bot=bot, turn=turn, artifacts=artifacts)
    fake_runtime_manager = _FakeRuntimeManager(turn_service=fake_turn_service, event_service=_FakeEventService())
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
    artifacts = _artifacts(answer_text="streamed answer")
    fake_turn_service = _FakeTurnService(
        chat=chat,
        bot=bot,
        turn=running_turn,
        artifacts=artifacts,
        query_turns=[running_turn, completed_turn],
    )
    fake_event_service = _FakeEventService(
        events=[SimpleNamespace(sequence=1, type="text.delta", data={"delta": "hello"})]
    )
    fake_runtime_manager = _FakeRuntimeManager(turn_service=fake_turn_service, event_service=fake_event_service)

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
