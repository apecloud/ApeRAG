from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

import aperag.domains.agent_runtime.services as agent_runtime_services
import aperag.domains.agent_runtime.storage as agent_runtime_storage
from aperag.agent_runtime.schemas import (
    AgentArtifactEnvelope,
    AgentTimelineEventEnvelope,
    AgentTurnSnapshot,
    CreateTurnRequest,
    UserActivityIntent,
)
from aperag.agent_runtime.services import EventService, HistoryWriter, TurnService
from aperag.db.models import AgentEventActor, AgentTurnStatus
from aperag.domains.agent_runtime.api import routes as agent_runtime_view


def _now():
    return datetime.now(timezone.utc)


class _FakeRedisStore:
    def __init__(self, *, events=None, runtime_state=None):
        self.events = list(events or [])
        self.runtime_state = runtime_state
        self.updated_states = []

    async def get_all_events(self, _turn_id):
        return list(self.events)

    async def get_runtime_state(self, _turn_id):
        return self.runtime_state

    async def update_runtime_state(self, turn_id, state):
        self.updated_states.append(("update", turn_id, state))
        self.runtime_state = state

    async def merge_runtime_state(self, turn_id, state):
        self.updated_states.append(("merge", turn_id, state))
        current = dict(self.runtime_state or {})
        current.update(state)
        self.runtime_state = current
        return current


class _FakeLeaseRedisClient:
    def __init__(self):
        self.values = {}

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def eval(self, script, _numkeys, key, token, *_args):
        if "EXPIRE" in script:
            return 1 if self.values.get(key) == token else 0
        if "DEL" in script:
            if self.values.get(key) == token:
                self.values.pop(key, None)
                return 1
            return 0
        raise AssertionError(f"Unexpected script: {script}")


class _FakeDbOps:
    def __init__(self, *, turn=None, persisted_events=None, artifacts=None):
        self.turn = turn
        self.persisted_events = list(persisted_events or [])
        self.artifacts = list(artifacts or [])

    async def query_agent_turn(self, _user, _chat_id, _turn_id):
        return self.turn

    async def query_agent_timeline_events(self, _turn_id, after_sequence=0, limit=2000):
        return [event for event in self.persisted_events if event.sequence > after_sequence][:limit]

    async def query_agent_artifacts_by_turn(self, _turn_id):
        return list(self.artifacts)


class _FakeCreateTurnDbOps:
    def __init__(self, *, existing_turn=None, raise_on_create=False):
        self.existing_turn = existing_turn
        self.raise_on_create = raise_on_create
        self.created_turn = existing_turn or _build_turn(id="turn-created")
        self.create_calls = 0
        self.query_calls = 0

    async def query_chat_by_id(self, _user, _chat_id):
        return SimpleNamespace(id="chat-1", bot_id="bot-1")

    async def query_bot(self, _user, _bot_id):
        return SimpleNamespace(id="bot-1", type="agent")

    async def query_agent_turn_by_idempotency(self, _user, _chat_id, _idempotency_key):
        self.query_calls += 1
        if self.raise_on_create and self.query_calls == 1:
            return None
        return self.existing_turn

    async def create_agent_turn(self, **_kwargs):
        self.create_calls += 1
        if self.raise_on_create:
            raise IntegrityError("insert into agent_turn", {}, Exception("duplicate key"))
        self.existing_turn = self.created_turn
        return self.created_turn


class _FakeHistoryDbOps:
    def __init__(self, *, turns=None, artifacts_by_turn=None):
        self.turns = list(turns or [])
        self.artifacts_by_turn = artifacts_by_turn or {}

    async def query_recent_agent_turns(self, _user, _chat_id, limit=8):
        return self.turns[-limit:]

    async def query_agent_artifacts_by_turn(self, turn_id):
        return list(self.artifacts_by_turn.get(turn_id, []))


def _build_turn(**overrides):
    values = {
        "id": "turn-1",
        "chat_id": "chat-1",
        "user": "user-1",
        "bot_id": "bot-1",
        "request_id": "req-1",
        "client_idempotency_key": "idem-1",
        "status": AgentTurnStatus.QUEUED,
        "input_text": "hello",
        "model_profile": {"model": "gpt"},
        "error_code": None,
        "error_message": None,
        "answer_artifact_id": None,
        "reference_bundle_artifact_id": None,
        "timeline_cursor": 1,
        "gmt_started": None,
        "gmt_finished": None,
        "gmt_created": _now(),
        "gmt_updated": _now(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _build_event(sequence: int, event_type: str, *, label=None, status=None, data=None):
    return SimpleNamespace(
        id=f"event-{sequence}",
        turn_id="turn-1",
        sequence=sequence,
        timestamp=_now(),
        type=event_type,
        label=label,
        status=status,
        actor=AgentEventActor.AGENT,
        data=data or {},
    )


def _build_artifact():
    return SimpleNamespace(
        id="artifact-1",
        turn_id="turn-1",
        artifact_type="answer",
        summary="answer",
        payload={"text": "done"},
        storage_ref=None,
        gmt_created=_now(),
        gmt_updated=_now(),
    )


class _FakeRequest:
    def __init__(self, *, base_url="http://testserver/", headers=None):
        self.base_url = base_url
        self.headers = headers or {}

    async def is_disconnected(self):
        return False


@pytest.mark.asyncio
async def test_turn_snapshot_merges_persisted_events_cached_events_and_runtime_state():
    turn = _build_turn(answer_artifact_id="artifact-db", timeline_cursor=0)
    persisted_event = _build_event(1, "turn.started", status="running")
    cached_event = AgentTimelineEventEnvelope(
        event_id="event-2",
        turn_id="turn-1",
        sequence=2,
        timestamp=_now(),
        type="text.delta",
        label="Streaming Answer",
        status="streaming",
        actor="agent",
        data={"delta": "hello"},
    ).model_dump(mode="json")
    artifact = _build_artifact()

    service = TurnService(
        db_ops=_FakeDbOps(
            turn=turn,
            persisted_events=[persisted_event],
            artifacts=[artifact],
        ),
        redis_store=_FakeRedisStore(
            events=[cached_event],
            runtime_state={
                "status": "RUNNING",
                "timeline_cursor": 1,
                "answer_artifact_id": None,
                "reference_bundle_artifact_id": "bundle-1",
            },
        ),
    )

    snapshot = await service.get_turn_snapshot("user-1", "chat-1", "turn-1")

    assert snapshot.turn.status == "RUNNING"
    assert snapshot.turn.timeline_cursor == 2
    assert snapshot.turn.answer_artifact_id == "artifact-db"
    assert snapshot.turn.reference_bundle_artifact_id == "bundle-1"
    assert [event.sequence for event in snapshot.timeline] == [1, 2]
    assert snapshot.artifacts[0].artifact_id == "artifact-1"


@pytest.mark.asyncio
async def test_event_service_to_event_envelope_adds_user_activity_contract():
    event = _build_event(
        3,
        "tool.started",
        label="search_collection",
        status="started",
        data={
            "tool_name": "search_collection",
            "args": {
                "query": "OpenAI API key",
                "collection_name": "Product Docs",
            },
        },
    )

    envelope = EventService.to_event_envelope(event)

    assert envelope.technical_type == "tool.started"
    assert envelope.user_activity is not None
    assert envelope.user_activity.intent == UserActivityIntent.SEARCHING_KNOWLEDGE
    assert envelope.user_activity.title_key == "activity.searching_knowledge.title"
    assert envelope.user_activity.subtitle_key == "activity.searching_knowledge.subtitle"
    assert envelope.user_activity.detail_key == "activity.searching_knowledge.detail.keyword"
    assert envelope.user_activity.context is not None
    assert envelope.user_activity.context.keyword == "OpenAI API key"
    assert envelope.user_activity.context.source_name == "Product Docs"
    assert envelope.user_activity.context.target_type == "knowledge_base"


@pytest.mark.asyncio
async def test_turn_snapshot_backfills_user_activity_for_legacy_cached_events():
    turn = _build_turn()
    legacy_cached_event = {
        "event_id": "event-3",
        "turn_id": "turn-1",
        "sequence": 3,
        "timestamp": _now().isoformat(),
        "type": "tool.finished",
        "label": "search_collection",
        "status": "success",
        "actor": "tool",
        "data": {
            "tool_name": "search_collection",
            "result": {
                "items": [{"id": "doc-1"}, {"id": "doc-2"}],
                "collection_name": "Product Docs",
            },
        },
    }

    service = TurnService(
        db_ops=_FakeDbOps(turn=turn),
        redis_store=_FakeRedisStore(events=[legacy_cached_event], runtime_state=None),
    )

    snapshot = await service.get_turn_snapshot("user-1", "chat-1", "turn-1")

    assert len(snapshot.timeline) == 1
    event = snapshot.timeline[0]
    assert event.technical_type == "tool.finished"
    assert event.user_activity is not None
    assert event.user_activity.intent == UserActivityIntent.SEARCHING_KNOWLEDGE
    assert event.user_activity.detail_key == "activity.searching_knowledge.detail.source_name"
    assert event.user_activity.context is not None
    assert event.user_activity.context.source_name == "Product Docs"
    assert event.user_activity.context.count == 2


@pytest.mark.asyncio
async def test_create_or_get_turn_recovers_existing_turn_after_idempotency_race():
    existing_turn = _build_turn(id="turn-existing")
    db_ops = _FakeCreateTurnDbOps(existing_turn=existing_turn, raise_on_create=True)
    service = TurnService(
        db_ops=db_ops,
        redis_store=_FakeRedisStore(),
    )

    chat, bot, turn, created = await service.create_or_get_turn(
        "user-1",
        "chat-1",
        CreateTurnRequest(query="hello", client_idempotency_key="idem-race"),
    )

    assert chat.id == "chat-1"
    assert bot.id == "bot-1"
    assert turn.id == "turn-existing"
    assert created is False
    assert db_ops.create_calls == 1


@pytest.mark.asyncio
async def test_agent_runtime_redis_store_turn_claim_renew_and_release(monkeypatch):
    client = _FakeLeaseRedisClient()

    async def _fake_get_async_client(_cls, redis_url=None):
        return client

    monkeypatch.setattr(
        agent_runtime_storage.RedisConnectionManager,
        "get_async_client",
        classmethod(_fake_get_async_client),
    )

    store = agent_runtime_storage.AgentRuntimeRedisStore(prefix="test-agent-runtime")

    assert await store.try_claim_turn("turn-1", "owner-a") is True
    assert await store.try_claim_turn("turn-1", "owner-b") is False
    assert await store.renew_turn_claim("turn-1", "owner-a") is True
    assert await store.renew_turn_claim("turn-1", "owner-b") is False
    assert await store.release_turn_claim("turn-1", "owner-b") is False
    assert await store.release_turn_claim("turn-1", "owner-a") is True
    assert await store.try_claim_turn("turn-1", "owner-b") is True


@pytest.mark.asyncio
async def test_history_writer_builds_context_from_v3_turns_only():
    completed_turn = _build_turn(
        id="turn-v3",
        status=AgentTurnStatus.COMPLETED,
        input_text="new question",
        answer_artifact_id="artifact-answer",
    )
    incomplete_answer_turn = _build_turn(
        id="turn-v3-missing-answer",
        status=AgentTurnStatus.COMPLETED,
        input_text="missing answer question",
    )
    answer_artifact = SimpleNamespace(
        id="artifact-answer",
        artifact_type="answer",
        payload={"text": "new answer"},
    )
    writer = HistoryWriter(
        db_ops=_FakeHistoryDbOps(
            turns=[completed_turn, incomplete_answer_turn],
            artifacts_by_turn={
                "turn-v3": [answer_artifact],
                "turn-v3-missing-answer": [],
            },
        )
    )

    context = await writer.build_history_context("user-1", "chat-1")

    assert "User: new question" in context
    assert "Assistant: new answer" in context
    assert "missing answer question" not in context


@pytest.mark.asyncio
async def test_agent_runtime_views_create_stream_snapshot_cancel_and_artifact(monkeypatch):
    turn = _build_turn(status=AgentTurnStatus.RUNNING, timeline_cursor=1)
    snapshot = AgentTurnSnapshot(
        turn=TurnService.to_turn_envelope(turn),
        timeline=[
            AgentTimelineEventEnvelope(
                event_id="event-1",
                turn_id="turn-1",
                sequence=1,
                timestamp=_now(),
                type="turn.started",
                label="Thinking",
                status="running",
                actor="system",
                data={"chat_id": "chat-1"},
            )
        ],
        artifacts=[],
    )
    artifact = AgentArtifactEnvelope(
        artifact_id="artifact-1",
        turn_id="turn-1",
        artifact_type="answer",
        summary="done",
        payload={"text": "done"},
        storage_ref=None,
        created_at=_now().isoformat(),
        updated_at=_now().isoformat(),
    )

    class _FakeTurnService:
        def __init__(self):
            async def _query_agent_turn(*_args, **_kwargs):
                return SimpleNamespace(
                    status=AgentTurnStatus.COMPLETED,
                    timeline_cursor=1,
                )

            self.db_ops = SimpleNamespace(query_agent_turn=_query_agent_turn)

        async def create_or_get_turn(self, _user, _chat_id, _body):
            return (
                SimpleNamespace(id="chat-1"),
                SimpleNamespace(id="bot-1"),
                turn,
                True,
            )

        async def get_turn_snapshot(self, _user, _chat_id, _turn_id):
            return snapshot

        def to_turn_envelope(self, _turn):
            return snapshot.turn

    class _FakeEventService:
        async def get_events_after(self, _turn_id, after_sequence=0, limit=500):
            return snapshot.timeline if after_sequence == 0 else []

    class _FakeArtifactService:
        async def get_artifact_for_user(self, _user, _artifact_id):
            return artifact

    class _FakeRuntimeManager:
        def __init__(self):
            self.turn_service = _FakeTurnService()
            self.event_service = _FakeEventService()
            self.artifact_service = _FakeArtifactService()
            self.tasks = {}
            self.claim_turn_id = None
            self.claim_turn_result = "lease-owner"
            self.launch_args = None
            self.cancelled_turn_id = None

        async def claim_turn(self, turn_id):
            self.claim_turn_id = turn_id
            return self.claim_turn_result

        def launch_turn(self, **kwargs):
            self.launch_args = kwargs

        async def cancel_turn(self, turn_id):
            self.cancelled_turn_id = turn_id

    fake_runtime_manager = _FakeRuntimeManager()
    monkeypatch.setattr(agent_runtime_view, "runtime_manager", fake_runtime_manager)

    request = _FakeRequest()
    body = CreateTurnRequest(query="hello")
    user = SimpleNamespace(id="user-1")

    create_response = await agent_runtime_view.create_turn_view(
        request,
        "chat-1",
        body,
        user=user,
    )
    assert create_response.turn.turn_id == "turn-1"
    assert create_response.stream_url.endswith("/api/v2/agent/chats/chat-1/turns/turn-1/events")
    assert fake_runtime_manager.claim_turn_id == "turn-1"
    assert fake_runtime_manager.launch_args["turn"].id == "turn-1"
    assert fake_runtime_manager.launch_args["lease_owner"] == "lease-owner"

    snapshot_response = await agent_runtime_view.get_turn_snapshot_view(
        "chat-1",
        "turn-1",
        user=user,
    )
    assert snapshot_response.turn.turn_id == "turn-1"

    artifact_response = await agent_runtime_view.get_artifact_view(
        "artifact-1",
        user=user,
    )
    assert artifact_response.artifact_id == "artifact-1"

    cancel_response = await agent_runtime_view.cancel_turn_view(
        "chat-1",
        "turn-1",
        user=user,
    )
    assert cancel_response.turn_id == "turn-1"
    assert fake_runtime_manager.cancelled_turn_id == "turn-1"

    stream_response = await agent_runtime_view.stream_turn_events_view(
        _FakeRequest(),
        "chat-1",
        "turn-1",
        after_sequence=0,
        user=user,
    )
    chunks = []
    async for chunk in stream_response.body_iterator:
        chunks.append(chunk)

    joined = "".join(chunks)
    assert "event: turn.started" in joined
    assert '"turn_id": "turn-1"' in joined


@pytest.mark.asyncio
async def test_agent_runtime_view_create_skips_launch_when_turn_claim_fails(monkeypatch):
    turn = _build_turn(status=AgentTurnStatus.QUEUED, timeline_cursor=0)

    class _FakeTurnService:
        async def create_or_get_turn(self, _user, _chat_id, _body):
            return (
                SimpleNamespace(id="chat-1"),
                SimpleNamespace(id="bot-1"),
                turn,
                True,
            )

        def to_turn_envelope(self, _turn):
            return TurnService.to_turn_envelope(turn)

    class _FakeRuntimeManager:
        def __init__(self):
            self.turn_service = _FakeTurnService()
            self.claim_turn_id = None
            self.launch_args = None

        async def claim_turn(self, turn_id):
            self.claim_turn_id = turn_id
            return None

        def launch_turn(self, **kwargs):
            self.launch_args = kwargs

    fake_runtime_manager = _FakeRuntimeManager()
    monkeypatch.setattr(agent_runtime_view, "runtime_manager", fake_runtime_manager)

    response = await agent_runtime_view.create_turn_view(
        _FakeRequest(),
        "chat-1",
        CreateTurnRequest(query="hello"),
        user=SimpleNamespace(id="user-1"),
    )

    assert response.turn.turn_id == "turn-1"
    assert fake_runtime_manager.claim_turn_id == "turn-1"
    assert fake_runtime_manager.launch_args is None
