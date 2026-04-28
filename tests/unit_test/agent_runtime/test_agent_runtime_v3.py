from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

import aperag.domains.agent_runtime.storage as agent_runtime_storage
from aperag.domains.agent_runtime.api import routes as agent_runtime_view
from aperag.domains.agent_runtime.db.models import AgentEventActor, AgentTurnStatus
from aperag.domains.agent_runtime.schemas import (
    AgentTimelineEventEnvelope,
    CreateTurnRequest,
    UserActivityIntent,
)
from aperag.domains.agent_runtime.services import EventService, HistoryWriter, TurnService
from aperag.domains.agent_runtime.uimessage import (
    AgentTurnSnapshot,
    CitationData,
    DataCitationPart,
    SourceUrlPart,
    TextPart,
    UIMessage,
    UrlCitationLocation,
)


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


class _FakeUIMessageStore:
    """Minimal in-memory UIMessageStore stand-in for unit tests."""

    def __init__(self, *, messages=None):
        self.messages = dict(messages or {})

    async def read(self, turn_id):
        return self.messages.get(turn_id)


class _FakeDbOps:
    def __init__(self, *, turn=None):
        self.turn = turn

    async def query_agent_turn(self, _user, _chat_id, _turn_id):
        return self.turn


class _FakeCreateTurnDbOps:
    def __init__(self, *, existing_turn=None, raise_on_create=False):
        self.existing_turn = existing_turn
        self.raise_on_create = raise_on_create
        self.created_turn = existing_turn or _build_turn(id="turn-created")
        self.create_calls = 0
        self.query_calls = 0

    async def query_chat_by_id(self, _user, _chat_id):
        return SimpleNamespace(id="chat-1", bot_id="bot-1")

    async def query_bot(self, _user, _bot_id, *, exclude_system: bool = True):
        # ``exclude_system`` ignored — the legacy AGENT bot used in this
        # fixture is non-system and would pass either filter.
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
    def __init__(self, *, turns=None):
        self.turns = list(turns or [])

    async def query_recent_agent_turns(self, _user, _chat_id, limit=8):
        return self.turns[-limit:]


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
        "timeline_cursor": 1,
        "gmt_started": None,
        "gmt_finished": None,
        "gmt_created": _now(),
        "gmt_updated": _now(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _build_envelope(sequence: int, event_type: str, *, label=None, status=None, data=None):
    return AgentTimelineEventEnvelope(
        event_id=f"event-{sequence}",
        turn_id="turn-1",
        sequence=sequence,
        timestamp=_now(),
        type=event_type,
        technical_type=event_type,
        label=label,
        status=status,
        actor=AgentEventActor.AGENT.value,
        data=data or {},
    )


class _FakeRequest:
    def __init__(self, *, base_url="http://testserver/", headers=None):
        self.base_url = base_url
        self.headers = headers or {}

    async def is_disconnected(self):
        return False


@pytest.mark.asyncio
async def test_turn_snapshot_returns_canonical_uimessage_parts_for_completed_turn():
    """Phase 8 D8.6 (#80) chunk-2: snapshot endpoint reads canonical
    ``UIMessagePart[]`` directly from ``UIMessageStore`` (no artifact
    fallback). The runtime now persists assistant text + citations as
    a single ``UIMessage`` row at end-of-turn.
    """

    turn = _build_turn(status=AgentTurnStatus.COMPLETED, timeline_cursor=2)
    persisted = UIMessage(
        id="msg-turn-1",
        role="assistant",
        parts=[
            TextPart(text="Hello world"),
            SourceUrlPart(source_id="src-1", url="https://example.com/a", title="Example"),
            DataCitationPart(
                data=CitationData(
                    cited_text="ApeRAG is great",
                    location=UrlCitationLocation(url="https://example.com/a", title="Example"),
                )
            ),
        ],
    )

    service = TurnService(
        db_ops=_FakeDbOps(turn=turn),
        redis_store=_FakeRedisStore(runtime_state={"status": "COMPLETED", "timeline_cursor": 2}),
        uimessage_store=_FakeUIMessageStore(messages={"turn-1": persisted}),
    )

    snapshot = await service.get_turn_snapshot("user-1", "chat-1", "turn-1")

    assert snapshot.turn_id == "turn-1"
    assert snapshot.chat_id == "chat-1"
    assert snapshot.role == "assistant"
    assert snapshot.status == "COMPLETED"
    assert snapshot.timeline_cursor == 2
    assert snapshot.error_text is None

    types = [getattr(part, "type", None) for part in snapshot.parts]
    assert types == ["text", "source-url", "data-citation"]
    assert snapshot.parts[0].text == "Hello world"
    assert snapshot.parts[1].source_id == "src-1"
    assert snapshot.parts[2].data.cited_text == "ApeRAG is great"
    assert snapshot.parts[2].data.location.url == "https://example.com/a"


@pytest.mark.asyncio
async def test_turn_snapshot_surfaces_error_text_for_failed_turn():
    """Phase 8 D8.6 (#80) chunk-2: a FAILED turn's ``error_text`` comes
    straight off the ``AgentTurn`` row — the legacy ``error_summary``
    artifact was removed when ``agent_artifact`` got dropped.
    """

    turn = _build_turn(
        status=AgentTurnStatus.FAILED,
        error_code="agent.tool_failure",
        error_message="upstream timeout",
        timeline_cursor=3,
    )

    service = TurnService(
        db_ops=_FakeDbOps(turn=turn),
        redis_store=_FakeRedisStore(runtime_state={"status": "FAILED"}),
        uimessage_store=_FakeUIMessageStore(),
    )

    snapshot = await service.get_turn_snapshot("user-1", "chat-1", "turn-1")

    assert snapshot.status == "FAILED"
    assert snapshot.error_text == "upstream timeout"
    # No assistant text was produced; error is not modelled as a part.
    assert snapshot.parts == []


@pytest.mark.asyncio
async def test_turn_snapshot_does_not_expose_legacy_keys():
    """Regression-guard: legacy ``{turn, timeline, artifacts}`` must
    never reappear on the new ``AgentTurnSnapshot`` shape (D8 §2
    wire/at-rest byte-equal canonical).
    """

    service = TurnService(
        db_ops=_FakeDbOps(turn=_build_turn(status=AgentTurnStatus.QUEUED)),
        redis_store=_FakeRedisStore(),
        uimessage_store=_FakeUIMessageStore(),
    )

    snapshot = await service.get_turn_snapshot("user-1", "chat-1", "turn-1")
    serialised = snapshot.model_dump(mode="json")

    forbidden = {"turn", "timeline", "artifacts"}
    assert forbidden.isdisjoint(serialised.keys()), (
        f"legacy keys leaked into AgentTurnSnapshot serialization: {forbidden & serialised.keys()}"
    )


@pytest.mark.asyncio
async def test_event_service_adapt_event_envelope_adds_user_activity_contract():
    """Phase 8 D8.6 (#80) chunk-3: ``to_event_envelope`` is gone with
    ``agent_timeline_event``. ``adapt_event_envelope`` is now the only
    surface for turning a wire envelope into the user-activity-tagged
    payload the FE renderer consumes.
    """

    envelope = _build_envelope(
        3,
        "tool.started",
        label="vector_search",
        status="started",
        data={
            # Post D10.h cutover the omnibus ``search_collection`` is
            # gone; ``_KNOWLEDGE_SEARCH_TOOLS`` (services.py) now keys
            # off the canonical D10.d split tools, so the activity
            # inference contract is exercised against ``vector_search``.
            "tool_name": "vector_search",
            "args": {
                "query": "OpenAI API key",
                "collection_name": "Product Docs",
            },
        },
    )

    adapted = EventService.adapt_event_envelope(envelope)

    assert adapted.technical_type == "tool.started"
    assert adapted.user_activity is not None
    assert adapted.user_activity.intent == UserActivityIntent.SEARCHING_KNOWLEDGE
    assert adapted.user_activity.title_key == "activity.searching_knowledge.title"
    assert adapted.user_activity.subtitle_key == "activity.searching_knowledge.subtitle"
    assert adapted.user_activity.detail_key == "activity.searching_knowledge.detail.keyword"
    assert adapted.user_activity.context is not None
    assert adapted.user_activity.context.keyword == "OpenAI API key"
    assert adapted.user_activity.context.source_name == "Product Docs"
    assert adapted.user_activity.context.target_type == "knowledge_base"


@pytest.mark.asyncio
async def test_turn_snapshot_user_activity_inference_runs_via_event_service():
    """``TurnService.get_turn_snapshot`` no longer projects timeline
    events. The user activity inference contract still belongs to
    ``EventService`` and is exercised directly.
    """

    turn = _build_turn()
    service = TurnService(
        db_ops=_FakeDbOps(turn=turn),
        redis_store=_FakeRedisStore(runtime_state=None),
        uimessage_store=_FakeUIMessageStore(),
    )

    snapshot = await service.get_turn_snapshot("user-1", "chat-1", "turn-1")

    # Legacy timeline / artifacts fields are gone (D8.4d canonical lock).
    assert not hasattr(snapshot, "timeline")
    assert not hasattr(snapshot, "artifacts")


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
    """Phase 8 D8.6 (#80) chunk-2: ``build_history_context`` now reads
    canonical ``UIMessage`` text parts per-turn instead of legacy
    ``answer`` artifact rows. Turns whose persisted message is empty
    contribute nothing to the prompt context.
    """

    completed_turn = _build_turn(
        id="turn-v3",
        status=AgentTurnStatus.COMPLETED,
        input_text="new question",
    )
    incomplete_answer_turn = _build_turn(
        id="turn-v3-missing-answer",
        status=AgentTurnStatus.COMPLETED,
        input_text="missing answer question",
    )
    persisted_message = UIMessage(
        id="msg-turn-v3",
        role="assistant",
        parts=[TextPart(text="new answer")],
    )
    writer = HistoryWriter(
        db_ops=_FakeHistoryDbOps(turns=[completed_turn, incomplete_answer_turn]),
        uimessage_store=_FakeUIMessageStore(messages={"turn-v3": persisted_message}),
    )

    context = await writer.build_history_context("user-1", "chat-1")

    assert "User: new question" in context
    assert "Assistant: new answer" in context
    assert "missing answer question" not in context


@pytest.mark.asyncio
async def test_agent_runtime_views_create_stream_snapshot_and_cancel(monkeypatch):
    """Phase 8 D8.6 (#80) chunk-2: ``/agent/artifacts/{id}`` route is
    deleted along with the artifact_service entry point. This test
    pins the remaining create / snapshot / cancel / stream routes.
    """

    turn = _build_turn(status=AgentTurnStatus.RUNNING, timeline_cursor=1)
    snapshot = AgentTurnSnapshot(
        turn_id="turn-1",
        chat_id="chat-1",
        status=AgentTurnStatus.RUNNING.value,
        parts=[],
        timeline_cursor=1,
        started_at=_now(),
        finished_at=None,
        created_at=_now(),
        updated_at=_now(),
    )
    timeline_for_stream = [
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
    ]

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
            return TurnService.to_turn_envelope(turn)

    class _FakeEventService:
        async def get_events_after(self, _turn_id, after_sequence=0, limit=500):
            return timeline_for_stream if after_sequence == 0 else []

    class _FakeRuntimeManager:
        def __init__(self):
            self.turn_service = _FakeTurnService()
            self.event_service = _FakeEventService()
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
    assert snapshot_response.turn_id == "turn-1"
    assert snapshot_response.role == "assistant"

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
    # Phase 8 D8.1 hard-cut: SSE wire is now AI SDK v5 stream parts
    # (no SSE ``event:`` field, JSON ``type`` discriminator only).
    # ``turn.started`` envelope fans out to ``[start, start-step]``.
    assert '"type":"start"' in joined
    assert '"type":"start-step"' in joined
    assert "event: turn.started" not in joined
    assert stream_response.headers["x-vercel-ai-ui-message-stream"] == "v1"


def test_agent_runtime_views_no_artifact_route():
    """Phase 8 D8.6 (#80) chunk-2: the legacy
    ``/agent/artifacts/{artifact_id}`` route + ``ArtifactService`` are
    removed. Pin their absence as a regression guard.
    """

    paths = {route.path for route in agent_runtime_view.router.routes}
    assert "/agent/artifacts/{artifact_id}" not in paths
    assert not hasattr(agent_runtime_view, "get_artifact_view")


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
