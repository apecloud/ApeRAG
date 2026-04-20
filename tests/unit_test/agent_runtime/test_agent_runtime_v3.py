from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from aperag.agent_runtime.schemas import (
    AgentArtifactEnvelope,
    AgentTimelineEventEnvelope,
    AgentTurnSnapshot,
    CreateTurnRequest,
)
from aperag.agent_runtime.services import TurnService
from aperag.db.models import AgentEventActor, AgentTurnStatus
from aperag.views import agent_runtime as agent_runtime_view


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


class _FakeDbOps:
    def __init__(self, *, turn=None, persisted_events=None, artifacts=None):
        self.turn = turn
        self.persisted_events = list(persisted_events or [])
        self.artifacts = list(artifacts or [])

    async def query_agent_turn(self, _user, _chat_id, _turn_id):
        return self.turn

    async def query_agent_timeline_events(self, _turn_id, after_sequence=0, limit=2000):
        return [
            event
            for event in self.persisted_events
            if event.sequence > after_sequence
        ][:limit]

    async def query_agent_artifacts_by_turn(self, _turn_id):
        return list(self.artifacts)


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
    turn = _build_turn()
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
                "timeline_cursor": 2,
                "answer_artifact_id": "artifact-1",
            },
        ),
    )

    snapshot = await service.get_turn_snapshot("user-1", "chat-1", "turn-1")

    assert snapshot.turn.status == "RUNNING"
    assert snapshot.turn.timeline_cursor == 2
    assert snapshot.turn.answer_artifact_id == "artifact-1"
    assert [event.sequence for event in snapshot.timeline] == [1, 2]
    assert snapshot.artifacts[0].artifact_id == "artifact-1"


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
            self.launch_args = None
            self.cancelled_turn_id = None

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
    assert create_response.stream_url.endswith(
        "/api/v2/agent/chats/chat-1/turns/turn-1/events"
    )
    assert fake_runtime_manager.launch_args["turn"].id == "turn-1"

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
    assert "\"turn_id\": \"turn-1\"" in joined
