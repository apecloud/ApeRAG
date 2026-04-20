# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import uuid
from typing import Any, Optional

from aperag.agent.agent_history_manager import AgentHistoryManager
from aperag.agent_runtime.schemas import (
    AgentArtifactEnvelope,
    AgentTimelineEventEnvelope,
    AgentTurnEnvelope,
    AgentTurnSnapshot,
    CreateTurnRequest,
    ReferenceBundleItem,
)
from aperag.agent_runtime.storage import AgentRuntimeRedisStore
from aperag.db.models import AgentEventActor, AgentTurnStatus, BotType
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import ChatNotFoundException, ResourceNotFoundException, ValidationException
from aperag.schema import view_models
from aperag.utils.history import query_chat_messages
from aperag.utils.utils import utc_now


def _parse_bot_config(bot) -> Optional[view_models.BotConfig]:
    if not bot or not bot.config:
        return None
    try:
        return view_models.BotConfig(**json.loads(bot.config))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


class TurnService:
    def __init__(self, db_ops: AsyncDatabaseOps | None = None, redis_store: AgentRuntimeRedisStore | None = None):
        self.db_ops = db_ops or async_db_ops
        self.redis_store = redis_store or AgentRuntimeRedisStore()

    async def get_chat_and_bot(self, user: str, chat_id: str):
        chat = await self.db_ops.query_chat_by_id(user, chat_id)
        if not chat:
            raise ChatNotFoundException(chat_id)

        bot = await self.db_ops.query_bot(user, chat.bot_id)
        if not bot:
            raise ResourceNotFoundException("Bot", chat.bot_id)
        if bot.type != BotType.AGENT:
            raise ValidationException("Only agent bots are supported")

        return chat, bot

    async def create_or_get_turn(self, user: str, chat_id: str, request: CreateTurnRequest):
        chat, bot = await self.get_chat_and_bot(user, chat_id)
        idempotency_key = request.client_idempotency_key or uuid.uuid4().hex
        existing = await self.db_ops.query_agent_turn_by_idempotency(user, chat_id, idempotency_key)
        if existing:
            return chat, bot, existing, False

        turn = await self.db_ops.create_agent_turn(
            chat_id=chat_id,
            user=user,
            bot_id=bot.id,
            request_id=uuid.uuid4().hex,
            client_idempotency_key=idempotency_key,
            input_text=request.query,
            model_profile=self._build_model_profile(request),
        )
        await self.redis_store.update_runtime_state(
            turn.id,
            {"status": turn.status, "timeline_cursor": turn.timeline_cursor, "chat_id": chat_id, "user": user},
        )
        return chat, bot, turn, True

    async def mark_running(self, turn_id: str) -> None:
        turn = await self.db_ops.update_agent_turn(turn_id, status=AgentTurnStatus.RUNNING, gmt_started=utc_now())
        if turn:
            await self.redis_store.merge_runtime_state(
                turn_id,
                {"status": turn.status, "timeline_cursor": turn.timeline_cursor, "chat_id": turn.chat_id},
            )

    async def mark_completed(
        self, turn_id: str, *, answer_artifact_id: Optional[str], reference_bundle_artifact_id: Optional[str], sequence: int
    ) -> None:
        turn = await self.db_ops.update_agent_turn(
            turn_id,
            status=AgentTurnStatus.COMPLETED,
            answer_artifact_id=answer_artifact_id,
            reference_bundle_artifact_id=reference_bundle_artifact_id,
            timeline_cursor=sequence,
            gmt_finished=utc_now(),
        )
        if turn:
            await self.redis_store.merge_runtime_state(
                turn_id,
                {
                    "status": turn.status,
                    "timeline_cursor": turn.timeline_cursor,
                    "chat_id": turn.chat_id,
                    "answer_artifact_id": answer_artifact_id,
                    "reference_bundle_artifact_id": reference_bundle_artifact_id,
                },
            )

    async def mark_failed(self, turn_id: str, *, error_code: str, error_message: str, sequence: int) -> None:
        turn = await self.db_ops.update_agent_turn(
            turn_id,
            status=AgentTurnStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
            timeline_cursor=sequence,
            gmt_finished=utc_now(),
        )
        if turn:
            await self.redis_store.merge_runtime_state(
                turn_id,
                {
                    "status": turn.status,
                    "timeline_cursor": turn.timeline_cursor,
                    "chat_id": turn.chat_id,
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )

    async def mark_cancelled(self, turn_id: str, *, sequence: int) -> None:
        turn = await self.db_ops.update_agent_turn(
            turn_id,
            status=AgentTurnStatus.CANCELLED,
            timeline_cursor=sequence,
            gmt_finished=utc_now(),
        )
        if turn:
            await self.redis_store.merge_runtime_state(
                turn_id,
                {"status": turn.status, "timeline_cursor": turn.timeline_cursor, "chat_id": turn.chat_id},
            )

    async def get_turn_snapshot(self, user: str, chat_id: str, turn_id: str) -> AgentTurnSnapshot:
        turn = await self.db_ops.query_agent_turn(user, chat_id, turn_id)
        if not turn:
            raise ResourceNotFoundException("Turn", turn_id)

        persisted_events = await self.db_ops.query_agent_timeline_events(turn_id, after_sequence=0, limit=2000)
        cached_events = [AgentTimelineEventEnvelope.model_validate(item) for item in await self.redis_store.get_all_events(turn_id)]
        merged_events: dict[int, AgentTimelineEventEnvelope] = {
            event.sequence: EventService.to_event_envelope(event) for event in persisted_events
        }
        for event in cached_events:
            merged_events[event.sequence] = event

        runtime_state = await self.redis_store.get_runtime_state(turn_id)
        artifacts = await self.db_ops.query_agent_artifacts_by_turn(turn_id)
        turn_envelope = self.to_turn_envelope(turn)
        timeline = [merged_events[key] for key in sorted(merged_events)]
        latest_sequence = timeline[-1].sequence if timeline else 0

        if runtime_state:
            timeline_cursor = runtime_state.get("timeline_cursor")
            turn_envelope = turn_envelope.model_copy(
                update={
                    "status": runtime_state.get("status", turn_envelope.status),
                    "timeline_cursor": max(
                        turn_envelope.timeline_cursor,
                        timeline_cursor if isinstance(timeline_cursor, int) else latest_sequence,
                        latest_sequence,
                    ),
                    "answer_artifact_id": runtime_state.get("answer_artifact_id", turn_envelope.answer_artifact_id),
                    "reference_bundle_artifact_id": runtime_state.get(
                        "reference_bundle_artifact_id", turn_envelope.reference_bundle_artifact_id
                    ),
                    "error_code": runtime_state.get("error_code", turn_envelope.error_code),
                    "error_message": runtime_state.get("error_message", turn_envelope.error_message),
                }
            )

        return AgentTurnSnapshot(
            turn=turn_envelope,
            timeline=timeline,
            artifacts=[ArtifactService.to_artifact_envelope(artifact) for artifact in artifacts],
        )

    @staticmethod
    def _build_model_profile(request: CreateTurnRequest) -> dict[str, Any]:
        if not request.completion:
            return {}
        return request.completion.model_dump(exclude_none=True)

    @staticmethod
    def to_turn_envelope(turn) -> AgentTurnEnvelope:
        return AgentTurnEnvelope(
            turn_id=turn.id,
            chat_id=turn.chat_id,
            user_id=turn.user,
            bot_id=turn.bot_id,
            request_id=turn.request_id,
            client_idempotency_key=turn.client_idempotency_key,
            status=turn.status,
            input_text=turn.input_text,
            model_profile=turn.model_profile or {},
            error_code=turn.error_code,
            error_message=turn.error_message,
            answer_artifact_id=turn.answer_artifact_id,
            reference_bundle_artifact_id=turn.reference_bundle_artifact_id,
            timeline_cursor=turn.timeline_cursor or 0,
            started_at=turn.gmt_started,
            finished_at=turn.gmt_finished,
            created_at=turn.gmt_created,
            updated_at=turn.gmt_updated,
        )


class EventService:
    def __init__(self, db_ops: AsyncDatabaseOps | None = None, redis_store: AgentRuntimeRedisStore | None = None):
        self.db_ops = db_ops or async_db_ops
        self.redis_store = redis_store or AgentRuntimeRedisStore()

    async def append_event(
        self,
        *,
        turn_id: str,
        sequence: int,
        event_type: str,
        actor: AgentEventActor,
        label: Optional[str] = None,
        status: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> AgentTimelineEventEnvelope:
        timestamp = utc_now()
        event = await self.db_ops.create_agent_timeline_event(
            turn_id=turn_id,
            sequence=sequence,
            timestamp=timestamp,
            event_type=event_type,
            label=label,
            status=status,
            actor=actor,
            data=data or {},
        )
        envelope = self.to_event_envelope(event)
        await self.redis_store.append_event(envelope)
        await self.redis_store.merge_runtime_state(turn_id, {"timeline_cursor": sequence})
        return envelope

    async def get_events_after(self, turn_id: str, after_sequence: int = 0, limit: int = 500) -> list[AgentTimelineEventEnvelope]:
        cached = await self.redis_store.get_events_after(turn_id, after_sequence=after_sequence, limit=limit)
        if cached:
            return [AgentTimelineEventEnvelope.model_validate(item) for item in cached]
        persisted = await self.db_ops.query_agent_timeline_events(turn_id, after_sequence=after_sequence, limit=limit)
        return [self.to_event_envelope(item) for item in persisted]

    @staticmethod
    def to_event_envelope(event) -> AgentTimelineEventEnvelope:
        actor_value = event.actor.value if hasattr(event.actor, "value") else event.actor
        return AgentTimelineEventEnvelope(
            event_id=event.id,
            turn_id=event.turn_id,
            sequence=event.sequence,
            timestamp=event.timestamp,
            type=event.type,
            label=event.label,
            status=event.status,
            actor=actor_value,
            data=event.data or {},
        )


class ArtifactService:
    def __init__(self, db_ops: AsyncDatabaseOps | None = None):
        self.db_ops = db_ops or async_db_ops

    async def create_artifact(
        self,
        *,
        turn_id: str,
        artifact_type,
        summary: Optional[str],
        payload: dict[str, Any],
        storage_ref: Optional[str] = None,
    ) -> AgentArtifactEnvelope:
        artifact = await self.db_ops.create_agent_artifact(
            turn_id=turn_id,
            artifact_type=artifact_type,
            summary=summary,
            payload=payload,
            storage_ref=storage_ref,
        )
        return self.to_artifact_envelope(artifact)

    async def get_artifact_for_user(self, user: str, artifact_id: str) -> AgentArtifactEnvelope:
        artifact = await self.db_ops.query_agent_artifact(artifact_id)
        if not artifact:
            raise ResourceNotFoundException("Artifact", artifact_id)

        turn = await self._query_turn_for_user(user, artifact.turn_id)
        if not turn:
            raise ResourceNotFoundException("Artifact", artifact_id)
        return self.to_artifact_envelope(artifact)

    async def _query_turn_for_user(self, user: str, turn_id: str):
        async def _query(session):
            from sqlalchemy import select

            from aperag.db.models import AgentTurn

            stmt = select(AgentTurn).where(AgentTurn.id == turn_id, AgentTurn.user == user)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self.db_ops._execute_query(_query)

    @staticmethod
    def to_artifact_envelope(artifact) -> AgentArtifactEnvelope:
        artifact_type = artifact.artifact_type.value if hasattr(artifact.artifact_type, "value") else artifact.artifact_type
        return AgentArtifactEnvelope(
            artifact_id=artifact.id,
            turn_id=artifact.turn_id,
            artifact_type=artifact_type,
            summary=artifact.summary,
            payload=artifact.payload or {},
            storage_ref=artifact.storage_ref,
            created_at=artifact.gmt_created,
            updated_at=artifact.gmt_updated,
        )


class HistoryWriter:
    def __init__(self, db_ops: AsyncDatabaseOps | None = None):
        self.db_ops = db_ops or async_db_ops
        self.history_manager = AgentHistoryManager()

    async def build_history_context(self, user: str, chat_id: str, limit: int = 8) -> str:
        turns = await self.db_ops.query_recent_agent_turns(user, chat_id, limit=limit)
        lines: list[str] = []

        for turn in turns:
            if turn.status != AgentTurnStatus.COMPLETED:
                continue
            artifacts = await self.db_ops.query_agent_artifacts_by_turn(turn.id)
            answer = next(
                (
                    artifact
                    for artifact in artifacts
                    if getattr(artifact.artifact_type, "value", artifact.artifact_type) == "answer"
                ),
                None,
            )
            answer_text = ""
            if answer and isinstance(answer.payload, dict):
                answer_text = answer.payload.get("text") or answer.payload.get("content") or ""
            if not answer_text:
                continue
            lines.append(f"User: {turn.input_text}")
            lines.append(f"Assistant: {answer_text}")

        if lines:
            return "Conversation so far:\n" + "\n".join(lines)

        legacy_history = await query_chat_messages(user, chat_id)
        legacy_lines: list[str] = []
        for turn_parts in legacy_history[-limit:]:
            human_texts = [part.data for part in turn_parts if part.role == "human" and part.type == "message" and part.data]
            ai_texts = [part.data for part in turn_parts if part.role == "ai" and part.type == "message" and part.data]
            if human_texts:
                legacy_lines.append(f"User: {' '.join(human_texts)}")
            if ai_texts:
                legacy_lines.append(f"Assistant: {' '.join(ai_texts)}")

        return "Conversation so far:\n" + "\n".join(legacy_lines) if legacy_lines else ""

    async def commit_completed_turn(
        self,
        *,
        turn,
        request: CreateTurnRequest,
        answer_text: str,
        tool_summaries: list[str],
        references: list[ReferenceBundleItem],
    ) -> bool:
        history = await self.history_manager.get_chat_history(turn.chat_id)
        return await self.history_manager.save_conversation_turn(
            message_id=turn.id,
            trace_id=turn.request_id,
            history=history,
            user_query=request.query,
            ai_response=answer_text,
            files=[file.model_dump(exclude_none=True) for file in request.files or []],
            tool_use_list=[{"data": summary} for summary in tool_summaries],
            tool_references=[item.model_dump(exclude_none=True) for item in references],
        )

    async def commit_failed_turn(
        self,
        *,
        turn,
        request: CreateTurnRequest,
        error_message: str,
        tool_summaries: list[str],
    ) -> bool:
        history = await self.history_manager.get_chat_history(turn.chat_id)
        return await self.history_manager.save_conversation_turn(
            message_id=turn.id,
            trace_id=turn.request_id,
            history=history,
            user_query=request.query,
            ai_response=error_message,
            files=[file.model_dump(exclude_none=True) for file in request.files or []],
            tool_use_list=[{"data": summary} for summary in tool_summaries],
            tool_references=[],
        )
