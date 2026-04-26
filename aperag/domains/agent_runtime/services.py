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

from sqlalchemy.exc import IntegrityError

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.domains.agent_runtime.db.models import AgentEventActor, AgentTurnStatus
from aperag.domains.agent_runtime.schemas import (
    AgentTimelineEventEnvelope,
    AgentTurnEnvelope,
    CreateTurnRequest,
    ReferenceBundleItem,
    UserActivityContext,
    UserActivityEnvelope,
    UserActivityIntent,
)
from aperag.domains.agent_runtime.storage import AgentRuntimeRedisStore
from aperag.domains.agent_runtime.uimessage import AgentTurnSnapshot, TextPart
from aperag.domains.agent_runtime.uimessage_store import UIMessageStore
from aperag.domains.conversation.db.models import BotType
from aperag.domains.conversation.schemas import BotConfig
from aperag.exceptions import ChatNotFoundException, ResourceNotFoundException, ValidationException
from aperag.utils.utils import utc_now


def _parse_bot_config(bot) -> Optional[BotConfig]:
    if not bot or not bot.config:
        return None
    try:
        return BotConfig(**json.loads(bot.config))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _coerce_timeline_cursor(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _apply_runtime_state_to_turn(
    turn_envelope: AgentTurnEnvelope, runtime_state: dict[str, Any] | None, latest_sequence: int
) -> AgentTurnEnvelope:
    timeline_cursor = max(_coerce_timeline_cursor(turn_envelope.timeline_cursor), latest_sequence)
    updates: dict[str, Any] = {"timeline_cursor": timeline_cursor}
    if not runtime_state:
        return turn_envelope.model_copy(update=updates)

    updates["timeline_cursor"] = max(timeline_cursor, _coerce_timeline_cursor(runtime_state.get("timeline_cursor")))
    if "status" in runtime_state:
        updates["status"] = runtime_state["status"]
    if runtime_state.get("error_code"):
        updates["error_code"] = runtime_state["error_code"]
    if runtime_state.get("error_message"):
        updates["error_message"] = runtime_state["error_message"]
    return turn_envelope.model_copy(update=updates)


_ACTIVITY_TITLE_KEYS = {
    UserActivityIntent.THINKING: "activity.thinking.title",
    UserActivityIntent.SEARCHING_KNOWLEDGE: "activity.searching_knowledge.title",
    UserActivityIntent.READING_SOURCE: "activity.reading_source.title",
    UserActivityIntent.COMPARING_RESULTS: "activity.comparing_results.title",
    UserActivityIntent.WRITING_ANSWER: "activity.writing_answer.title",
    UserActivityIntent.WAITING: "activity.waiting.title",
    UserActivityIntent.COMPLETED: "activity.completed.title",
    UserActivityIntent.ERROR: "activity.error.title",
}

_ACTIVITY_SUBTITLE_KEYS = {
    UserActivityIntent.THINKING: "activity.thinking.subtitle",
    UserActivityIntent.SEARCHING_KNOWLEDGE: "activity.searching_knowledge.subtitle",
    UserActivityIntent.READING_SOURCE: "activity.reading_source.subtitle",
    UserActivityIntent.COMPARING_RESULTS: "activity.comparing_results.subtitle",
    UserActivityIntent.WRITING_ANSWER: "activity.writing_answer.subtitle",
    UserActivityIntent.WAITING: "activity.waiting.subtitle",
    UserActivityIntent.COMPLETED: "activity.completed.subtitle",
    UserActivityIntent.ERROR: "activity.error.subtitle",
}

_KNOWLEDGE_SEARCH_TOOLS = {"list_collections", "search_collection"}
_WEB_SEARCH_TOOLS = {"search_web", "web_search"}
_READING_TOOLS = {"read_document", "web_read"}


def _normalize_activity_text(value: Any, *, max_length: int = 160) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().split())
    if not normalized:
        return None
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def _iter_activity_payloads(data: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if isinstance(data, dict):
        payloads.append(data)
        for key in ("args", "result"):
            nested = data.get(key)
            if isinstance(nested, dict):
                payloads.append(nested)
    return payloads


def _extract_activity_string(data: Any, *keys: str, max_length: int = 160) -> Optional[str]:
    for payload in _iter_activity_payloads(data):
        for key in keys:
            value = _normalize_activity_text(payload.get(key), max_length=max_length)
            if value:
                return value
    return None


def _extract_activity_count(data: Any) -> Optional[int]:
    for payload in _iter_activity_payloads(data):
        for key in ("count", "total", "total_count", "result_count"):
            value = payload.get(key)
            if isinstance(value, int) and value >= 0:
                return value
        for key in ("items", "results"):
            items = payload.get(key)
            if isinstance(items, list):
                return len(items)
    return None


def _extract_tool_name(label: Optional[str], data: dict[str, Any]) -> Optional[str]:
    return _extract_activity_string(data, "tool_name") or _normalize_activity_text(label)


def _infer_target_type(tool_name: Optional[str]) -> Optional[str]:
    if not tool_name:
        return None
    if tool_name in _KNOWLEDGE_SEARCH_TOOLS:
        return "knowledge_base"
    if tool_name in _READING_TOOLS:
        return "document" if tool_name == "read_document" else "web"
    if tool_name in _WEB_SEARCH_TOOLS:
        return "web"
    return None


def _build_activity_context(data: dict[str, Any], tool_name: Optional[str]) -> Optional[UserActivityContext]:
    values: dict[str, Any] = {}

    keyword = _extract_activity_string(data, "query", "keyword", "keywords", "search_query")
    if keyword:
        values["keyword"] = keyword

    source_name = _extract_activity_string(
        data,
        "source_name",
        "collection_name",
        "collection_title",
        "document_title",
        "title",
        "name",
        max_length=120,
    )
    if source_name:
        values["source_name"] = source_name

    count = _extract_activity_count(data)
    if count is not None:
        values["count"] = count

    target_type = _infer_target_type(tool_name)
    if target_type:
        values["target_type"] = target_type

    scope_label = _extract_activity_string(data, "collection_id", "document_id", "url", max_length=120)
    if scope_label and scope_label != values.get("source_name"):
        values["scope_label"] = scope_label

    if not values:
        return None
    return UserActivityContext.model_validate(values)


def _detail_key_for_activity(intent: UserActivityIntent, context: Optional[UserActivityContext]) -> Optional[str]:
    if not context:
        return None
    if intent == UserActivityIntent.SEARCHING_KNOWLEDGE:
        if context.keyword:
            return "activity.searching_knowledge.detail.keyword"
        if context.source_name:
            return "activity.searching_knowledge.detail.source_name"
        if context.count is not None:
            return "activity.searching_knowledge.detail.count"
    if intent == UserActivityIntent.READING_SOURCE and context.source_name:
        return "activity.reading_source.detail.source_name"
    if intent == UserActivityIntent.COMPARING_RESULTS and context.count is not None:
        return "activity.comparing_results.detail.count"
    return None


def _build_user_activity(
    intent: UserActivityIntent, context: Optional[UserActivityContext] = None
) -> UserActivityEnvelope:
    return UserActivityEnvelope(
        intent=intent,
        title_key=_ACTIVITY_TITLE_KEYS[intent],
        subtitle_key=_ACTIVITY_SUBTITLE_KEYS[intent],
        detail_key=_detail_key_for_activity(intent, context),
        context=context,
    )


def _infer_tool_activity_intent(tool_name: Optional[str]) -> Optional[UserActivityIntent]:
    if not tool_name:
        return None
    if tool_name in _KNOWLEDGE_SEARCH_TOOLS or tool_name in _WEB_SEARCH_TOOLS:
        return UserActivityIntent.SEARCHING_KNOWLEDGE
    if tool_name in _READING_TOOLS:
        return UserActivityIntent.READING_SOURCE
    return None


def _build_user_activity_for_event(
    *,
    technical_type: str,
    label: Optional[str],
    status: Optional[str],
    data: dict[str, Any],
) -> UserActivityEnvelope:
    normalized_status = str(status or "").lower()
    tool_name = _extract_tool_name(label, data)
    context = _build_activity_context(data, tool_name)

    if technical_type == "agent.state.changed":
        if normalized_status == "thinking":
            return _build_user_activity(UserActivityIntent.THINKING)
        if normalized_status == "searching":
            return _build_user_activity(UserActivityIntent.SEARCHING_KNOWLEDGE, context=context)
        if normalized_status == "calling_tool":
            return _build_user_activity(_infer_tool_activity_intent(tool_name) or UserActivityIntent.WAITING, context)
        if normalized_status == "reading_result":
            return _build_user_activity(UserActivityIntent.COMPARING_RESULTS, context=context)
        if normalized_status in {"composing", "streaming"}:
            return _build_user_activity(UserActivityIntent.WRITING_ANSWER)
        if normalized_status == "done":
            return _build_user_activity(UserActivityIntent.COMPLETED)
        if normalized_status in {"error", "failed"}:
            return _build_user_activity(UserActivityIntent.ERROR)
        return _build_user_activity(UserActivityIntent.WAITING)

    if technical_type in {"tool.started", "external_action.started"}:
        return _build_user_activity(_infer_tool_activity_intent(tool_name) or UserActivityIntent.WAITING, context)

    if technical_type in {"tool.finished", "external_action.finished"}:
        if normalized_status in {"failed", "error"}:
            return _build_user_activity(UserActivityIntent.ERROR)
        return _build_user_activity(
            _infer_tool_activity_intent(tool_name) or UserActivityIntent.COMPARING_RESULTS, context
        )

    if technical_type == "text.delta":
        return _build_user_activity(UserActivityIntent.WRITING_ANSWER)

    if technical_type == "turn.started":
        return _build_user_activity(UserActivityIntent.THINKING)
    if technical_type == "turn.completed":
        return _build_user_activity(UserActivityIntent.COMPLETED)
    if technical_type in {"turn.failed", "turn.cancelled"}:
        return _build_user_activity(UserActivityIntent.ERROR)

    return _build_user_activity(UserActivityIntent.WAITING)


class TurnService:
    def __init__(
        self,
        db_ops: AsyncDatabaseOps | None = None,
        redis_store: AgentRuntimeRedisStore | None = None,
        uimessage_store: UIMessageStore | None = None,
    ):
        self.db_ops = db_ops or async_db_ops
        self.redis_store = redis_store or AgentRuntimeRedisStore()
        self.uimessage_store = uimessage_store

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

        try:
            turn = await self.db_ops.create_agent_turn(
                chat_id=chat_id,
                user=user,
                bot_id=bot.id,
                request_id=uuid.uuid4().hex,
                client_idempotency_key=idempotency_key,
                input_text=request.query,
                model_profile=self._build_model_profile(request),
            )
        except IntegrityError:
            existing = await self.db_ops.query_agent_turn_by_idempotency(user, chat_id, idempotency_key)
            if not existing:
                raise
            return chat, bot, existing, False

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

    async def mark_completed(self, turn_id: str, *, sequence: int) -> None:
        turn = await self.db_ops.update_agent_turn(
            turn_id,
            status=AgentTurnStatus.COMPLETED,
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
        """Return the canonical ``AgentTurnSnapshot`` from the at-rest UIMessage store.

        Phase 8 D8.6 (#80) chunk-2 hard-cut removed the legacy
        ``AgentArtifact`` projection fallback. The live runtime now
        writes ``UIMessageStore`` at end-of-turn (D8.2 #74 store +
        D8.6 #80 wire-in), so the snapshot endpoint reads the
        canonical ``UIMessage`` directly. ``error_text`` falls back
        to the ``AgentTurn`` row's ``error_message`` for FAILED /
        CANCELLED turns.
        """

        turn = await self.db_ops.query_agent_turn(user, chat_id, turn_id)
        if not turn:
            raise ResourceNotFoundException("Turn", turn_id)

        runtime_state = await self.redis_store.get_runtime_state(turn_id) or {}
        status_value = runtime_state.get("status") or turn.status
        status_str = status_value.value if hasattr(status_value, "value") else str(status_value)
        timeline_cursor = max(
            _coerce_timeline_cursor(runtime_state.get("timeline_cursor")),
            _coerce_timeline_cursor(turn.timeline_cursor),
        )

        parts: list[Any] = []
        if self.uimessage_store is not None:
            persisted_message = await self.uimessage_store.read(turn_id)
            if persisted_message is not None and persisted_message.parts:
                parts = list(persisted_message.parts)

        error_text: Optional[str] = None
        if status_str in {AgentTurnStatus.FAILED.value, AgentTurnStatus.CANCELLED.value}:
            error_text = runtime_state.get("error_message") or turn.error_message

        return AgentTurnSnapshot(
            turn_id=turn.id,
            chat_id=turn.chat_id,
            runtime_kind="agent_runtime",
            status=status_str,
            parts=parts,
            error_text=error_text,
            timeline_cursor=timeline_cursor,
            started_at=turn.gmt_started,
            finished_at=turn.gmt_finished,
            created_at=turn.gmt_created,
            updated_at=turn.gmt_updated,
            input_text=turn.input_text,
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

    async def get_events_after(
        self, turn_id: str, after_sequence: int = 0, limit: int = 500
    ) -> list[AgentTimelineEventEnvelope]:
        cached = await self.redis_store.get_events_after(turn_id, after_sequence=after_sequence, limit=limit)
        if cached:
            return [self.adapt_event_envelope(AgentTimelineEventEnvelope.model_validate(item)) for item in cached]
        persisted = await self.db_ops.query_agent_timeline_events(turn_id, after_sequence=after_sequence, limit=limit)
        return [self.to_event_envelope(item) for item in persisted]

    @staticmethod
    def to_event_envelope(event) -> AgentTimelineEventEnvelope:
        actor_value = event.actor.value if hasattr(event.actor, "value") else event.actor
        envelope = AgentTimelineEventEnvelope(
            event_id=event.id,
            turn_id=event.turn_id,
            sequence=event.sequence,
            timestamp=event.timestamp,
            type=event.type,
            technical_type=event.type,
            label=event.label,
            status=event.status,
            actor=actor_value,
            data=event.data or {},
        )
        return EventService.adapt_event_envelope(envelope)

    @staticmethod
    def adapt_event_envelope(event: AgentTimelineEventEnvelope) -> AgentTimelineEventEnvelope:
        technical_type = event.technical_type or event.type
        return event.model_copy(
            update={
                "technical_type": technical_type,
                "user_activity": _build_user_activity_for_event(
                    technical_type=technical_type,
                    label=event.label,
                    status=event.status,
                    data=event.data or {},
                ),
            }
        )


class HistoryWriter:
    def __init__(
        self,
        db_ops: AsyncDatabaseOps | None = None,
        uimessage_store: UIMessageStore | None = None,
    ):
        self.db_ops = db_ops or async_db_ops
        self.uimessage_store = uimessage_store

    async def build_history_context(self, user: str, chat_id: str, limit: int = 8) -> str:
        """Compose a plain-text history of recent COMPLETED turns for prompt context.

        Phase 8 D8.6 (#80) chunk-2 hard-cut switched the source from
        the legacy ``AgentArtifact`` rows to the canonical
        ``UIMessage`` parts written by the runtime at end-of-turn.
        Each turn contributes its persisted assistant ``TextPart``
        text (joined when multiple) plus the original ``input_text``.
        """

        if self.uimessage_store is None:
            return ""

        turns = await self.db_ops.query_recent_agent_turns(user, chat_id, limit=limit)
        lines: list[str] = []
        for turn in turns:
            if turn.status != AgentTurnStatus.COMPLETED:
                continue
            message = await self.uimessage_store.read(turn.id)
            if message is None or not message.parts:
                continue
            chunks = [part.text for part in message.parts if isinstance(part, TextPart) and part.text]
            answer_text = "\n".join(chunks).strip()
            if not answer_text:
                continue
            lines.append(f"User: {turn.input_text}")
            lines.append(f"Assistant: {answer_text}")

        return "Conversation so far:\n" + "\n".join(lines) if lines else ""

    async def commit_completed_turn(
        self,
        *,
        turn,
        request: CreateTurnRequest,
        answer_text: str,
        tool_summaries: list[str],
        references: list[ReferenceBundleItem],
    ) -> bool:
        return True

    async def commit_failed_turn(
        self,
        *,
        turn,
        request: CreateTurnRequest,
        error_message: str,
        tool_summaries: list[str],
    ) -> bool:
        return True
