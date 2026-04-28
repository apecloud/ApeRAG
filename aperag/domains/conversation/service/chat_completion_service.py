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

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.domains.agent_runtime.db.models import AgentTurnStatus
from aperag.domains.agent_runtime.runtime import agent_runtime_manager as runtime_manager
from aperag.domains.agent_runtime.schemas import CreateTurnRequest
from aperag.domains.agent_runtime.services import _parse_bot_config
from aperag.domains.conversation.service.chat_service import chat_service_global
from aperag.exceptions import ValidationException
from aperag.schema.common import ModelSpec
from aperag.utils.constant import DOC_QA_REFERENCES

logger = logging.getLogger(__name__)

_TERMINAL_TURN_STATUSES = {
    AgentTurnStatus.COMPLETED,
    AgentTurnStatus.FAILED,
    AgentTurnStatus.CANCELLED,
}
_SUPPORTED_LANGUAGES = {
    "en-US",
    "zh-CN",
    "zh-TW",
    "ja-JP",
    "ko-KR",
    "fr-FR",
    "de-DE",
    "es-ES",
    "it-IT",
    "pt-BR",
    "ru-RU",
}
_DEFAULT_TIMEOUT_SECONDS = 300
_POLL_INTERVAL_SECONDS = 0.25


@dataclass
class APIRequest:
    """API request parameters for direct API calls"""

    user: str
    bot_id: str
    msg_id: str
    stream: bool
    messages: List[Dict[str, str]]


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    timeout: Optional[int] = None


class OpenAIChatCompletionMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class OpenAIChatCompletionChoice(BaseModel):
    index: int
    message: OpenAIChatCompletionMessage
    finish_reason: Optional[str] = None


class OpenAIChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[OpenAIChatCompletionChoice]
    usage: OpenAIChatCompletionUsage


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    code: str


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorDetail


class OpenAIFormatter:
    """Format responses according to OpenAI API specification"""

    @staticmethod
    def format_stream_start(msg_id: str) -> Dict[str, Any]:
        """Format the start event for streaming"""
        return {
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "aperag",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

    @staticmethod
    def format_stream_content(msg_id: str, content: str) -> Dict[str, Any]:
        """Format a content chunk for streaming"""
        return {
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "aperag",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }

    @staticmethod
    def format_stream_end(msg_id: str) -> Dict[str, Any]:
        """Format the end event for streaming"""
        return {
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "aperag",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    @staticmethod
    def format_complete_response(msg_id: str, content: str) -> Dict[str, Any]:
        """Format a complete response for non-streaming mode"""
        return {
            "id": msg_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "aperag",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    @staticmethod
    def format_error(error: str) -> Dict[str, Any]:
        """Format an error response"""
        return {"error": {"message": error, "type": "server_error", "code": "internal_error"}}


class ChatCompletionService:
    """Chat completion service that handles business logic for OpenAI-compatible API"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def stream_openai_sse_response(self, generator: AsyncGenerator[str, None], formatter, msg_id: str):
        """Stream SSE response for OpenAI API format"""
        yield f"data: {json.dumps(formatter.format_stream_start(msg_id))}\n\n"
        async for chunk in generator:
            await asyncio.sleep(0.001)
            yield f"data: {json.dumps(formatter.format_stream_content(msg_id, chunk))}\n\n"
        yield f"data: {json.dumps(formatter.format_stream_end(msg_id))}\n\n"

    async def openai_chat_completions(self, user, body_data, query_params, headers=None):
        """Handle OpenAI-compatible chat completions through Agent Runtime V3."""
        try:
            payload = OpenAIChatCompletionRequest.model_validate(body_data or {})
            bot_id = str(query_params.get("bot_id") or "").strip()
            chat_id = str(query_params.get("chat_id") or "").strip() or None
            language = str(query_params.get("language") or "zh-CN").strip() or "zh-CN"
            if language not in _SUPPORTED_LANGUAGES:
                raise ValidationException(f"Unsupported language '{language}'")
            if not bot_id:
                raise ValidationException("Query parameter 'bot_id' is required")

            chat, bot, ephemeral_chat = await self._resolve_chat_and_bot(user=user, bot_id=bot_id, chat_id=chat_id)
            idempotency_key = None
            if headers is not None:
                idempotency_key = headers.get("Idempotency-Key") or headers.get("X-Idempotency-Key")

            turn_request = self._build_turn_request(
                payload=payload,
                bot=bot,
                language=language,
                idempotency_key=idempotency_key,
            )
            chat, bot, turn, created = await runtime_manager.turn_service.create_or_get_turn(
                user, chat.id, turn_request
            )

            if created or (
                self._status_value(turn.status) in {"QUEUED", "RUNNING"} and turn.id not in runtime_manager.tasks
            ):
                lease_owner = await runtime_manager.claim_turn(turn.id)
                if lease_owner:
                    runtime_manager.launch_turn(
                        turn=turn,
                        chat=chat,
                        bot=bot,
                        user=user,
                        request=turn_request,
                        lease_owner=lease_owner,
                    )

            if payload.stream:
                return self._stream_openai_turn(
                    user=user,
                    bot_id=bot.id,
                    chat_id=chat.id,
                    turn_id=turn.id,
                    ephemeral_chat=ephemeral_chat,
                    timeout_seconds=payload.timeout or _DEFAULT_TIMEOUT_SECONDS,
                ), None

            response = await self._collect_openai_turn_response(
                user=user,
                bot_id=bot.id,
                chat_id=chat.id,
                turn_id=turn.id,
                ephemeral_chat=ephemeral_chat,
                timeout_seconds=payload.timeout or _DEFAULT_TIMEOUT_SECONDS,
            )
            return None, response
        except ValidationException as exc:
            return None, OpenAIFormatter.format_error(str(exc))
        except ValidationError as exc:
            return None, OpenAIFormatter.format_error(f"Invalid chat completion request: {exc.errors()[0]['msg']}")
        except Exception as exc:
            logger.exception("OpenAI-compatible completion request failed")
            return None, OpenAIFormatter.format_error(str(exc))

    async def _resolve_chat_and_bot(self, *, user: str, bot_id: str, chat_id: Optional[str]):
        if chat_id:
            chat, bot = await runtime_manager.turn_service.get_chat_and_bot(user, chat_id)
            if str(bot.id) != bot_id:
                raise ValidationException("The provided chat_id does not belong to the provided bot_id")
            return chat, bot, False

        chat_view = await chat_service_global.create_chat(user, bot_id)
        chat, bot = await runtime_manager.turn_service.get_chat_and_bot(user, chat_view.id)
        return chat, bot, True

    def _build_turn_request(
        self,
        *,
        payload: OpenAIChatCompletionRequest,
        bot,
        language: str,
        idempotency_key: Optional[str],
    ) -> CreateTurnRequest:
        query = self._extract_latest_user_message(payload.messages)
        bot_config = _parse_bot_config(bot)
        default_completion = bot_config.agent.completion if bot_config and bot_config.agent else None
        override_requested = any(
            (
                payload.model not in (None, "", "aperag"),
                payload.temperature is not None,
                payload.max_tokens is not None,
                payload.max_completion_tokens is not None,
                payload.timeout is not None,
            )
        )

        completion = None
        if override_requested:
            completion = ModelSpec(
                model_id=payload.model
                if payload.model not in (None, "", "aperag")
                else getattr(default_completion, "model_id", None),
                temperature=payload.temperature
                if payload.temperature is not None
                else getattr(default_completion, "temperature", None),
                max_tokens=payload.max_tokens
                if payload.max_tokens is not None
                else getattr(default_completion, "max_tokens", None),
                max_completion_tokens=payload.max_completion_tokens
                if payload.max_completion_tokens is not None
                else getattr(default_completion, "max_completion_tokens", None),
                timeout=payload.timeout
                if payload.timeout is not None
                else getattr(default_completion, "timeout", None),
            )
            if not completion.model_id:
                raise ValidationException("Custom model overrides require a model id")

        return CreateTurnRequest(
            query=query,
            completion=completion,
            language=language,
            client_idempotency_key=idempotency_key,
        )

    @staticmethod
    def _extract_latest_user_message(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            text = ChatCompletionService._coerce_message_content_to_text(message.get("content"))
            if text:
                return text
        raise ValidationException("At least one user message with textual content is required")

    @staticmethod
    def _coerce_message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                text = part.get("text") or part.get("content")
                if part_type in (None, "text", "input_text") and isinstance(text, str):
                    text_parts.append(text)
            return "\n".join(item.strip() for item in text_parts if isinstance(item, str) and item.strip()).strip()
        return ""

    async def _collect_openai_turn_response(
        self,
        *,
        user: str,
        bot_id: str,
        chat_id: str,
        turn_id: str,
        ephemeral_chat: bool,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        try:
            turn = await self._wait_for_terminal_turn(
                user=user, chat_id=chat_id, turn_id=turn_id, timeout_seconds=timeout_seconds
            )
            if self._status_value(turn.status) != AgentTurnStatus.COMPLETED.value:
                return OpenAIFormatter.format_error(turn.error_message or "Chat completion failed")

            # Phase 8 D8.6 (#80) chunk-2 hard-cut: pull the canonical
            # UIMessage parts persisted by the runtime at end-of-turn
            # and project them into the OpenAI-compat completion
            # content shape (answer text + references payload).
            persisted = await runtime_manager.uimessage_store.read(turn_id)
            parts = list(persisted.parts) if persisted and persisted.parts else []
            content = self._build_completion_content(parts)
            return OpenAIFormatter.format_complete_response(turn_id, content)
        finally:
            if ephemeral_chat:
                await chat_service_global.delete_chat(user, bot_id, chat_id)

    async def _stream_openai_turn(
        self,
        *,
        user: str,
        bot_id: str,
        chat_id: str,
        turn_id: str,
        ephemeral_chat: bool,
        timeout_seconds: int,
    ) -> AsyncGenerator[str, None]:
        current_after = 0
        deadline = time.monotonic() + timeout_seconds
        try:
            yield f"data: {json.dumps(OpenAIFormatter.format_stream_start(turn_id))}\n\n"
            while True:
                events = await runtime_manager.event_service.get_events_after(
                    turn_id, after_sequence=current_after, limit=500
                )
                for event in events:
                    current_after = event.sequence
                    if event.type != "text.delta":
                        continue
                    delta = event.data.get("delta")
                    if isinstance(delta, str) and delta:
                        yield f"data: {json.dumps(OpenAIFormatter.format_stream_content(turn_id, delta))}\n\n"

                turn = await runtime_manager.turn_service.db_ops.query_agent_turn(user, chat_id, turn_id)
                if turn and self._status_value(turn.status) in {
                    AgentTurnStatus.COMPLETED.value,
                    AgentTurnStatus.FAILED.value,
                    AgentTurnStatus.CANCELLED.value,
                }:
                    if self._status_value(turn.status) == AgentTurnStatus.COMPLETED.value:
                        yield f"data: {json.dumps(OpenAIFormatter.format_stream_end(turn_id))}\n\n"
                        return
                    error_message = turn.error_message or "Chat completion failed"
                    yield f"data: {json.dumps(OpenAIFormatter.format_error(error_message))}\n\n"
                    return

                if time.monotonic() >= deadline:
                    await runtime_manager.cancel_turn(turn_id)
                    yield f"data: {json.dumps(OpenAIFormatter.format_error('Chat completion timed out'))}\n\n"
                    return

                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        finally:
            if ephemeral_chat:
                await chat_service_global.delete_chat(user, bot_id, chat_id)

    async def _wait_for_terminal_turn(self, *, user: str, chat_id: str, turn_id: str, timeout_seconds: int):
        deadline = time.monotonic() + timeout_seconds
        while True:
            turn = await runtime_manager.turn_service.db_ops.query_agent_turn(user, chat_id, turn_id)
            if turn and self._status_value(turn.status) in {status.value for status in _TERMINAL_TURN_STATUSES}:
                return turn
            if time.monotonic() >= deadline:
                await runtime_manager.cancel_turn(turn_id)
                raise ValidationException("Chat completion timed out")
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    @staticmethod
    def _status_value(status: Any) -> str:
        return status.value if hasattr(status, "value") else str(status)

    @staticmethod
    def _build_completion_content(parts) -> str:
        """Build OpenAI-compat completion content from canonical UIMessage parts.

        Phase 8 D8.6 (#80) chunk-2 switched the source from legacy
        ``AgentArtifact`` rows to the at-rest ``UIMessage`` parts the
        runtime now writes at end-of-turn. The answer text comes from
        joined ``TextPart`` content; the references payload is
        derived from ``DataCitationPart`` entries (one dict per
        citation, preserving ``url`` / ``title`` / ``cited_text`` so
        the OpenAI-compat envelope keeps the existing answer +
        references shape).
        """
        from aperag.domains.agent_runtime.uimessage import (
            DataCitationPart,
            TextPart,
            UrlCitationLocation,
        )

        text_chunks: list[str] = []
        references_payload: list[dict[str, Any]] = []

        for part in parts:
            if isinstance(part, TextPart) and part.text:
                text_chunks.append(part.text)
                continue
            if isinstance(part, DataCitationPart):
                location = part.data.location if part.data else None
                url = location.url if isinstance(location, UrlCitationLocation) else None
                title = location.title if isinstance(location, UrlCitationLocation) else None
                cited_text = part.data.cited_text if part.data else ""
                references_payload.append(
                    {
                        "uri": url or "",
                        "title": title,
                        "snippet": cited_text,
                    }
                )

        answer_text = "".join(text_chunks)
        if references_payload:
            return f"{answer_text}{DOC_QA_REFERENCES}{json.dumps(references_payload, ensure_ascii=False)}"
        return answer_text


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
chat_completion_service = ChatCompletionService()
