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
import os
from dataclasses import dataclass
from typing import Any, Optional

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai import messages as pai_messages
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from aperag.agent_runtime.schemas import CreateTurnRequest, ReferenceBundleItem, VisibleAgentState
from aperag.agent_runtime.services import ArtifactService, EventService, HistoryWriter, TurnService, _parse_bot_config
from aperag.db.models import AgentArtifactType, AgentEventActor, AgentTurnStatus
from aperag.db.ops import async_db_ops
from aperag.exceptions import ResourceNotFoundException, ValidationException
from aperag.schema import view_models
from aperag.service.prompt_template_service import build_agent_query_prompt, prompt_template_service

logger = logging.getLogger(__name__)


@dataclass
class ResolvedAgentRequest:
    agent_message: view_models.AgentMessage
    system_prompt: str
    query_prompt_template: str
    provider_base_url: str
    provider_api_key: str
    aperag_api_key: str


class AgentRuntime:
    async def run_turn(self, *, turn, chat, bot, user: str, request: CreateTurnRequest) -> None:
        raise NotImplementedError()

    async def cancel_turn(self, turn_id: str) -> None:
        raise NotImplementedError()


class AgentRuntimeTaskManager:
    def __init__(self):
        self.tasks: dict[str, asyncio.Task] = {}
        self.turn_service = TurnService()
        self.event_service = EventService()
        self.artifact_service = ArtifactService()
        self.history_writer = HistoryWriter()
        self.runtime: AgentRuntime = PydanticAIRuntime(
            self.turn_service, self.event_service, self.artifact_service, self.history_writer
        )

    def launch_turn(self, *, turn, chat, bot, user: str, request: CreateTurnRequest) -> None:
        async def _runner():
            try:
                await self.runtime.run_turn(turn=turn, chat=chat, bot=bot, user=user, request=request)
            finally:
                self.tasks.pop(turn.id, None)

        task = asyncio.create_task(_runner(), name=f"agent-runtime-{turn.id}")
        self.tasks[turn.id] = task

    async def cancel_turn(self, turn_id: str) -> None:
        await self.runtime.cancel_turn(turn_id)
        task = self.tasks.get(turn_id)
        if task and not task.done():
            task.cancel()
            return

        turn = await self.turn_service.db_ops.query_agent_turn_by_id(turn_id)
        if not turn or turn.status in {
            AgentTurnStatus.COMPLETED,
            AgentTurnStatus.FAILED,
            AgentTurnStatus.CANCELLED,
        }:
            return

        sequence = turn.timeline_cursor or 0
        sequence += 1
        await self.event_service.append_event(
            turn_id=turn_id,
            sequence=sequence,
            event_type="turn.cancelled",
            actor=AgentEventActor.SYSTEM,
            label="Cancelled",
            status="cancelled",
            data={},
        )
        await self.turn_service.mark_cancelled(turn_id, sequence=sequence)


class PydanticAIRuntime(AgentRuntime):
    def __init__(
        self,
        turn_service: TurnService,
        event_service: EventService,
        artifact_service: ArtifactService,
        history_writer: HistoryWriter,
    ):
        self.turn_service = turn_service
        self.event_service = event_service
        self.artifact_service = artifact_service
        self.history_writer = history_writer

    async def cancel_turn(self, turn_id: str) -> None:
        await self.turn_service.redis_store.mark_cancelled(turn_id)

    async def run_turn(self, *, turn, chat, bot, user: str, request: CreateTurnRequest) -> None:
        sequence = 0
        text_chunks: list[str] = []
        reference_items: list[ReferenceBundleItem] = []
        tool_summaries: list[str] = []

        async def emit(
            event_type: str,
            *,
            actor: AgentEventActor,
            label: Optional[str] = None,
            status: Optional[str] = None,
            data: Optional[dict[str, Any]] = None,
        ):
            nonlocal sequence
            sequence += 1
            return await self.event_service.append_event(
                turn_id=turn.id,
                sequence=sequence,
                event_type=event_type,
                actor=actor,
                label=label,
                status=status,
                data=data or {},
            )

        try:
            await self.turn_service.redis_store.clear_cancelled(turn.id)
            resolved_request = await self._resolve_request(user=user, chat_id=chat.id, bot=bot, request=request)
            await self.turn_service.mark_running(turn.id)
            await emit("turn.started", actor=AgentEventActor.SYSTEM, status="running", data={"chat_id": chat.id})
            await emit(
                "agent.state.changed",
                actor=AgentEventActor.AGENT,
                label=VisibleAgentState.THINKING.value,
                status="thinking",
            )

            history_context = await self.history_writer.build_history_context(user, chat.id)
            from aperag.service.chat_document_service import chat_document_service

            has_chat_files = await chat_document_service.has_documents_in_chat(chat.id, user)
            prompt = build_agent_query_prompt(
                chat.id,
                agent_message=resolved_request.agent_message,
                user=user,
                template=resolved_request.query_prompt_template,
                has_chat_files=has_chat_files,
            )
            if history_context:
                prompt = f"{history_context}\n\nCurrent turn:\n{prompt}"

            model = OpenAIChatModel(
                resolved_request.agent_message.completion.model,
                provider=OpenAIProvider(
                    base_url=resolved_request.provider_base_url,
                    api_key=resolved_request.provider_api_key,
                ),
            )

            async with MCPServerStreamableHTTP(
                os.getenv("APERAG_MCP_URL", "http://localhost:8000/mcp/"),
                headers={"Authorization": f"Bearer {resolved_request.aperag_api_key}"},
                timeout=30,
                read_timeout=300,
            ) as toolset:
                agent = Agent(
                    model=model,
                    system_prompt=resolved_request.system_prompt,
                    toolsets=[toolset],
                    tool_timeout=120,
                )

                async for event in agent.run_stream_events(prompt):
                    await self._check_cancelled(turn.id)

                    if isinstance(event, pai_messages.FunctionToolCallEvent):
                        tool_name = event.part.tool_name
                        tool_summaries.append(f"Calling tool: {tool_name}")
                        await emit(
                            "agent.state.changed",
                            actor=AgentEventActor.AGENT,
                            label=self._tool_label(tool_name),
                            status="calling_tool",
                            data={"tool_name": tool_name},
                        )
                        await emit(
                            "tool.started",
                            actor=AgentEventActor.TOOL,
                            label=tool_name,
                            status="started",
                            data={"tool_name": tool_name, "args": self._normalize_jsonish(event.part.args)},
                        )
                        if self._is_external_action(tool_name):
                            await emit(
                                "external_action.started",
                                actor=AgentEventActor.TOOL,
                                label=tool_name,
                                status="started",
                                data={"tool_name": tool_name},
                            )
                        continue

                    if isinstance(event, pai_messages.FunctionToolResultEvent):
                        tool_name = event.result.tool_name
                        normalized = self._normalize_jsonish(event.result.content)
                        reference_items.extend(self._extract_reference_items(tool_name, normalized))
                        outcome = getattr(event.result, "outcome", "success")
                        tool_summaries.append(f"Tool {tool_name} finished with status: {outcome}")
                        await emit(
                            "tool.finished",
                            actor=AgentEventActor.TOOL,
                            label=tool_name,
                            status=outcome,
                            data={"tool_name": tool_name, "result": normalized},
                        )
                        if self._is_external_action(tool_name):
                            await emit(
                                "external_action.finished",
                                actor=AgentEventActor.TOOL,
                                label=tool_name,
                                status="finished",
                                data={"tool_name": tool_name},
                            )
                        await emit(
                            "agent.state.changed",
                            actor=AgentEventActor.AGENT,
                            label=VisibleAgentState.READING_RESULT.value,
                            status="reading_result",
                            data={"tool_name": tool_name},
                        )
                        continue

                    if isinstance(event, pai_messages.PartDeltaEvent):
                        if isinstance(event.delta, pai_messages.ThinkingPartDelta):
                            await emit(
                                "agent.state.changed",
                                actor=AgentEventActor.AGENT,
                                label=VisibleAgentState.THINKING.value,
                                status="thinking",
                            )
                            continue

                        if isinstance(event.delta, pai_messages.TextPartDelta):
                            text_chunks.append(event.delta.content_delta)
                            await emit(
                                "agent.state.changed",
                                actor=AgentEventActor.AGENT,
                                label=VisibleAgentState.STREAMING_ANSWER.value,
                                status="composing",
                            )
                            await emit(
                                "text.delta",
                                actor=AgentEventActor.AGENT,
                                label=VisibleAgentState.STREAMING_ANSWER.value,
                                status="streaming",
                                data={"delta": event.delta.content_delta},
                            )
                            continue

                    if isinstance(event, AgentRunResultEvent):
                        final_text = str(event.result.output or "")
                        if final_text and not text_chunks:
                            text_chunks.append(final_text)
                            await emit(
                                "text.delta",
                                actor=AgentEventActor.AGENT,
                                label=VisibleAgentState.STREAMING_ANSWER.value,
                                status="streaming",
                                data={"delta": final_text},
                            )

            answer_text = "".join(text_chunks).strip()
            answer_artifact = await self.artifact_service.create_artifact(
                turn_id=turn.id,
                artifact_type=AgentArtifactType.ANSWER,
                summary=answer_text[:200] if answer_text else "",
                payload={"text": answer_text, "query": request.query},
            )
            reference_artifact = None
            if reference_items:
                reference_artifact = await self.artifact_service.create_artifact(
                    turn_id=turn.id,
                    artifact_type=AgentArtifactType.REFERENCE_BUNDLE,
                    summary=f"{len(reference_items)} references",
                    payload={"items": [item.model_dump() for item in reference_items]},
                )
                await emit(
                    "artifact.created",
                    actor=AgentEventActor.SYSTEM,
                    label="reference_bundle",
                    status="ready",
                    data={
                        "artifact_id": reference_artifact.artifact_id,
                        "artifact_type": reference_artifact.artifact_type,
                    },
                )

            await emit(
                "artifact.created",
                actor=AgentEventActor.SYSTEM,
                label="answer",
                status="ready",
                data={"artifact_id": answer_artifact.artifact_id, "artifact_type": answer_artifact.artifact_type},
            )
            await emit(
                "agent.state.changed",
                actor=AgentEventActor.AGENT,
                label=VisibleAgentState.COMPLETED.value,
                status="done",
            )
            await emit(
                "turn.completed",
                actor=AgentEventActor.SYSTEM,
                label=VisibleAgentState.COMPLETED.value,
                status="completed",
            )
            await self.turn_service.mark_completed(
                turn.id,
                answer_artifact_id=answer_artifact.artifact_id,
                reference_bundle_artifact_id=reference_artifact.artifact_id if reference_artifact else None,
                sequence=sequence,
            )
            await self.history_writer.commit_completed_turn(
                turn=turn,
                request=request,
                answer_text=answer_text,
                tool_summaries=tool_summaries,
                references=reference_items,
            )
        except asyncio.CancelledError:
            await emit(
                "agent.state.changed",
                actor=AgentEventActor.AGENT,
                label=VisibleAgentState.FAILED.value,
                status="error",
                data={"reason": "cancelled"},
            )
            await emit(
                "turn.cancelled",
                actor=AgentEventActor.SYSTEM,
                label="Cancelled",
                status="cancelled",
            )
            await self.turn_service.mark_cancelled(turn.id, sequence=sequence)
            raise
        except Exception as exc:
            logger.exception("Agent Runtime V3 turn failed: %s", turn.id)
            error_artifact = await self.artifact_service.create_artifact(
                turn_id=turn.id,
                artifact_type=AgentArtifactType.ERROR_SUMMARY,
                summary=str(exc),
                payload={"message": str(exc), "type": exc.__class__.__name__},
            )
            await emit(
                "artifact.created",
                actor=AgentEventActor.SYSTEM,
                label="error_summary",
                status="ready",
                data={"artifact_id": error_artifact.artifact_id, "artifact_type": error_artifact.artifact_type},
            )
            await emit(
                "agent.state.changed",
                actor=AgentEventActor.AGENT,
                label=VisibleAgentState.FAILED.value,
                status="error",
            )
            await emit(
                "turn.failed",
                actor=AgentEventActor.SYSTEM,
                label=VisibleAgentState.FAILED.value,
                status="failed",
                data={"error": str(exc), "error_type": exc.__class__.__name__},
            )
            await self.turn_service.mark_failed(
                turn.id,
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                sequence=sequence,
            )
            await self.history_writer.commit_failed_turn(
                turn=turn,
                request=request,
                error_message=str(exc),
                tool_summaries=tool_summaries,
            )

    async def _resolve_request(
        self, *, user: str, chat_id: str, bot, request: CreateTurnRequest
    ) -> ResolvedAgentRequest:
        bot_config = _parse_bot_config(bot)
        final_completion = request.completion or (
            bot_config.agent.completion if bot_config and bot_config.agent else None
        )
        if not final_completion or not final_completion.model or not final_completion.model_service_provider:
            raise ValidationException("Model specification is required for agent runtime v3")

        final_collections = request.collections
        if not final_collections and bot_config and bot_config.agent and bot_config.agent.collections:
            collection_ids = [collection.id for collection in bot_config.agent.collections if collection.id]
            db_collections = await async_db_ops.query_collections_by_ids(user, collection_ids)
            final_collections = [
                view_models.Collection(
                    id=item.id,
                    title=item.title,
                    description=item.description,
                    type=item.type,
                    status=getattr(item, "status", None),
                    created=item.gmt_created,
                    updated=item.gmt_updated,
                )
                for item in db_collections
            ]

        provider = await async_db_ops.query_llm_provider_by_name(final_completion.model_service_provider)
        if not provider:
            raise ResourceNotFoundException("Provider", final_completion.model_service_provider)
        api_key = await async_db_ops.query_provider_api_key(
            final_completion.model_service_provider, user_id=user, need_public=True
        )
        if not api_key:
            raise ValidationException(f"No API key available for provider '{final_completion.model_service_provider}'")

        system_api_keys = await async_db_ops.query_api_keys(user, is_system=True)
        aperag_api_key = next((item.key for item in system_api_keys if item.key), None)
        if not aperag_api_key:
            created = await async_db_ops.create_api_key(user=user, description="aperag", is_system=True)
            aperag_api_key = created.key

        system_prompt = await prompt_template_service.resolve_agent_system_prompt(bot=bot, user_id=user)
        query_prompt_template = await prompt_template_service.resolve_agent_query_prompt(bot=bot, user_id=user)

        return ResolvedAgentRequest(
            agent_message=view_models.AgentMessage(
                query=request.query,
                completion=final_completion,
                collections=final_collections,
                web_search_enabled=request.web_search_enabled,
                language=request.language,
                files=request.files,
            ),
            system_prompt=system_prompt,
            query_prompt_template=query_prompt_template,
            provider_base_url=provider.base_url,
            provider_api_key=api_key,
            aperag_api_key=aperag_api_key,
        )

    async def _check_cancelled(self, turn_id: str) -> None:
        if await self.turn_service.redis_store.is_cancelled(turn_id):
            raise asyncio.CancelledError()

    @staticmethod
    def _tool_label(tool_name: str) -> str:
        return (
            VisibleAgentState.SEARCHING.value
            if PydanticAIRuntime._is_external_action(tool_name)
            else VisibleAgentState.CALLING_TOOL.value
        )

    @staticmethod
    def _is_external_action(tool_name: str) -> bool:
        return tool_name in {"web_search", "web_read"}

    @staticmethod
    def _normalize_jsonish(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        if isinstance(value, tuple):
            return [PydanticAIRuntime._normalize_jsonish(item) for item in value]
        if isinstance(value, list):
            return [PydanticAIRuntime._normalize_jsonish(item) for item in value]
        if isinstance(value, dict):
            return {key: PydanticAIRuntime._normalize_jsonish(item) for key, item in value.items()}
        return value

    @staticmethod
    def _extract_reference_items(tool_name: str, payload: Any) -> list[ReferenceBundleItem]:
        items: list[ReferenceBundleItem] = []
        if not isinstance(payload, dict):
            return items

        result_items = payload.get("items")
        if isinstance(result_items, list):
            for item in result_items:
                if not isinstance(item, dict):
                    continue
                metadata = item.get("metadata") or {}
                items.append(
                    ReferenceBundleItem(
                        source_type=tool_name,
                        source_id=metadata.get("source") or metadata.get("document_id"),
                        title=metadata.get("source") or metadata.get("title"),
                        snippet=item.get("content"),
                        score=item.get("score"),
                        uri=metadata.get("url"),
                        metadata=metadata,
                    )
                )
            return items

        if payload.get("results") and isinstance(payload["results"], list):
            for item in payload["results"]:
                if not isinstance(item, dict):
                    continue
                items.append(
                    ReferenceBundleItem(
                        source_type=tool_name,
                        source_id=item.get("id") or item.get("url"),
                        title=item.get("title"),
                        snippet=item.get("snippet") or item.get("content"),
                        score=item.get("score"),
                        uri=item.get("url"),
                        metadata=item,
                    )
                )
        return items


agent_runtime_manager = AgentRuntimeTaskManager()
