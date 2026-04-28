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

# Wave 3 hard-cut moved this trivial token generator inline so the
# legacy ``aperag.tasks.processing_lease`` module can be deleted
# without leaving an external dependency on the agent-runtime path
# (per architect msg=3890c9d7 Item 1 ruling: "如实际用到，提取小
# helper 到 agent_runtime 自己 module").
import uuid as _uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai import messages as pai_messages
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets.filtered import FilteredToolset

from aperag.config import get_async_session
from aperag.db.ops import async_db_ops
from aperag.domains.agent_runtime.db.models import AgentEventActor, AgentTurnStatus
from aperag.domains.agent_runtime.ports import PromptTemplateOps
from aperag.domains.agent_runtime.schemas import (
    AgentMessage,
    CreateTurnRequest,
    ReferenceBundleItem,
    VisibleAgentState,
)
from aperag.domains.agent_runtime.services import (
    EventService,
    HistoryWriter,
    TurnService,
    _parse_bot_config,
)
from aperag.domains.agent_runtime.storage import DEFAULT_AGENT_TURN_LEASE_TTL_SECONDS, AgentRuntimeRedisStore
from aperag.domains.agent_runtime.tools.safe_name import sanitize_tool_name
from aperag.domains.agent_runtime.uimessage import (
    MAX_REASONING_PART_CHARS,
    CitationData,
    DataCitationPart,
    ReasoningPart,
    SourceUrlPart,
    TextPart,
    ToolPart,
    UIMessage,
    UIMessagePart,
    UrlCitationLocation,
)
from aperag.domains.agent_runtime.uimessage_store import UIMessageDbOps, UIMessageStore
from aperag.domains.knowledge_base.schemas import Collection as KBCollectionSchema
from aperag.domains.model_platform.schemas import ModelCapability
from aperag.exceptions import ResourceNotFoundException, ValidationException
from aperag.llm.runtime.resolver import resolve_model_invocation_from_records


def generate_processing_token() -> str:
    """Return a fresh hex token used to claim a turn's processing
    lease. The token is opaque to callers; the only contract is
    uniqueness."""
    return _uuid.uuid4().hex


# ``prompt_template_service`` is reached via a ``PromptTemplateOps`` DI
# slot rather than a direct import — it still lives in
# ``aperag/service/prompt_template_service.py`` (legacy provider, Phase
# 6 cleanup candidate), so a direct ``from aperag.service.*`` would
# trip G1. ``aperag/app.py`` wires the concrete legacy singleton in
# at startup. The narrow surface matches the two methods the runtime
# actually calls — ``resolve_agent_system_prompt`` +
# ``resolve_agent_query_prompt`` — plus the standalone
# ``build_agent_query_prompt`` helper, which is bound via the same
# slot for symmetry.
_prompt_template_ops: Optional[PromptTemplateOps] = None


def set_prompt_template_ops(ops: PromptTemplateOps) -> None:
    global _prompt_template_ops
    _prompt_template_ops = ops


def _get_prompt_template_ops() -> PromptTemplateOps:
    if _prompt_template_ops is None:
        raise RuntimeError(
            "PromptTemplateOps not wired — call "
            "aperag.domains.agent_runtime.runtime.set_prompt_template_ops() "
            "in aperag/app.py startup before serving requests."
        )
    return _prompt_template_ops


logger = logging.getLogger(__name__)

DEFAULT_AGENT_TURN_LEASE_RENEW_INTERVAL_SECONDS = int(os.getenv("APERAG_AGENT_TURN_LEASE_RENEW_INTERVAL_SECONDS", "30"))


# Wave 10 §K.13: hardcoded read-only tool subset for ``BotType.SUMMARY``.
# Mapping by bot type lives in the runtime layer (per design doc) so we
# do not introduce a ``Bot.tool_subset`` column. New bot types that need
# narrowed tools register here; everything else gets the full toolset.
_BOT_TYPE_ALLOWED_TOOLS: dict[str, frozenset[str]] = {
    "summary": frozenset(
        {
            "list_collections",
            "vector_search",
            "fulltext_search",
            "graph_search",
            "query_graph_entities",
            "expand_graph_subgraph",
            "get_entity_detail",
            "read_document",
            "read_document_section",
            "read_document_outline",
            "read_document_chunk",
            "get_collection_metadata",
            "get_document_metadata",
        }
    ),
}


def _allowed_tool_names_for_bot(bot) -> frozenset[str] | None:
    """Return the allowed tool name set for ``bot``, or ``None`` for the
    full toolset (no filtering)."""
    bot_type = getattr(bot, "type", None)
    if bot_type is None:
        return None
    type_value = getattr(bot_type, "value", bot_type)
    return _BOT_TYPE_ALLOWED_TOOLS.get(str(type_value))


def _wrap_toolset_for_bot(toolset, bot):
    """Wrap ``toolset`` in a :class:`FilteredToolset` if ``bot.type`` has
    a hardcoded subset; otherwise return ``toolset`` unchanged."""
    allowed = _allowed_tool_names_for_bot(bot)
    if not allowed:
        return toolset

    def _filter(_ctx, tool_def) -> bool:
        return tool_def.name in allowed

    return FilteredToolset(toolset, _filter)


@dataclass
class _PersistedToolCall:
    """Collapsed tool lifecycle for durable UIMessage reload."""

    tool_call_id: str
    tool_name: str
    state: Literal["input-available", "output-available", "output-error"] = "input-available"
    output: Any = None
    error_text: Optional[str] = None
    # User-facing one-line summary extracted from the tool's input args
    # (e.g. "搜索：张飞牛肉" / "阅读：example.com/page"). Persisted as
    # ``ToolPart.summary`` so the FE can render the activity subtitle
    # post-refresh, when the raw ``input`` is intentionally NOT
    # persisted (D9 §A7). Bounded to ``_TOOL_SUMMARY_MAX_LEN`` chars.
    summary: Optional[str] = None


@dataclass
class _PersistedReasoning:
    """One reasoning chunk (the model's thinking text emitted between
    tool calls). Lives on the chronological timeline alongside
    :class:`_PersistedToolCall` entries so the FE can render
    Claude/Cursor-style "思考1 → 工具a → 思考2 → 工具b" interleaving.

    The runtime accumulates ``ThinkingPartDelta.content_delta`` into
    a buffer; ``_close_reasoning_chunk`` flushes the buffer into a
    ``_PersistedReasoning`` entry and appends it to the timeline:
    when a tool call interrupts the reasoning stream, OR when the
    buffer reaches ``MAX_REASONING_PART_CHARS``, OR when the turn
    completes.
    """

    text: str


# Maximum length of the user-facing tool-call summary written to
# ``ToolPart.summary``. Keeps ``agent_message.parts`` size bounded (per
# architect msg=2639aeea size-budget concern); long URLs / queries
# truncate with an ellipsis at extraction.
_TOOL_SUMMARY_MAX_LEN: int = 200


def _truncate_summary(value: str) -> str:
    if len(value) <= _TOOL_SUMMARY_MAX_LEN:
        return value
    keep = max(_TOOL_SUMMARY_MAX_LEN - 1, 1)
    return value[:keep] + "…"


def _compact_url_label(value: str) -> str:
    """Mirror the FE ``compactUrlLabel`` (agent-turn-renderer.tsx) so
    the persisted summary reads identical to live SSE rendering — same
    user-visible string in both paths."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            return value
        path = "" if parsed.path in ("", "/") else parsed.path
        label = f"{parsed.netloc}{path}".rstrip("/")
        return label or value
    except Exception:
        return value


def _extract_tool_summary(args: Any) -> Optional[str]:
    """Project the raw tool input into a small user-facing summary.

    Recognises common tool-input shapes and returns ``"搜索：…"`` /
    ``"阅读：…"`` style copy. Returns ``None`` when the args don't fit
    a recognised shape — the FE falls back to its generic per-tool
    label in that case.

    Bounded to ``_TOOL_SUMMARY_MAX_LEN`` chars so persisted rows stay
    small. Raw args are NEVER persisted in full (D9 §A7) — this
    function intentionally reads only the keys needed for the visible
    summary.
    """
    if not isinstance(args, dict):
        return None
    for key in ("query", "q", "keyword", "keywords"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate_summary(f"搜索:{value.strip()}")
    for key in ("url", "uri", "link"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate_summary(f"阅读:{_compact_url_label(value.strip())}")
    return None


def _tool_part_type(tool_name: str) -> str:
    safe = sanitize_tool_name(tool_name or "tool") or "tool"
    return f"tool-{safe}"


def _is_tool_failure_status(status: Any) -> bool:
    raw = getattr(status, "value", status)
    return str(raw or "").lower() in {"error", "failed", "failure"}


def _compose_assistant_parts(
    *,
    turn_id: str,
    answer_text: str,
    references: list[ReferenceBundleItem],
    tool_calls: list[_PersistedToolCall] | None = None,
    timeline: list[Any] | None = None,
) -> list[UIMessagePart]:
    """Project the runtime's accumulated answer + references into a
    canonical at-rest ``UIMessagePart`` list for ``UIMessageStore.write``.

    Order:
    - ``timeline`` entries first, in their original chronological order
      — interleaved ``ReasoningPart`` / ``ToolPart`` records so FE
      renders Claude/Cursor-style "思考1 → 工具a → 思考2 → 工具b" flow
      (Wave 9 task #2 followup #2, architect ratify msg=2639aeea).
    - ``TextPart`` (final answer) next.
    - ``SourceUrlPart`` / ``DataCitationPart`` pairs from each reference.

    ``tool_calls`` is the legacy parameter (Wave 9 PR #1798); when
    ``timeline`` is supplied it takes precedence and ``tool_calls`` is
    ignored. Callers that don't track reasoning (older / test paths)
    can keep using ``tool_calls`` for backward compat.
    """

    parts: list[UIMessagePart] = []
    if timeline is not None:
        for entry in timeline:
            if isinstance(entry, _PersistedReasoning):
                if entry.text and entry.text.strip():
                    parts.append(ReasoningPart(text=entry.text))
            elif isinstance(entry, _PersistedToolCall):
                parts.append(
                    ToolPart(
                        type=_tool_part_type(entry.tool_name),
                        tool_call_id=entry.tool_call_id,
                        state=entry.state,
                        output=entry.output,
                        error_text=entry.error_text,
                        summary=entry.summary,
                        metadata={"mcpToolName": entry.tool_name},
                    )
                )
    else:
        for call in tool_calls or []:
            parts.append(
                ToolPart(
                    type=_tool_part_type(call.tool_name),
                    tool_call_id=call.tool_call_id,
                    state=call.state,
                    output=call.output,
                    error_text=call.error_text,
                    summary=call.summary,
                    metadata={"mcpToolName": call.tool_name},
                )
            )
    if answer_text:
        parts.append(TextPart(text=answer_text))
    for index, ref in enumerate(references):
        metadata = ref.metadata if isinstance(ref.metadata, dict) else {}
        url = ref.uri or metadata.get("url")
        title = ref.title or metadata.get("title")
        snippet = ref.snippet or ""
        source_id = ref.source_id or f"{turn_id}-ref-{index}"
        if url:
            parts.append(SourceUrlPart(source_id=str(source_id), url=str(url), title=title))
        parts.append(
            DataCitationPart(
                data=CitationData(
                    cited_text=str(snippet),
                    location=UrlCitationLocation(url=str(url) if url else "", title=title),
                )
            )
        )
    return parts


@dataclass
class ResolvedAgentRequest:
    agent_message: AgentMessage
    system_prompt: str
    query_prompt_template: str
    provider_model_id: str
    provider_base_url: str
    provider_api_key: str
    aperag_api_key: str


class TurnLeaseLostError(RuntimeError):
    pass


class TurnLeaseGuard:
    def __init__(
        self,
        *,
        turn_service: TurnService,
        turn_id: str,
        owner_token: str,
        ttl_seconds: int = DEFAULT_AGENT_TURN_LEASE_TTL_SECONDS,
        renew_interval_seconds: int = DEFAULT_AGENT_TURN_LEASE_RENEW_INTERVAL_SECONDS,
    ):
        self.turn_service = turn_service
        self.turn_id = turn_id
        self.owner_token = owner_token
        self.ttl_seconds = max(ttl_seconds, 1)
        self.renew_interval_seconds = max(renew_interval_seconds, 1)
        self._stop_event = asyncio.Event()
        self._ownership_lost = asyncio.Event()
        self._renew_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._renew_task is not None:
            return
        self._renew_task = asyncio.create_task(
            self._renew_loop(),
            name=f"agent-runtime-lease-renew-{self.turn_id}",
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._renew_task is not None:
            self._renew_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._renew_task
        try:
            await self.turn_service.redis_store.release_turn_claim(self.turn_id, self.owner_token)
        except Exception:
            logger.exception("Failed to release turn lease for %s", self.turn_id)

    @property
    def ownership_lost(self) -> bool:
        return self._ownership_lost.is_set()

    def ensure_owned(self) -> None:
        if self.ownership_lost:
            raise TurnLeaseLostError(f"Turn lease ownership lost for {self.turn_id}")

    async def _renew_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.renew_interval_seconds)
                return
            except asyncio.TimeoutError:
                pass

            try:
                renewed = await self.turn_service.redis_store.renew_turn_claim(
                    self.turn_id,
                    self.owner_token,
                    ttl_seconds=self.ttl_seconds,
                )
            except Exception:
                logger.exception("Failed to renew turn lease for %s", self.turn_id)
                continue

            if renewed:
                continue

            self._ownership_lost.set()
            logger.warning("Turn lease ownership lost for %s", self.turn_id)
            self._stop_event.set()
            return


class AgentRuntime:
    async def run_turn(self, *, turn, chat, bot, user: str, request: CreateTurnRequest, lease_owner: str) -> None:
        raise NotImplementedError()

    async def cancel_turn(self, turn_id: str) -> None:
        raise NotImplementedError()


class AgentRuntimeTaskManager:
    def __init__(self):
        self.tasks: dict[str, asyncio.Task] = {}
        self.uimessage_store = UIMessageStore(
            db_ops=UIMessageDbOps(session_factory=get_async_session),
            redis_store=AgentRuntimeRedisStore(),
        )
        self.turn_service = TurnService(uimessage_store=self.uimessage_store)
        self.event_service = EventService()
        self.history_writer = HistoryWriter(uimessage_store=self.uimessage_store)
        self.runtime: AgentRuntime = PydanticAIRuntime(
            self.turn_service, self.event_service, self.history_writer, self.uimessage_store
        )

    async def claim_turn(self, turn_id: str) -> str | None:
        owner_token = generate_processing_token()
        claimed = await self.turn_service.redis_store.try_claim_turn(turn_id, owner_token)
        return owner_token if claimed else None

    def launch_turn(self, *, turn, chat, bot, user: str, request: CreateTurnRequest, lease_owner: str) -> None:
        async def _runner():
            try:
                await self.runtime.run_turn(
                    turn=turn,
                    chat=chat,
                    bot=bot,
                    user=user,
                    request=request,
                    lease_owner=lease_owner,
                )
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
        history_writer: HistoryWriter,
        uimessage_store: UIMessageStore,
    ):
        self.turn_service = turn_service
        self.event_service = event_service
        self.history_writer = history_writer
        self.uimessage_store = uimessage_store

    async def cancel_turn(self, turn_id: str) -> None:
        await self.turn_service.redis_store.mark_cancelled(turn_id)

    async def run_turn(self, *, turn, chat, bot, user: str, request: CreateTurnRequest, lease_owner: str) -> None:
        sequence = 0
        text_chunks: list[str] = []
        reference_items: list[ReferenceBundleItem] = []
        tool_summaries: list[str] = []
        persisted_tool_calls: list[_PersistedToolCall] = []
        persisted_tool_index: dict[str, _PersistedToolCall] = {}
        # Chronological timeline of reasoning chunks + tool calls.
        # ``_compose_assistant_parts`` reads this in order so the FE
        # gets Claude/Cursor-style "思考1 → 工具a → 思考2 → 工具b"
        # interleaving (Wave 9 task #2 followup #2).
        persisted_timeline: list[Any] = []
        # Reasoning text accumulator: ThinkingPartDelta deltas append
        # here; a flush (tool-call interrupt / size cap / turn end)
        # converts the buffer into a ``_PersistedReasoning`` entry on
        # ``persisted_timeline`` and resets the buffer.
        reasoning_buffer: list[str] = []

        def _flush_reasoning_chunk() -> None:
            """Push the accumulated reasoning text onto the timeline as
            a ``_PersistedReasoning`` entry, then reset the buffer.

            Called on tool-call interrupt, size cap, and turn end —
            the chunk boundaries are what give the FE its discrete
            "思考N" blocks rather than one ever-growing reasoning
            block at the bottom."""
            text = "".join(reasoning_buffer).strip()
            reasoning_buffer.clear()
            if text:
                persisted_timeline.append(_PersistedReasoning(text=text))

        lease_guard = TurnLeaseGuard(turn_service=self.turn_service, turn_id=turn.id, owner_token=lease_owner)

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
            await lease_guard.start()
            await self.turn_service.redis_store.clear_cancelled(turn.id)
            lease_guard.ensure_owned()
            resolved_request = await self._resolve_request(user=user, chat_id=chat.id, bot=bot, request=request)
            lease_guard.ensure_owned()
            await self.turn_service.mark_running(turn.id)
            await emit("turn.started", actor=AgentEventActor.SYSTEM, status="running", data={"chat_id": chat.id})
            await emit(
                "agent.state.changed",
                actor=AgentEventActor.AGENT,
                label=VisibleAgentState.THINKING.value,
                status="thinking",
            )

            history_context = await self.history_writer.build_history_context(user, chat.id)
            from aperag.domains.conversation.service.chat_document_service import chat_document_service

            has_chat_files = await chat_document_service.has_documents_in_chat(chat.id, user)
            prompt = _get_prompt_template_ops().build_agent_query_prompt(
                chat.id,
                agent_message=resolved_request.agent_message,
                user=user,
                template=resolved_request.query_prompt_template,
                has_chat_files=has_chat_files,
            )
            if history_context:
                prompt = f"{history_context}\n\nCurrent turn:\n{prompt}"

            model = OpenAIChatModel(
                resolved_request.provider_model_id,
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
                # Wave 10 §K.13: hardcoded read-only tool subset for
                # ``BotType.SUMMARY`` — bots created by the
                # collection-summary regen pipeline must not call
                # write/mutating tools. The filter is applied at the
                # toolset layer (not the agent prompt) so the LLM
                # cannot route around it via tool-name confusion.
                effective_toolset = _wrap_toolset_for_bot(toolset, bot)
                agent = Agent(
                    model=model,
                    system_prompt=resolved_request.system_prompt,
                    toolsets=[effective_toolset],
                    tool_timeout=120,
                )

                async for event in agent.run_stream_events(prompt):
                    await self._check_cancelled(turn.id, lease_guard=lease_guard)

                    if isinstance(event, pai_messages.FunctionToolCallEvent):
                        tool_name = event.part.tool_name
                        tool_call_id = event.part.tool_call_id
                        # Tool call interrupts the reasoning stream —
                        # close the current chunk so the FE renders
                        # this thinking block before the upcoming tool
                        # action card.
                        _flush_reasoning_chunk()
                        tool_call = _PersistedToolCall(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            state="input-available",
                            summary=_extract_tool_summary(self._normalize_jsonish(event.part.args)),
                        )
                        persisted_tool_calls.append(tool_call)
                        persisted_tool_index[tool_call_id] = tool_call
                        persisted_timeline.append(tool_call)
                        tool_summaries.append(f"Calling tool: {tool_name}")
                        await emit(
                            "agent.state.changed",
                            actor=AgentEventActor.AGENT,
                            label=self._tool_label(tool_name),
                            status="calling_tool",
                            data={"tool_name": tool_name, "tool_call_id": tool_call_id},
                        )
                        await emit(
                            "tool.started",
                            actor=AgentEventActor.TOOL,
                            label=tool_name,
                            status="started",
                            data={
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                "args": self._normalize_jsonish(event.part.args),
                            },
                        )
                        if self._is_external_action(tool_name):
                            await emit(
                                "external_action.started",
                                actor=AgentEventActor.TOOL,
                                label=tool_name,
                                status="started",
                                data={"tool_name": tool_name, "tool_call_id": tool_call_id},
                            )
                        continue

                    if isinstance(event, pai_messages.FunctionToolResultEvent):
                        tool_name = event.result.tool_name
                        tool_call_id = event.result.tool_call_id
                        normalized = self._normalize_jsonish(event.result.content)
                        reference_items.extend(self._extract_reference_items(tool_name, normalized))
                        outcome = getattr(event.result, "outcome", "success")
                        tool_call = persisted_tool_index.get(tool_call_id)
                        if tool_call is None:
                            tool_call = _PersistedToolCall(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                            )
                            persisted_tool_calls.append(tool_call)
                            persisted_tool_index[tool_call_id] = tool_call
                            persisted_timeline.append(tool_call)
                        if _is_tool_failure_status(outcome):
                            tool_call.state = "output-error"
                            tool_call.error_text = (
                                normalized
                                if isinstance(normalized, str)
                                else json.dumps(normalized, ensure_ascii=False, default=str)
                            )
                        else:
                            tool_call.state = "output-available"
                        tool_summaries.append(f"Tool {tool_name} finished with status: {outcome}")
                        await emit(
                            "tool.finished",
                            actor=AgentEventActor.TOOL,
                            label=tool_name,
                            status=outcome,
                            data={"tool_name": tool_name, "tool_call_id": tool_call_id, "result": normalized},
                        )
                        if self._is_external_action(tool_name):
                            await emit(
                                "external_action.finished",
                                actor=AgentEventActor.TOOL,
                                label=tool_name,
                                status="finished",
                                data={"tool_name": tool_name, "tool_call_id": tool_call_id},
                            )
                        await emit(
                            "agent.state.changed",
                            actor=AgentEventActor.AGENT,
                            label=VisibleAgentState.READING_RESULT.value,
                            status="reading_result",
                            data={"tool_name": tool_name, "tool_call_id": tool_call_id},
                        )
                        continue

                    if isinstance(event, pai_messages.PartDeltaEvent):
                        if isinstance(event.delta, pai_messages.ThinkingPartDelta):
                            delta_text = event.delta.content_delta or ""
                            if delta_text:
                                # Capture the actual reasoning text so
                                # we can persist it as a ReasoningPart
                                # (not just emit a status badge).
                                reasoning_buffer.append(delta_text)
                                # Auto-flush when the chunk fills the
                                # per-part budget — keeps any single
                                # ReasoningPart small enough for
                                # incremental FE rendering.
                                if sum(len(s) for s in reasoning_buffer) >= MAX_REASONING_PART_CHARS:
                                    _flush_reasoning_chunk()
                                # Live SSE: emit a reasoning delta so
                                # the FE renderer can stream the
                                # current reasoning chunk text — same
                                # convention as ``text.delta``.
                                await emit(
                                    "reasoning.delta",
                                    actor=AgentEventActor.AGENT,
                                    label=VisibleAgentState.THINKING.value,
                                    status="thinking",
                                    data={"delta": delta_text},
                                )
                            else:
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

            lease_guard.ensure_owned()
            answer_text = "".join(text_chunks).strip()

            # Final flush — the model may have emitted reasoning text
            # after its last tool call (the close-out thinking that
            # leads into the answer). Without this, that trailing
            # block would be silently dropped from the persisted
            # timeline.
            _flush_reasoning_chunk()

            await self.uimessage_store.write(
                turn_id=turn.id,
                chat_id=chat.id,
                message=UIMessage(
                    id=f"msg-{turn.id}",
                    role="assistant",
                    parts=_compose_assistant_parts(
                        turn_id=turn.id,
                        answer_text=answer_text,
                        references=reference_items,
                        timeline=persisted_timeline,
                    ),
                ),
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
            await self.turn_service.mark_completed(turn.id, sequence=sequence)
            await self.history_writer.commit_completed_turn(
                turn=turn,
                request=request,
                answer_text=answer_text,
                tool_summaries=tool_summaries,
                references=reference_items,
            )
        except TurnLeaseLostError:
            logger.warning("Agent Runtime V3 lease lost for turn %s; exiting duplicate runner", turn.id)
            return
        except asyncio.CancelledError:
            if lease_guard.ownership_lost:
                logger.warning("Agent Runtime V3 cancelled after lease loss for turn %s", turn.id)
                return
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
            if lease_guard.ownership_lost:
                logger.warning(
                    "Agent Runtime V3 suppressing failure for turn %s after lease loss: %s",
                    turn.id,
                    exc,
                )
                return
            logger.exception("Agent Runtime V3 turn failed: %s", turn.id)
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
        finally:
            await lease_guard.stop()

    async def _resolve_request(
        self, *, user: str, chat_id: str, bot, request: CreateTurnRequest
    ) -> ResolvedAgentRequest:
        bot_config = _parse_bot_config(bot)
        final_completion = request.completion or (
            bot_config.agent.completion if bot_config and bot_config.agent else None
        )
        if final_completion is None:
            raise ValidationException("Model specification is required for agent runtime v3")
        if not final_completion.model_id and final_completion.has_legacy_triple():
            # Pre-#1697 back-compat: resolve the stashed legacy triple
            # via the model-platform service (Weston msg=80e873c1).
            from aperag.schema.utils import resolve_model_spec_legacy

            await resolve_model_spec_legacy(
                final_completion,
                user_id=user,
                capability=ModelCapability.CHAT,
            )
        if not final_completion.model_id:
            raise ValidationException("Model specification is required for agent runtime v3")

        final_collections = request.collections
        if not final_collections and bot_config and bot_config.agent and bot_config.agent.collections:
            collection_ids = [collection.id for collection in bot_config.agent.collections if collection.id]
            db_collections = await async_db_ops.query_collections_by_ids(user, collection_ids)
            final_collections = [
                KBCollectionSchema(
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

        runtime_row = await async_db_ops.query_model_runtime(final_completion.model_id, user)
        if not runtime_row:
            raise ResourceNotFoundException("Model", final_completion.model_id)
        model_record, account_record = runtime_row
        invocation = resolve_model_invocation_from_records(model=model_record, account=account_record)

        system_api_keys = await async_db_ops.query_api_keys(user, is_system=True)
        aperag_api_key = next((item.key for item in system_api_keys if item.key), None)
        if not aperag_api_key:
            created = await async_db_ops.create_api_key(user=user, description="aperag", is_system=True)
            aperag_api_key = created.key

        prompt_ops = _get_prompt_template_ops()
        system_prompt = await prompt_ops.resolve_agent_system_prompt(bot=bot, user_id=user)
        query_prompt_template = await prompt_ops.resolve_agent_query_prompt(bot=bot, user_id=user)

        return ResolvedAgentRequest(
            agent_message=AgentMessage(
                query=request.query,
                completion=final_completion,
                collections=final_collections,
                web_search_enabled=request.web_search_enabled,
                language=request.language,
                files=request.files,
            ),
            system_prompt=system_prompt,
            query_prompt_template=query_prompt_template,
            provider_model_id=invocation.provider_model_id,
            provider_base_url=invocation.base_url,
            provider_api_key=invocation.api_key,
            aperag_api_key=aperag_api_key,
        )

    async def _check_cancelled(self, turn_id: str, lease_guard: TurnLeaseGuard | None = None) -> None:
        if lease_guard:
            lease_guard.ensure_owned()
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
