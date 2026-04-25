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

"""Agent-runtime FastAPI routes moved to the domain home in Phase 5
step 5-S5b.

Mirrors the Phase 4 model_platform + Phase 5 conversation route
patterns — the legacy ``aperag/views/agent_runtime.py`` module is
retained as a re-export shim; this module holds the canonical
``router``. URL / method / response_model / SSE streaming shape stays
byte-stable against the pre-move file (``scripts/export_openapi.py
--check`` exit 0).

Handler ``user`` parameters are typed against a per-domain
``AuthenticatedUser(Protocol)`` rather than the identity-owned
``User`` ORM (G16 canonical). ``AgentTurnStatus`` is imported from
the agent_runtime domain's own ``db/models`` per Phase 5 5-S2b
layout. ``required_user`` still resolves through the identity
routes, which is G1-legal cross-domain.
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Protocol

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from aperag.domains.agent_runtime.db.models import AgentTurnStatus
from aperag.domains.agent_runtime.runtime import agent_runtime_manager as runtime_manager
from aperag.domains.agent_runtime.schemas import (
    AgentArtifactEnvelope,
    AgentTurnSnapshot,
    CancelTurnResponse,
    CreateTurnRequest,
    CreateTurnResponse,
)
from aperag.domains.agent_runtime.wire import (
    StreamPart,
    TranslatorState,
    dump_part,
    translate_envelope,
)
from aperag.domains.identity.service.auth_dependencies import required_user

logger = logging.getLogger(__name__)

# AI SDK v5 ``UI Message Stream Protocol`` advertises itself via this
# response header. The FE ``@ai-sdk/react`` consumer (#76 dongdong)
# refuses to parse a stream that lacks the marker — landed here in
# Phase 8 D8.1 (#73) as part of the hard-cut from the legacy envelope
# wire to the v5 part wire.
AI_SDK_V5_HEADER = "x-vercel-ai-ui-message-stream"
AI_SDK_V5_HEADER_VALUE = "v1"


class AuthenticatedUser(Protocol):
    """Per-domain auth context for agent_runtime routes (G16 compliance)."""

    id: object


router = APIRouter(tags=["agent-runtime"])


def _format_part_frame(part: StreamPart, *, event_id: int | None = None) -> str:
    """Format a single AI SDK v5 stream part as one SSE frame.

    Wire shape:
        ``id: <sequence>\\ndata: <single-line JSON>\\n\\n``

    Notes:
    * No SSE ``event:`` field — AI SDK v5 carries the part type
      inside the JSON ``type`` discriminator, not in the SSE event
      name. The FE consumer keys on ``data`` only.
    * Single-line JSON only. The spec forbids multi-line ``data:``
      splitting because the consumer concatenates frames, so we keep
      ``ensure_ascii=False`` (UTF-8 wire) but never inject ``\\n``.
    * ``event_id`` is optional: only the LAST part of a fan-out
      group (multiple parts mapping to the same envelope sequence)
      gets the SSE ``id:`` line; intermediate parts are emitted as
      continuation frames so ``Last-Event-ID`` resume still points
      at the next envelope. See the translator module docstring for
      the rationale.
    """

    payload = json.dumps(dump_part(part), ensure_ascii=False, default=str, separators=(",", ":"))
    if event_id is not None:
        return f"id: {event_id}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


@router.post("/agent/chats/{chat_id}/turns")
async def create_turn_view(
    request: Request, chat_id: str, body: CreateTurnRequest, user: AuthenticatedUser = Depends(required_user)
) -> CreateTurnResponse:
    chat, bot, turn, created = await runtime_manager.turn_service.create_or_get_turn(str(user.id), chat_id, body)
    if created or turn.status in {AgentTurnStatus.QUEUED, AgentTurnStatus.RUNNING}:
        lease_owner = await runtime_manager.claim_turn(turn.id)
        if lease_owner:
            runtime_manager.launch_turn(
                turn=turn,
                chat=chat,
                bot=bot,
                user=str(user.id),
                request=body,
                lease_owner=lease_owner,
            )

    return CreateTurnResponse(
        turn=runtime_manager.turn_service.to_turn_envelope(turn),
        stream_url=f"{request.base_url}api/v2/agent/chats/{chat_id}/turns/{turn.id}/events",
    )


@router.get("/agent/chats/{chat_id}/turns/{turn_id}")
async def get_turn_snapshot_view(
    chat_id: str, turn_id: str, user: AuthenticatedUser = Depends(required_user)
) -> AgentTurnSnapshot:
    return await runtime_manager.turn_service.get_turn_snapshot(str(user.id), chat_id, turn_id)


@router.post("/agent/chats/{chat_id}/turns/{turn_id}/cancel")
async def cancel_turn_view(
    chat_id: str, turn_id: str, user: AuthenticatedUser = Depends(required_user)
) -> CancelTurnResponse:
    await runtime_manager.turn_service.get_turn_snapshot(str(user.id), chat_id, turn_id)
    await runtime_manager.cancel_turn(turn_id)
    return CancelTurnResponse(turn_id=turn_id, status=AgentTurnStatus.CANCELLED)


@router.get("/agent/artifacts/{artifact_id}")
async def get_artifact_view(
    artifact_id: str, user: AuthenticatedUser = Depends(required_user)
) -> AgentArtifactEnvelope:
    return await runtime_manager.artifact_service.get_artifact_for_user(str(user.id), artifact_id)


@router.get(
    "/agent/chats/{chat_id}/turns/{turn_id}/events",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent timeline events for one agent turn.",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": (
                            "AI SDK v5 UI Message Stream Protocol frames "
                            "(see https://sdk.vercel.ai/docs). The response "
                            "header `x-vercel-ai-ui-message-stream: v1` "
                            "advertises the protocol version."
                        ),
                    }
                }
            },
        }
    },
)
async def stream_turn_events_view(
    request: Request,
    chat_id: str,
    turn_id: str,
    after_sequence: int = Query(default=0, ge=0),
    user: AuthenticatedUser = Depends(required_user),
) -> StreamingResponse:
    await runtime_manager.turn_service.get_turn_snapshot(str(user.id), chat_id, turn_id)

    async def event_stream() -> AsyncIterator[str]:
        current_after_sequence = after_sequence
        header = request.headers.get("last-event-id")
        if header:
            try:
                current_after_sequence = int(header)
            except ValueError:
                current_after_sequence = after_sequence

        heartbeat_interval = 5.0
        last_heartbeat = asyncio.get_event_loop().time()
        translator_state = TranslatorState()

        try:
            while True:
                if await request.is_disconnected():
                    break

                events = await runtime_manager.event_service.get_events_after(
                    turn_id, after_sequence=current_after_sequence, limit=500
                )
                if events:
                    for event in events:
                        current_after_sequence = event.sequence
                        parts = translate_envelope(event, translator_state)
                        if not parts:
                            # No FE-visible parts for this envelope.
                            # Cursor still advanced so resume keeps
                            # moving past the no-op envelope.
                            continue
                        # Only the LAST part of the fan-out gets the
                        # SSE id: line — see translator module
                        # docstring for the resume invariant.
                        last_index = len(parts) - 1
                        for index, part in enumerate(parts):
                            event_id = event.sequence if index == last_index else None
                            yield _format_part_frame(part, event_id=event_id)
                    last_heartbeat = asyncio.get_event_loop().time()
                else:
                    turn = await runtime_manager.turn_service.db_ops.query_agent_turn(str(user.id), chat_id, turn_id)
                    if (
                        turn
                        and turn.status
                        in {
                            AgentTurnStatus.COMPLETED,
                            AgentTurnStatus.FAILED,
                            AgentTurnStatus.CANCELLED,
                        }
                        and current_after_sequence >= (turn.timeline_cursor or 0)
                    ):
                        break

                    now = asyncio.get_event_loop().time()
                    if now - last_heartbeat >= heartbeat_interval:
                        # SSE comment heartbeat — invisible to the
                        # AI SDK v5 consumer (which keys on
                        # ``data:`` only) but keeps proxies awake.
                        yield ": heartbeat\n\n"
                        last_heartbeat = now
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("agent SSE stream crashed for turn=%s", turn_id)
            error_payload = json.dumps(
                {"type": "error", "errorText": f"stream failed: {exc.__class__.__name__}"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            yield f"data: {error_payload}\n\n"
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={AI_SDK_V5_HEADER: AI_SDK_V5_HEADER_VALUE},
    )
