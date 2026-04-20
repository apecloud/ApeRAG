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
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from aperag.agent_runtime import (
    AgentTurnSnapshot,
    CancelTurnResponse,
    CreateTurnRequest,
    CreateTurnResponse,
)
from aperag.agent_runtime.runtime import agent_runtime_manager as runtime_manager
from aperag.db.models import AgentTurnStatus, User
from aperag.views.auth import required_user

router = APIRouter(tags=["agent-runtime"])


def _format_sse(event: str, data: dict, event_id: int | None = None) -> str:
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = json.dumps(data, ensure_ascii=False, default=str)
    for line in payload.splitlines():
        lines.append(f"data: {line}")
    lines.append("")
    return "\n".join(lines)


@router.post("/agent/chats/{chat_id}/turns")
async def create_turn_view(
    request: Request, chat_id: str, body: CreateTurnRequest, user: User = Depends(required_user)
) -> CreateTurnResponse:
    chat, bot, turn, created = await runtime_manager.turn_service.create_or_get_turn(str(user.id), chat_id, body)
    if created or (turn.status in {AgentTurnStatus.QUEUED, AgentTurnStatus.RUNNING} and turn.id not in runtime_manager.tasks):
        runtime_manager.launch_turn(turn=turn, chat=chat, bot=bot, user=str(user.id), request=body)

    return CreateTurnResponse(
        turn=runtime_manager.turn_service.to_turn_envelope(turn),
        stream_url=f"{request.base_url}api/v2/agent/chats/{chat_id}/turns/{turn.id}/events",
    )


@router.get("/agent/chats/{chat_id}/turns/{turn_id}")
async def get_turn_snapshot_view(chat_id: str, turn_id: str, user: User = Depends(required_user)) -> AgentTurnSnapshot:
    return await runtime_manager.turn_service.get_turn_snapshot(str(user.id), chat_id, turn_id)


@router.post("/agent/chats/{chat_id}/turns/{turn_id}/cancel")
async def cancel_turn_view(chat_id: str, turn_id: str, user: User = Depends(required_user)) -> CancelTurnResponse:
    await runtime_manager.turn_service.get_turn_snapshot(str(user.id), chat_id, turn_id)
    await runtime_manager.cancel_turn(turn_id)
    return CancelTurnResponse(turn_id=turn_id, status=AgentTurnStatus.CANCELLED)


@router.get("/agent/artifacts/{artifact_id}")
async def get_artifact_view(artifact_id: str, user: User = Depends(required_user)):
    return await runtime_manager.artifact_service.get_artifact_for_user(str(user.id), artifact_id)


@router.get("/agent/chats/{chat_id}/turns/{turn_id}/events")
async def stream_turn_events_view(
    request: Request,
    chat_id: str,
    turn_id: str,
    after_sequence: int = Query(default=0, ge=0),
    user: User = Depends(required_user),
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

        while True:
            if await request.is_disconnected():
                break

            events = await runtime_manager.event_service.get_events_after(
                turn_id, after_sequence=current_after_sequence, limit=500
            )
            if events:
                for event in events:
                    current_after_sequence = event.sequence
                    yield _format_sse(event.type, event.model_dump(mode="json"), event.sequence)
                last_heartbeat = asyncio.get_event_loop().time()
            else:
                turn = await runtime_manager.turn_service.db_ops.query_agent_turn(str(user.id), chat_id, turn_id)
                if turn and turn.status in {
                    AgentTurnStatus.COMPLETED,
                    AgentTurnStatus.FAILED,
                    AgentTurnStatus.CANCELLED,
                } and current_after_sequence >= (turn.timeline_cursor or 0):
                    break

                now = asyncio.get_event_loop().time()
                if now - last_heartbeat >= heartbeat_interval:
                    yield _format_sse("heartbeat", {"turn_id": turn_id})
                    last_heartbeat = now
                await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
