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
from typing import Any, AsyncIterator, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aperag.domains.agent_runtime.db.models import AgentEventActor, AgentTurnStatus
from aperag.domains.agent_runtime.runtime import agent_runtime_manager as runtime_manager
from aperag.domains.agent_runtime.schemas import (
    CancelTurnResponse,
    CreateTurnRequest,
    CreateTurnResponse,
)
from aperag.domains.agent_runtime.tools.consent import ConsentOwnershipError, ConsentService
from aperag.domains.agent_runtime.tools.elicitation import ElicitationOwnershipError, ElicitationService
from aperag.domains.agent_runtime.tools.lifecycle import translate_lifecycle_envelope
from aperag.domains.agent_runtime.uimessage import AgentTurnSnapshot
from aperag.domains.agent_runtime.wire import (
    StreamPart,
    TranslatorState,
    dump_part,
    translate_envelope,
)
from aperag.domains.identity.service.auth_dependencies import required_user

# D8.3 lane (#75) -- consent / elicitation services live as
# process-singletons here so the SSE route, the runtime worker, and
# the POST handlers all share one in-memory state. Production wires
# the audit logger via app startup; defaults are no-op for tests +
# local dev. The runtime task manager imports these via this module
# (cf. ``runtime.AgentRuntimeTaskManager``).
consent_service = ConsentService()
elicitation_service = ElicitationService()

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


# -- D8.3 (#75) consent + elicitation HTTP surfaces ---------------


class ConsentDecisionBody(BaseModel):
    """Body for ``POST /agent/turns/{turn_id}/consent/{tool_call_id}``.

    ``decision`` is one of ``approved`` / ``denied``. The runtime
    transitions ``state="expired"`` independently when the consent
    times out -- not allowed as an inbound decision.
    """

    decision: str = Field(pattern="^(approved|denied)$")


class ConsentDecisionResponse(BaseModel):
    """Response for the consent POST -- echoes the resolved part."""

    consent_id: str
    decision: str
    state: str


class ElicitationSubmitBody(BaseModel):
    """Body for ``POST /agent/turns/{turn_id}/elicit/{elicitation_id}``."""

    response: dict[str, Any]


class ElicitationSubmitResponse(BaseModel):
    elicitation_id: str
    state: str


@router.post("/agent/chats/{chat_id}/turns/{turn_id}/consent/{tool_call_id}")
async def post_consent_decision_view(
    chat_id: str,
    turn_id: str,
    tool_call_id: str,
    body: ConsentDecisionBody,
    user: AuthenticatedUser = Depends(required_user),
) -> ConsentDecisionResponse:
    """Record the user's consent decision (D9 §3.2 step 4).

    Tenant-bound per D9 §2 multi-tenant boundary (architect canonical
    lock msg=19f2c9a9):

    1. ``required_user`` enforces authentication.
    2. ``turn_service.get_turn_snapshot`` enforces that the
       authenticated user owns ``(chat_id, turn_id)``;
       ``ResourceNotFoundException`` on miss surfaces as 404 via the
       global exception handler.
    3. ``ConsentService.decide(*, expected_turn_id=...)`` enforces
       that the consent's stashed binding matches both
       ``actor_user_id`` and ``turn_id``. Defense-in-depth so even a
       misrouted request cannot resolve someone else's consent.

    Returns ``404`` for unknown turn / unknown consent, ``403`` when
    the consent is bound to a different user / turn, ``409`` when a
    decision was already recorded.
    """

    # Steps 1+2: ownership pre-check (raises ResourceNotFoundException
    # -> 404 via global handler when chat or turn does not belong to
    # the authenticated user).
    await runtime_manager.turn_service.get_turn_snapshot(str(user.id), chat_id, turn_id)

    try:
        result = await consent_service.decide(
            tool_call_id,
            body.decision,
            actor_user_id=str(user.id),  # type: ignore[arg-type]
            expected_turn_id=turn_id,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no pending consent for {tool_call_id!r}")
    except ConsentOwnershipError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # The runtime worker that owns the turn is awaiting
    # ``ConsentService.wait_for_decision``; recording the decision
    # above wakes that waiter. We additionally append a
    # ``tool.consent.decided`` envelope to the timeline so the SSE
    # stream replays it on reconnect.
    sequence = await _next_sequence(turn_id)
    await runtime_manager.event_service.append_event(
        turn_id=turn_id,
        sequence=sequence,
        event_type="tool.consent.decided",
        actor=AgentEventActor.SYSTEM,
        label=tool_call_id,
        status=body.decision,
        data={"data": result.payload.model_dump(mode="json", by_alias=True)},
    )
    return ConsentDecisionResponse(
        consent_id=tool_call_id,
        decision=body.decision,
        state=result.payload.state,
    )


@router.post("/agent/chats/{chat_id}/turns/{turn_id}/elicit/{elicitation_id}")
async def post_elicitation_response_view(
    chat_id: str,
    turn_id: str,
    elicitation_id: str,
    body: ElicitationSubmitBody,
    user: AuthenticatedUser = Depends(required_user),
) -> ElicitationSubmitResponse:
    """Submit an elicitation response (D9 §5.2 step 4).

    Tenant-bound the same way as :func:`post_consent_decision_view`
    (per D9 §2 multi-tenant boundary; architect canonical lock
    msg=19f2c9a9): HTTP-layer ``turn_service.get_turn_snapshot``
    ownership pre-check + service-layer
    :class:`ElicitationOwnershipError` defense-in-depth.

    Schema validation is handled inside
    :meth:`ElicitationService.submit`; a validation failure surfaces
    as ``422`` and the elicitation stays pending so the FE can
    re-prompt with the corrected value.
    """

    await runtime_manager.turn_service.get_turn_snapshot(str(user.id), chat_id, turn_id)

    try:
        result = await elicitation_service.submit(
            elicitation_id,
            body.response,
            actor_user_id=str(user.id),  # type: ignore[arg-type]
            expected_turn_id=turn_id,
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"no pending elicitation for {elicitation_id!r}",
        )
    except ElicitationOwnershipError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        # Validation error vs already-resolved error are both 4xx but
        # carry different semantics; the message body distinguishes
        # them.
        if "already resolved" in str(exc):
            raise HTTPException(status_code=409, detail=str(exc))
        raise HTTPException(status_code=422, detail=str(exc))

    sequence = await _next_sequence(turn_id)
    await runtime_manager.event_service.append_event(
        turn_id=turn_id,
        sequence=sequence,
        event_type="tool.elicitation.resolved",
        actor=AgentEventActor.SYSTEM,
        label=elicitation_id,
        status=result.outcome,
        data={"data": result.payload.model_dump(mode="json", by_alias=True)},
    )
    return ElicitationSubmitResponse(elicitation_id=elicitation_id, state=result.payload.state)


async def _next_sequence(turn_id: str) -> int:
    """Allocate the next envelope sequence for ``turn_id``.

    The runtime worker normally owns sequence allocation, but the
    consent / elicitation HTTP handlers run outside that loop, so
    they read the current ``timeline_cursor`` from the turn row and
    bump it. Race risk is bounded -- envelope-atomic replay (#73
    Option 2) tolerates the cursor advancing in either path because
    each envelope carries a stable id (here: ``tool_call_id`` /
    ``elicitation_id``) the FE consumer can dedup on.
    """

    # Lazy import keeps this module import-cycle-safe (the runtime
    # imports routes.py via app.py at startup; routes.py importing
    # the runtime module top-level would close the loop).
    from aperag.domains.agent_runtime.runtime import agent_runtime_manager

    snapshot = await agent_runtime_manager.turn_service.db_ops.query_agent_turn_by_id(turn_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"unknown turn {turn_id!r}")
    cursor = snapshot.timeline_cursor or 0
    return cursor + 1


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
                        # D8.1 wire translator handles canonical
                        # tool/text/source/finish parts; D8.3 (#75)
                        # adds consent + elicitation parts via the
                        # lifecycle translator extension. The two
                        # are concatenated so a single envelope can
                        # produce both kinds of parts in one
                        # fan-out group (uncommon today but the
                        # translator extension is purely additive).
                        parts = translate_envelope(event, translator_state) + translate_lifecycle_envelope(event)
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
        headers={
            AI_SDK_V5_HEADER: AI_SDK_V5_HEADER_VALUE,
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
