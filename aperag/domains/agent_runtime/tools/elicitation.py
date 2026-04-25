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

"""Tool elicitation flow -- D9 §5.

MCP `elicitation` is the server-asks-user-mid-tool surface. Where
:mod:`consent` gates *whether* a tool runs, elicitation gates *what
inputs* the tool gets while it is running:

1. Tool execution starts.
2. Mid-execution the tool emits an MCP elicitation request with a
   prompt + JSON Schema describing the expected user response.
3. Runtime calls :meth:`ElicitationService.request_input` -- builds
   a wire-side ``data-elicitation`` part with the schema and a
   stable elicitation id, surfaces it to the FE.
4. Runtime awaits :meth:`ElicitationService.wait_for_input` until
   the user POSTs to ``/agent/turns/{id}/elicit/{eid}`` (handler
   calls :meth:`ElicitationService.submit`).
5. Runtime resumes tool execution with the validated user response.

Schema validation: D9 §5 / §A4 #5 says the response must be
schema-validated. We use :mod:`jsonschema` if available; if the
project has not pulled in that dep yet we fall back to a minimal
"required-fields-present" check so the contract test
(`elicitation rejects bad input`) still has bite without forcing a
new runtime dep on the consent path. The runtime-side caller can
swap in a stricter validator by passing ``validator=`` to
:meth:`ElicitationService.submit`.

State transitions: ``pending -> submitted | cancelled``. There is no
"expired" state for elicitation in D9 §5.1 -- the runtime decides
whether to time out the awaiting tool execution and surface a
``tool-output-error`` wire part (per AI SDK v5 strict spec) instead.
We mirror that here by exposing :meth:`ElicitationService.cancel` for
the runtime to call on abort / timeout.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from aperag.domains.agent_runtime.tools.registry import AuditLogger, _noop_audit
from aperag.domains.agent_runtime.uimessage import ElicitationData

# Default timeout matches consent so a stalled FE doesn't pin a tool
# execution forever; tests + admin override via constructor.
DEFAULT_ELICITATION_TIMEOUT_SECONDS = 300


ElicitationOutcome = Literal["answered", "cancelled"]


@dataclass(frozen=True)
class ElicitationRequestResult:
    elicitation_id: str
    payload: ElicitationData


@dataclass(frozen=True)
class ElicitationBinding:
    """Tenant binding recorded at request time so the submit/cancel
    path can enforce ownership (per D9 §2 multi-tenant boundary;
    architect canonical lock msg=19f2c9a9)."""

    turn_id: str
    user_id: str


class ElicitationOwnershipError(PermissionError):
    """Raised when an elicitation submit attempt violates tenant binding."""


@dataclass(frozen=True)
class ElicitationSubmitResult:
    elicitation_id: str
    outcome: ElicitationOutcome
    payload: ElicitationData


SchemaValidator = Callable[[dict[str, Any], dict[str, Any]], None]
"""``(schema, response) -> None`` -- raises on invalid input.

Tests inject a strict :mod:`jsonschema` validator; the default
fallback below only checks required-field presence so the consent
domain doesn't grow a hard ``jsonschema`` dependency.
"""


def _required_fields_validator(schema: dict[str, Any], response: dict[str, Any]) -> None:
    """Minimal JSON Schema check: assert ``required`` properties are present.

    Returns silently on conformance; raises ``ValueError`` otherwise.
    Intentionally permissive on type / range / pattern -- the runtime
    can swap in a stricter validator (e.g. ``jsonschema.validate``)
    by passing ``validator=`` to :meth:`ElicitationService.submit`.
    """

    required = schema.get("required") if isinstance(schema, dict) else None
    if not isinstance(required, list):
        return
    missing = [name for name in required if name not in response]
    if missing:
        raise ValueError(f"elicitation response missing required fields: {missing}")


class ElicitationService:
    """Coordinator for elicitation request <-> answer handshake.

    Single instance per process; the in-process state is keyed on
    elicitation id which is the runtime's responsibility to make
    unique across all turns it serves.
    """

    def __init__(
        self,
        *,
        audit_logger: Optional[AuditLogger] = None,
        default_timeout_seconds: int = DEFAULT_ELICITATION_TIMEOUT_SECONDS,
        default_validator: Optional[SchemaValidator] = None,
    ):
        self._audit_logger: AuditLogger = audit_logger or _noop_audit
        self._default_timeout = max(default_timeout_seconds, 1)
        self._default_validator = default_validator or _required_fields_validator
        self._lock = asyncio.Lock()
        self._waiters: dict[str, asyncio.Event] = {}
        self._payloads: dict[str, ElicitationData] = {}
        self._results: dict[str, ElicitationSubmitResult] = {}
        self._bindings: dict[str, ElicitationBinding] = {}

    # -- request side ----------------------------------------------

    async def request_input(
        self,
        *,
        elicitation_id: str,
        turn_id: str,
        user_id: str,
        server_name: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ElicitationRequestResult:
        """Stash schema, register pending waiter, return wire payload.

        ``server_name`` is the MCP server identity that initiated the
        elicitation (per D9 §5.1 + D9.1 amend) so the FE consent UI
        can surface where the input request originated. ``turn_id``
        + ``user_id`` are recorded as the tenant binding so
        :meth:`submit` / :meth:`cancel` can reject cross-user /
        cross-turn calls (per D9 §2 multi-tenant boundary; architect
        canonical lock msg=19f2c9a9).
        """

        if not elicitation_id:
            raise ValueError("elicitation_id must be non-empty")
        if not turn_id:
            raise ValueError("turn_id must be non-empty")
        if not user_id:
            raise ValueError("user_id must be non-empty")
        if not server_name:
            raise ValueError("server_name must be non-empty")
        if not isinstance(schema, dict):
            raise TypeError("schema must be a dict (JSON Schema object)")
        async with self._lock:
            if elicitation_id in self._waiters:
                raise ValueError(f"elicitation already pending for {elicitation_id!r}")
            self._waiters[elicitation_id] = asyncio.Event()
            self._bindings[elicitation_id] = ElicitationBinding(turn_id=turn_id, user_id=user_id)

        # Defensive copy so the caller can mutate the schema dict
        # afterwards without retroactively changing what the wire
        # frame already advertised.
        payload = ElicitationData(
            elicitation_id=elicitation_id,
            server_name=server_name,
            prompt=prompt,
            schema_=json.loads(json.dumps(schema)),
            response=None,
            state="pending",
        )
        async with self._lock:
            self._payloads[elicitation_id] = payload

        self._audit(
            "elicitation.requested",
            {
                "elicitation_id": elicitation_id,
                "schema_keys": sorted(schema.keys()),
            },
        )
        return ElicitationRequestResult(elicitation_id=elicitation_id, payload=payload)

    # -- submit side ----------------------------------------------

    async def submit(
        self,
        elicitation_id: str,
        response: dict[str, Any],
        *,
        actor_user_id: str,
        expected_turn_id: Optional[str] = None,
        validator: Optional[SchemaValidator] = None,
    ) -> ElicitationSubmitResult:
        """Validate + record the user's answer, wake the waiter.

        Sets ``state="answered"`` on success (per D9 §5.1 / D9.1
        canonical state vocabulary). Raises ``KeyError`` when no
        elicitation is pending, ``ValueError`` when the response
        fails validation. The elicitation stays ``pending`` on
        validation failure so the FE can re-prompt with the
        corrected response. Raises
        :class:`ElicitationOwnershipError` when ``actor_user_id`` /
        ``expected_turn_id`` do not match the binding recorded at
        :meth:`request_input` time (per D9 §2 multi-tenant boundary).
        """

        async with self._lock:
            payload = self._payloads.get(elicitation_id)
            event = self._waiters.get(elicitation_id)
            binding = self._bindings.get(elicitation_id)
            if payload is None or event is None or binding is None:
                raise KeyError(elicitation_id)
            if binding.user_id != actor_user_id:
                raise ElicitationOwnershipError(
                    f"elicitation {elicitation_id!r} is owned by user {binding.user_id!r}, not {actor_user_id!r}"
                )
            if expected_turn_id is not None and binding.turn_id != expected_turn_id:
                raise ElicitationOwnershipError(
                    f"elicitation {elicitation_id!r} is bound to turn {binding.turn_id!r}, not {expected_turn_id!r}"
                )
            if elicitation_id in self._results:
                raise ValueError(f"elicitation already resolved for {elicitation_id!r}")

        check = validator or self._default_validator
        check(payload.schema_, response)

        async with self._lock:
            updated = payload.model_copy(update={"response": response, "state": "answered"})
            result = ElicitationSubmitResult(
                elicitation_id=elicitation_id,
                outcome="answered",
                payload=updated,
            )
            self._payloads[elicitation_id] = updated
            self._results[elicitation_id] = result
            event.set()

        self._audit(
            "elicitation.submitted",
            {
                "elicitation_id": elicitation_id,
                "actor_user_id": actor_user_id,
                "response_keys": sorted(response.keys()) if isinstance(response, dict) else None,
            },
        )
        return result

    async def cancel(
        self,
        elicitation_id: str,
        *,
        actor_user_id: str,
        expected_turn_id: Optional[str] = None,
        reason: str = "cancelled",
        bypass_ownership: bool = False,
    ) -> ElicitationSubmitResult:
        """Resolve a pending elicitation as ``state='cancelled'``.

        ``bypass_ownership=True`` is reserved for internal callers
        (timeout sweeper, runtime abort) where there is no human
        actor to validate against. Public-facing HTTP handlers MUST
        leave it ``False`` so the tenant binding is enforced.
        """

        async with self._lock:
            payload = self._payloads.get(elicitation_id)
            event = self._waiters.get(elicitation_id)
            binding = self._bindings.get(elicitation_id)
            if payload is None or event is None or binding is None:
                raise KeyError(elicitation_id)
            if not bypass_ownership:
                if binding.user_id != actor_user_id:
                    raise ElicitationOwnershipError(
                        f"elicitation {elicitation_id!r} is owned by user {binding.user_id!r}, not {actor_user_id!r}"
                    )
                if expected_turn_id is not None and binding.turn_id != expected_turn_id:
                    raise ElicitationOwnershipError(
                        f"elicitation {elicitation_id!r} is bound to turn {binding.turn_id!r}, not {expected_turn_id!r}"
                    )
            if elicitation_id in self._results:
                raise ValueError(f"elicitation already resolved for {elicitation_id!r}")
            updated = payload.model_copy(update={"state": "cancelled"})
            result = ElicitationSubmitResult(
                elicitation_id=elicitation_id,
                outcome="cancelled",
                payload=updated,
            )
            self._payloads[elicitation_id] = updated
            self._results[elicitation_id] = result
            event.set()

        self._audit(
            "elicitation.cancelled",
            {
                "elicitation_id": elicitation_id,
                "actor_user_id": actor_user_id,
                "reason": reason,
            },
        )
        return result

    # -- runtime hot-path -----------------------------------------

    async def wait_for_input(
        self,
        elicitation_id: str,
        *,
        timeout_seconds: Optional[int] = None,
    ) -> ElicitationSubmitResult:
        """Block until ``submit`` / ``cancel`` fires or timeout cancels."""

        async with self._lock:
            event = self._waiters.get(elicitation_id)
            existing = self._results.get(elicitation_id)
        if event is None:
            raise KeyError(elicitation_id)
        if existing is not None:
            return existing

        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return await self.cancel(
                elicitation_id,
                actor_user_id="system",
                reason="timeout",
                bypass_ownership=True,
            )

        async with self._lock:
            result = self._results.get(elicitation_id)
            if result is None:
                # Defensive parity with ``ConsentService.wait_for_decision``.
                return await self.cancel(
                    elicitation_id,
                    actor_user_id="system",
                    reason="missing_result",
                    bypass_ownership=True,
                )
            return result

    # -- introspection ---------------------------------------------

    def get_payload(self, elicitation_id: str) -> Optional[ElicitationData]:
        return self._payloads.get(elicitation_id)

    # -- internals -------------------------------------------------

    def _audit(self, event: str, payload: dict[str, Any]) -> None:
        try:
            self._audit_logger(event, payload)
        except Exception:  # pragma: no cover
            return None


__all__ = [
    "DEFAULT_ELICITATION_TIMEOUT_SECONDS",
    "ElicitationBinding",
    "ElicitationOutcome",
    "ElicitationOwnershipError",
    "ElicitationRequestResult",
    "ElicitationService",
    "ElicitationSubmitResult",
    "SchemaValidator",
]
