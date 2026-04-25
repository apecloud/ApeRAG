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

"""Tool consent flow -- D9 §3 + §A7.

Sequence (per D9 §3.2):

1. Agent decides to call a side-effect tool.
2. Runtime calls :meth:`ConsentService.request_consent` -- this:
    * stashes the raw args in the backend-private
      :class:`RawArgsCache` keyed on the consent id (= tool call id);
    * builds a :class:`ToolConsentData` payload with redacted preview
      + stable hash + risk classification + ``state="pending"``;
    * surfaces it on the wire / at-rest store via the runtime's
      envelope emit path (the translator routes the new envelope
      type to a ``data-tool-consent`` part).
3. Runtime awaits :meth:`ConsentService.wait_for_decision` until the
   user POSTs to ``/agent/turns/{id}/consent`` (handler calls
   :meth:`ConsentService.decide`) or the consent times out
   (``state="expired"``).
4. On approval, the runtime fetches raw args via
   :meth:`ConsentService.consume_raw_args` and dispatches the tool.
   On denial / expiry, the runtime emits a ``tool-output-error``
   wire part (per AI SDK v5 strict spec) to surface the rejection.

Design points:

* The service is async-end-to-end. ``wait_for_decision`` uses an
  ``asyncio.Event`` per consent id so multiple concurrent agent
  loops in the same process never block each other.
* Raw args are fetched via ``consume_raw_args`` (single-use) so a
  duplicate request after dispatch returns ``None`` -- defends
  against accidental re-execution of the same approved consent.
* Audit trail is wired through the same :class:`AuditLogger`
  protocol the registry uses, so the system has a single audit sink
  and tests can pass an in-memory list lambda.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from aperag.domains.agent_runtime.tools.args_cache import (
    DEFAULT_RAW_ARGS_TTL_SECONDS,
    InMemoryRawArgsCache,
    RawArgsCache,
    args_hash,
    args_preview,
)
from aperag.domains.agent_runtime.tools.registry import AuditLogger, _noop_audit
from aperag.domains.agent_runtime.uimessage import ToolConsentData

# D9 §3.3: pending consent > 5 min -> ``state: "expired"``. Kept
# slightly under the raw-args TTL so an expired decision still has
# the raw args available for audit trail / debugging on the failure
# path.
DEFAULT_CONSENT_TIMEOUT_SECONDS = 300


ConsentDecision = Literal["approved", "denied", "expired"]


@dataclass(frozen=True)
class ConsentRequestResult:
    """Outcome of :meth:`ConsentService.request_consent`.

    Carries the wire-side payload the runtime must surface plus the
    consent id (= tool call id) the runtime threads through the rest
    of the dispatch.
    """

    consent_id: str
    payload: ToolConsentData


@dataclass(frozen=True)
class ConsentBinding:
    """Tenant binding recorded at request time so the decide path can
    enforce ownership (per D9 §2 multi-tenant boundary).

    Architect canonical lock (msg=19f2c9a9): a pending consent MUST
    record ``turn_id`` + ``user_id``; ``decide()`` MUST reject
    decisions that do not match. Defense-in-depth alongside the
    HTTP-layer turn ownership check via
    ``turn_service.get_turn_snapshot``.
    """

    turn_id: str
    user_id: str


class ConsentOwnershipError(PermissionError):
    """Raised when a consent decision attempt violates tenant binding."""


@dataclass(frozen=True)
class ConsentDecisionResult:
    """Outcome of :meth:`ConsentService.wait_for_decision`."""

    consent_id: str
    decision: ConsentDecision
    payload: ToolConsentData


class ConsentService:
    """Coordinator for the consent request <-> decision handshake.

    Single instance per process is sufficient; the in-process state
    is keyed on consent id which is globally unique (== tool call
    id). No per-turn isolation needed because each consent id is
    independent.
    """

    def __init__(
        self,
        *,
        raw_args_cache: Optional[RawArgsCache] = None,
        audit_logger: Optional[AuditLogger] = None,
        default_timeout_seconds: int = DEFAULT_CONSENT_TIMEOUT_SECONDS,
        default_args_ttl_seconds: int = DEFAULT_RAW_ARGS_TTL_SECONDS,
    ):
        self._raw_args_cache: RawArgsCache = raw_args_cache or InMemoryRawArgsCache()
        self._audit_logger: AuditLogger = audit_logger or _noop_audit
        self._default_timeout = max(default_timeout_seconds, 1)
        self._default_args_ttl = max(default_args_ttl_seconds, self._default_timeout + 30)
        self._lock = asyncio.Lock()
        self._waiters: dict[str, asyncio.Event] = {}
        self._decisions: dict[str, ConsentDecisionResult] = {}
        self._payloads: dict[str, ToolConsentData] = {}
        self._bindings: dict[str, ConsentBinding] = {}

    # -- request side (runtime emits consent prompt) ---------------

    async def request_consent(
        self,
        *,
        consent_id: str,
        turn_id: str,
        user_id: str,
        tool_name: str,
        raw_args: Any,
        risk: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConsentRequestResult:
        """Stash raw args, build wire payload, register a pending waiter.

        ``consent_id`` should be the tool call id so the FE consent
        UI's POST back can be correlated to the same id. ``turn_id``
        + ``user_id`` are recorded as the tenant binding so
        :meth:`decide` can reject cross-user / cross-turn decisions
        (per D9 §2 multi-tenant boundary; architect canonical lock
        msg=19f2c9a9). Raises ``ValueError`` on duplicate consent id
        (re-emitting consent for an already-pending tool call is a
        programming bug).
        """

        if not consent_id:
            raise ValueError("consent_id must be non-empty")
        if not turn_id:
            raise ValueError("turn_id must be non-empty")
        if not user_id:
            raise ValueError("user_id must be non-empty")
        async with self._lock:
            if consent_id in self._waiters:
                raise ValueError(f"consent already pending for {consent_id!r}")
            self._waiters[consent_id] = asyncio.Event()

        payload = ToolConsentData(
            tool_call_id=consent_id,
            tool_name=tool_name,
            metadata=dict(metadata or {}),
            args_preview=args_preview(raw_args),
            args_hash=args_hash(raw_args),
            risk=risk,  # type: ignore[arg-type]
            requested_at=_iso_now(),
            state="pending",
        )
        await self._raw_args_cache.put(consent_id, raw_args, ttl_seconds=self._default_args_ttl)
        async with self._lock:
            self._payloads[consent_id] = payload
            self._bindings[consent_id] = ConsentBinding(turn_id=turn_id, user_id=user_id)

        self._audit(
            "consent.requested",
            {
                "consent_id": consent_id,
                "tool_name": tool_name,
                "risk": risk,
                "args_hash": payload.args_hash,
                "metadata": payload.metadata,
            },
        )
        return ConsentRequestResult(consent_id=consent_id, payload=payload)

    # -- decide side (HTTP handler / admin override) --------------

    async def decide(
        self,
        consent_id: str,
        decision: ConsentDecision,
        *,
        actor_user_id: str,
        expected_turn_id: Optional[str] = None,
    ) -> ConsentDecisionResult:
        """Record the user's decision and wake the runtime waiter.

        Raises ``KeyError`` when no consent is pending for
        ``consent_id`` (idempotent re-decide is rejected -- tests +
        admin rollback should explicitly cancel + re-request rather
        than silently double-decide). Raises
        :class:`ConsentOwnershipError` when ``actor_user_id`` does
        not match the user that the consent was issued for, or when
        ``expected_turn_id`` is provided and does not match the turn
        the consent was bound to (per D9 §2 multi-tenant boundary).
        """

        if decision not in ("approved", "denied"):
            raise ValueError(f"decision must be 'approved' or 'denied', got {decision!r}")

        async with self._lock:
            event = self._waiters.get(consent_id)
            payload = self._payloads.get(consent_id)
            binding = self._bindings.get(consent_id)
            if event is None or payload is None or binding is None:
                raise KeyError(consent_id)
            if binding.user_id != actor_user_id:
                raise ConsentOwnershipError(
                    f"consent {consent_id!r} is owned by user {binding.user_id!r}, not {actor_user_id!r}"
                )
            if expected_turn_id is not None and binding.turn_id != expected_turn_id:
                raise ConsentOwnershipError(
                    f"consent {consent_id!r} is bound to turn {binding.turn_id!r}, not {expected_turn_id!r}"
                )
            if consent_id in self._decisions:
                raise ValueError(f"decision already recorded for {consent_id!r}")
            updated = payload.model_copy(update={"state": decision})
            result = ConsentDecisionResult(consent_id=consent_id, decision=decision, payload=updated)
            self._payloads[consent_id] = updated
            self._decisions[consent_id] = result
            event.set()

        if decision == "denied":
            # Drop raw args eagerly on denial so a leaked consent id
            # cannot fish them out post-decision.
            await self._raw_args_cache.delete(consent_id)

        self._audit(
            "consent.decided",
            {
                "consent_id": consent_id,
                "decision": decision,
                "actor_user_id": actor_user_id,
                "args_hash": payload.args_hash,
            },
        )
        return result

    # -- runtime hot-path -----------------------------------------

    async def wait_for_decision(
        self,
        consent_id: str,
        *,
        timeout_seconds: Optional[int] = None,
    ) -> ConsentDecisionResult:
        """Block the runtime until the user decides or timeout fires.

        On timeout the consent transitions to ``state="expired"`` and
        raw args are dropped from the cache. The runtime should treat
        ``decision="expired"`` the same as denial for tool dispatch
        but may surface a different UI message.
        """

        async with self._lock:
            event = self._waiters.get(consent_id)
            existing = self._decisions.get(consent_id)
        if event is None:
            raise KeyError(consent_id)
        if existing is not None:
            return existing

        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return await self._mark_expired(consent_id)

        async with self._lock:
            decision = self._decisions.get(consent_id)
            if decision is None:
                # Should be unreachable -- decide() always populates
                # _decisions before set()ing the event. Treat as
                # expired to keep the runtime moving.
                return await self._mark_expired(consent_id)
            return decision

    async def consume_raw_args(self, consent_id: str) -> Optional[Any]:
        """Single-use raw-args fetch on the approved-dispatch path.

        Returns ``None`` if (a) consent was denied/expired (cache
        already cleared), or (b) raw args were already consumed.
        Callers must treat ``None`` as "do not dispatch" rather than
        retrying.
        """

        value = await self._raw_args_cache.get(consent_id)
        if value is None:
            return None
        await self._raw_args_cache.delete(consent_id)
        return value

    # -- introspection --------------------------------------------

    def get_payload(self, consent_id: str) -> Optional[ToolConsentData]:
        return self._payloads.get(consent_id)

    # -- internals ------------------------------------------------

    async def _mark_expired(self, consent_id: str) -> ConsentDecisionResult:
        async with self._lock:
            payload = self._payloads.get(consent_id)
            if payload is None:
                raise KeyError(consent_id)
            existing = self._decisions.get(consent_id)
            if existing is not None:
                return existing
            updated = payload.model_copy(update={"state": "expired"})
            result = ConsentDecisionResult(consent_id=consent_id, decision="expired", payload=updated)
            self._payloads[consent_id] = updated
            self._decisions[consent_id] = result

        await self._raw_args_cache.delete(consent_id)
        self._audit(
            "consent.expired",
            {
                "consent_id": consent_id,
                "args_hash": payload.args_hash,
            },
        )
        return result

    def _audit(self, event: str, payload: dict[str, Any]) -> None:
        try:
            self._audit_logger(event, payload)
        except Exception:  # pragma: no cover
            return None


def _iso_now() -> str:
    """ISO-8601 timestamp with explicit UTC offset for D9 §3.1
    ``requestedAt``. Stable across daylight-savings boundaries."""

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ConsentBinding",
    "ConsentDecision",
    "ConsentDecisionResult",
    "ConsentOwnershipError",
    "ConsentRequestResult",
    "ConsentService",
    "DEFAULT_CONSENT_TIMEOUT_SECONDS",
]
