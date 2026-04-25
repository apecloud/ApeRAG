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

"""Tool lifecycle adapter for D9 §6 + §3 + §5.

Bridges the runtime's ``EventService.append_event`` envelope flow
into the new D9 lifecycle stream parts. Three responsibilities:

1. **Event-type constants** -- the runtime emits these strings into
   the existing envelope ``type`` field; they are scoped under
   ``tool.consent.*`` / ``tool.elicitation.*`` so the existing
   translator's ``technical_type`` adapter for user-activity stays
   well-defined (nothing maps to a user-activity icon).
2. **Translator extension hook** --
   :func:`translate_lifecycle_envelope` consumes an
   :class:`AgentTimelineEventEnvelope` whose ``type`` is one of the
   constants above and returns the AI SDK v5 wire parts to emit.
   The wire-side discriminator land in
   :mod:`aperag.domains.agent_runtime.wire.parts`
   (``DataToolConsentPart`` / ``DataElicitationPart``); the inner
   payload classes (``ToolConsentData`` / ``ElicitationData``) come
   from :mod:`aperag.domains.agent_runtime.uimessage` to enforce
   wire/at-rest same-schema canonical (per D8 §2 round-trip lock).
3. **Runtime integration helper** --
   :class:`LifecycleEmitter` wraps :class:`ConsentService` and
   :class:`ElicitationService` so the runtime can ``await
   emitter.consent_request(...)`` and get the envelope payload to
   feed into ``EventService.append_event`` in one call. Keeps the
   runtime's emit code focussed on sequencing rather than wire shape
   construction.

The envelope-event-type approach lets us extend the wire layer
without forking the SSE route or the existing
``translate_envelope``: ``api/routes.py`` already dispatches every
envelope through both the canonical translator and (in #75) the
lifecycle translator extension, then concatenates the part lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from aperag.domains.agent_runtime.schemas import AgentTimelineEventEnvelope
from aperag.domains.agent_runtime.tools.consent import ConsentService
from aperag.domains.agent_runtime.tools.elicitation import ElicitationService
from aperag.domains.agent_runtime.uimessage import ElicitationData, ToolConsentData
from aperag.domains.agent_runtime.wire.parts import (
    DataElicitationPart,
    DataToolConsentPart,
    StreamPart,
)

# -- envelope event-type constants -------------------------------------

# D9 §3.2 consent flow envelopes. Both transitions emit a separate
# envelope so the ``data-tool-consent`` part renders twice (pending
# then resolved) and the timeline preserves the full audit trail.
EVENT_TOOL_CONSENT_REQUESTED = "tool.consent.requested"
EVENT_TOOL_CONSENT_DECIDED = "tool.consent.decided"

# D9 §5.2 elicitation flow envelopes.
EVENT_TOOL_ELICITATION_REQUESTED = "tool.elicitation.requested"
EVENT_TOOL_ELICITATION_RESOLVED = "tool.elicitation.resolved"

LIFECYCLE_EVENT_TYPES = frozenset(
    {
        EVENT_TOOL_CONSENT_REQUESTED,
        EVENT_TOOL_CONSENT_DECIDED,
        EVENT_TOOL_ELICITATION_REQUESTED,
        EVENT_TOOL_ELICITATION_RESOLVED,
    }
)


# -- envelope -> wire translator extension ------------------------------


def translate_lifecycle_envelope(envelope: AgentTimelineEventEnvelope) -> list[StreamPart]:
    """Map a D9 lifecycle envelope to its wire part(s).

    Returns an empty list for envelopes whose ``type`` is not one of
    the lifecycle constants -- safe to call on every envelope so the
    SSE route can blindly chain it after
    :func:`aperag.domains.agent_runtime.wire.translator.translate_envelope`.
    """

    event_type = envelope.type
    if event_type not in LIFECYCLE_EVENT_TYPES:
        return []

    payload = envelope.data or {}
    if event_type in (EVENT_TOOL_CONSENT_REQUESTED, EVENT_TOOL_CONSENT_DECIDED):
        # The runtime stashes the wrapped ``data`` field on the
        # envelope so the translator does not need to know whether
        # the inner shape is ``ToolConsentData`` or a dict; pydantic
        # validates either form.
        data = ToolConsentData.model_validate(payload.get("data") or payload)
        return [DataToolConsentPart(data=data)]

    if event_type in (EVENT_TOOL_ELICITATION_REQUESTED, EVENT_TOOL_ELICITATION_RESOLVED):
        data = ElicitationData.model_validate(payload.get("data") or payload)
        return [DataElicitationPart(data=data)]

    return []  # pragma: no cover - guarded by the membership test above


# -- runtime integration helper ----------------------------------------


@dataclass(frozen=True)
class ConsentEnvelopeEmission:
    """Bundle returned from :meth:`LifecycleEmitter.request_consent`.

    ``event_type`` + ``data`` go straight into
    ``EventService.append_event(turn_id=..., sequence=..., type=event_type,
    data={"data": data.model_dump(mode="json", by_alias=True)}, ...)``;
    ``payload`` is also handed back as a typed handle for the runtime
    to await via :meth:`ConsentService.wait_for_decision`.
    """

    event_type: str
    payload: ToolConsentData
    envelope_data: dict[str, Any]


@dataclass(frozen=True)
class ElicitationEnvelopeEmission:
    """Like :class:`ConsentEnvelopeEmission` but for elicitation."""

    event_type: str
    payload: ElicitationData
    envelope_data: dict[str, Any]


class LifecycleEmitter:
    """Glue between the consent/elicitation services and the runtime
    envelope emit path.

    Single instance per runtime worker. Holds references to the
    consent / elicitation services so the runtime does not have to
    plumb them separately when it wants to emit a lifecycle envelope.
    """

    def __init__(
        self,
        *,
        consent: ConsentService,
        elicitation: ElicitationService,
    ):
        self._consent = consent
        self._elicitation = elicitation

    @property
    def consent(self) -> ConsentService:
        return self._consent

    @property
    def elicitation(self) -> ElicitationService:
        return self._elicitation

    # -- consent ----------------------------------------------------

    async def request_consent(
        self,
        *,
        consent_id: str,
        tool_name: str,
        raw_args: Any,
        risk: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConsentEnvelopeEmission:
        """Build the consent envelope payload + register the waiter.

        Pure construction -- the caller is responsible for actually
        feeding ``envelope_data`` into ``EventService.append_event``
        with the appropriate ``turn_id`` / ``sequence`` / ``actor``.
        """

        result = await self._consent.request_consent(
            consent_id=consent_id,
            tool_name=tool_name,
            raw_args=raw_args,
            risk=risk,
            metadata=metadata,
        )
        return ConsentEnvelopeEmission(
            event_type=EVENT_TOOL_CONSENT_REQUESTED,
            payload=result.payload,
            envelope_data=_envelope_data(result.payload),
        )

    async def consent_decision_envelope(self, consent_id: str) -> ConsentEnvelopeEmission:
        """Construct the post-decision envelope payload.

        Caller has already invoked :meth:`ConsentService.wait_for_decision`
        and got the typed result; this helper just wraps the resolved
        payload so the runtime emits both ``requested`` and
        ``decided`` envelopes through the same code path.
        """

        payload = self._consent.get_payload(consent_id)
        if payload is None:
            raise KeyError(consent_id)
        return ConsentEnvelopeEmission(
            event_type=EVENT_TOOL_CONSENT_DECIDED,
            payload=payload,
            envelope_data=_envelope_data(payload),
        )

    # -- elicitation ------------------------------------------------

    async def request_elicitation(
        self,
        *,
        elicitation_id: str,
        server_name: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ElicitationEnvelopeEmission:
        result = await self._elicitation.request_input(
            elicitation_id=elicitation_id,
            server_name=server_name,
            prompt=prompt,
            schema=schema,
        )
        return ElicitationEnvelopeEmission(
            event_type=EVENT_TOOL_ELICITATION_REQUESTED,
            payload=result.payload,
            envelope_data=_envelope_data(result.payload),
        )

    async def elicitation_resolved_envelope(self, elicitation_id: str) -> ElicitationEnvelopeEmission:
        payload = self._elicitation.get_payload(elicitation_id)
        if payload is None:
            raise KeyError(elicitation_id)
        return ElicitationEnvelopeEmission(
            event_type=EVENT_TOOL_ELICITATION_RESOLVED,
            payload=payload,
            envelope_data=_envelope_data(payload),
        )


def _envelope_data(payload: Any) -> dict[str, Any]:
    """Stash a wrapped ``{data: {...}}`` payload onto the envelope.

    The translator extension reads the wrapped form so future
    callers that hand-craft envelopes (tests, replay tools) can
    pass either shape without having to know the wrapping rule.
    """

    return {"data": payload.model_dump(mode="json", by_alias=True)}


__all__ = [
    "ConsentEnvelopeEmission",
    "ElicitationEnvelopeEmission",
    "EVENT_TOOL_CONSENT_DECIDED",
    "EVENT_TOOL_CONSENT_REQUESTED",
    "EVENT_TOOL_ELICITATION_REQUESTED",
    "EVENT_TOOL_ELICITATION_RESOLVED",
    "LIFECYCLE_EVENT_TYPES",
    "LifecycleEmitter",
    "translate_lifecycle_envelope",
]
