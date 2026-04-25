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

"""Three-level tool authorization policy -- D9 §2.

Each MCP tool the runtime offers gets evaluated against a policy that
returns three independent answers:

* **Visibility** -- does this user see the tool in their agent's
  available_tools list at all? Hides admin-only tools from
  non-admin users so the LLM never even attempts them.
* **Invocation** -- can the agent call the tool automatically, or
  must it route through a consent gate first?
* **Consent gate** -- when invocation is gated, what consent state
  must the runtime collect before dispatching the call?

Defaults follow the D9 §2.2 table; per-tool overrides come from the
runtime / admin layer (e.g. an MCP server's declared risk metadata).
The :class:`ToolAuthorizationPolicy` is intentionally pure logic --
no IO, no DB lookups -- so the runtime can compose it with whatever
identity / role lookup it already has.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ToolRiskClassification(str, Enum):
    """D9 §2.2 risk buckets that drive the default policy.

    The enum name mirrors the D9 doc table category, the value is
    the wire-side risk literal that ``data-tool-consent`` parts
    surface to the FE (so renderers can colour-code the consent
    prompt). Keep the values stable -- they are part of the D9 §3.1
    contract.
    """

    READ_ONLY = "read_only"
    """``search_collection`` / ``web_search`` / ``read_document``
    style tools that produce no side effects. Auto-invocable for
    everyone."""

    SIDE_EFFECT_USER_DATA = "writes_user_data"
    """Tools that mutate the user's data (e.g. ``write_file``,
    ``create_note``). Visible to all but blocked unless consent."""

    SIDE_EFFECT_EXTERNAL = "calls_external_api"
    """Tools that call external APIs with side effects (e.g.
    ``send_email``, ``post_to_slack``). Same gating as
    ``writes_user_data`` -- the wire-side risk literal differs so
    the FE can render the prompt with the correct context."""

    SIDE_EFFECT_SYSTEM = "modifies_system"
    """Tools that mutate ApeRAG-system state (e.g. ``database_modify``,
    quota changes). Consent-gated for everyone, even admins."""

    ADMIN_ONLY = "admin_only"
    """Tools that bypass visibility for non-admin users entirely
    (e.g. ``system_config``, ``user_management``). Auto-invocable
    for admins, hidden for everyone else."""

    USER_PERSONAL = "user_personal"
    """Tools backed by a user-personal MCP server (per D9 §2.2 row
    4). Visible only to the owning user; per-server policy decides
    whether to gate."""


@dataclass(frozen=True)
class AuthorizationDecision:
    """Result of evaluating one tool against the policy.

    ``visible=False`` is a hard filter -- the runtime must not even
    surface the tool to the LLM.

    ``can_invoke_auto=True`` means the runtime may dispatch the call
    immediately on agent intent.

    ``requires_consent=True`` means the runtime must emit a
    ``data-tool-consent`` part and wait for the user decision before
    dispatching.

    ``risk`` is the wire-side risk literal that the consent UI
    surfaces -- ``None`` for ``ADMIN_ONLY`` tools because they never
    reach the consent gate.
    """

    visible: bool
    can_invoke_auto: bool
    requires_consent: bool
    risk: Optional[str]
    reason: str
    """Human-readable rationale for audit log + debugging. Not
    surfaced on wire -- consent UI uses ``risk`` only."""


# -- principal descriptor ------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """Identity carried into the policy.

    Kept dataclass-only so the runtime can populate it from whatever
    identity surface it already has (e.g. ``required_user`` -> ORM
    User -> map to admin flag) without coupling the policy to the
    identity domain.
    """

    user_id: str
    is_admin: bool = False


# -- the policy ---------------------------------------------------------


class ToolAuthorizationPolicy:
    """Three-level decision per tool.

    Construct with a ``risk_resolver`` callable that maps a tool's
    safe name to its :class:`ToolRiskClassification`. Tests pass a
    dict-backed lambda; production wires the runtime's tool-cache
    metadata. When the resolver returns ``None`` the policy falls
    back to ``READ_ONLY`` -- a deliberately conservative-on-the-LLM
    default so that an unclassified tool is auto-invocable but never
    triggers consent (the runtime that adds new tool types must
    explicitly classify them as side-effecting if they are).
    """

    def __init__(
        self,
        *,
        risk_resolver: callable,  # type: ignore[valid-type]
        owner_resolver: Optional[callable] = None,  # type: ignore[valid-type]
    ):
        # ``owner_resolver`` is consulted only for USER_PERSONAL
        # tools to know which user owns the per-user MCP server. It
        # returns the owning ``user_id`` (or ``None`` to reject all
        # callers).
        self._risk_resolver = risk_resolver
        self._owner_resolver = owner_resolver

    def evaluate(
        self,
        principal: Principal,
        tool_safe_name: str,
        tool_args: Optional[dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        # ``tool_args`` is reserved for future per-arg risk evaluation
        # (e.g. ``write_file(path=/etc/passwd)`` could escalate beyond
        # the per-tool risk). The current policy does not branch on
        # args; the parameter is part of the public surface so callers
        # don't need to refactor when arg-aware policy lands.
        del tool_args

        risk = self._risk_resolver(tool_safe_name) or ToolRiskClassification.READ_ONLY

        if risk is ToolRiskClassification.READ_ONLY:
            return AuthorizationDecision(
                visible=True,
                can_invoke_auto=True,
                requires_consent=False,
                risk=None,
                reason="read-only tool, auto-invocable",
            )

        if risk is ToolRiskClassification.ADMIN_ONLY:
            if principal.is_admin:
                return AuthorizationDecision(
                    visible=True,
                    can_invoke_auto=True,
                    requires_consent=False,
                    risk=None,
                    reason="admin-only tool, principal is admin",
                )
            return AuthorizationDecision(
                visible=False,
                can_invoke_auto=False,
                requires_consent=False,
                risk=None,
                reason="admin-only tool hidden from non-admin principal",
            )

        if risk is ToolRiskClassification.USER_PERSONAL:
            owner = self._owner_resolver(tool_safe_name) if self._owner_resolver else None
            if owner is None or owner != principal.user_id:
                return AuthorizationDecision(
                    visible=False,
                    can_invoke_auto=False,
                    requires_consent=False,
                    risk=None,
                    reason="user-personal tool only visible to owner",
                )
            return AuthorizationDecision(
                visible=True,
                can_invoke_auto=True,
                requires_consent=False,
                risk=None,
                reason="user-personal tool, principal is owner",
            )

        # All side-effect categories converge on consent-gated:
        # visible to everyone, blocked from auto-invocation, surface
        # the matching wire-side risk literal so the FE renders the
        # right consent prompt.
        return AuthorizationDecision(
            visible=True,
            can_invoke_auto=False,
            requires_consent=True,
            risk=risk.value,
            reason=f"side-effect tool ({risk.value}); consent required per default policy",
        )

    # -- convenience helpers ---------------------------------------

    def filter_visible(self, principal: Principal, tool_safe_names: list[str]) -> list[str]:
        """Return the subset of ``tool_safe_names`` ``principal`` may see.

        Helper for the runtime's "build the LLM tool list" call site
        so it doesn't loop and branch on the decision dataclass.
        Order is preserved.
        """

        return [name for name in tool_safe_names if self.evaluate(principal, name).visible]


def default_policy(
    risk_resolver: callable,  # type: ignore[valid-type]
    owner_resolver: Optional[callable] = None,  # type: ignore[valid-type]
) -> ToolAuthorizationPolicy:
    """Construct the default policy backed by ``risk_resolver``.

    Hand the resulting policy directly to the runtime; it carries
    no per-call state so a single instance per process is fine.
    """

    return ToolAuthorizationPolicy(risk_resolver=risk_resolver, owner_resolver=owner_resolver)


__all__ = [
    "AuthorizationDecision",
    "Principal",
    "ToolAuthorizationPolicy",
    "ToolRiskClassification",
    "default_policy",
]
