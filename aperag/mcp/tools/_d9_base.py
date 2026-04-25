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

"""D9 base reuse helpers for D10.c read primitives.

Per ``docs/modularization/d10-design-pack.md`` §F.4 (Weston msg=52616723
hard lock): tenancy + 3-level authorization MUST be enforced via the
existing ApeRAG D9 base — primitives MUST NOT self-attest tenancy from
client-supplied identifiers. The single source-of-truth tenancy gate is
``db_ops.query_collection(user, collection_id)`` which raises
``CollectionNotFoundException`` on a non-owner / non-subscriber miss.

This module exposes three small wrappers that compose the locked
sequence (per cuiwenbo msg=13a4139a + 明书 msg=c5e3be15):

1. ``resolve_authenticated_user()`` — surface the in-process user from
   the active FastMCP HTTP request (Authorization: Bearer <api_key>).
2. ``tenancy_gate(user, collection_id)`` — call canonical
   ``async_db_ops.query_collection`` and raise on miss.
3. ``authorization_gate(user, collection_id)`` — apply read-only +
   visibility decision via ``ToolAuthorizationPolicy`` (D9 §2). Read
   primitives are READ_ONLY by classification, so the gate is a fast
   pass for any authenticated user that survived ``tenancy_gate``; the
   call is preserved here so D10.g cache wire-in CANNOT skip it.

These helpers are shared across the 8 read primitive bodies so the
sequence is identical and audit-traceable. D10.g (#99) wires cache by
modifying the final ``await fetch_*_authoritative(...)`` call — these
gates are guaranteed un-cached per §E.7.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastmcp.server.dependencies import get_http_headers
from sqlalchemy import select

from aperag.config import get_async_session
from aperag.db.ops import async_db_ops
from aperag.domains.governance.db.models import ApiKey, ApiKeyStatus
from aperag.domains.identity.db.models import Role, User
from aperag.exceptions import (
    CollectionNotFoundException,
    PermissionDeniedError,
)


def _lazy_authz_imports():
    """Defer the agent-runtime authorization import to first-call time.

    Direct top-level import triggers a circular path through
    ``agent_runtime.uimessage`` ↔ ``conversation.schemas`` (an existing
    fragility unrelated to D10.c). Importing inside the helper is safe
    because by the time a primitive runs, FastAPI app startup has fully
    initialized the agent_runtime package.
    """

    from aperag.domains.agent_runtime.tools.authorization import (
        AuthorizationDecision,
        Principal,
        ToolAuthorizationPolicy,
        ToolRiskClassification,
    )

    return AuthorizationDecision, Principal, ToolAuthorizationPolicy, ToolRiskClassification


logger = logging.getLogger(__name__)


# ----- API key resolution ----------------------------------------------------


def _extract_api_key_from_request() -> Optional[str]:
    """Pull the bearer token from the in-flight FastMCP HTTP request.

    Falls back to ``APERAG_API_KEY`` env var (matches the pattern in
    ``aperag/mcp/server.py`` ``get_api_key()`` so stdio / inproc usage
    keeps working for tests + smoke).
    """

    try:
        headers = get_http_headers(include={"authorization"})
        if headers:
            auth_header = headers.get("Authorization") or headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                return auth_header[7:]
    except Exception as e:
        # Not in HTTP request context (stdio transport / direct unit-test call).
        logger.debug("Could not extract API key from headers: %s", e)
    return os.getenv("APERAG_API_KEY")


# ----- D9 base step 1: authenticated user resolution ------------------------


async def resolve_authenticated_user() -> User:
    """Resolve the authenticated ``User`` for the current MCP request.

    Mirrors ``aperag.domains.identity.service.auth_dependencies`` —
    looks up the API key in the ``api_key`` table (must be ACTIVE +
    not soft-deleted), then fetches the owning ``User``. Raises
    :class:`PermissionDeniedError` on any miss, which the FastMCP
    framework surfaces to the LLM client as a tool error per the §F.4
    Weston lock — ApeRAG never trusts client-asserted identity.
    """

    api_key = _extract_api_key_from_request()
    if not api_key:
        raise PermissionDeniedError("Missing API key for authenticated read primitive")

    async for session in get_async_session():
        result = await session.execute(
            select(ApiKey).where(
                ApiKey.key == api_key,
                ApiKey.status == ApiKeyStatus.ACTIVE,
                ApiKey.gmt_deleted.is_(None),
            )
        )
        ak = result.scalars().first()
        if not ak:
            raise PermissionDeniedError("Invalid or inactive API key")
        result = await session.execute(
            select(User).where(
                User.id == ak.user,
                User.is_active.is_(True),
                User.gmt_deleted.is_(None),
            )
        )
        user = result.scalars().first()
        if not user:
            raise PermissionDeniedError("API key owner not found / inactive")
        return user

    # Defensive: get_async_session generator yielded nothing.
    raise PermissionDeniedError("Could not establish DB session for authentication")


# ----- D9 base step 2: tenancy gate -----------------------------------------


async def tenancy_gate(user: User, collection_id: str):
    """Enforce per-§F.4 canonical tenancy gate.

    Returns the resolved ``Collection`` row (so callers don't double-
    fetch). Raises :class:`CollectionNotFoundException` on a miss —
    same exception surface as
    ``CollectionService.get_collection`` so error mapping stays
    consistent across the FastAPI + MCP entry points.
    """

    collection = await async_db_ops.query_collection(str(user.id), collection_id)
    if collection is None:
        # 404 over 403 to avoid leaking collection existence across tenants
        # (matches ``CollectionService.get_collection`` behavior).
        raise CollectionNotFoundException(collection_id)
    return collection


# ----- D9 base step 3: 3-level authorization gate ---------------------------


_policy = None  # populated lazily on first authorization_gate call


def _get_policy():
    global _policy
    if _policy is None:
        _AuthorizationDecision, _Principal, ToolAuthorizationPolicy, ToolRiskClassification = _lazy_authz_imports()
        # All D10.c read primitives are READ_ONLY per design pack §A.
        _policy = (
            ToolAuthorizationPolicy(
                risk_resolver=lambda _name: ToolRiskClassification.READ_ONLY,
            ),
            _Principal,
        )
    return _policy


async def authorization_gate(user: User, tool_safe_name: str):
    """Apply D9 §2 three-level authorization decision.

    For READ_ONLY primitives this is effectively a pass-through but
    the call MUST stay un-cached and on every invocation (§E.7) so that
    when the policy hardens (e.g. per-tool admin gates), the D10 read
    surface honors it without a cache-shortcut bypass.
    """

    policy, Principal = _get_policy()
    is_admin = bool(getattr(user, "role", None) == Role.ADMIN)
    decision = policy.evaluate(
        Principal(user_id=str(user.id), is_admin=is_admin),
        tool_safe_name,
    )
    if not decision.visible or not decision.can_invoke_auto:
        raise PermissionDeniedError(f"Tool {tool_safe_name!r} not authorized for user {user.id!r}: {decision.reason}")
    return decision


__all__ = [
    "authorization_gate",
    "resolve_authenticated_user",
    "tenancy_gate",
]
