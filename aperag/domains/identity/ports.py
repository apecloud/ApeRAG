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

"""Consumer-owned Protocols the identity domain needs from peers
plus the per-domain ``AuthenticatedUser`` contract (lesson 9a-ter).

Phase 4 design-lock (@符炫炜 msg=d47fa490 Q4 + msg=896584ee) routes
the three user-init side effects in ``UserManager.on_after_register``
through three consumer-owned Protocols. Today the concrete
implementations live at ``aperag/service/{bot,chat_collection,quota}_service``
or their Phase 5 conversation-domain successors; ``aperag/app.py``
startup wires the instances into the identity DI slots so this module
never static-imports the provider code.

Same lesson 9a-quad rule every other domain follows:

* The *consumer* (identity) owns the Protocol.
* The *provider* (legacy bot_service / chat_collection_service /
  quota_service, or their Phase 5 domain versions) structurally
  satisfies the shape — no provider-side import of this module.
* Phase 6 cleanup may collapse or rename these once every call site
  is migrated.

Per-domain ``AuthenticatedUser(Protocol)`` (msg=d47fa490 Section 3
microadjust) keeps identity's route handlers (and every other
domain's) decoupled from ``aperag.db.models.User``; only the
attributes actually read are pinned. Identity never promotes
``AuthenticatedUser`` to a cross-domain shared contract — each
consumer domain owns its own narrow Protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuthenticatedUser(Protocol):
    """Per-domain auth context (lesson 9a-ter canonical).

    Identity route handlers receive ``Depends(required_user)`` typed
    as this Protocol so they do not bind to ``aperag.db.models.User``.
    The attributes pinned here are the ones identity's own handlers
    read today — ``id`` (stable user identifier) and ``role`` (string
    literal compared per Phase 4 G15 canonical: ``user.role ==
    "admin"``, no enum import across domains).
    """

    id: Any
    role: str


@runtime_checkable
class BotInitOps(Protocol):
    """Minimum bot-init surface identity's ``UserManager.on_after_register``
    calls into.

    Today the legacy ``aperag.service.bot_service.bot_service`` creates
    a ``BotType.AGENT`` titled ``"Default Agent Bot"`` via
    ``create_bot(user, BotCreate(...), skip_quota_check=True)``; Phase
    4 ``aperag/app.py`` startup will wire a thin adapter that exposes
    that sequence under the ``create_default_bot_for_user`` Protocol
    method. Phase 5 moves bot_service to the conversation domain and
    the adapter collapses into a direct structural satisfaction.
    """

    async def create_default_bot_for_user(self, user_id: str) -> None: ...


@runtime_checkable
class ChatInitOps(Protocol):
    """Minimum chat-init surface identity's ``UserManager.on_after_register``
    calls into.

    Today the legacy ``aperag.service.chat_collection_service.chat_collection_service``
    bootstraps the user's per-tenant chat collection via
    ``initialize_user_chat_collection(user_id)``; Phase 4 wires a thin
    adapter exposing that as ``create_default_chat_for_user``. Phase 5
    moves the service into the conversation domain and the adapter
    collapses.
    """

    async def create_default_chat_for_user(self, user_id: str) -> None: ...


@runtime_checkable
class QuotaInitOps(Protocol):
    """Minimum quota-init surface identity's ``UserManager.on_after_register``
    calls into.

    Today the legacy ``aperag.service.quota_service.quota_service``
    seeds the per-user quota rows via ``initialize_user_quotas(user_id)``;
    Phase 4 wires a thin adapter exposing that as
    ``initialize_user_quota``. Quota's permanent home is deferred to
    Phase 5 / Phase 6 (msg=896584ee Section 2 patch); until then the
    legacy service stays in ``aperag/service/``.
    """

    async def initialize_user_quota(self, user_id: str) -> None: ...


__all__ = [
    "AuthenticatedUser",
    "BotInitOps",
    "ChatInitOps",
    "QuotaInitOps",
]
