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

"""Cross-domain contracts owned by the ``conversation`` domain.

Lesson 9a-quad (consumer-owned Protocol): the conversation domain
owns Protocols describing every out-of-domain collaborator it reads or
calls. ``KnowledgeBaseCollectionView`` was seeded in Phase 3 step 1 so
``chat_collection_service`` could switch its type bindings before the
Collection ORM moved. Phase 5 adds:

* ``AuthenticatedUser`` — per-domain auth context (lesson 9a-ter):
  conversation handlers read ``id`` and occasionally ``role`` without
  importing the identity domain's concrete User row.
* ``QuotaOps`` — ``bot_service`` consumer of the legacy
  ``aperag.service.quota_service.check_and_consume_quota`` /
  ``release_quota`` pair; Phase 5 canonical Q1 (ruling C,
  ``msg=4a93c97e``) keeps quota in ``aperag/service/`` through Phase 6
  cleanup, so the singleton is wired via ``aperag/app.py`` at startup.

All Protocols are ``@runtime_checkable`` so the G18 alt runtime smoke
can ``isinstance``-probe the injected singletons.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KnowledgeBaseCollectionView(Protocol):
    """Structural view of a Collection row readable by ``conversation``.

    The conversation domain consumes Collection rows at two points:

    * Chat-create / chat-update: the user picks a Collection id; the
      service dereferences it to surface ``title`` / ``description``
      in the chat payload and to gate access control on ``user``.
    * Pipeline wiring: the stored ``config`` JSON is parsed into the
      retrieval / flow engine's typed config so the assistant can run
      the right recall pipeline.

    ``type`` is included because conversation selects different bot
    wiring for ``document`` / ``graph`` collection types. ``config``
    is ``Any`` because the legacy model stores it as a JSON string
    hydrated by ``aperag.schema.utils.parseCollectionConfig`` at the
    point of use.
    """

    id: str
    user: str
    title: str
    description: Any  # Nullable Text column.
    type: Any  # EnumColumn(CollectionType), kept Any to avoid coupling.
    config: Any


@runtime_checkable
class AuthenticatedUser(Protocol):
    """Per-domain auth context (lesson 9a-ter).

    Conversation handlers need ``id`` for ownership checks on chats
    and bots, and occasionally ``role`` to short-circuit admin-only
    paths. The identity domain's concrete ``User`` row structurally
    satisfies this Protocol — declaring it locally means conversation
    never has to ``from aperag.db.models import User`` and Phase 4's
    identity move stays invisible to consumers.
    """

    id: Any
    role: str


@runtime_checkable
class QuotaOps(Protocol):
    """``bot_service``'s view of the legacy quota service.

    Only two methods are consumed — bot create calls
    ``check_and_consume_quota(user, "max_bot_count", 1, session)`` and
    bot delete calls ``release_quota(user, "max_bot_count", 1, session)``.
    The Protocol stays narrower than the knowledge_base variant (which
    also reads ``get_user_quotas``) because conversation has no need
    for the lookup form.

    ``session`` threads an optional AsyncSession so the call joins the
    current transaction; the legacy implementation already exposes
    that keyword argument, so structural satisfaction is free.
    """

    async def check_and_consume_quota(
        self,
        user_id: str,
        quota_type: str,
        amount: int = 1,
        session: Any = None,
    ) -> None: ...

    async def release_quota(
        self,
        user_id: str,
        quota_type: str,
        amount: int = 1,
        session: Any = None,
    ) -> None: ...


@runtime_checkable
class DefaultModelOps(Protocol):
    """``chat_title_service``'s view of ``default_model_service``.

    Phase 5 step 5-S4e canonical (``msg=b93904b6`` Q-extension): even
    though Phase 4 has moved ``default_model_service`` into the
    ``model_platform`` domain where a direct cross-domain import is
    G1-legal, the Phase 5 system canonical stays consistent —
    ``chat_title_service`` reaches its single call site through a
    consumer-owned Protocol + DI slot, matching ``QuotaOps`` /
    ``ChatCollectionServiceOps`` / Q2 ``ChatDocumentOps`` patterns.

    The Protocol exposes only the one method the service actually
    uses. ``aperag/app.py`` wires the concrete
    ``aperag.domains.model_platform.service.default_model_service``
    singleton in at startup; Phase 6 cleanup may collapse the
    Protocol + DI seam into a direct import as part of the broader
    "simplify narrow DI with stable provider" sweep.
    """

    async def get_default_background_task_config(self, user_id: str) -> Any: ...


@runtime_checkable
class ChatCollectionServiceOps(Protocol):
    """``chat_document_service``'s view of ``chat_collection_service``.

    Phase 5 step 5-S4d moves ``chat_document_service`` into the
    conversation domain before ``chat_collection_service`` itself —
    ``chat_collection_service.create_user_chat_collection`` reaches
    ``session.get(User, ...)`` which requires the Phase 4 identity
    domain merge before it can be rewritten without leaking a G1
    legacy-aggregate ``User`` import into the domain module.

    Until the sibling service follows (5-S4f after ``#1633`` merges,
    per PM ``msg=0f9f10c4`` β ruling), the two surfaces
    ``chat_document_service`` actually calls —
    ``get_user_chat_collection`` and ``create_user_chat_collection``
    — are exposed here as a consumer-owned Protocol. ``aperag/app.py``
    wires the legacy singleton in at startup; the legacy instance
    structurally satisfies the Protocol verbatim.
    """

    async def get_user_chat_collection(self, user_id: str) -> Any: ...

    async def create_user_chat_collection(self, user_id: str) -> Any: ...


__all__ = [
    "KnowledgeBaseCollectionView",
    "AuthenticatedUser",
    "QuotaOps",
    "DefaultModelOps",
    "ChatCollectionServiceOps",
]
