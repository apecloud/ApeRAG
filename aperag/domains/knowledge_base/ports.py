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

"""Consumer-owned Protocols the knowledge_base domain needs from the
Phase 4 marketplace / Phase 2 retrieval peers and the infra-level
quota service.

Lesson 9a-quad (consumer-owned bridge): ``collection_service`` and
``document_service`` (the two KB services that land in Step 5b2 /
Step 5b3) reach across to four services that are not yet inside
``aperag/domains/`` — ``marketplace_service`` / ``marketplace_collection_service``
(Phase 4 scope), ``search_pipeline_service`` (pre-domain Phase 2
leftover), and ``quota_service`` (cross-cutting infra whose permanent
home Phase 5 decides per msg=896584ee). KB declares the Protocols
here, at the consumer side, so those four service modules can stay
where they are this phase. Phase 4 / Phase 5 and the eventual
retrieval clean-up only need to have their concrete implementations
structurally satisfy the shapes below — no code under
``aperag/service/`` has to import ``aperag.domains.knowledge_base``.

Surface is scoped to the exact method signatures ``collection_service``
and ``document_service`` call today — grepped from call sites:

- ``marketplace_service.validate_marketplace_collection(collection_id)``
  (collection_service L204; document_service L493 / L592 / L935)
- ``marketplace_collection_service.check_marketplace_access(user, collection_id)``
  (collection_service L382; legacy method name is
  ``_check_marketplace_access`` — Phase 4 marketplace_collection_service
  move drops the ``_`` prefix per msg=6ab7d211 Q2. The legacy shim at
  ``aperag/service/collection_service.py`` wraps the current
  underscore method through a small adapter so the Protocol surface is
  already public today.)
- ``search_pipeline_service.execute_search(data, collection_id, search_user_id, chat_id)``
  (collection_service L363)
- ``quota_service.check_and_consume_quota(user_id, quota_type, amount, session)``
  and ``quota_service.release_quota(user_id, quota_type, amount, session)``
  (collection_service L80 / L336; document_service L173 / L636).
  Satisfied by ``aperag/service/quota_service.py`` today; Phase 5
  design-lock decides permanent domain home (msg=896584ee).

Phase 6 cleanup may collapse or rename these once the owning domains
ship; until then the Protocols pin the minimum contract KB depends on.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuthenticatedUser(Protocol):
    """Per-domain auth context (lesson 9a-ter canonical).

    KB route handlers receive ``Depends(required_user)`` typed as this
    Protocol so they do not bind to ``aperag.db.models.User``. Only the
    attributes KB routes actually read — the user ``id`` — are pinned
    here. Phase 4 identity domain may promote a shared
    ``AuthenticatedUser`` once a cross-domain canonical is needed; KB's
    own Protocol stays until then.
    """

    id: Any


@runtime_checkable
class MarketplaceOps(Protocol):
    """Minimum marketplace access the KB domain calls into.

    - ``validate_marketplace_collection`` raises when the caller (an
      un-authenticated public-read path) tries to touch a collection
      that is not published on the marketplace.
    - ``get_sharing_status`` / ``publish_collection`` /
      ``unpublish_collection`` back the KB ``/sharing`` routes that
      moved to the KB domain in Step 5a; the concrete
      ``marketplace_service`` already provides these methods today, so
      the Protocol simply pins the surface KB reads without
      structurally importing the service.
    """

    async def validate_marketplace_collection(self, collection_id: str) -> Any: ...

    async def get_sharing_status(self, user_id: Any, collection_id: str) -> tuple[bool, Any]: ...

    async def publish_collection(self, user_id: Any, collection_id: str) -> Any: ...

    async def unpublish_collection(self, user_id: Any, collection_id: str) -> Any: ...


@runtime_checkable
class MarketplaceCollectionOps(Protocol):
    """Minimum marketplace-collection access the KB domain calls into.

    Used by the marketplace-subscriber search fallback in
    ``collection_service.create_search`` to resolve the owner user id
    when a subscriber searches a published collection. Method is
    ``check_marketplace_access`` (public, per msg=6ab7d211 Q2); the
    pre-Phase-4 concrete service still exposes the method under its
    original ``_check_marketplace_access`` name, so the legacy shim at
    ``aperag/service/collection_service.py`` wires a small adapter.
    """

    async def check_marketplace_access(self, user_id: str, collection_id: str) -> dict: ...


@runtime_checkable
class SearchPipelineOps(Protocol):
    """Minimum search-pipeline surface KB's ``execute_search_flow``
    delegates to. Return shape is ``(items, flow_debug_str)``; typed as
    ``Any`` because the concrete types live outside this domain and the
    call site treats the tuple opaquely before forwarding to its own
    caller.
    """

    async def execute_search(
        self,
        *,
        data: Any,
        collection_id: str,
        search_user_id: str,
        chat_id: Any = None,
    ) -> Any: ...


@runtime_checkable
class QuotaOps(Protocol):
    """Minimum quota surface KB consumes for per-tenant resource caps.

    ``check_and_consume_quota`` is called inside the KB write-path
    transactions (collection + document create) to enforce
    ``max_collection_count`` / ``max_document_count`` atomically with
    the write. ``release_quota`` mirrors it on delete. ``session`` is
    the SQLAlchemy async session the KB transaction is already running
    in; the quota implementation reuses it so the quota decrement joins
    the same transaction boundary.

    ``get_user_quotas`` returns the full user-quota dictionary (keyed
    by quota_type with ``quota_limit`` / ``current_usage`` / ``remaining``
    sub-fields). Used by ``document_service`` to enforce the
    ``max_document_count_per_collection`` cap — a read-only limit
    lookup that pairs with an in-transaction ``COUNT(*)`` over
    ``Document``. Limits are admin-controlled and rarely mutate, so a
    separate read session is acceptable here.
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

    async def get_user_quotas(self, user_id: str) -> dict: ...


__all__ = [
    "AuthenticatedUser",
    "MarketplaceCollectionOps",
    "MarketplaceOps",
    "QuotaOps",
    "SearchPipelineOps",
]
