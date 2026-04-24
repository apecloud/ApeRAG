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
Phase 4 marketplace / Phase 2 retrieval peers.

Lesson 9a-quad (consumer-owned bridge): ``collection_service`` and
``document_service`` (the two KB services that land in Step 5b2 /
Step 5b3) reach across to three services that are not yet inside
``aperag/domains/`` — ``marketplace_service`` / ``marketplace_collection_service``
(Phase 4 scope) and ``search_pipeline_service`` (pre-domain Phase 2
leftover). KB declares the Protocols here, at the consumer side, so
those three service modules can stay where they are this phase. Phase 4
and the eventual retrieval clean-up only need to have their concrete
implementations structurally satisfy the shapes below — no code under
``aperag/service/`` has to import ``aperag.domains.knowledge_base``.

Surface is scoped to the exact method signatures ``collection_service``
and ``document_service`` call today — grepped from call sites:

- ``marketplace_service.validate_marketplace_collection(collection_id)``
  (collection_service L204; document_service L493 / L592 / L935)
- ``marketplace_collection_service._check_marketplace_access(user, collection_id)``
  (collection_service L382)
- ``search_pipeline_service.execute_search(data, collection_id, search_user_id, chat_id)``
  (collection_service L363)

Phase 6 cleanup may collapse or rename these once Phase 4 marketplace
ships; until then the Protocols pin the minimum contract KB depends on.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MarketplaceOps(Protocol):
    """Minimum marketplace access the KB domain calls into.

    Today the only method is ``validate_marketplace_collection``, which
    raises when the caller (an un-authenticated public-read path) tries
    to touch a collection that is not published on the marketplace.
    """

    async def validate_marketplace_collection(self, collection_id: str) -> Any: ...


@runtime_checkable
class MarketplaceCollectionOps(Protocol):
    """Minimum marketplace-collection access the KB domain calls into.

    Used by the marketplace-subscriber search fallback in
    ``collection_service.create_search`` to resolve the owner user id
    when a subscriber searches a published collection.
    """

    async def _check_marketplace_access(self, user_id: str, collection_id: str) -> dict: ...


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


__all__ = [
    "MarketplaceCollectionOps",
    "MarketplaceOps",
    "SearchPipelineOps",
]
