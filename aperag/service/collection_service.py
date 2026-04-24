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

"""Legacy re-export + transitional DI wire-up shim.

Phase 3 Step 5b2b moved the ``CollectionService`` body to
``aperag.domains.knowledge_base.service.collection_service``. Pre-Phase-3
callers and FastAPI handlers that still do
``from aperag.service.collection_service import collection_service``
continue to resolve through this module.

The KB domain uses consumer-owned Protocols (Phase 3 Step 5b2a) for its
four cross-domain dependencies: ``MarketplaceOps``,
``MarketplaceCollectionOps``, ``SearchPipelineOps`` and ``QuotaOps``.
Step 5b2c moves the concrete-instance wire-up into ``aperag/app.py``
startup; until then this module bootstraps the DI setters at import
time so the legacy import path keeps working unchanged.

The ``MarketplaceCollectionOps`` Protocol uses the public method name
``check_marketplace_access`` (msg=6ab7d211 Q2); the current concrete
service still exposes it as ``_check_marketplace_access``. A thin
adapter class bridges the two names until Phase 4 marketplace
collection service drops the ``_`` prefix at its canonical location.
Phase 6 cleanup removes this whole module once every caller is migrated
to the domain path.
"""

from __future__ import annotations

from aperag.domains.knowledge_base.service.collection_service import *  # noqa: F401, F403
from aperag.domains.knowledge_base.service.collection_service import (  # noqa: F401
    CollectionService,
    collection_service,
    set_marketplace_collection_ops,
    set_marketplace_ops,
    set_quota_ops,
    set_search_pipeline_ops,
)


def _bootstrap_knowledge_base_collection_service_di() -> None:
    """Import legacy service singletons and wire the Phase 3 Step 5b2a
    consumer-owned Protocol setters. Kept internal and idempotent so a
    re-import (e.g. test reset) does not double-wire. Step 5b2c replaces
    this with explicit ``aperag/app.py`` startup wire-up."""

    from aperag.service.marketplace_collection_service import marketplace_collection_service
    from aperag.service.marketplace_service import marketplace_service
    from aperag.service.quota_service import quota_service
    from aperag.service.search_pipeline_service import search_pipeline_service

    class _MarketplaceCollectionOpsAdapter:
        """Bridge the public ``check_marketplace_access`` Protocol name
        onto the legacy ``_check_marketplace_access`` method. Phase 4
        drops the ``_`` prefix at the concrete service; at that point
        this adapter collapses to a straight passthrough (or is removed
        once the service structurally satisfies the Protocol directly)."""

        async def check_marketplace_access(self, user_id: str, collection_id: str) -> dict:
            return await marketplace_collection_service._check_marketplace_access(user_id, collection_id)

    set_marketplace_ops(marketplace_service)
    set_marketplace_collection_ops(_MarketplaceCollectionOpsAdapter())
    set_search_pipeline_ops(search_pipeline_service)
    set_quota_ops(quota_service)


_bootstrap_knowledge_base_collection_service_di()
