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

"""Legacy re-export shim — the service body was moved to
``aperag.domains.knowledge_base.service.collection_service`` in Phase 3
Step 5b2b. Pre-Phase-3 callers and FastAPI handlers that still do
``from aperag.service.collection_service import collection_service``
continue to resolve through this module.

The concrete-instance wire-up for the four consumer-owned Protocols
(``MarketplaceOps`` / ``MarketplaceCollectionOps`` /
``SearchPipelineOps`` / ``QuotaOps``) happens in ``aperag/app.py``
startup (Phase 3 Step 5b2c). This shim therefore has no runtime side
effects; Phase 6 cleanup removes it once every caller is migrated to
the domain path.
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
