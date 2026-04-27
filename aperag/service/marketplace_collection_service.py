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

"""Legacy re-export shim — the service body moved to
``aperag.domains.marketplace.service.marketplace_collection_service``
in Phase 4 Step 4-S4. Pre-Phase-4 callers keep working through this
module.

Note — Q2 public rename: the method previously named
``_check_marketplace_access`` is now ``check_marketplace_access``
(dropping the underscore). Callers that still use the old
underscore-prefixed name will break — they should migrate to the
public method name before Phase 6 cleanup.
"""

from __future__ import annotations

from aperag.domains.marketplace.service.marketplace_collection_service import *  # noqa: F401, F403
from aperag.domains.marketplace.service.marketplace_collection_service import (  # noqa: F401
    MarketplaceCollectionService,
    marketplace_collection_service,
)
