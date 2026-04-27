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
``aperag.domains.marketplace.service.marketplace_service`` in Phase 4
Step 4-S4. Pre-Phase-4 callers keep working through this module.
Phase 6 cleanup removes the shim.
"""

from __future__ import annotations

from aperag.domains.marketplace.service.marketplace_service import *  # noqa: F401, F403
from aperag.domains.marketplace.service.marketplace_service import (  # noqa: F401
    MarketplaceService,
    marketplace_service,
)
