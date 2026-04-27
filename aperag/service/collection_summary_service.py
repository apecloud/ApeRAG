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
``aperag.domains.knowledge_base.service.collection_summary_service`` in
Phase 3 Step 5b1. Tasks, workers, and any other pre-Phase-3 caller that
still imports ``aperag.service.collection_summary_service`` continues to
work unchanged through this module. Phase 6 cleanup removes the shim
after every caller is migrated to the domain path.
"""

from aperag.domains.knowledge_base.service.collection_summary_service import *  # noqa: F401, F403
from aperag.domains.knowledge_base.service.collection_summary_service import (  # noqa: F401
    CollectionSummaryService,
    collection_summary_service,
)
