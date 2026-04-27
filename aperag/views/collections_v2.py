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

"""Legacy router-import shim — the collection-v2 and document-v2 HTTP
handlers were merged into the canonical KB domain router at
``aperag.domains.knowledge_base.api.routes.router`` in Phase 3 Step 5a.
Callers that still do ``from aperag.views.collections_v2 import
router`` continue to resolve the same ``APIRouter`` instance through
this re-export; Phase 6 cleanup removes the shim once every import has
been migrated to the domain path.
"""

from __future__ import annotations

from aperag.domains.knowledge_base.api.routes import router  # noqa: F401
