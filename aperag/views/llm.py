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

"""Legacy router-import shim — the embed / rerank OpenAI-compat
handlers moved to ``aperag.domains.model_platform.api.llm_routes`` in
Phase 4 Step 4-S6. Test modules that import private helpers such as
``_build_rerank_response_items`` keep working via the explicit
re-export below.
"""

from __future__ import annotations

from aperag.domains.model_platform.api.llm_routes import (  # noqa: F401
    _build_rerank_response_items,
    router,
)
