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
``aperag.domains.knowledge_base.service.document_service`` in Phase 3
Step 5b3. Pre-Phase-3 callers and FastAPI handlers that still do
``from aperag.service.document_service import document_service``
continue to resolve through this module.

``document_service`` consumes the same ``MarketplaceOps`` / ``QuotaOps``
Protocol DI slots as ``collection_service`` (both sit in the KB domain
and share the wire-up), so this module has no runtime side effects —
the ``aperag/app.py`` startup in Step 5b2c wires the concrete
instances once and both services see them through the sibling-import
accessor functions. Phase 6 cleanup removes this shim after every
caller is migrated to the domain path.
"""

from __future__ import annotations

from aperag.domains.knowledge_base.service.document_service import *  # noqa: F401, F403
from aperag.domains.knowledge_base.service.document_service import (  # noqa: F401
    DocumentService,
    document_service,
)
