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

"""Consumer-owned Protocols + per-domain auth context for the
marketplace domain (MarketplaceCollection + MarketplaceSubscription).

Marketplace route handlers accept ``Depends(required_user)`` typed as
``AuthenticatedUser`` so they do not bind to ``aperag.db.models.User``.
Only the attributes the handlers actually read are pinned.

Phase 4-S4 grep audit may reveal additional cross-domain consumer
surface (e.g. marketplace needing a read-only view of KB
``Collection`` — candidate ``CollectionOps`` Protocol). If so the
surface lands here as a minimum additive extension. Today the
observed marketplace services only consume ``User`` (owner id /
subscriber id) and their own ORM — no KB / governance / model_platform
consumer Protocols required.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuthenticatedUser(Protocol):
    """Per-domain auth context (lesson 9a-ter). Marketplace route
    handlers only read the user ``id`` (publish / subscribe ownership
    checks); ``role`` is not part of marketplace's authz surface.
    """

    id: Any


__all__ = [
    "AuthenticatedUser",
]
