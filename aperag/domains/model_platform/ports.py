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
model_platform domain (LLMProvider + LLMProviderModel metadata).

Model_platform route handlers accept ``Depends(required_user)`` typed
as ``AuthenticatedUser`` so they do not bind to ``aperag.db.models.User``.

``aperag.llm.*`` (embedding / rerank / completion runtime wrappers
over HTTP APIs) is **not** part of this domain — it stays as shared
infrastructure (Phase 4 canonical msg=d47fa490 Section 7, sibling of
``aperag.config`` / ``aperag.db.base``) and can be imported freely
by this domain and any other.

Phase 4-S6 grep audit may reveal additional cross-domain consumer
Protocols (e.g. model_platform services needing a read-only user view
for provider-ACL scoping). If so the surface lands here as minimum
additive extension.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuthenticatedUser(Protocol):
    """Per-domain auth context (lesson 9a-ter). Model_platform route
    handlers read the user ``id`` for per-user provider ACL scoping;
    ``role`` may be needed for admin-only provider CRUD — pinned as
    plain ``str`` to allow G15-compliant literal compare.
    """

    id: Any
    role: str


__all__ = [
    "AuthenticatedUser",
]
