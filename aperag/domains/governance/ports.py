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
governance domain (audit logs + API keys).

Governance's route handlers accept ``Depends(required_user)`` typed
as ``AuthenticatedUser`` so they never bind to ``aperag.db.models.User``.
``UserView`` is the narrow read-only user shape governance services
use when they need to render who-did-what (audit event subject /
admin-only permission checks). Both Protocols expose ``role`` as a
plain ``str`` — governance compares ``user.role == "admin"`` by
literal (Phase 4 G15 canonical), never importing the identity domain's
``Role`` enum.

Lesson 9a-quad rules: consumer owns the Protocol, provider (identity)
structurally satisfies by exposing ``id`` + ``role`` on its ``User``
ORM / domain schema. No provider-side import of this module.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuthenticatedUser(Protocol):
    """Per-domain auth context (lesson 9a-ter). Used by governance
    route handlers for ``Depends(required_user)`` parameter types.

    Pins ``id`` + ``role`` — ``role`` is compared by string literal
    (``"admin"`` / ``"user"``) per G15 canonical; the identity
    domain's ``Role`` enum never crosses into governance via import.
    """

    id: Any
    role: str


@runtime_checkable
class UserView(Protocol):
    """Read-only user view consumed by governance services for
    audit-subject lookup and admin-only permission checks.

    Structurally satisfied by the identity ``User`` ORM class
    (``User.id`` + ``User.role``). Governance never depends on the
    concrete ORM — only the attributes pinned here.
    """

    id: Any
    role: str


__all__ = [
    "AuthenticatedUser",
    "UserView",
]
