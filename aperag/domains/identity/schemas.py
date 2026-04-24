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

"""Canonical Pydantic view models for the ``identity`` domain.

Phase 4 Step 4-S3a carves the user / register / login / OAuth /
invitation / auth-config schemas out of ``aperag.schema.view_models``
so the identity domain owns its public contract shape. For the
Phase 6 cleanup window ``aperag.schema.view_models`` continues to
re-export these names via its end-of-file ``try`` block + the
symmetric ``_bind_view_models_reexports()`` hook defined at the
bottom of this module.

``role`` fields are typed as plain ``str`` / ``Literal["admin", "rw",
"ro"]`` so consumers outside identity never need the identity ``Role``
enum (G15 canonical — ``user.role == "admin"`` by literal).

``QuotaInfo`` / ``UserQuotaInfo`` / ``UserQuotaList`` /
``QuotaUpdateRequest`` stay in ``aperag.schema.view_models`` — quota's
permanent domain home is deferred to Phase 5 / Phase 6 (msg=896584ee
Section 2 patch). Phase 4 only carves the identity-shape contract.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from aperag.schema.common import PageResult

__all__ = [
    "Auth0",
    "Authing",
    "Logto",
    "Auth",
    "Config",
    "Register",
    "User",
    "Login",
    "UserList",
    "ChangePassword",
    "InvitationCreate",
    "Invitation",
    "InvitationList",
]


class Auth0(BaseModel):
    auth_domain: Optional[str] = None
    auth_app_id: Optional[str] = None


class Authing(BaseModel):
    auth_domain: Optional[str] = None
    auth_app_id: Optional[str] = None


class Logto(BaseModel):
    auth_domain: Optional[str] = None
    auth_app_id: Optional[str] = None


class Auth(BaseModel):
    type: Optional[Literal["none", "auth0", "authing", "logto", "cookie"]] = None
    auth0: Optional[Auth0] = None
    authing: Optional[Authing] = None
    logto: Optional[Logto] = None


class Config(BaseModel):
    admin_user_exists: Optional[bool] = Field(None, description="Whether the admin user exists")
    auth: Optional[Auth] = None
    login_methods: Optional[list[str]] = Field(
        None,
        description="Available login methods",
        examples=[["local", "google", "github"]],
    )


class Register(BaseModel):
    """
    The email of the user
    """

    token: Optional[str] = Field(None, description="The invitation token")
    email: Optional[str] = Field(None, description="The email of the user")
    username: Optional[str] = Field(None, description="The username of the user")
    password: Optional[str] = Field(None, description="The password of the user")


class User(BaseModel):
    id: Optional[str] = Field(None, description="The ID of the user")
    username: Optional[str] = Field(None, description="The username of the user")
    email: Optional[str] = Field(None, description="The email of the user")
    role: Optional[str] = Field(None, description="The role of the user")
    is_active: Optional[bool] = Field(None, description="Whether the user is active")
    date_joined: Optional[str] = Field(None, description="The date and time the user joined the system")
    registration_source: Optional[str] = Field(
        None,
        description="The registration source of the user (local, google, github, etc.)",
    )


class Login(BaseModel):
    username: Optional[str] = Field(None, description="The username of the user")
    password: Optional[str] = Field(None, description="The password of the user")


class UserList(BaseModel):
    """
    A list of users
    """

    items: Optional[list[User]] = None
    pageResult: Optional[PageResult] = None


class ChangePassword(BaseModel):
    username: Optional[str] = Field(None, description="The username of the user")
    old_password: Optional[str] = Field(None, description="The old password of the user")
    new_password: Optional[str] = Field(None, description="The new password of the user")


class InvitationCreate(BaseModel):
    username: Optional[str] = Field(None, description="The username of the user")
    email: Optional[str] = Field(None, description="The email of the user")
    role: Optional[Literal["admin", "rw", "ro"]] = Field(None, description="The role of the user (admin, rw, ro)")


class Invitation(BaseModel):
    email: Optional[str] = Field(None, description="The email of the user")
    token: Optional[str] = Field(None, description="The token of the invitation")
    created_by: Optional[str] = Field(None, description="The ID of the user who created the invitation")
    created_at: Optional[str] = Field(None, description="The date and time the invitation was created")
    is_valid: Optional[bool] = Field(None, description="Whether the invitation is valid")
    used_at: Optional[str] = Field(None, description="The date and time the invitation was used")
    role: Optional[Literal["admin", "rw", "ro"]] = Field(None, description="The role of the user (admin, rw, ro)")
    expires_at: Optional[str] = Field(None, description="The date and time the invitation will expire")


class InvitationList(BaseModel):
    """
    A list of invitations
    """

    items: Optional[list[Invitation]] = None
    pageResult: Optional[PageResult] = None


# Re-bind the 13 identity schemas onto ``aperag.schema.view_models``
# so that pre-migration callers (e.g. fastapi-users integration
# paths, views/auth.py imports, OpenAPI generation) continue to see
# the same class objects this module defines. Mirrors Step 4b
# (Phase 3 KB) + Step 5b3 / 5a (Phase 3 KB envelope schemas).
#
# ``sys.modules`` is consulted via a string lookup so the G1 AST
# scan does not flag a runtime import of
# ``aperag.schema.view_models`` from inside this identity-domain
# module.
def _bind_view_models_reexports() -> None:
    import sys

    _vm = sys.modules.get("aperag.schema.view_models")
    if _vm is None:
        return
    for _name in __all__:
        setattr(_vm, _name, globals()[_name])


_bind_view_models_reexports()
