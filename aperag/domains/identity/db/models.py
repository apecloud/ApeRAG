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

"""Identity-domain SQLAlchemy models.

Owns the three entities the identity domain is responsible for:

* ``User`` — the authenticated principal (fastapi-users-backed).
* ``OAuthAccount`` — OAuth provider link rows owned by ``User``.
* ``Role`` — the per-user authorization level enum
  (``admin`` / ``rw`` / ``ro``).

Moved here from ``aperag.db.models`` in Phase 4 Step 4-S2a; the
legacy aggregate module retains a re-export shim with the full
Phase 3 + Phase 4 symbol list until Phase 6 cleanup.

G15 canonical note: cross-domain consumers (governance /
model_platform / marketplace / Phase 3 KB / Phase 5 conversation)
**do not** import ``Role`` from this module. They compare
``user.role == "admin"`` / ``"rw"`` / ``"ro"`` by string literal
against the per-domain ``AuthenticatedUser`` / ``UserView`` Protocols
whose ``role`` attribute is typed ``str``. The enum stays
identity-internal.
"""

from __future__ import annotations

import random
import uuid
from enum import Enum

from fastapi_users.db import SQLAlchemyBaseOAuthAccountTable
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aperag.db.base import Base
from aperag.utils.utils import utc_now


def _random_id() -> str:
    """Local copy of ``aperag.db.models.random_id`` so this module
    does not have to import from the G1 strict-ban aggregate. Phase 6
    cleanup collapses the helper twins (one in each per-domain DB
    module) onto ``aperag.db.base``.
    """

    return "".join(random.sample(uuid.uuid4().hex, 16))


def _enum_column(enum_class):
    """Mirror of ``aperag.db.models.EnumColumn``. Formula:
    ``max(max_value_len + 20, 50)``. Keep in sync until Phase 6
    consolidates these helper twins.
    """

    max_length = max(len(e.value) for e in enum_class) if enum_class and len(enum_class) > 0 else 50
    max_length = max(max_length + 20, 50)
    return String(length=max_length)


class Role(str, Enum):
    ADMIN = "admin"
    RW = "rw"
    RO = "ro"


class User(Base):
    __tablename__ = "user"

    id = Column(String(24), primary_key=True, default=lambda: "user" + _random_id())
    username = Column(String(256), unique=True, nullable=True)  # Unified with other user fields
    email = Column(String(254), unique=True, nullable=True)
    role = Column(_enum_column(Role), nullable=False, default=Role.RO)
    hashed_password = Column(String(128), nullable=False)  # fastapi-users expects hashed_password
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=True, nullable=False)  # fastapi-users requires is_verified
    is_staff = Column(Boolean, default=False, nullable=False)
    chat_collection_id = Column(String(24), nullable=True, index=True)  # Chat collection for user
    date_joined = Column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )  # Unified naming with other time fields
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)
    oauth_accounts: Mapped[list["OAuthAccount"]] = relationship("OAuthAccount", lazy="joined", back_populates="user")

    @property
    def password(self):
        raise AttributeError("password is not a readable attribute")

    @password.setter
    def password(self, value):
        self.hashed_password = value


class OAuthAccount(SQLAlchemyBaseOAuthAccountTable[str], Base):
    __tablename__ = "oauth_account"

    id = Column(String(24), primary_key=True, default=lambda: "oauth" + _random_id())
    user_id: Mapped[str] = mapped_column(String, ForeignKey("user.id", ondelete="cascade"), nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="oauth_accounts")


__all__ = [
    "OAuthAccount",
    "Role",
    "User",
]
