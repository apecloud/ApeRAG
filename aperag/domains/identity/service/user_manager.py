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

"""Identity-domain ``UserManager`` (Phase 4 Step 4-S7).

Moved from ``aperag/views/auth.py``. ``on_after_register`` is the
hook fastapi-users invokes after a user row is committed (both
email-password flow and OAuth callback paths go through it). It
triggers three user-initialization side effects that live in other
domains:

* default bot creation (``BotInitOps``)
* default chat-collection creation (``ChatInitOps``)
* per-user quota row seeding (``QuotaInitOps``)

Phase 4 canonical ``msg=d47fa490`` Q4: the identity domain owns
those three Protocols (``aperag.domains.identity.ports``) and the
concrete providers (today: legacy ``bot_service`` /
``chat_collection_service`` / ``quota_service``; Phase 5 moves them
into the conversation domain) structurally satisfy them. This module
exposes module-level DI setters that ``aperag/app.py`` startup wires
before the first registration request arrives.

System API-key seeding and the first-user-is-admin heuristic stay in
this module — they only need ``aperag.db.ops`` + ``utc_now`` + the
identity ``Role`` enum and therefore don't cross domain boundaries.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from fastapi_users import BaseUserManager

from aperag.config import settings
from aperag.db.ops import async_db_ops
from aperag.domains.identity.db.models import Role, User
from aperag.domains.identity.ports import BotInitOps, ChatInitOps, QuotaInitOps

logger = logging.getLogger(__name__)


# ---------- Consumer-owned Protocol DI setters (lesson 9a-quad) ---------- #
#
# Module-level singletons wired by ``aperag/app.py`` startup with
# concrete adapters. The wire-up happens once at module import time
# (before FastAPI begins serving requests), so ``on_after_register``
# always sees non-``None`` instances. A fresh registration that fires
# the hook without DI wired raises ``RuntimeError`` with an actionable
# message rather than falling through to ``None.method(...)``.

_bot_init_ops: Optional[BotInitOps] = None
_chat_init_ops: Optional[ChatInitOps] = None
_quota_init_ops: Optional[QuotaInitOps] = None


def set_bot_init_ops(ops: BotInitOps) -> None:
    global _bot_init_ops
    _bot_init_ops = ops


def set_chat_init_ops(ops: ChatInitOps) -> None:
    global _chat_init_ops
    _chat_init_ops = ops


def set_quota_init_ops(ops: QuotaInitOps) -> None:
    global _quota_init_ops
    _quota_init_ops = ops


def _get_bot_init_ops() -> BotInitOps:
    if _bot_init_ops is None:
        raise RuntimeError("identity.user_manager: bot_init_ops not wired. Call set_bot_init_ops() at app startup.")
    return _bot_init_ops


def _get_chat_init_ops() -> ChatInitOps:
    if _chat_init_ops is None:
        raise RuntimeError("identity.user_manager: chat_init_ops not wired. Call set_chat_init_ops() at app startup.")
    return _chat_init_ops


def _get_quota_init_ops() -> QuotaInitOps:
    if _quota_init_ops is None:
        raise RuntimeError("identity.user_manager: quota_init_ops not wired. Call set_quota_init_ops() at app startup.")
    return _quota_init_ops


class UserManager(BaseUserManager[User, str]):
    reset_password_token_secret = settings.jwt_secret
    verification_token_secret = settings.jwt_secret

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        """Set the first registered user as admin and seed per-user
        resources. Works for both email-password registration and
        OAuth callback (fastapi-users invokes the same hook).
        """
        user_count = await async_db_ops.query_user_count()
        if user_count == 1 and user.role != Role.ADMIN:
            user.role = Role.ADMIN
            self.user_db.session.add(user)
            await self.user_db.session.commit()
            await self.user_db.session.refresh(user)

        # For GitHub OAuth users, fetch username from GitHub API
        # await self._fetch_github_username_if_needed(user)

        # Initialize user resources for all new users (including OAuth users).
        # All three side effects reach across domain boundaries through
        # consumer-owned Protocols wired at app startup.
        try:
            await _get_quota_init_ops().initialize_user_quota(str(user.id))

            # System + default API keys are identity-internal, direct db_ops call.
            await async_db_ops.create_api_key(user=str(user.id), description="system", is_system=True)
            await async_db_ops.create_api_key(user=str(user.id), description="default", is_system=False)

            await _get_bot_init_ops().create_default_bot_for_user(str(user.id))
            await _get_chat_init_ops().create_default_chat_for_user(str(user.id))

            logger.info(f"Initialized resources for user {user.username or user.email} ({user.id})")
        except Exception as e:
            logger.error(f"Failed to initialize resources for user {user.username or user.email} ({user.id}): {e}")

    async def _fetch_github_username_if_needed(self, user: User):
        """
        For GitHub OAuth users, fetch username from GitHub API using account_id
        """
        try:
            # Check if user has GitHub OAuth account
            github_oauth_account = None
            for oauth_account in user.oauth_accounts:
                if oauth_account.oauth_name == "github":
                    github_oauth_account = oauth_account
                    break

            if not github_oauth_account:
                return  # Not a GitHub OAuth user

            if user.username:
                return  # Username already set

            # Fetch username from GitHub API
            import httpx

            github_user_id = github_oauth_account.account_id
            github_api_url = f"https://api.github.com/user/{github_user_id}"

            async with httpx.AsyncClient() as client:
                response = await client.get(github_api_url)
                if response.status_code == 200:
                    github_user_data = response.json()
                    github_username = github_user_data.get("login")

                    if github_username:
                        user.username = github_username
                        self.user_db.session.add(user)
                        await self.user_db.session.commit()
                        await self.user_db.session.refresh(user)
                        logger.info(f"Updated GitHub user {user.id} with username: {github_username}")
                else:
                    logger.warning(f"Failed to fetch GitHub user data for user {user.id}: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to fetch GitHub username for user {user.id}: {e}")

    def parse_id(self, value: any) -> str:
        """Parse ID from any type to str"""
        if isinstance(value, str):
            return value
        return str(value)


__all__ = [
    "UserManager",
    "set_bot_init_ops",
    "set_chat_init_ops",
    "set_quota_init_ops",
]
