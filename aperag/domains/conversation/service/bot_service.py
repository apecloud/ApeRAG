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

"""Bot service moved to the ``conversation`` domain in Phase 5 step 5-S4c.

Quota dependency rewiring — canonical Q1 ruling C
(``msg=4a93c97e``): ``aperag/service/quota_service.py`` stays as the
legacy provider through Phase 6 cleanup. ``bot_service`` must not
import it directly (G1 bans ``aperag.service.*``), so we introduce a
consumer-owned ``QuotaOps`` Protocol DI slot and let ``aperag/app.py``
wire the legacy singleton in at startup. The Protocol matches the
surface ``bot_service`` actually uses —
``check_and_consume_quota(user, "max_bot_count", 1, session)`` on
create and ``release_quota(user, "max_bot_count", 1, session)`` on
delete — and is *narrower* than the knowledge_base domain's
``QuotaOps`` (which also needs ``get_user_quotas``). Both Protocols
live independently, and the legacy ``quota_service`` structurally
satisfies both.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.domains.conversation.db.models import Bot as BotRow
from aperag.domains.conversation.db.models import BotStatus, BotType
from aperag.domains.conversation.ports import QuotaOps
from aperag.domains.conversation.schemas import Bot, BotConfig, BotCreate, BotList, BotUpdate
from aperag.exceptions import ResourceNotFoundException, ValidationException
from aperag.utils.utils import utc_now

_quota_ops: Optional[QuotaOps] = None


def set_quota_ops(ops: QuotaOps) -> None:
    """Wire the ``QuotaOps`` singleton at app startup.

    ``aperag/app.py`` calls this once with a legacy-adapter instance
    (``aperag.service.quota_service.quota_service`` satisfies the
    Protocol structurally, so the adapter is a trivial passthrough
    until Phase 6 cleanup removes the shim).
    """
    global _quota_ops
    _quota_ops = ops


def _get_quota_ops() -> QuotaOps:
    if _quota_ops is None:
        raise RuntimeError(
            "QuotaOps not wired — call aperag.domains.conversation.service.bot_service.set_quota_ops()"
            " in aperag/app.py startup before serving requests."
        )
    return _quota_ops


class BotService:
    """Bot service that handles business logic for bots"""

    def __init__(self, session: AsyncSession = None):
        if session is None:
            self.db_ops = async_db_ops
        else:
            self.db_ops = AsyncDatabaseOps(session)

    async def build_bot_response(self, bot: BotRow) -> Bot:
        """Build Bot response object for API return."""
        bot_config = None
        if bot.config:
            config_dict = json.loads(bot.config)
            bot_config = BotConfig(**config_dict)

        return Bot(
            id=bot.id,
            title=bot.title,
            description=bot.description,
            type=bot.type,
            config=bot_config,
            created=bot.gmt_created.isoformat(),
            updated=bot.gmt_updated.isoformat(),
        )

    async def validate_collections(self, user: str, bot_config: BotConfig):
        if bot_config and bot_config.agent and bot_config.agent.collections:
            collection_ids = [collection.id for collection in bot_config.agent.collections]
            collections = await self.db_ops.query_collections_by_ids(user, collection_ids)
            if not collections or len(collections) != len(collection_ids):
                raise ResourceNotFoundException("Collection", collection_ids)

    async def validate_agent_model(self, user: str, bot_config: BotConfig):
        completion = bot_config.agent.completion if bot_config and bot_config.agent else None
        if not completion or not completion.model_id:
            return
        from aperag.domains.model_platform.schemas import ModelUseScenario
        from aperag.domains.model_platform.service.model_service import model_platform_service

        await model_platform_service.ensure_model_allowed_for_scenario(
            user, completion.model_id, ModelUseScenario.AGENT_CHAT
        )

    async def create_bot(self, user: str, bot_in: BotCreate, skip_quota_check: bool = False) -> Bot:
        bot_type = bot_in.type or BotType.AGENT
        if bot_type != BotType.AGENT:
            raise ValidationException("Only agent bots are supported")

        async def _create_bot_atomically(session):
            if not skip_quota_check:
                await _get_quota_ops().check_and_consume_quota(user, "max_bot_count", 1, session)

            await self.validate_collections(user, bot_in.config)
            await self.validate_agent_model(user, bot_in.config)

            config_str = "{}"
            if bot_in.config:
                config_str = json.dumps(bot_in.config.model_dump(exclude_none=True))

            bot = BotRow(
                user=user,
                title=bot_in.title,
                type=bot_type,
                status=BotStatus.ACTIVE,
                description=bot_in.description,
                config=config_str,
            )
            session.add(bot)
            await session.flush()
            await session.refresh(bot)

            return bot

        bot = await self.db_ops.execute_with_transaction(_create_bot_atomically)
        return await self.build_bot_response(bot)

    async def list_bots(self, user: str) -> BotList:
        # Wave 10 §K.13 — ``exclude_system`` defaults to True at the
        # ``db_ops`` layer so system bots (Wave 10 hidden per-user
        # summary bot) never reach the response builder. Filtering at
        # the DB layer (vs. service-layer post-filter) keeps the query
        # narrow as the per-user bot count grows.
        bots = await self.db_ops.query_bots([user])
        return BotList(items=[await self.build_bot_response(bot) for bot in bots])

    async def get_bot(self, user: str, bot_id: str) -> Bot:
        # ``query_bot`` defaults to ``exclude_system=True`` so a system
        # bot returns None here — caller sees a 404, not the bot.
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            raise ResourceNotFoundException("Bot", bot_id)

        return await self.build_bot_response(bot)

    async def update_bot(self, user: str, bot_id: str, bot_in: BotUpdate) -> Bot:
        # ``query_bot`` defaults to excluding system bots, so a user
        # can never update / mutate the hidden summary bot via this
        # path; they get a 404 instead.
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            raise ResourceNotFoundException("Bot", bot_id)

        new_config_str = None
        if bot_in.config:
            new_config_str = json.dumps(bot_in.config.model_dump(exclude_none=True))

        await self.validate_collections(user, bot_in.config)
        await self.validate_agent_model(user, bot_in.config)

        async def _update_bot_atomically(session):
            from sqlalchemy import select

            stmt = select(BotRow).where(BotRow.id == bot_id, BotRow.user == user, BotRow.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            bot_to_update = result.scalars().first()

            if not bot_to_update:
                raise ResourceNotFoundException("Bot", bot_id)

            if bot_in.title is not None:
                bot_to_update.title = bot_in.title
            if bot_in.description is not None:
                bot_to_update.description = bot_in.description
            if new_config_str is not None:
                bot_to_update.config = new_config_str
            session.add(bot_to_update)
            await session.flush()
            await session.refresh(bot_to_update)

            return bot_to_update

        updated_bot = await self.db_ops.execute_with_transaction(_update_bot_atomically)
        return await self.build_bot_response(updated_bot)

    async def delete_bot(self, user: str, bot_id: str) -> Optional[Bot]:
        """Delete bot by ID (idempotent operation).

        Returns the deleted bot or None if already deleted/not found.
        """
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            return None

        async def _delete_bot_atomically(session):
            from sqlalchemy import select

            stmt = select(BotRow).where(BotRow.id == bot_id, BotRow.user == user, BotRow.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            bot_to_delete = result.scalars().first()

            if not bot_to_delete:
                return None

            bot_to_delete.status = BotStatus.DELETED
            bot_to_delete.gmt_deleted = utc_now()
            session.add(bot_to_delete)
            await session.flush()
            await session.refresh(bot_to_delete)

            # Release quota within the transaction (only for non-system bots).
            if bot_to_delete.title != "Default Agent Bot":
                await _get_quota_ops().release_quota(user, "max_bot_count", 1, session)

            return bot_to_delete

        deleted_bot = await self.db_ops.execute_with_transaction(_delete_bot_atomically)

        if deleted_bot:
            return await self.build_bot_response(deleted_bot)

        return None


# Global service instance — wired by the legacy shim at
# ``aperag.service.bot_service`` at module load time so the
# pre-migration caller base (``views/bots_v2.py`` /
# ``domains/identity/service/user_manager.py`` adapter /
# ``evaluation_v2.worker``) keeps resolving the same singleton.
bot_service = BotService()
