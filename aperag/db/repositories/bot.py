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

from typing import List, Optional

from sqlalchemy import desc, select

from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.domains.conversation.db.models import (
    Bot,
    BotStatus,
)
from aperag.utils.utils import utc_now


class AsyncBotRepositoryMixin(AsyncRepositoryProtocol):
    async def query_bot(self, user: str, bot_id: str, *, exclude_system: bool = True):
        # ``exclude_system`` defaults to True so user-facing read paths
        # never accidentally surface the Wave 10 §K.13 hidden summary
        # bot. Internal regen plumbing fetches its system bot via the
        # dedicated ``get_or_create_summary_bot_for_user`` helper which
        # talks directly to the ``Bot`` ORM, bypassing this method.
        async def _query(session):
            conds = [Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED]
            if exclude_system:
                conds.append(Bot.is_system.is_(False))
            stmt = select(Bot).where(*conds)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_bots(self, users: List[str], *, exclude_system: bool = True):
        # See ``query_bot`` — same default for the same reason.
        async def _query(session):
            conds = [Bot.user.in_(users), Bot.status != BotStatus.DELETED]
            if exclude_system:
                conds.append(Bot.is_system.is_(False))
            stmt = select(Bot).where(*conds).order_by(desc(Bot.gmt_created))
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_bots_count(self, user: str, *, exclude_system: bool = True):
        async def _query(session):
            from sqlalchemy import func

            conds = [Bot.user == user, Bot.status != BotStatus.DELETED]
            if exclude_system:
                conds.append(Bot.is_system.is_(False))
            stmt = select(func.count()).select_from(Bot).where(*conds)
            return await session.scalar(stmt)

        return await self._execute_query(_query)

    # Bot Operations
    async def create_bot(self, user: str, title: str, description: str, bot_type, config: str = "{}") -> Bot:
        """Create a new bot in database"""

        async def _operation(session):
            instance = Bot(
                user=user,
                title=title,
                type=bot_type,
                status=BotStatus.ACTIVE,
                description=description,
                config=config,
            )
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def update_bot_by_id(
        self, user: str, bot_id: str, title: str, description: str, bot_type, config: str
    ) -> Optional[Bot]:
        """Update bot by ID"""

        async def _operation(session):
            stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                instance.title = title
                instance.description = description
                instance.type = bot_type
                instance.config = config
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def update_bot_config_by_id(self, user: str, bot_id: str, config: str) -> Optional[Bot]:
        """Update bot config by ID without affecting other fields"""

        async def _operation(session):
            stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                instance.config = config
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def delete_bot_by_id(self, user: str, bot_id: str) -> Optional[Bot]:
        """Soft delete bot by ID"""

        async def _operation(session):
            stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                instance.status = BotStatus.DELETED
                instance.gmt_deleted = utc_now()
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)
