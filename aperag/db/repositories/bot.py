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

from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc, select

from aperag.db.models import (
    Bot,
    BotStatus,
)
from aperag.db.repositories.base import AsyncRepositoryProtocol


class AsyncBotRepositoryMixin(AsyncRepositoryProtocol):
    async def query_bot(self, user: str, bot_id: str):
        async def _query(session):
            stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_bots(self, users: List[str]):
        async def _query(session):
            stmt = (
                select(Bot).where(Bot.user.in_(users), Bot.status != BotStatus.DELETED).order_by(desc(Bot.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_bots_count(self, user: str):
        async def _query(session):
            from sqlalchemy import func

            stmt = select(func.count()).select_from(Bot).where(Bot.user == user, Bot.status != BotStatus.DELETED)
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

    async def delete_bot_by_id(self, user: str, bot_id: str) -> Optional[Bot]:
        """Soft delete bot by ID"""

        async def _operation(session):
            stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                instance.status = BotStatus.DELETED
                instance.gmt_deleted = datetime.utcnow()
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def create_bot_collection_relation(self, bot_id: str, collection_id: str):
        """Create bot-collection relation"""
        from aperag.db.models import BotCollectionRelation

        async def _operation(session):
            relation = BotCollectionRelation(bot_id=bot_id, collection_id=collection_id)
            session.add(relation)
            await session.flush()
            return relation

        return await self.execute_with_transaction(_operation)

    async def delete_bot_collection_relations(self, bot_id: str):
        """Soft delete all bot-collection relations for a bot"""
        from aperag.db.models import BotCollectionRelation

        async def _operation(session):
            stmt = select(BotCollectionRelation).where(
                BotCollectionRelation.bot_id == bot_id, BotCollectionRelation.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            relations = result.scalars().all()
            for rel in relations:
                rel.gmt_deleted = datetime.utcnow()
                session.add(rel)
            await session.flush()
            return len(relations)

        return await self.execute_with_transaction(_operation)
