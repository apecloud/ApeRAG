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

from sqlalchemy import desc, func, select

from aperag.db.models import Collection, CollectionStatus
from aperag.db.repositories.base import (
    AsyncRepositoryProtocol,
    SyncRepositoryProtocol,
)
from aperag.utils.utils import utc_now


class CollectionRepositoryMixin(SyncRepositoryProtocol):
    def query_collection_by_id(self, collection_id: str, ignore_deleted: bool = True) -> Collection:
        def _query(session):
            stmt = select(Collection).where(Collection.id == collection_id)
            if ignore_deleted:
                stmt = stmt.where(Collection.status != CollectionStatus.DELETED)
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)

    def update_collection(self, collection: Collection):
        session = self._get_session()
        session.add(collection)
        session.commit()
        session.refresh(collection)
        return collection


class AsyncCollectionRepositoryMixin(AsyncRepositoryProtocol):
    # Collection Operations
    async def create_collection(
        self, user: str, title: str, description: str, collection_type, config: str = None
    ) -> Collection:
        """Create a new collection in database"""

        async def _operation(session):
            instance = Collection(
                user=user,
                type=collection_type,
                status=CollectionStatus.INACTIVE,
                title=title,
                description=description,
                config=config,
            )
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def update_collection_by_id(
        self, user: str, collection_id: str, title: str, description: str, config: str
    ) -> Optional[Collection]:
        """Update collection by ID"""

        async def _operation(session):
            stmt = select(Collection).where(
                Collection.id == collection_id, Collection.user == user, Collection.status != CollectionStatus.DELETED
            )
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                instance.title = title
                instance.description = description
                instance.config = config
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def delete_collection_by_id(self, user: str, collection_id: str) -> Optional[Collection]:
        """Soft delete collection by ID"""

        async def _operation(session):
            stmt = select(Collection).where(
                Collection.id == collection_id, Collection.user == user, Collection.status != CollectionStatus.DELETED
            )
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                # Check if collection has related bots
                collection_bots = await instance.bots(session, only_ids=True)
                if len(collection_bots) > 0:
                    raise ValueError(f"Collection has related to bots {','.join(collection_bots)}, can not be deleted")

                instance.status = CollectionStatus.DELETED
                instance.gmt_deleted = utc_now()
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def query_collection(self, user: str, collection_id: str):
        async def _query(session):
            stmt = select(Collection).where(
                Collection.id == collection_id, Collection.user == user, Collection.status != CollectionStatus.DELETED
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_collections(self, users: List[str]):
        async def _query(session):
            stmt = (
                select(Collection)
                .where(Collection.user.in_(users), Collection.status != CollectionStatus.DELETED)
                .order_by(desc(Collection.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_collections_count(self, user: str):
        async def _query(session):
            stmt = (
                select(func.count())
                .select_from(Collection)
                .where(Collection.user == user, Collection.status != CollectionStatus.DELETED)
            )
            return await session.scalar(stmt)

        return await self._execute_query(_query)

    async def query_collection_without_user(self, collection_id: str):
        async def _query(session):
            stmt = select(Collection).where(
                Collection.id == collection_id, Collection.status != CollectionStatus.DELETED
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_collections_by_ids(self, user: str, collection_ids: List[str]):
        """Query multiple collections by their IDs in a single database call"""

        async def _query(session):
            stmt = select(Collection).where(
                Collection.id.in_(collection_ids),
                Collection.user == user,
                Collection.status != CollectionStatus.DELETED,
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)
