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

from typing import List

from sqlalchemy import desc, select

from aperag.db.models import SearchHistory
from aperag.db.repositories.base import AsyncRepositoryProtocol


class AsyncSearchRepositoryMixin(AsyncRepositoryProtocol):
    async def create_search(
        self,
        user: str,
        collection_id: str,
        query: str,
        vector_search: dict = None,
        fulltext_search: dict = None,
        graph_search: dict = None,
        summary_search: dict = None,
        items: List[dict] = None,
    ) -> SearchHistory:
        async def _operation(session):
            instance = SearchHistory(
                user=user,
                collection_id=collection_id,
                query=query,
                vector_search=vector_search,
                fulltext_search=fulltext_search,
                graph_search=graph_search,
                summary_search=summary_search,
                items=items,
            )
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def query_searches(self, user: str, collection_id: str) -> List[SearchHistory]:
        """Query searches by user and collection"""

        async def _query(session):
            stmt = (
                select(SearchHistory)
                .where(SearchHistory.user == user, SearchHistory.collection_id == collection_id)
                .order_by(desc(SearchHistory.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def delete_search(self, user: str, collection_id: str, search_id: str) -> bool:
        """Delete search by ID"""

        async def _operation(session):
            stmt = select(SearchHistory).where(
                SearchHistory.id == search_id,
                SearchHistory.user == user,
                SearchHistory.collection_id == collection_id,
            )
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                await session.delete(instance)
                await session.flush()
                return True
            return False

        return await self.execute_with_transaction(_operation)
