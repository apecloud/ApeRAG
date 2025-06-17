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

from typing import Optional

from sqlalchemy import select

from aperag.db.models import ApiKey, ApiKeyStatus
from aperag.db.repositories.base import AsyncRepositoryProtocol


class AsyncApiKeyRepositoryMixin(AsyncRepositoryProtocol):
    async def query_api_keys(self, user: str):
        """List all active API keys for a user"""

        async def _query(session):
            stmt = select(ApiKey).where(
                ApiKey.user == user, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def create_api_key(self, user: str, description: Optional[str] = None) -> ApiKey:
        """Create a new API key for a user"""

        async def _operation(session):
            api_key = ApiKey(user=user, description=description, status=ApiKeyStatus.ACTIVE)
            session.add(api_key)
            await session.flush()
            await session.refresh(api_key)
            return api_key

        return await self.execute_with_transaction(_operation)

    async def delete_api_key(self, user: str, key_id: str) -> bool:
        """Delete an API key (soft delete)"""

        async def _operation(session):
            stmt = select(ApiKey).where(
                ApiKey.id == key_id,
                ApiKey.user == user,
                ApiKey.status == ApiKeyStatus.ACTIVE,
                ApiKey.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            api_key = result.scalars().first()
            if not api_key:
                return None

            from datetime import datetime as dt

            api_key.status = ApiKeyStatus.DELETED
            api_key.gmt_deleted = dt.utcnow()
            session.add(api_key)
            await session.flush()
            return api_key

        return await self.execute_with_transaction(_operation)

    async def get_api_key_by_id(self, user: str, id: str) -> Optional[ApiKey]:
        """Get API key by id string"""

        async def _query(session):
            stmt = select(ApiKey).where(
                ApiKey.user == user, ApiKey.id == id, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def get_api_key_by_key(self, key: str) -> Optional[ApiKey]:
        """Get API key by key string"""

        async def _query(session):
            stmt = select(ApiKey).where(
                ApiKey.key == key, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def update_api_key_by_id(self, user: str, key_id: str, description: str) -> Optional[ApiKey]:
        """Update API key description"""

        async def _operation(session):
            stmt = select(ApiKey).where(
                ApiKey.user == user,
                ApiKey.id == key_id,
                ApiKey.status == ApiKeyStatus.ACTIVE,
                ApiKey.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            api_key = result.scalars().first()

            if api_key:
                api_key.description = description
                session.add(api_key)
                await session.flush()
                await session.refresh(api_key)

            return api_key

        return await self.execute_with_transaction(_operation)
