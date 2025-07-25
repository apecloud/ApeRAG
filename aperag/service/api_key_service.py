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

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import ApiKey
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import ResourceNotFoundException
from aperag.schema.view_models import ApiKey as ApiKeyModel
from aperag.schema.view_models import ApiKeyCreate, ApiKeyList, ApiKeyUpdate


class ApiKeyService:
    """API Key service that handles business logic for API keys"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    # Convert database ApiKey model to API response model
    def to_api_key_model(self, apikey: ApiKey) -> ApiKeyModel:
        return ApiKeyModel(
            id=str(apikey.id),
            key=apikey.key,
            description=apikey.description,
            created_at=apikey.gmt_created,
            updated_at=apikey.gmt_updated,
            last_used_at=apikey.last_used_at,
        )

    async def list_api_keys(self, user: str) -> ApiKeyList:
        """List all API keys for the current user"""
        tokens = await self.db_ops.query_api_keys(user, is_system=False)
        items = []
        for token in tokens:
            items.append(self.to_api_key_model(token))
        return ApiKeyList(items=items)

    async def create_api_key(self, user: str, api_key_create: ApiKeyCreate) -> ApiKeyModel:
        """Create a new API key"""
        # For single operations, use DatabaseOps directly
        token = await self.db_ops.create_api_key(user, api_key_create.description, is_system=False)
        return self.to_api_key_model(token)

    async def delete_api_key(self, user: str, apikey_id: str) -> Optional[bool]:
        """Delete an API key (idempotent operation)

        Returns True if deleted, None if already deleted/not found
        """
        # Check if API key exists - if not, silently succeed (idempotent)
        existing_keys = await self.db_ops.query_api_keys(user, is_system=False)
        key_exists = any(str(key.id) == apikey_id for key in existing_keys)

        if not key_exists:
            return None  # Idempotent operation, not found is success

        # For single operations, use DatabaseOps directly
        result = await self.db_ops.delete_api_key(user, apikey_id)
        return result

    async def update_api_key(self, user: str, apikey_id: str, api_key_update: ApiKeyUpdate) -> Optional[ApiKeyModel]:
        """Update an API key"""
        # For single operations, use DatabaseOps directly
        updated_key = await self.db_ops.update_api_key_by_id(user, apikey_id, api_key_update.description)
        if not updated_key:
            raise ResourceNotFoundException("API key", apikey_id)
        return self.to_api_key_model(updated_key)


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
api_key_service = ApiKeyService()
