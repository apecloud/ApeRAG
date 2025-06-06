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

from http import HTTPStatus
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import ApiKey
from aperag.db.ops import DatabaseOps, db_ops
from aperag.schema.view_models import ApiKey as ApiKeyModel
from aperag.schema.view_models import ApiKeyCreate, ApiKeyList, ApiKeyUpdate
from aperag.views.utils import fail, success


class ApiKeyService:
    """API Key service that handles business logic for API keys"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = db_ops  # Use global instance
        else:
            self.db_ops = DatabaseOps(session)  # Create custom instance for transaction control

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
        tokens = await self.db_ops.query_api_keys(user)
        items = []
        for token in tokens:
            items.append(self.to_api_key_model(token))
        return success(ApiKeyList(items=items))

    async def create_api_key(self, user: str, api_key_create: ApiKeyCreate) -> ApiKeyModel:
        """Create a new API key"""
        try:
            # For single operations, use DatabaseOps directly
            token = await self.db_ops.create_api_key(user, api_key_create.description)
            return success(self.to_api_key_model(token))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to create API key: {str(e)}")

    async def delete_api_key(self, user: str, apikey_id: str):
        """Delete an API key"""
        # First check if API key exists
        api_key = await self.db_ops.get_api_key_by_id(user, apikey_id)
        if not api_key:
            return fail(HTTPStatus.NOT_FOUND, "API key not found")

        async def _delete_operation(session):
            # Use DatabaseOps to delete API key
            db_ops_session = DatabaseOps(session)
            deleted = await db_ops_session.delete_api_key(user, apikey_id)
            if not deleted:
                raise ValueError("API key not found")
            return {}

        try:
            result = await self.db_ops.execute_with_transaction(_delete_operation)
            return success(result)
        except ValueError as e:
            return fail(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to delete API key: {str(e)}")

    async def update_api_key(self, user: str, apikey_id: str, api_key_update: ApiKeyUpdate) -> Optional[ApiKeyModel]:
        """Update an API key"""
        # First check if API key exists
        api_key = await self.db_ops.get_api_key_by_id(user, apikey_id)
        if not api_key:
            return fail(HTTPStatus.NOT_FOUND, "API key not found")

        async def _update_operation(session):
            # Use DatabaseOps to update API key
            db_ops_session = DatabaseOps(session)
            updated_key = await db_ops_session.update_api_key_by_id(user, apikey_id, api_key_update.description)
            if not updated_key:
                raise ValueError("API key not found")
            return self.to_api_key_model(updated_key)

        try:
            result = await self.db_ops.execute_with_transaction(_update_operation)
            return success(result)
        except ValueError as e:
            return fail(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to update API key: {str(e)}")


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
api_key_service = ApiKeyService()
