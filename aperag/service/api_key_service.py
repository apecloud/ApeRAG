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

from django.core.cache import cache

from aperag.auth.authentication import build_api_key_cache_key
from aperag.db import ops
from aperag.db.models import ApiKey
from aperag.schema.view_models import ApiKey as ApiKeyModel
from aperag.schema.view_models import ApiKeyCreate, ApiKeyList, ApiKeyUpdate


# Convert database ApiKey model to API response model
def to_api_key_model(apikey: ApiKey) -> ApiKeyModel:
    return ApiKeyModel(
        id=str(apikey.id),
        key=apikey.key,
        description=apikey.description,
        created_at=apikey.gmt_created,
        updated_at=apikey.gmt_updated,
        last_used_at=apikey.last_used_at,
    )


async def list_api_keys(user) -> ApiKeyList:
    """
    List all API keys for the current user
    """
    tokens = await ops.list_user_api_keys(user)
    items = []
    async for token in tokens:
        items.append(to_api_key_model(token))
    return ApiKeyList(items=items)


async def create_api_key(user, api_key_create: ApiKeyCreate) -> ApiKeyModel:
    """
    Create a new API key
    """
    token = await ops.create_api_key(user, api_key_create.description)
    return to_api_key_model(token)


async def delete_api_key(user, apikey_id: str):
    """
    Delete an API key
    """
    api_key = await ops.get_api_key_by_id(user, apikey_id)
    if not api_key:
        return None
    await ops.delete_api_key(user, apikey_id)
    cache.delete(build_api_key_cache_key(api_key.key))
    return True


async def update_api_key(user, apikey_id: str, api_key_update: ApiKeyUpdate) -> Optional[ApiKeyModel]:
    """
    Update an API key
    """
    api_key = await ops.get_api_key_by_id(user, apikey_id)
    if not api_key:
        return None
    api_key.description = api_key_update.description
    await api_key.asave()
    return to_api_key_model(api_key)
