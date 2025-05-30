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

from ninja import Router

from aperag.db.models import ApiKey
from aperag.schema.view_models import ApiKey as ApiKeyModel
from aperag.schema.view_models import ApiKeyCreate, ApiKeyList, ApiKeyUpdate
from aperag.service.api_key_service import (
    create_api_key,
    delete_api_key,
    list_api_keys,
    update_api_key,
)
from aperag.utils.request import get_user
from aperag.views.utils import fail, success

router = Router()


def to_api_key_model(apikey: ApiKey) -> ApiKeyModel:
    """Convert database ApiKey model to API response model"""
    return success(
        ApiKeyModel(
            id=str(apikey.id),
            key=apikey.key,
            description=apikey.description,
            created_at=apikey.gmt_created,
            updated_at=apikey.gmt_updated,
            last_used_at=apikey.last_used_at,
        )
    )


@router.get("/apikeys")
async def list_api_keys_view(request) -> ApiKeyList:
    """List all API keys for the current user"""
    user = get_user(request)
    result = await list_api_keys(user)
    return success(result)


@router.post("/apikeys")
async def create_api_key_view(request, api_key_create: ApiKeyCreate):
    """Create a new API key"""
    user = get_user(request)
    result = await create_api_key(user, api_key_create)
    return success(result)


@router.delete("/apikeys/{apikey_id}")
async def delete_api_key_view(request, apikey_id: str):
    """Delete an API key"""
    user = get_user(request)
    ok = await delete_api_key(user, apikey_id)
    if not ok:
        return fail(HTTPStatus.NOT_FOUND, message="API key not found")
    return success({})


@router.put("/apikeys/{apikey_id}")
async def update_api_key_view(request, apikey_id: str, api_key_update: ApiKeyUpdate):
    """Update an API key"""
    user = get_user(request)
    result = await update_api_key(user, apikey_id, api_key_update)
    if not result:
        return fail(HTTPStatus.NOT_FOUND, message="API key not found")
    return success(result)
