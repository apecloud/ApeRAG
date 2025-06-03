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

from fastapi import APIRouter, Depends, Request

from aperag.db.models import ApiKey, User
from aperag.schema.view_models import ApiKey as ApiKeyModel
from aperag.schema.view_models import ApiKeyCreate, ApiKeyList, ApiKeyUpdate
from aperag.service.api_key_service import (
    create_api_key,
    delete_api_key,
    list_api_keys,
    update_api_key,
)
from aperag.views.auth import get_current_user_with_state
from aperag.views.utils import fail, success

router = APIRouter()


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
async def list_api_keys_view(request: Request, user: User = Depends(get_current_user_with_state)) -> ApiKeyList:
    """List all API keys for the current user"""
    result = await list_api_keys(str(user.id))
    return success(result)


@router.post("/apikeys")
async def create_api_key_view(
    request: Request, api_key_create: ApiKeyCreate, user: User = Depends(get_current_user_with_state)
):
    """Create a new API key"""
    result = await create_api_key(str(user.id), api_key_create)
    return success(result)


@router.delete("/apikeys/{apikey_id}")
async def delete_api_key_view(request: Request, apikey_id: str, user: User = Depends(get_current_user_with_state)):
    """Delete an API key"""
    ok = await delete_api_key(str(user.id), apikey_id)
    if not ok:
        return fail(HTTPStatus.NOT_FOUND, message="API key not found")
    return success({})


@router.put("/apikeys/{apikey_id}")
async def update_api_key_view(
    request: Request, apikey_id: str, api_key_update: ApiKeyUpdate, user: User = Depends(get_current_user_with_state)
):
    """Update an API key"""
    result = await update_api_key(str(user.id), apikey_id, api_key_update)
    if not result:
        return fail(HTTPStatus.NOT_FOUND, message="API key not found")
    return success(result)
