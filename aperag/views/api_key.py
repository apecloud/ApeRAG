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


from fastapi import APIRouter, Depends, Request

from aperag.db.models import User
from aperag.schema.view_models import ApiKeyCreate, ApiKeyList, ApiKeyUpdate
from aperag.service.api_key_service import api_key_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import current_user

router = APIRouter()


@router.get("/apikeys", tags=["api_keys"])
async def list_api_keys_view(request: Request, user: User = Depends(current_user)) -> ApiKeyList:
    """List all API keys for the current user"""
    return await api_key_service.list_api_keys(str(user.id))


@router.post("/apikeys", tags=["api_keys"])
@audit(resource_type="api_key", api_name="CreateApiKey")
async def create_api_key_view(
    request: Request,
    api_key_create: ApiKeyCreate,
    user: User = Depends(current_user),
):
    """Create a new API key"""
    return await api_key_service.create_api_key(str(user.id), api_key_create)


@router.delete("/apikeys/{apikey_id}", tags=["api_keys"])
@audit(resource_type="api_key", api_name="DeleteApiKey")
async def delete_api_key_view(request: Request, apikey_id: str, user: User = Depends(current_user)):
    """Delete an API key"""
    return await api_key_service.delete_api_key(str(user.id), apikey_id)


@router.put("/apikeys/{apikey_id}", tags=["api_keys"])
@audit(resource_type="api_key", api_name="UpdateApiKey")
async def update_api_key_view(
    request: Request,
    apikey_id: str,
    api_key_update: ApiKeyUpdate,
    user: User = Depends(current_user),
):
    """Update an API key"""
    return await api_key_service.update_api_key(str(user.id), apikey_id, api_key_update)
