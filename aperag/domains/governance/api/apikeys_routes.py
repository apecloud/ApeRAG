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

"""Governance API key routes.

Carved out of ``aperag/domains/governance/api/routes.py`` in Phase 8
task #50 (G4a) — the legacy combined ``governance_router`` (which
mounted both API keys and audit logs) is split into two single-purpose
routers so that #50 (audit-logs) and #51 (apikeys) can migrate to
``/api/v2`` independently per architect canonical D7-2 (msg=25a30445).

Phase 8 #50 lands this file with no URL change — apikeys stays at
``/api/v1/apikeys`` until #51 G4b flips the mount to ``/api/v2``.
"""

from fastapi import APIRouter, Depends, Request

from aperag.domains.governance.ports import AuthenticatedUser
from aperag.domains.governance.schemas import ApiKeyCreate, ApiKeyList, ApiKeyUpdate
from aperag.domains.governance.service.api_key_service import api_key_service
from aperag.domains.identity.service.auth_dependencies import required_user
from aperag.utils.audit_decorator import audit

router = APIRouter()


@router.get("/apikeys", tags=["api_keys"])
async def list_api_keys_view(request: Request, user: AuthenticatedUser = Depends(required_user)) -> ApiKeyList:
    """List all API keys for the current user"""
    return await api_key_service.list_api_keys(str(user.id))


@router.post("/apikeys", tags=["api_keys"])
@audit(resource_type="api_key", api_name="CreateApiKey")
async def create_api_key_view(
    request: Request,
    api_key_create: ApiKeyCreate,
    user: AuthenticatedUser = Depends(required_user),
):
    """Create a new API key"""
    return await api_key_service.create_api_key(str(user.id), api_key_create)


@router.delete("/apikeys/{apikey_id}", tags=["api_keys"])
@audit(resource_type="api_key", api_name="DeleteApiKey")
async def delete_api_key_view(request: Request, apikey_id: str, user: AuthenticatedUser = Depends(required_user)):
    """Delete an API key"""
    return await api_key_service.delete_api_key(str(user.id), apikey_id)


@router.put("/apikeys/{apikey_id}", tags=["api_keys"])
@audit(resource_type="api_key", api_name="UpdateApiKey")
async def update_api_key_view(
    request: Request,
    apikey_id: str,
    api_key_update: ApiKeyUpdate,
    user: AuthenticatedUser = Depends(required_user),
):
    """Update an API key"""
    return await api_key_service.update_api_key(str(user.id), apikey_id, api_key_update)
