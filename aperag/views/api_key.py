from typing import Optional

from django.http import HttpRequest
from django.utils import timezone
from ninja import Router
from ninja.security import HttpBearer

from aperag.db.models import ApiKey
from aperag.db.ops import (
    create_api_key,
    delete_api_key,
    get_api_key_by_id,
    list_user_api_keys,
)
from aperag.schema.view_models import ApiKey as ApiKeyModel
from aperag.schema.view_models import ApiKeyList, PageResult, ApiKeyCreate, ApiKeyUpdate
from aperag.views.utils import success, fail
from http import HTTPStatus
from aperag.utils.request import get_user
from aperag.service.api_key_service import (
    list_api_keys,
    create_api_key,
    delete_api_key,
    update_api_key,
)

router = Router()

def to_api_key_model(apikey: ApiKey) -> ApiKeyModel:
    """Convert database ApiKey model to API response model"""
    return success(ApiKeyModel(
        id=str(apikey.id),
        key=apikey.key,
        description=apikey.description,
        created_at=apikey.gmt_created,
        updated_at=apikey.gmt_updated,
        last_used_at=apikey.last_used_at
    ))

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
