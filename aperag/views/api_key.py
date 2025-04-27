from typing import Optional

from django.http import HttpRequest
from django.utils import timezone
from ninja import Router
from ninja.security import HttpBearer

from aperag.db.models import ApiKey
from aperag.db.ops import (
    create_api_key,
    delete_api_key,
    get_api_key_by_token,
    list_user_api_keys,
)
from aperag.views.models import ApiKey as ApiKeyModel
from aperag.views.models import ApiKeyList, PageResult, ApiKeyCreate, ApiKeyUpdate
from aperag.views.utils import success, fail
from http import HTTPStatus

router = Router()

def to_api_key_model(apikey: ApiKey) -> ApiKeyModel:
    """Convert database ApiKey model to API response model"""
    return ApiKeyModel(
        id=str(apikey.id),
        key=apikey.key,
        description=apikey.description,
        created_at=apikey.gmt_created,
        updated_at=apikey.gmt_updated,
        last_used_at=apikey.last_used_at
    )

@router.get("/apikeys")
async def list_api_keys(request) -> ApiKeyList:
    """List all API keys for the current user"""
    tokens = await list_user_api_keys(request.user.username)
    items = [to_api_key_model(token) for token in tokens]
    return ApiKeyList(
        items=items,
        pageResult=PageResult(
            page_number=1,
            page_size=len(items),
            count=len(items)
        )
    )

@router.post("/apikeys")
async def create_api_key_view(request, api_key_create: ApiKeyCreate) -> ApiKeyModel:
    """Create a new API key"""
    token = await create_api_key(request.user.username, api_key_create.description)
    return to_api_key_model(token)

@router.delete("/apikeys/{apikey_id}")
async def delete_api_key_view(request, apikey_id: str):
    """Delete an API key"""
    api_key = await get_api_key_by_token(apikey_id)
    if not api_key:
        return fail(HTTPStatus.NOT_FOUND, message="API key not found")

    await delete_api_key(request.user.username, apikey_id)
    return success({})

@router.put("/apikeys/{apikey_id}")
async def update_api_key_view(request, apikey_id: str, api_key_update: ApiKeyUpdate):
    """Update an API key"""
    api_key = await get_api_key_by_token(apikey_id)
    if not api_key:
        return fail(HTTPStatus.NOT_FOUND, message="API key not found")
    
    api_key.description = api_key_update.description
    await api_key.asave()
    return to_api_key_model(api_key)


class ApiKeyAuth(HttpBearer):
    async def authenticate(self, request: HttpRequest, token: str) -> Optional[ApiKey]:
        """Authenticate request using API key"""
        api_key = await get_api_key_by_token(token)
        if api_key:
            request.user = type('User', (), {'username': api_key.user})()
        return api_key 