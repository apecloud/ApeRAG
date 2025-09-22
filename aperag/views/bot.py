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

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from aperag.db.models import User
from aperag.exceptions import (
    BotNotFoundException,
    BotNotPublishedError,
    PermissionDeniedError,
)
from aperag.schema import view_models
from aperag.service.bot_marketplace_service import bot_marketplace_service
from aperag.service.bot_service import bot_service
from aperag.service.flow_service import flow_service_global
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bots"])


@router.post("/bots")
@audit(resource_type="bot", api_name="CreateBot")
async def create_bot_view(
    request: Request,
    bot_in: view_models.BotCreate,
    user: User = Depends(required_user),
) -> view_models.Bot:
    return await bot_service.create_bot(str(user.id), bot_in)


@router.get("/bots")
async def list_bots_view(request: Request, user: User = Depends(required_user)) -> view_models.BotList:
    return await bot_service.list_bots(str(user.id))


@router.get("/bots/{bot_id}")
async def get_bot_view(request: Request, bot_id: str, user: User = Depends(required_user)) -> view_models.Bot:
    return await bot_service.get_bot(str(user.id), bot_id)


@router.put("/bots/{bot_id}")
@audit(resource_type="bot", api_name="UpdateBot")
async def update_bot_view(
    request: Request,
    bot_id: str,
    bot_in: view_models.BotUpdate,
    user: User = Depends(required_user),
) -> view_models.Bot:
    return await bot_service.update_bot(str(user.id), bot_id, bot_in)


@router.delete("/bots/{bot_id}")
@audit(resource_type="bot", api_name="DeleteBot")
async def delete_bot_view(request: Request, bot_id: str, user: User = Depends(required_user)):
    await bot_service.delete_bot(str(user.id), bot_id)
    return Response(status_code=204)


@router.post("/bots/{bot_id}/flow/debug", tags=["flows"])
async def debug_flow_stream_view(
    request: Request,
    bot_id: str,
    debug: view_models.DebugFlowRequest,
    user: User = Depends(required_user),
):
    return await flow_service_global.debug_flow_stream(str(user.id), bot_id, debug)


# Bot sharing endpoints
@router.get("/bots/{bot_id}/sharing", tags=["bots"])
async def get_bot_sharing_status(
    bot_id: str,
    user: User = Depends(required_user),
) -> view_models.SharingStatusResponse:
    """Get bot sharing status (owner only)"""
    try:
        is_published, published_at = await bot_marketplace_service.get_sharing_status(str(user.id), bot_id)
        return view_models.SharingStatusResponse(is_published=is_published, published_at=published_at)
    except BotNotFoundException:
        raise HTTPException(status_code=404, detail="Bot not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Error getting bot sharing status {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/bots/{bot_id}/sharing", tags=["bots"])
async def publish_bot_to_marketplace(
    bot_id: str,
    user: User = Depends(required_user),
):
    """Publish bot to marketplace (owner only)"""
    try:
        await bot_marketplace_service.publish_bot(str(user.id), bot_id)
        return Response(status_code=204)
    except BotNotFoundException:
        raise HTTPException(status_code=404, detail="Bot not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Error publishing bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/bots/{bot_id}/sharing", tags=["bots"])
async def unpublish_bot_from_marketplace(
    bot_id: str,
    user: User = Depends(required_user),
):
    """Unpublish bot from marketplace (owner only)"""
    try:
        await bot_marketplace_service.unpublish_bot(str(user.id), bot_id)
        return Response(status_code=204)
    except BotNotFoundException:
        raise HTTPException(status_code=404, detail="Bot not found")
    except BotNotPublishedError:
        raise HTTPException(status_code=400, detail="Bot is not published to marketplace")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Error unpublishing bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
