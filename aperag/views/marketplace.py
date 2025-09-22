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
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query

from aperag.db.models import User
from aperag.exceptions import (
    AlreadySubscribedError,
    AlreadySubscribedToBotError,
    BotNotFoundException,
    BotNotPublishedError,
    CollectionNotPublishedError,
    SelfBotSubscriptionError,
    SelfSubscriptionError,
)
from aperag.schema import view_models
from aperag.service.bot_marketplace_service import bot_marketplace_service
from aperag.service.marketplace_service import marketplace_service
from aperag.views.auth import optional_user, required_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["marketplace"])


@router.get("/marketplace/collections", response_model=view_models.SharedCollectionList)
async def list_marketplace_collections(
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
    user: User = Depends(optional_user),
) -> view_models.SharedCollectionList:
    """List all published Collections in marketplace"""
    try:
        # Allow unauthenticated access - use empty user_id for anonymous users
        user_id = user.id if user else ""
        result = await marketplace_service.list_published_collections(user_id, page, page_size)
        return result
    except Exception as e:
        logger.error(f"Error listing marketplace collections: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/marketplace/collections/subscriptions", response_model=view_models.SharedCollectionList)
async def list_user_subscribed_collections(
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
    user: User = Depends(required_user),
) -> view_models.SharedCollectionList:
    """Get user's subscribed Collections"""
    try:
        result = await marketplace_service.list_user_subscribed_collections(user.id, page, page_size)
        return result
    except Exception as e:
        logger.error(f"Error listing user subscribed collections: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/marketplace/collections/{collection_id}/subscribe", response_model=view_models.SharedCollection)
async def subscribe_collection(
    collection_id: str,
    user: User = Depends(required_user),
) -> view_models.SharedCollection:
    """Subscribe to a Collection"""
    try:
        result = await marketplace_service.subscribe_collection(user.id, collection_id)
        return result
    except CollectionNotPublishedError:
        raise HTTPException(status_code=400, detail="Collection is not published to marketplace")
    except SelfSubscriptionError:
        raise HTTPException(status_code=400, detail="Cannot subscribe to your own collection")
    except AlreadySubscribedError:
        raise HTTPException(status_code=409, detail="Already subscribed to this collection")
    except Exception as e:
        logger.error(f"Error subscribing to collection {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/marketplace/collections/{collection_id}/subscribe")
async def unsubscribe_collection(
    collection_id: str,
    user: User = Depends(required_user),
) -> Dict[str, Any]:
    """Unsubscribe from a Collection"""
    try:
        await marketplace_service.unsubscribe_collection(user.id, collection_id)
        return {"message": "Successfully unsubscribed"}
    except Exception as e:
        logger.error(f"Error unsubscribing from collection {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Bot marketplace endpoints
@router.get("/marketplace/bots", response_model=view_models.SharedBotList)
async def list_marketplace_bots(
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
    type: str = Query(None, description="Bot type filter"),
    user: User = Depends(optional_user),
) -> view_models.SharedBotList:
    """List all published bots in marketplace"""
    try:
        # Allow unauthenticated access - use empty user_id for anonymous users
        user_id = str(user.id) if user else ""
        result = await bot_marketplace_service.list_published_bots(user_id, page, page_size, type)
        return result
    except Exception as e:
        logger.error(f"Error listing marketplace bots: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/marketplace/bots/subscriptions", response_model=view_models.SharedBotList)
async def list_user_subscribed_bots(
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
    user: User = Depends(required_user),
) -> view_models.SharedBotList:
    """Get user's subscribed bots"""
    try:
        result = await bot_marketplace_service.list_user_subscribed_bots(str(user.id), page, page_size)
        return result
    except Exception as e:
        logger.error(f"Error listing user subscribed bots: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/marketplace/bots/{bot_id}/subscribe", response_model=view_models.SharedBot)
async def subscribe_bot(
    bot_id: str,
    user: User = Depends(required_user),
) -> view_models.SharedBot:
    """Subscribe to a bot"""
    try:
        result = await bot_marketplace_service.subscribe_bot(str(user.id), bot_id)
        return result
    except BotNotPublishedError:
        raise HTTPException(status_code=400, detail="Bot is not published to marketplace")
    except SelfBotSubscriptionError:
        raise HTTPException(status_code=400, detail="Cannot subscribe to your own bot")
    except AlreadySubscribedToBotError:
        raise HTTPException(status_code=409, detail="Already subscribed to this bot")
    except BotNotFoundException:
        raise HTTPException(status_code=404, detail="Bot not found")
    except Exception as e:
        logger.error(f"Error subscribing to bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/marketplace/bots/{bot_id}/subscribe")
async def unsubscribe_bot(
    bot_id: str,
    user: User = Depends(required_user),
) -> Dict[str, Any]:
    """Unsubscribe from a bot"""
    try:
        await bot_marketplace_service.unsubscribe_bot(str(user.id), bot_id)
        return {"message": "Successfully unsubscribed"}
    except Exception as e:
        logger.error(f"Error unsubscribing from bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/marketplace/bots/{bot_id}", response_model=view_models.SharedBot)
async def get_marketplace_bot(
    bot_id: str,
    user: User = Depends(required_user),
) -> view_models.SharedBot:
    """Get marketplace bot details (read-only access)"""
    try:
        # Check if user has subscription or is owner
        subscription = await bot_marketplace_service.get_user_subscription(str(user.id), bot_id)
        if subscription is None:
            # Check if user is the owner
            from aperag.service.bot_service import bot_service
            try:
                bot = await bot_service.get_bot(str(user.id), bot_id)
                if bot.user != str(user.id):
                    raise HTTPException(status_code=403, detail="Access denied - subscription required")
            except:
                raise HTTPException(status_code=403, detail="Access denied - subscription required")
        
        # Get bot details from marketplace
        bots_data, _ = await bot_marketplace_service.db_ops.list_published_bots_with_subscription_status(
            user_id=str(user.id), page=1, page_size=1
        )
        
        if not bots_data:
            raise HTTPException(status_code=404, detail="Bot not found in marketplace")
            
        # Find the specific bot
        bot_data = None
        for data in bots_data:
            if data["id"] == bot_id:
                bot_data = data
                break
                
        if not bot_data:
            raise HTTPException(status_code=404, detail="Bot not found in marketplace")
            
        # Convert to SharedBot
        from aperag.schema.utils import parseBotConfig, convertToSharedBotConfig
        bot_config = parseBotConfig(bot_data["config"])
        shared_config = convertToSharedBotConfig(bot_config)
        
        return view_models.SharedBot(
            id=bot_data["id"],
            title=bot_data["title"],
            description=bot_data["description"],
            type=bot_data["type"],
            owner_user_id=bot_data["owner_user_id"],
            owner_username=bot_data["owner_username"],
            subscription_id=bot_data["subscription_id"],
            gmt_subscribed=bot_data["gmt_subscribed"],
            config=shared_config,
            is_subscribed=bot_data["subscription_id"] is not None,
            is_owner=bot_data["owner_user_id"] == str(user.id),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting marketplace bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
