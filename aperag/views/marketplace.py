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
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from aperag.db.models import User
from aperag.exceptions import (
    AlreadySubscribedError,
    CollectionNotPublishedError,
    SelfSubscriptionError,
)
from aperag.schema import view_models
from aperag.service.marketplace_service import marketplace_service
from aperag.views.auth import current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["marketplace"])


@router.get("/marketplace/collections", response_model=view_models.SharedCollectionList)
async def list_marketplace_collections(
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
    user: Optional[User] = Depends(current_user),
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
    user: User = Depends(current_user),
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
    user: User = Depends(current_user),
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
    user: User = Depends(current_user),
) -> Dict[str, Any]:
    """Unsubscribe from a Collection"""
    try:
        await marketplace_service.unsubscribe_collection(user.id, collection_id)
        return {"message": "Successfully unsubscribed"}
    except Exception as e:
        logger.error(f"Error unsubscribing from collection {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
