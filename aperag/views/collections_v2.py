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

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from aperag.db.models import User
from aperag.exceptions import CollectionNotFoundException, PermissionDeniedError
from aperag.schema import view_models
from aperag.service.collection_service import collection_service
from aperag.service.collection_summary_service import collection_summary_service
from aperag.service.marketplace_service import marketplace_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["collections-v2"])


# Static-path routes must be declared before `/collections/{collection_id}` so FastAPI
# matches them first and provider/collection names cannot shadow them.


@router.post(
    "/collections/test-mineru-token",
    response_model=view_models.MineruTokenTestResponse,
)
async def test_mineru_token_view(
    body: view_models.MineruTokenTestRequest,
    user: User = Depends(required_user),
) -> view_models.MineruTokenTestResponse:
    """Probe an upstream MinerU API token and echo the status envelope."""

    result = await collection_service.test_mineru_token(body.token)
    return view_models.MineruTokenTestResponse(
        status_code=int(result.get("status_code", 500)),
        data=result.get("data") or {},
    )


@router.post("/collections", response_model=view_models.Collection)
@audit(resource_type="collection", api_name="CreateCollectionV2")
async def create_collection_view(
    collection: view_models.CollectionCreate,
    user: User = Depends(required_user),
) -> view_models.Collection:
    """Create a collection owned by the current user."""

    return await collection_service.create_collection(str(user.id), collection)


@router.get("/collections", response_model=view_models.CollectionViewList)
async def list_collections_view(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    include_subscribed: bool = Query(True),
    user: User = Depends(required_user),
) -> view_models.CollectionViewList:
    """List collections visible to the current user."""

    return await collection_service.list_collections_view(str(user.id), include_subscribed, page, page_size)


@router.get("/collections/{collection_id}", response_model=view_models.Collection)
async def get_collection_view(
    collection_id: str,
    user: User = Depends(required_user),
) -> view_models.Collection:
    """Return one collection owned by the current user."""

    return await collection_service.get_collection(str(user.id), collection_id)


@router.put("/collections/{collection_id}", response_model=view_models.Collection)
@audit(resource_type="collection", api_name="UpdateCollectionV2")
async def update_collection_view(
    collection_id: str,
    body: view_models.CollectionUpdate,
    user: User = Depends(required_user),
) -> view_models.Collection:
    """Update a collection owned by the current user."""

    return await collection_service.update_collection(str(user.id), collection_id, body)


@router.delete("/collections/{collection_id}", status_code=204)
@audit(resource_type="collection", api_name="DeleteCollectionV2")
async def delete_collection_view(
    collection_id: str,
    user: User = Depends(required_user),
) -> Response:
    """Delete a collection owned by the current user. Idempotent."""

    await collection_service.delete_collection(str(user.id), collection_id)
    return Response(status_code=204)


@router.post(
    "/collections/{collection_id}/summary/generate",
    response_model=view_models.CollectionSummaryTriggerResponse,
)
@audit(resource_type="collection", api_name="GenerateCollectionSummaryV2")
async def generate_collection_summary_view(
    collection_id: str,
    user: User = Depends(required_user),
) -> view_models.CollectionSummaryTriggerResponse:
    """Trigger background summary generation for one collection."""

    collection = await collection_service.get_collection(str(user.id), collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    task_triggered = await collection_summary_service.trigger_collection_summary_generation(collection)
    if task_triggered:
        return view_models.CollectionSummaryTriggerResponse(
            collection_id=collection_id,
            success=True,
            message="Collection summary generation started",
            summary_status="PENDING",
        )
    return view_models.CollectionSummaryTriggerResponse(
        collection_id=collection_id,
        success=False,
        message="Collection summary generation already in progress or disabled",
        summary_status="GENERATING",
    )


@router.get(
    "/collections/{collection_id}/sharing",
    response_model=view_models.SharingStatusResponse,
)
async def get_collection_sharing_status_view(
    collection_id: str,
    user: User = Depends(required_user),
) -> view_models.SharingStatusResponse:
    """Return marketplace sharing status for a collection owned by the current user."""

    try:
        is_published, published_at = await marketplace_service.get_sharing_status(user.id, collection_id)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    return view_models.SharingStatusResponse(is_published=is_published, published_at=published_at)


@router.post("/collections/{collection_id}/sharing", status_code=204)
@audit(resource_type="collection", api_name="PublishCollectionV2")
async def publish_collection_sharing_view(
    collection_id: str,
    user: User = Depends(required_user),
) -> Response:
    """Publish a collection to the marketplace. Owner only."""

    try:
        await marketplace_service.publish_collection(user.id, collection_id)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    return Response(status_code=204)


@router.delete("/collections/{collection_id}/sharing", status_code=204)
@audit(resource_type="collection", api_name="UnpublishCollectionV2")
async def unpublish_collection_sharing_view(
    collection_id: str,
    user: User = Depends(required_user),
) -> Response:
    """Unpublish a collection from the marketplace. Owner only."""

    try:
        await marketplace_service.unpublish_collection(user.id, collection_id)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    return Response(status_code=204)
