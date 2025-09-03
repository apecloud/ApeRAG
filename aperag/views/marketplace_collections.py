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

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from aperag.db.models import User
from aperag.exceptions import (
    CollectionMarketplaceAccessDeniedError,
    CollectionNotPublishedError,
)
from aperag.schema import view_models
from aperag.service.document_service import document_service
from aperag.service.marketplace_collection_service import marketplace_collection_service
from aperag.views.auth import required_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["marketplace-collections"])


@router.get("/marketplace/collections/{collection_id}", response_model=view_models.SharedCollection)
async def get_marketplace_collection(
    collection_id: str,
    user: User = Depends(required_user),
) -> view_models.SharedCollection:
    """Get MarketplaceCollection details (read-only)"""
    try:
        result = await marketplace_collection_service.get_marketplace_collection(user.id, collection_id)
        return result
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting marketplace collection {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/marketplace/collections/{collection_id}/documents")
async def list_marketplace_collection_documents(
    request: Request,
    collection_id: str,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    sort_by: str = Query("created", description="Field to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    search: str = Query(None, description="Search documents by name"),
    user: User = Depends(required_user),
):
    """List documents in MarketplaceCollection (read-only) with pagination, sorting and search capabilities"""
    try:
        # Check marketplace access first (all logged-in users can view published collections)
        marketplace_info = await marketplace_collection_service._check_marketplace_access(user.id, collection_id)

        # Use the collection owner's user_id to query documents, not the current user's id
        owner_user_id = marketplace_info["owner_user_id"]
        result = await document_service.list_documents(
            user=str(owner_user_id),
            collection_id=collection_id,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
            search=search,
        )

        return {
            "items": result.items,
            "total": result.total,
            "page": result.page,
            "page_size": result.page_size,
            "total_pages": result.total_pages,
            "has_next": result.has_next,
            "has_prev": result.has_prev,
        }
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing marketplace collection documents {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/marketplace/collections/{collection_id}/documents/{document_id}/preview",
    tags=["documents"],
    operation_id="get_marketplace_document_preview",
)
async def get_marketplace_collection_document_preview(
    collection_id: str,
    document_id: str,
    user: User = Depends(required_user),
):
    """Preview document in MarketplaceCollection (read-only)"""
    try:
        # Check marketplace access first (all logged-in users can view published collections)
        marketplace_info = await marketplace_collection_service._check_marketplace_access(user.id, collection_id)

        # Use the collection owner's user_id to query document, not the current user's id
        owner_user_id = marketplace_info["owner_user_id"]
        return await document_service.get_document_preview(owner_user_id, collection_id, document_id)
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting marketplace collection document preview {collection_id}/{document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/marketplace/collections/{collection_id}/documents/{document_id}/object",
    tags=["documents"],
    operation_id="get_marketplace_document_object",
)
async def get_marketplace_collection_document_object(
    request: Request,
    collection_id: str,
    document_id: str,
    path: str = Query(..., description="Object path within the document"),
    user: User = Depends(required_user),
):
    """Get document object from MarketplaceCollection (read-only)"""
    try:
        # Check marketplace access first (all logged-in users can view published collections)
        marketplace_info = await marketplace_collection_service._check_marketplace_access(user.id, collection_id)

        # Use the collection owner's user_id to get document object, not the current user's id
        owner_user_id = marketplace_info["owner_user_id"]
        range_header = request.headers.get("range")
        return await document_service.get_document_object(owner_user_id, collection_id, document_id, path, range_header)
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting marketplace collection document object {collection_id}/{document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/marketplace/collections/{collection_id}/graph", tags=["graph"])
async def get_marketplace_collection_graph(
    request: Request,
    collection_id: str,
    label: str = Query("*"),
    max_nodes: int = Query(1000, ge=1, le=10000),
    max_depth: int = Query(3, ge=1, le=10),
    user: User = Depends(required_user),
) -> Dict[str, Any]:
    """Get knowledge graph for MarketplaceCollection (read-only)"""
    from aperag.service.graph_service import graph_service

    # Validate parameters (same as regular collections)
    if not (1 <= max_nodes <= 10000):
        raise HTTPException(status_code=400, detail="max_nodes must be between 1 and 10000")
    if not (1 <= max_depth <= 10):
        raise HTTPException(status_code=400, detail="max_depth must be between 1 and 10")

    try:
        # Check marketplace access first (all logged-in users can view published collections)
        marketplace_info = await marketplace_collection_service._check_marketplace_access(user.id, collection_id)

        # Use the collection owner's user_id to query graph, not the current user's id
        owner_user_id = marketplace_info["owner_user_id"]
        return await graph_service.get_knowledge_graph(str(owner_user_id), collection_id, label, max_depth, max_nodes)
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting marketplace collection graph {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
