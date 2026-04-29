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

"""Marketplace HTTP router (Phase 4 Step 4-S4).

Merges the former ``aperag/views/marketplace.py`` +
``aperag/views/marketplace_collections.py`` into a single canonical
``APIRouter`` at the marketplace domain location. Handlers keep the
same URLs, methods, response_model, status codes so the
FastAPI-generated OpenAPI spec stays byte-stable across the move.

Handler parameter types for ``Depends(required_user)`` /
``Depends(optional_user)`` use the per-domain
``AuthenticatedUser(Protocol)`` from ``aperag.domains.marketplace.ports``
instead of binding to ``aperag.db.models.User`` (lesson 9a-ter + G16
canonical).

The ``_check_marketplace_access`` calls from the legacy file are
rewritten to the Q2 public name ``check_marketplace_access`` per
msg=6ab7d211.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from aperag.domains.identity.service.auth_dependencies import optional_user, required_user
from aperag.domains.knowledge_graph.schemas import (
    GraphEmbeddingMapResponse,
    GraphEntitiesSearchResponse,
    GraphEvidenceResponse,
    GraphHybridResponse,
    GraphRelationEvidenceRequest,
    KnowledgeGraph,
)
from aperag.domains.knowledge_graph.service import graph_service
from aperag.domains.marketplace.ports import AuthenticatedUser
from aperag.domains.marketplace.schemas import SharedCollection, SharedCollectionList
from aperag.domains.marketplace.service.marketplace_collection_service import marketplace_collection_service
from aperag.domains.marketplace.service.marketplace_service import marketplace_service
from aperag.exceptions import (
    AlreadySubscribedError,
    CollectionMarketplaceAccessDeniedError,
    CollectionNotPublishedError,
    SelfSubscriptionError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["marketplace"])


# ---------- Marketplace listing / subscribe (from former views/marketplace.py) ---------- #


@router.get("/marketplace/collections", response_model=SharedCollectionList)
async def list_marketplace_collections(
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
    user: AuthenticatedUser = Depends(optional_user),
) -> SharedCollectionList:
    """List all published Collections in marketplace"""
    try:
        # Allow unauthenticated access - use empty user_id for anonymous users
        user_id = user.id if user else ""
        result = await marketplace_service.list_published_collections(user_id, page, page_size)
        return result
    except Exception as e:
        logger.error(f"Error listing marketplace collections: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/marketplace/collections/subscriptions", response_model=SharedCollectionList)
async def list_user_subscribed_collections(
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
    user: AuthenticatedUser = Depends(required_user),
) -> SharedCollectionList:
    """Get user's subscribed Collections"""
    try:
        result = await marketplace_service.list_user_subscribed_collections(user.id, page, page_size)
        return result
    except Exception as e:
        logger.error(f"Error listing user subscribed collections: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/marketplace/collections/{collection_id}/subscribe", response_model=SharedCollection)
async def subscribe_collection(
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> SharedCollection:
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
    user: AuthenticatedUser = Depends(required_user),
) -> Dict[str, Any]:
    """Unsubscribe from a Collection"""
    try:
        await marketplace_service.unsubscribe_collection(user.id, collection_id)
        return {"message": "Successfully unsubscribed"}
    except Exception as e:
        logger.error(f"Error unsubscribing from collection {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------- Marketplace collection read-only access (from former views/marketplace_collections.py) ---------- #


@router.get("/marketplace/collections/{collection_id}", response_model=SharedCollection)
async def get_marketplace_collection(
    collection_id: str,
    user: AuthenticatedUser = Depends(optional_user),
) -> SharedCollection:
    """Get MarketplaceCollection details (read-only)"""
    try:
        user_id = str(user.id) if user else ""
        result = await marketplace_collection_service.get_marketplace_collection(user_id, collection_id)
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
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    search: str = Query(None, description="Search documents by name"),
    user: AuthenticatedUser = Depends(optional_user),
):
    """List documents in MarketplaceCollection (read-only) with pagination, sorting and search capabilities"""
    try:
        # Check marketplace access first (all logged-in users can view published collections)
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service.check_marketplace_access(user_id, collection_id)

        # Lazy import to avoid import-time coupling into KB domain
        from aperag.domains.knowledge_base.service.document_service import document_service

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
    user: AuthenticatedUser = Depends(optional_user),
):
    """Preview document in MarketplaceCollection (read-only)"""
    try:
        # Check marketplace access first (all logged-in users can view published collections)
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service.check_marketplace_access(user_id, collection_id)

        from aperag.domains.knowledge_base.service.document_service import document_service

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
    user: AuthenticatedUser = Depends(optional_user),
):
    """Get document object from MarketplaceCollection (read-only)"""
    try:
        # Check marketplace access first (all logged-in users can view published collections)
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service.check_marketplace_access(user_id, collection_id)

        from aperag.domains.knowledge_base.service.document_service import document_service

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


@router.get("/marketplace/collections/{collection_id}/graph", tags=["graph"], response_model=KnowledgeGraph)
async def get_marketplace_collection_graph(
    request: Request,
    collection_id: str,
    label: str = Query("*"),
    max_nodes: int = Query(1000, ge=1, le=10000),
    max_depth: int = Query(3, ge=1, le=10),
    user: AuthenticatedUser = Depends(optional_user),
) -> KnowledgeGraph:
    """Get knowledge graph for MarketplaceCollection (read-only)"""
    # Validate parameters (same as regular collections)
    if not (1 <= max_nodes <= 10000):
        raise HTTPException(status_code=400, detail="max_nodes must be between 1 and 10000")
    if not (1 <= max_depth <= 10):
        raise HTTPException(status_code=400, detail="max_depth must be between 1 and 10")

    try:
        # Check marketplace access first (all logged-in users can view published collections)
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service.check_marketplace_access(user_id, collection_id)

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


@router.get(
    "/marketplace/collections/{collection_id}/graph/embedding-map",
    tags=["graph"],
    response_model=GraphEmbeddingMapResponse,
)
async def get_marketplace_collection_graph_embedding_map(
    request: Request,
    collection_id: str,
    max_entities: int = Query(1000, ge=1, le=5000),
    user: AuthenticatedUser = Depends(optional_user),
) -> GraphEmbeddingMapResponse:
    """Get projected entity coordinates for MarketplaceCollection (read-only)."""
    try:
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service.check_marketplace_access(user_id, collection_id)
        owner_user_id = marketplace_info["owner_user_id"]
        return await graph_service.get_embedding_map(
            str(owner_user_id),
            collection_id,
            max_entities=max_entities,
        )
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting marketplace collection graph embedding map {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/marketplace/collections/{collection_id}/graph/hybrid",
    tags=["graph"],
    response_model=GraphHybridResponse,
)
async def get_marketplace_collection_graph_hybrid(
    request: Request,
    collection_id: str,
    max_entities: int = Query(1000, ge=1, le=5000),
    user: AuthenticatedUser = Depends(optional_user),
) -> GraphHybridResponse:
    """Get positioned graph-hybrid data for MarketplaceCollection (read-only)."""
    try:
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service.check_marketplace_access(user_id, collection_id)
        owner_user_id = marketplace_info["owner_user_id"]
        return await graph_service.get_hybrid_graph(
            str(owner_user_id),
            collection_id,
            max_entities=max_entities,
        )
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting marketplace collection graph hybrid {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/marketplace/collections/{collection_id}/graph/entities/search",
    tags=["graph"],
    response_model=GraphEntitiesSearchResponse,
)
async def search_marketplace_collection_graph_entities(
    request: Request,
    collection_id: str,
    q: str,
    top_k: int = Query(10, ge=1, le=100),
    user: AuthenticatedUser = Depends(optional_user),
) -> GraphEntitiesSearchResponse:
    """Vector-recall graph entities for MarketplaceCollection (read-only)."""
    try:
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service.check_marketplace_access(user_id, collection_id)
        owner_user_id = marketplace_info["owner_user_id"]
        return await graph_service.search_entities(
            str(owner_user_id),
            collection_id,
            query=q,
            top_k=top_k,
        )
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching marketplace collection graph entities {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/marketplace/collections/{collection_id}/graph/entities/{name}/evidence",
    tags=["graph"],
    response_model=GraphEvidenceResponse,
)
async def get_marketplace_collection_graph_entity_evidence(
    request: Request,
    collection_id: str,
    name: str,
    limit: int = Query(5, ge=1, le=20),
    user: AuthenticatedUser = Depends(optional_user),
) -> GraphEvidenceResponse:
    """Get bounded source chunks for a MarketplaceCollection graph entity."""
    try:
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service.check_marketplace_access(user_id, collection_id)
        owner_user_id = marketplace_info["owner_user_id"]
        return await graph_service.get_entity_evidence(
            str(owner_user_id),
            collection_id,
            entity_name=name,
            limit=limit,
        )
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting marketplace collection graph entity evidence {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/marketplace/collections/{collection_id}/graph/relations/evidence",
    tags=["graph"],
    response_model=GraphEvidenceResponse,
)
async def get_marketplace_collection_graph_relation_evidence(
    request: Request,
    collection_id: str,
    payload: GraphRelationEvidenceRequest,
    user: AuthenticatedUser = Depends(optional_user),
) -> GraphEvidenceResponse:
    """Get bounded source chunks for a MarketplaceCollection graph relation."""
    try:
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service.check_marketplace_access(user_id, collection_id)
        owner_user_id = marketplace_info["owner_user_id"]
        return await graph_service.get_relation_evidence(
            str(owner_user_id),
            collection_id,
            source=payload.source,
            target=payload.target,
            relation_type=payload.relation_type,
            limit=payload.limit,
        )
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting marketplace collection graph relation evidence {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
