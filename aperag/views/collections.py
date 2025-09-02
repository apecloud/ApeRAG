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
from typing import List

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, Request, Response, UploadFile

from aperag.db.models import User
from aperag.exceptions import CollectionNotFoundException
from aperag.schema import view_models
from aperag.service.collection_service import collection_service
from aperag.service.collection_summary_service import collection_summary_service
from aperag.service.document_service import document_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import current_user, get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/collections", tags=["collections"])
@audit(resource_type="collection", api_name="CreateCollection")
async def create_collection_view(
    request: Request,
    collection: view_models.CollectionCreate,
    user: User = Depends(current_user),
) -> view_models.Collection:
    return await collection_service.create_collection(str(user.id), collection)


@router.get("/collections", tags=["collections"])
async def list_collections_view(
    request: Request,
    page: int = Query(1),
    page_size: int = Query(50),
    include_subscribed: bool = Query(True),
    user: User = Depends(current_user),
) -> view_models.CollectionViewList:
    return await collection_service.list_collections_view(str(user.id), include_subscribed, page, page_size)


@router.get("/collections/{collection_id}", tags=["collections"])
async def get_collection_view(
    request: Request, collection_id: str, user: User = Depends(current_user)
) -> view_models.Collection:
    return await collection_service.get_collection(str(user.id), collection_id)


@router.put("/collections/{collection_id}", tags=["collections"])
@audit(resource_type="collection", api_name="UpdateCollection")
async def update_collection_view(
    request: Request,
    collection_id: str,
    collection: view_models.CollectionUpdate,
    user: User = Depends(current_user),
) -> view_models.Collection:
    instance = await collection_service.update_collection(str(user.id), collection_id, collection)
    return instance


@router.delete("/collections/{collection_id}", tags=["collections"])
@audit(resource_type="collection", api_name="DeleteCollection")
async def delete_collection_view(
    request: Request, collection_id: str, user: User = Depends(current_user)
) -> view_models.Collection:
    return await collection_service.delete_collection(str(user.id), collection_id)


@router.post("/collections/{collection_id}/summary/generate", tags=["collections"])
@audit(resource_type="collection", api_name="GenerateCollectionSummary")
async def generate_collection_summary_view(
    request: Request, collection_id: str, user: User = Depends(current_user)
) -> dict:
    """Trigger collection summary generation as background task"""

    # Check if collection exists
    collection = await collection_service.get_collection(str(user.id), collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Trigger async summary generation
    task_triggered = await collection_summary_service.trigger_collection_summary_generation(collection)

    if task_triggered:
        return {
            "collection_id": collection_id,
            "success": True,
            "message": "Collection summary generation started",
            "summary_status": "PENDING",
        }
    else:
        return {
            "collection_id": collection_id,
            "success": False,
            "message": "Collection summary generation already in progress or disabled",
            "summary_status": "GENERATING",
        }


@router.post("/collections/test-mineru-token", tags=["collections"])
async def test_mineru_token_view(
    request: Request,
    data: dict = Body(...),
    user: User = Depends(current_user),
):
    token = data.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")
    return await collection_service.test_mineru_token(token)


# Collection sharing endpoints
@router.get("/collections/{collection_id}/sharing", tags=["collections"])
async def get_collection_sharing_status(
    collection_id: str,
    user: User = Depends(current_user),
) -> view_models.SharingStatusResponse:
    """Get collection sharing status (owner only)"""
    from aperag.exceptions import CollectionNotFoundException, PermissionDeniedError
    from aperag.service.marketplace_service import marketplace_service

    try:
        is_published, published_at = await marketplace_service.get_sharing_status(user.id, collection_id)
        return view_models.SharingStatusResponse(is_published=is_published, published_at=published_at)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Error getting collection sharing status {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/collections/{collection_id}/sharing", tags=["collections"])
async def publish_collection_to_marketplace(
    collection_id: str,
    user: User = Depends(current_user),
):
    """Publish collection to marketplace (owner only)"""
    from aperag.exceptions import CollectionNotFoundException, PermissionDeniedError
    from aperag.service.marketplace_service import marketplace_service

    try:
        await marketplace_service.publish_collection(user.id, collection_id)
        return Response(status_code=204)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Error publishing collection {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/collections/{collection_id}/sharing", tags=["collections"])
async def unpublish_collection_from_marketplace(
    collection_id: str,
    user: User = Depends(current_user),
):
    """Unpublish collection from marketplace (owner only)"""
    from aperag.exceptions import CollectionNotFoundException, PermissionDeniedError
    from aperag.service.marketplace_service import marketplace_service

    try:
        await marketplace_service.unpublish_collection(user.id, collection_id)
        return Response(status_code=204)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Error unpublishing collection {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Collection search endpoints
@router.post("/collections/{collection_id}/searches", tags=["search"])
@audit(resource_type="search", api_name="CreateSearch")
async def create_search_view(
    request: Request,
    collection_id: str,
    data: view_models.SearchRequest,
    user: User = Depends(current_user),
) -> view_models.SearchResult:
    return await collection_service.create_search(str(user.id), collection_id, data)


@router.delete("/collections/{collection_id}/searches/{search_id}", tags=["search"], name="DeleteSearch")
@audit(resource_type="search", api_name="DeleteSearch")
async def delete_search_view(
    request: Request,
    collection_id: str,
    search_id: str,
    user: User = Depends(current_user),
):
    return await collection_service.delete_search(str(user.id), collection_id, search_id)


@router.get("/collections/{collection_id}/searches", tags=["search"])
async def list_searches_view(
    request: Request, collection_id: str, user: User = Depends(current_user)
) -> view_models.SearchResultList:
    return await collection_service.list_searches(str(user.id), collection_id)


@router.post("/collections/{collection_id}/documents", tags=["documents"])
@audit(resource_type="document", api_name="CreateDocuments")
async def create_documents_view(
    request: Request,
    collection_id: str,
    files: List[UploadFile] = File(...),
    user: User = Depends(current_user),
) -> view_models.DocumentList:
    return await document_service.create_documents(str(user.id), collection_id, files)


@router.get("/collections/{collection_id}/documents", tags=["documents"])
async def list_documents_view(
    request: Request,
    collection_id: str,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    sort_by: str = Query("created", description="Field to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    search: str = Query(None, description="Search documents by name"),
    user: User = Depends(current_user),
):
    """List documents with pagination, sorting and search capabilities"""

    result = await document_service.list_documents(
        user=str(user.id),
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


@router.get("/collections/{collection_id}/documents/{document_id}", tags=["documents"])
async def get_document_view(
    request: Request,
    collection_id: str,
    document_id: str,
    user: User = Depends(current_user),
) -> view_models.Document:
    return await document_service.get_document(str(user.id), collection_id, document_id)


@router.delete("/collections/{collection_id}/documents/{document_id}", tags=["documents"])
@audit(resource_type="document", api_name="DeleteDocument")
async def delete_document_view(
    request: Request,
    collection_id: str,
    document_id: str,
    user: User = Depends(current_user),
) -> view_models.Document:
    return await document_service.delete_document(str(user.id), collection_id, document_id)


@router.delete("/collections/{collection_id}/documents", tags=["documents"])
@audit(resource_type="document", api_name="DeleteDocuments")
async def delete_documents_view(
    request: Request,
    collection_id: str,
    document_ids: List[str],
    user: User = Depends(current_user),
):
    return await document_service.delete_documents(str(user.id), collection_id, document_ids)


@router.get(
    "/collections/{collection_id}/documents/{document_id}/preview",
    tags=["documents"],
    operation_id="get_document_preview",
)
async def get_document_preview(
    collection_id: str,
    document_id: str,
    user: User = Depends(current_user),
):
    return await document_service.get_document_preview(user.id, collection_id, document_id)


@router.get(
    "/collections/{collection_id}/documents/{document_id}/object",
    tags=["documents"],
    operation_id="get_document_object",
)
async def get_document_object(
    request: Request,
    collection_id: str,
    document_id: str,
    path: str,
    user: User = Depends(current_user),
):
    range_header = request.headers.get("range")
    return await document_service.get_document_object(user.id, collection_id, document_id, path, range_header)


@router.post("/collections/{collection_id}/documents/{document_id}/rebuild_indexes", tags=["documents"])
@audit(resource_type="document", api_name="RebuildDocumentIndexes")
async def rebuild_document_indexes_view(
    request: Request,
    collection_id: str,
    document_id: str,
    rebuild_request: view_models.RebuildIndexesRequest,
    user: User = Depends(current_user),
):
    """Rebuild specified indexes for a document"""
    return await document_service.rebuild_document_indexes(
        str(user.id), collection_id, document_id, rebuild_request.index_types
    )


@router.post("/collections/{collection_id}/rebuild_failed_indexes", tags=["documents"])
@audit(resource_type="collection", api_name="RebuildFailedIndexes")
async def rebuild_failed_indexes_view(
    request: Request,
    collection_id: str,
    user: User = Depends(get_current_active_user),
):
    """Rebuild all failed indexes for all documents in a collection"""
    return await document_service.rebuild_failed_indexes(str(user.id), collection_id)


# Knowledge Graph API endpoints
@router.get("/collections/{collection_id}/graphs/labels", tags=["graph"])
async def get_graph_labels_view(
    request: Request,
    collection_id: str,
    user: User = Depends(current_user),
) -> view_models.GraphLabelsResponse:
    """Get all available node labels in the collection's knowledge graph"""
    from aperag.service.graph_service import graph_service

    try:
        result = await graph_service.get_graph_labels(str(user.id), collection_id)
        return result
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# New upload-related endpoints
@router.post("/collections/{collection_id}/documents/upload", tags=["documents"])
@audit(resource_type="document", api_name="UploadDocument")
async def upload_document_view(
    request: Request,
    collection_id: str,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
) -> view_models.UploadDocumentResponse:
    """Upload a single document file to temporary storage"""
    return await document_service.upload_document(str(user.id), collection_id, file)


@router.post("/collections/{collection_id}/documents/confirm", tags=["documents"])
@audit(resource_type="document", api_name="ConfirmDocuments")
async def confirm_documents_view(
    request: Request,
    collection_id: str,
    confirm_request: view_models.ConfirmDocumentsRequest,
    user: User = Depends(current_user),
) -> view_models.ConfirmDocumentsResponse:
    """Confirm uploaded documents and add them to the collection"""
    return await document_service.confirm_documents(str(user.id), collection_id, confirm_request.document_ids)


@router.get("/collections/{collection_id}/graphs", tags=["graph"])
async def get_knowledge_graph_view(
    request: Request,
    collection_id: str,
    label: str = "*",
    max_nodes: int = 1000,
    max_depth: int = 3,
    user: User = Depends(current_user),
):
    """Get knowledge graph - overview mode or subgraph mode"""
    from aperag.service.graph_service import graph_service

    # Validate parameters
    if not (1 <= max_nodes <= 10000):
        raise HTTPException(status_code=400, detail="max_nodes must be between 1 and 10000")
    if not (1 <= max_depth <= 10):
        raise HTTPException(status_code=400, detail="max_depth must be between 1 and 10")

    try:
        result = await graph_service.get_knowledge_graph(str(user.id), collection_id, label, max_depth, max_nodes)
        return result
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
