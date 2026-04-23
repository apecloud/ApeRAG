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

"""Residual v1 `/api/v1/collections/*` routes.

Collection CRUD / sharing / summary / mineru-token and the read-side
document routes were replaced by `aperag/views/collections_v2.py`
(#1573) and `aperag/views/documents_v2.py` (#1575) and the corresponding
FE adapters (#1579 / #1581 / #1585). Those v1 handlers were removed in
the `#26` final sweep.

Still here (no v2 replacement yet or upload flow intentionally deferred):

- Document upload flow: `upload`, `confirm`, `fetch-url`, `staged` listing.
  The upload queue will move under a dedicated slice together with the
  cross-page persistent upload queue that was flagged by `#1580`.
- Search routes under `/collections/{id}/searches*`.
- Graph routes under `/collections/{id}/graphs`.

New v2 work for any of those should land its own router file and a FE
adapter pair, same pattern as `collections_v2.py` + `features/collection/*`.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from aperag.db.models import User
from aperag.exceptions import CollectionNotFoundException
from aperag.schema import view_models
from aperag.service.collection_service import collection_service
from aperag.service.document_service import document_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

logger = logging.getLogger(__name__)

router = APIRouter()


# Collection search endpoints
@router.post("/collections/{collection_id}/searches", tags=["search"])
@audit(resource_type="search", api_name="CreateSearch")
async def create_search_view(
    request: Request,
    collection_id: str,
    data: view_models.SearchRequest,
    user: User = Depends(required_user),
) -> view_models.SearchResult:
    return await collection_service.create_search(str(user.id), collection_id, data)


@router.delete("/collections/{collection_id}/searches/{search_id}", tags=["search"], name="DeleteSearch")
@audit(resource_type="search", api_name="DeleteSearch")
async def delete_search_view(
    request: Request,
    collection_id: str,
    search_id: str,
    user: User = Depends(required_user),
):
    return await collection_service.delete_search(str(user.id), collection_id, search_id)


@router.get("/collections/{collection_id}/searches", tags=["search"])
async def list_searches_view(
    request: Request, collection_id: str, user: User = Depends(required_user)
) -> view_models.SearchResultList:
    return await collection_service.list_searches(str(user.id), collection_id)


@router.get("/collections/{collection_id}/documents/staged", tags=["documents"])
async def list_staged_documents_view(
    request: Request,
    collection_id: str,
    user: User = Depends(required_user),
) -> view_models.StagedDocumentsResponse:
    """Return all UPLOADED (staged) documents awaiting confirmation."""
    return await document_service.get_staged_documents(str(user.id), collection_id)


# Knowledge Graph API endpoints
@router.get("/collections/{collection_id}/graphs/labels", tags=["graph"])
async def get_graph_labels_view(
    request: Request,
    collection_id: str,
    user: User = Depends(required_user),
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


# Upload-related endpoints. Upload flow stays on v1 until the cross-page
# persistent upload queue lands (see #1580 for the underlying regression
# hotfix and #30 for its guards).
@router.post("/collections/{collection_id}/documents/upload", tags=["documents"])
@audit(resource_type="document", api_name="UploadDocument")
async def upload_document_view(
    request: Request,
    collection_id: str,
    file: UploadFile = File(...),
    user: User = Depends(required_user),
) -> view_models.UploadDocumentResponse:
    """Upload a single document file to temporary storage"""
    return await document_service.upload_document(str(user.id), collection_id, file)


@router.post("/collections/{collection_id}/documents/confirm", tags=["documents"])
@audit(resource_type="document", api_name="ConfirmDocuments")
async def confirm_documents_view(
    request: Request,
    collection_id: str,
    confirm_request: view_models.ConfirmDocumentsRequest,
    user: User = Depends(required_user),
) -> view_models.ConfirmDocumentsResponse:
    """Confirm uploaded documents and add them to the collection"""
    return await document_service.confirm_documents(str(user.id), collection_id, confirm_request.document_ids)


@router.post("/collections/{collection_id}/documents/fetch-url", tags=["documents"])
@audit(resource_type="document", api_name="FetchUrlDocument")
async def fetch_url_document_view(
    request: Request,
    collection_id: str,
    fetch_request: view_models.FetchUrlRequest,
    user: User = Depends(required_user),
) -> view_models.FetchUrlResponse:
    """
    Fetch web page content from URLs and create UPLOADED documents.

    Each URL is fetched via the web read service (JINA with Trafilatura fallback).
    Successful results are wrapped as virtual UploadFile objects and passed to
    upload_document(), producing UPLOADED documents identical to file uploads.
    Use POST /documents/confirm to move them to PENDING and start indexing.
    """
    return await document_service.fetch_url_documents(str(user.id), collection_id, fetch_request.urls)


@router.get("/collections/{collection_id}/graphs", tags=["graph"], response_model=view_models.KnowledgeGraph)
async def get_knowledge_graph_view(
    request: Request,
    collection_id: str,
    label: str = "*",
    max_nodes: int = 1000,
    max_depth: int = 3,
    user: User = Depends(required_user),
) -> view_models.KnowledgeGraph:
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
