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

"""Knowledge-base HTTP router (Phase 3 Step 5a).

Merges the two route modules ``aperag/views/collections_v2.py`` and
``aperag/views/documents_v2.py`` into the canonical KB domain
location. Handlers continue to expose the same URLs (``/api/v2``
prefix, same paths, same response shapes, same audit decorators) so
the FastAPI-generated OpenAPI spec stays byte-stable.

Two consumer-owned Protocols are relied on here:

* ``AuthenticatedUser`` — the handler parameter types for
  ``Depends(required_user)``. Replaces the banned
  ``aperag.db.models.User`` ORM class; only the attributes these
  handlers actually read (user ``id``) are pinned on the Protocol.
* ``MarketplaceOps`` — the ``/collections/{id}/sharing`` group calls
  into it for ``get_sharing_status`` / ``publish_collection`` /
  ``unpublish_collection``. Step 5b2c wires the legacy
  ``marketplace_service`` instance at app startup; this module reaches
  it through ``_get_marketplace_ops()`` imported from the sibling
  collection_service module.
"""

import logging
import uuid as uuid_lib
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from starlette.responses import StreamingResponse

from aperag.domains.identity.service.auth_dependencies import required_user
from aperag.domains.knowledge_base.ports import AuthenticatedUser
from aperag.domains.knowledge_base.schemas import (
    Collection,
    CollectionCreate,
    CollectionRegenTriggerResponse,
    CollectionUpdate,
    CollectionViewList,
    ConfirmDocumentsRequest,
    ConfirmDocumentsResponse,
    DeleteDocumentsRequest,
    DeleteDocumentsResponse,
    Document,
    DocumentList,
    DocumentPreview,
    FetchUrlRequest,
    FetchUrlResponse,
    MineruTokenTestRequest,
    MineruTokenTestResponse,
    RebuildIndexesRequest,
    RebuildIndexesResponse,
    SharingStatusResponse,
    StagedDocumentsResponse,
    UploadDocumentResponse,
)
from aperag.domains.knowledge_base.service.collection_service import (
    _get_marketplace_ops,
    collection_service,
)
from aperag.domains.knowledge_base.service.document_service import document_service
from aperag.exceptions import CollectionNotFoundException, PermissionDeniedError
from aperag.utils.audit_decorator import audit

logger = logging.getLogger(__name__)

router = APIRouter(tags=["collections-v2"])


# ---------- Collection routes (from former aperag/views/collections_v2.py) ---------- #
# Static-path routes must be declared before ``/collections/{collection_id}`` so FastAPI
# matches them first and provider/collection names cannot shadow them.


@router.post(
    "/collections/test-mineru-token",
    response_model=MineruTokenTestResponse,
)
async def test_mineru_token_view(
    body: MineruTokenTestRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> MineruTokenTestResponse:
    """Probe an upstream MinerU API token and echo the status envelope."""

    result = await collection_service.test_mineru_token(body.token)
    return MineruTokenTestResponse(
        status_code=int(result.get("status_code", 500)),
        data=result.get("data") or {},
    )


@router.post("/collections", response_model=Collection)
@audit(resource_type="collection", api_name="CreateCollectionV2")
async def create_collection_view(
    collection: CollectionCreate,
    user: AuthenticatedUser = Depends(required_user),
) -> Collection:
    """Create a collection owned by the current user."""

    return await collection_service.create_collection(str(user.id), collection)


@router.get("/collections", response_model=CollectionViewList)
async def list_collections_view(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    include_subscribed: bool = Query(True),
    user: AuthenticatedUser = Depends(required_user),
) -> CollectionViewList:
    """List collections visible to the current user."""

    return await collection_service.list_collections_view(str(user.id), include_subscribed, page, page_size)


@router.get("/collections/{collection_id}", response_model=Collection)
async def get_collection_view(
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Collection:
    """Return one collection owned by the current user."""

    return await collection_service.get_collection(str(user.id), collection_id)


@router.put("/collections/{collection_id}", response_model=Collection)
@audit(resource_type="collection", api_name="UpdateCollectionV2")
async def update_collection_view(
    collection_id: str,
    body: CollectionUpdate,
    user: AuthenticatedUser = Depends(required_user),
) -> Collection:
    """Update a collection owned by the current user."""

    return await collection_service.update_collection(str(user.id), collection_id, body)


@router.delete("/collections/{collection_id}", status_code=204)
@audit(resource_type="collection", api_name="DeleteCollectionV2")
async def delete_collection_view(
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Response:
    """Delete a collection owned by the current user. Idempotent."""

    await collection_service.delete_collection(str(user.id), collection_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------
# Wave 10 §K.13 — collection auto-description endpoints (Chunk C+D).
# ---------------------------------------------------------------------
#
# These two endpoints are explicit operator/external overrides that
# bypass the debounce / min-stale guards in
# ``reconcile_collection_descriptions_hook`` (Chunk E). Both return
# 202 Accepted and dispatch the regen as a fire-and-forget asyncio
# task; lease conflict is surfaced as 409 to keep the contract honest
# (per huangheng API contract — no silent ignore).


@router.post(
    "/collections/{collection_id}/summary/regen",
    response_model=CollectionRegenTriggerResponse,
    status_code=202,
)
@audit(resource_type="collection", api_name="RegenCollectionSummaryV2")
async def regen_collection_summary_view(
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> CollectionRegenTriggerResponse:
    """Trigger Stage 1 summary regen for ``collection_id``.

    Runs agent-runtime free-explore over the collection (3-tier
    fallback chain) and writes the result to ``Collection.summary``.
    Stage 2 (description derive) is automatically picked up by the
    reconciler hook on the next sweep — callers don't need a
    separate trigger unless they want to re-derive description with
    summary unchanged (use ``/description/regen`` for that).

    Returns 404 if collection doesn't exist or caller doesn't own it,
    409 if the regen lease is already held (concurrent regen in
    flight), 202 with task_id otherwise.
    """
    import asyncio

    from aperag.domains.knowledge_base.service.collection_regen_service import regen_summary

    collection = await collection_service.get_collection(str(user.id), collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    task_id = uuid_lib.uuid4().hex
    asyncio.create_task(regen_summary(collection_id))
    return CollectionRegenTriggerResponse(
        collection_id=collection_id,
        stage="summary",
        task_id=task_id,
        estimated_completion_seconds=60,
    )


@router.post(
    "/collections/{collection_id}/description/regen",
    response_model=CollectionRegenTriggerResponse,
    status_code=202,
)
@audit(resource_type="collection", api_name="RegenCollectionDescriptionV2")
async def regen_collection_description_view(
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> CollectionRegenTriggerResponse:
    """Trigger Stage 2 description derive from existing
    ``Collection.summary``.

    Cheap path — single LLM call, no agent multi-turn (~10s). Useful
    when the summary is current but you want to refresh description
    formatting / language / length.

    Returns 400 if ``Collection.summary IS NULL`` (must regen summary
    first), 404 if collection doesn't exist or caller doesn't own it,
    409 if regen lease is held, 202 with task_id otherwise.
    """
    import asyncio

    from aperag.domains.knowledge_base.service.collection_regen_service import regen_description

    collection = await collection_service.get_collection(str(user.id), collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    if not getattr(collection, "summary", None):
        raise HTTPException(
            status_code=400,
            detail="Collection has no summary yet — call /summary/regen first.",
        )

    task_id = uuid_lib.uuid4().hex
    asyncio.create_task(regen_description(collection_id))
    return CollectionRegenTriggerResponse(
        collection_id=collection_id,
        stage="description",
        task_id=task_id,
        estimated_completion_seconds=10,
    )


@router.get(
    "/collections/{collection_id}/sharing",
    response_model=SharingStatusResponse,
)
async def get_collection_sharing_status_view(
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> SharingStatusResponse:
    """Return marketplace sharing status for a collection owned by the current user."""

    try:
        is_published, published_at = await _get_marketplace_ops().get_sharing_status(user.id, collection_id)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    return SharingStatusResponse(is_published=is_published, published_at=published_at)


@router.post("/collections/{collection_id}/sharing", status_code=204)
@audit(resource_type="collection", api_name="PublishCollectionV2")
async def publish_collection_sharing_view(
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Response:
    """Publish a collection to the marketplace. Owner only."""

    try:
        await _get_marketplace_ops().publish_collection(user.id, collection_id)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    return Response(status_code=204)


@router.delete("/collections/{collection_id}/sharing", status_code=204)
@audit(resource_type="collection", api_name="UnpublishCollectionV2")
async def unpublish_collection_sharing_view(
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Response:
    """Unpublish a collection from the marketplace. Owner only."""

    try:
        await _get_marketplace_ops().unpublish_collection(user.id, collection_id)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except PermissionDeniedError:
        raise HTTPException(status_code=403, detail="Permission denied")
    return Response(status_code=204)


# ---------- Document routes (from former aperag/views/documents_v2.py) ---------- #


@router.post("/collections/{collection_id}/documents", response_model=DocumentList)
@audit(resource_type="document", api_name="CreateDocumentsV2")
async def create_documents_v2_view(
    request: Request,
    collection_id: str,
    files: list[UploadFile] = File(...),
    user: AuthenticatedUser = Depends(required_user),
) -> DocumentList:
    return await document_service.create_documents(str(user.id), collection_id, files)


@router.get("/collections/{collection_id}/documents", response_model=DocumentList)
async def list_documents_v2_view(
    collection_id: str,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    sort_by: Literal["name", "created", "updated", "size", "status"] = Query(
        "created",
        description="Field to sort by",
    ),
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort order"),
    search: str | None = Query(None, description="Search documents by name"),
    user: AuthenticatedUser = Depends(required_user),
) -> DocumentList:
    result = await document_service.list_documents(
        user=str(user.id),
        collection_id=collection_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        search=search,
    )
    return DocumentList.model_validate(result.model_dump())


@router.get("/collections/{collection_id}/documents/staged", response_model=StagedDocumentsResponse)
async def list_staged_documents_v2_view(
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> StagedDocumentsResponse:
    return await document_service.get_staged_documents(str(user.id), collection_id)


@router.get("/collections/{collection_id}/documents/{document_id}", response_model=Document)
async def get_document_v2_view(
    collection_id: str,
    document_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Document:
    return await document_service.get_document(str(user.id), collection_id, document_id)


@router.get(
    "/collections/{collection_id}/documents/{document_id}/download",
    response_class=StreamingResponse,
    responses={200: {"content": {"application/octet-stream": {}}}},
)
@audit(resource_type="document", api_name="DownloadDocumentV2")
async def download_document_v2_view(
    request: Request,
    collection_id: str,
    document_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> StreamingResponse:
    return await document_service.download_document(str(user.id), collection_id, document_id)


@router.delete("/collections/{collection_id}/documents/{document_id}", status_code=204)
@audit(resource_type="document", api_name="DeleteDocumentV2")
async def delete_document_v2_view(
    request: Request,
    collection_id: str,
    document_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Response:
    await document_service.delete_document(str(user.id), collection_id, document_id)
    return Response(status_code=204)


@router.delete("/collections/{collection_id}/documents", response_model=DeleteDocumentsResponse)
@audit(resource_type="document", api_name="DeleteDocumentsV2")
async def delete_documents_v2_view(
    request: Request,
    collection_id: str,
    body: DeleteDocumentsRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> DeleteDocumentsResponse:
    result = await document_service.delete_documents(str(user.id), collection_id, body.document_ids)
    return DeleteDocumentsResponse.model_validate(result)


@router.get(
    "/collections/{collection_id}/documents/{document_id}/preview",
    response_model=DocumentPreview,
)
async def get_document_preview_v2_view(
    collection_id: str,
    document_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> DocumentPreview:
    return await document_service.get_document_preview(str(user.id), collection_id, document_id)


@router.get(
    "/collections/{collection_id}/documents/{document_id}/object",
    response_class=StreamingResponse,
    responses={200: {"content": {"application/octet-stream": {}}}, 206: {"content": {"application/octet-stream": {}}}},
)
async def get_document_object_v2_view(
    collection_id: str,
    document_id: str,
    path: str,
    request: Request,
    user: AuthenticatedUser = Depends(required_user),
) -> StreamingResponse:
    range_header = request.headers.get("range")
    return await document_service.get_document_object(str(user.id), collection_id, document_id, path, range_header)


@router.post(
    "/collections/{collection_id}/documents/{document_id}/rebuild_indexes",
    response_model=RebuildIndexesResponse,
)
@audit(resource_type="document", api_name="RebuildDocumentIndexesV2")
async def rebuild_document_indexes_v2_view(
    request: Request,
    collection_id: str,
    document_id: str,
    body: RebuildIndexesRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> RebuildIndexesResponse:
    result = await document_service.rebuild_document_indexes(str(user.id), collection_id, document_id, body.index_types)
    return RebuildIndexesResponse.model_validate(result)


@router.post(
    "/collections/{collection_id}/rebuild_failed_indexes",
    response_model=RebuildIndexesResponse,
)
@audit(resource_type="collection", api_name="RebuildFailedIndexesV2")
async def rebuild_failed_indexes_v2_view(
    request: Request,
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> RebuildIndexesResponse:
    result = await document_service.rebuild_failed_indexes(str(user.id), collection_id)
    return RebuildIndexesResponse.model_validate(result)


@router.post("/collections/{collection_id}/documents/upload", response_model=UploadDocumentResponse)
@audit(resource_type="document", api_name="UploadDocumentV2")
async def upload_document_v2_view(
    request: Request,
    collection_id: str,
    file: UploadFile = File(...),
    user: AuthenticatedUser = Depends(required_user),
) -> UploadDocumentResponse:
    return await document_service.upload_document(str(user.id), collection_id, file)


@router.post("/collections/{collection_id}/documents/confirm", response_model=ConfirmDocumentsResponse)
@audit(resource_type="document", api_name="ConfirmDocumentsV2")
async def confirm_documents_v2_view(
    request: Request,
    collection_id: str,
    body: ConfirmDocumentsRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> ConfirmDocumentsResponse:
    return await document_service.confirm_documents(str(user.id), collection_id, body.document_ids)


@router.post("/collections/{collection_id}/documents/fetch-url", response_model=FetchUrlResponse)
@audit(resource_type="document", api_name="FetchUrlDocumentV2")
async def fetch_url_document_v2_view(
    request: Request,
    collection_id: str,
    body: FetchUrlRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> FetchUrlResponse:
    return await document_service.fetch_url_documents(str(user.id), collection_id, body.urls)
