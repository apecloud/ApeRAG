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

from typing import Literal

from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile
from starlette.responses import StreamingResponse

from aperag.db.models import User
from aperag.schema import view_models
from aperag.service.document_service import document_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

router = APIRouter(tags=["documents-v2"])


@router.post("/collections/{collection_id}/documents", response_model=view_models.DocumentList)
@audit(resource_type="document", api_name="CreateDocumentsV2")
async def create_documents_v2_view(
    request: Request,
    collection_id: str,
    files: list[UploadFile] = File(...),
    user: User = Depends(required_user),
) -> view_models.DocumentList:
    return await document_service.create_documents(str(user.id), collection_id, files)


@router.get("/collections/{collection_id}/documents", response_model=view_models.DocumentList)
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
    user: User = Depends(required_user),
) -> view_models.DocumentList:
    result = await document_service.list_documents(
        user=str(user.id),
        collection_id=collection_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        search=search,
    )
    return view_models.DocumentList.model_validate(result.model_dump())


@router.get("/collections/{collection_id}/documents/staged", response_model=view_models.StagedDocumentsResponse)
async def list_staged_documents_v2_view(
    collection_id: str,
    user: User = Depends(required_user),
) -> view_models.StagedDocumentsResponse:
    return await document_service.get_staged_documents(str(user.id), collection_id)


@router.get("/collections/{collection_id}/documents/{document_id}", response_model=view_models.Document)
async def get_document_v2_view(
    collection_id: str,
    document_id: str,
    user: User = Depends(required_user),
) -> view_models.Document:
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
    user: User = Depends(required_user),
) -> StreamingResponse:
    return await document_service.download_document(str(user.id), collection_id, document_id)


@router.delete("/collections/{collection_id}/documents/{document_id}", status_code=204)
@audit(resource_type="document", api_name="DeleteDocumentV2")
async def delete_document_v2_view(
    request: Request,
    collection_id: str,
    document_id: str,
    user: User = Depends(required_user),
) -> Response:
    await document_service.delete_document(str(user.id), collection_id, document_id)
    return Response(status_code=204)


@router.delete("/collections/{collection_id}/documents", response_model=view_models.DeleteDocumentsResponse)
@audit(resource_type="document", api_name="DeleteDocumentsV2")
async def delete_documents_v2_view(
    request: Request,
    collection_id: str,
    body: view_models.DeleteDocumentsRequest,
    user: User = Depends(required_user),
) -> view_models.DeleteDocumentsResponse:
    result = await document_service.delete_documents(str(user.id), collection_id, body.document_ids)
    return view_models.DeleteDocumentsResponse.model_validate(result)


@router.get(
    "/collections/{collection_id}/documents/{document_id}/preview",
    response_model=view_models.DocumentPreview,
)
async def get_document_preview_v2_view(
    collection_id: str,
    document_id: str,
    user: User = Depends(required_user),
) -> view_models.DocumentPreview:
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
    user: User = Depends(required_user),
) -> StreamingResponse:
    range_header = request.headers.get("range")
    return await document_service.get_document_object(str(user.id), collection_id, document_id, path, range_header)


@router.post(
    "/collections/{collection_id}/documents/{document_id}/rebuild_indexes",
    response_model=view_models.RebuildIndexesResponse,
)
@audit(resource_type="document", api_name="RebuildDocumentIndexesV2")
async def rebuild_document_indexes_v2_view(
    request: Request,
    collection_id: str,
    document_id: str,
    body: view_models.RebuildIndexesRequest,
    user: User = Depends(required_user),
) -> view_models.RebuildIndexesResponse:
    result = await document_service.rebuild_document_indexes(str(user.id), collection_id, document_id, body.index_types)
    return view_models.RebuildIndexesResponse.model_validate(result)


@router.post(
    "/collections/{collection_id}/rebuild_failed_indexes",
    response_model=view_models.RebuildIndexesResponse,
)
@audit(resource_type="collection", api_name="RebuildFailedIndexesV2")
async def rebuild_failed_indexes_v2_view(
    request: Request,
    collection_id: str,
    user: User = Depends(required_user),
) -> view_models.RebuildIndexesResponse:
    result = await document_service.rebuild_failed_indexes(str(user.id), collection_id)
    return view_models.RebuildIndexesResponse.model_validate(result)


@router.post("/collections/{collection_id}/documents/upload", response_model=view_models.UploadDocumentResponse)
@audit(resource_type="document", api_name="UploadDocumentV2")
async def upload_document_v2_view(
    request: Request,
    collection_id: str,
    file: UploadFile = File(...),
    user: User = Depends(required_user),
) -> view_models.UploadDocumentResponse:
    return await document_service.upload_document(str(user.id), collection_id, file)


@router.post("/collections/{collection_id}/documents/confirm", response_model=view_models.ConfirmDocumentsResponse)
@audit(resource_type="document", api_name="ConfirmDocumentsV2")
async def confirm_documents_v2_view(
    request: Request,
    collection_id: str,
    body: view_models.ConfirmDocumentsRequest,
    user: User = Depends(required_user),
) -> view_models.ConfirmDocumentsResponse:
    return await document_service.confirm_documents(str(user.id), collection_id, body.document_ids)


@router.post("/collections/{collection_id}/documents/fetch-url", response_model=view_models.FetchUrlResponse)
@audit(resource_type="document", api_name="FetchUrlDocumentV2")
async def fetch_url_document_v2_view(
    request: Request,
    collection_id: str,
    body: view_models.FetchUrlRequest,
    user: User = Depends(required_user),
) -> view_models.FetchUrlResponse:
    return await document_service.fetch_url_documents(str(user.id), collection_id, body.urls)
