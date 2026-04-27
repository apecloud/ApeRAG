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

The retrieval + knowledge_graph read / subgraph / curation routes that
previously lived here were hard-cut by the Phase 2 domain split and
now live at `aperag/domains/retrieval/api/routes.py` +
`aperag/domains/knowledge_graph/api/routes.py`; they are mounted under
`/api/v2/` by `aperag/app.py`. See
`docs/modularization/breaking-changes/phase2-retrieval-knowledge_graph.md`.

Still here (no v2 replacement yet or upload flow intentionally deferred):

- Document upload flow: `upload`, `confirm`, `fetch-url`, `staged` listing.
  The upload queue will move under a dedicated slice together with the
  cross-page persistent upload queue that was flagged by `#1580`.

New v2 work for any of those should land its own router file and a FE
adapter pair, same pattern as `collections_v2.py` + `features/collection/*`.
"""

import logging

from fastapi import APIRouter, Depends, File, Request, UploadFile

from aperag.db.models import User
from aperag.schema import view_models
from aperag.service.document_service import document_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/collections/{collection_id}/documents/staged", tags=["documents"])
async def list_staged_documents_view(
    request: Request,
    collection_id: str,
    user: User = Depends(required_user),
) -> view_models.StagedDocumentsResponse:
    """Return all UPLOADED (staged) documents awaiting confirmation."""
    return await document_service.get_staged_documents(str(user.id), collection_id)


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
