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

"""Canonical Pydantic view models for the `knowledge_base` domain.

Phase 3 Step 4b (msg=1505044c / msg=eb271056) extracted the 11
collection- and document-shaped schemas out of
``aperag.schema.view_models`` so the knowledge_base domain owns its
public contract shape. For the Phase 6 cleanup window
``aperag.schema.view_models`` continues to re-export these names (via
``_bind_view_models_reexports`` at the bottom of this module) so that
pre-migration callers and FastAPI handlers still referencing
``view_models.Collection`` / ``view_models.Document`` continue to see
the same class objects.

Shared cross-domain primitives — ``CollectionConfig``, ``PageResult``,
``PaginatedResponse``, ``Chunk``, ``VisionChunk`` — live in
``aperag.schema.common`` because they are consumed by non-KB domains
(source / bots / retrieval / views). That module is intentionally
outside the Phase 3 G1 legacy-aggregate ban list, so importing from
it here is a legitimate shared-schema dependency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import AnyUrl, BaseModel, Field, conint

from aperag.schema.common import (
    Chunk,
    CollectionConfig,
    PageResult,
    PaginatedResponse,
    VectorBackendInfo,
    VisionChunk,
)

__all__ = [
    "Collection",
    "CollectionView",
    "CollectionViewList",
    "CollectionCreate",
    "CollectionUpdate",
    "Document",
    "DocumentList",
    "DocumentPreview",
    "RebuildIndexesRequest",
    "RebuildIndexesResponse",
    "CollectionRegenTriggerResponse",  # Wave 10 §K.13
    # Step 5b3: document-upload / confirm / fetch-url envelope schemas
    # moved from ``aperag.schema.view_models`` so the KB
    # ``document_service`` does not have to bind to the aggregate.
    "UploadDocumentResponse",
    "FailedDocument",
    "ConfirmDocumentsResponse",
    "FetchUrlResultItem",
    "FetchUrlResponse",
    "StagedDocumentsResponse",
    # Step 5a: request bodies + sharing + mineru token schemas consumed
    # by the KB routes that move to ``aperag.domains.knowledge_base.api``.
    "ConfirmDocumentsRequest",
    "FetchUrlRequest",
    "DeleteDocumentsRequest",
    "DeleteDocumentsResponse",
    "SharingStatusResponse",
    "MineruTokenTestRequest",
    "MineruTokenTestResponse",
    "ExportTaskResponse",
]


class Collection(BaseModel):
    """
    Collection is a collection of documents
    """

    id: Optional[str] = None
    title: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    # Wave 10 §K.13: long-form auto-generated summary (~5-10 K chars).
    # The settings page renders it in the read-only textarea right
    # under ``description``; without this field on the response schema
    # the FE always sees ``summary=null`` and shows the placeholder
    # even after a successful regen (regression caught via @earayu2
    # msg=e4120886 reproducer).
    summary: Optional[str] = Field(None, description="Auto-generated long-form knowledge-base summary")
    config: Optional[CollectionConfig] = None
    # task #61 P1-D3 (PR for #87): read-only projection of the deployment
    # vector backend identity + capability matrix. **Intentionally placed
    # on Collection (read response) and NOT inside CollectionConfig.**
    # ``CollectionConfig`` is reused as the create/update input shape, so
    # putting vector_backend there would expose the field on the OpenAPI
    # ``CollectionCreate``/``CollectionUpdate`` input schemas and let
    # callers mistake a deployment-wide setting for a per-collection
    # editable knob (per dongdong msg=c2593fdd + PM msg=caf7e4df + spec
    # P1-D3 read-only projection lock). Projected by
    # ``collection_service.build_collection_response()`` from
    # ``settings.vector_db_type``; ``None`` for unknown backends so the FE
    # can render a placeholder without a hard failure.
    vector_backend: Optional[VectorBackendInfo] = Field(
        None,
        description=(
            "Read-only deployment vector backend identity + static capability "
            "matrix. Projected from ``settings.vector_db_type``; not editable "
            "per collection."
        ),
    )
    status: Optional[Literal["ACTIVE", "INACTIVE", "DELETED"]] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    is_published: Optional[bool] = Field(False, description="Whether the collection is published to marketplace")
    published_at: Optional[datetime] = Field(None, description="Publication time, null when not published")


class Document(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    status: Optional[
        Literal[
            "UPLOADED",
            "EXPIRED",
            "PENDING",
            "RUNNING",
            "COMPLETE",
            "FAILED",
            "DELETING",
            "DELETED",
        ]
    ] = None
    # Wave 3 §F.2 hard-cut: per-modality status Literal aligns to the
    # 4-state IndexStatus enum (PENDING / RUNNING / ACTIVE / FAILED).
    # The legacy 6-state values (CREATING / DELETING /
    # DELETION_IN_PROGRESS) are gone; RUNNING covers both create and
    # delete in-flight, and "modality not enabled" is expressed by the
    # field being absent (None) rather than a sentinel "SKIPPED" — the
    # row simply does not exist in document_index. Per architect msg
    # post-pass-5 + PM msg=79683cc0 ruling.
    vector_index_status: Optional[Literal["PENDING", "RUNNING", "ACTIVE", "FAILED"]] = None
    fulltext_index_status: Optional[Literal["PENDING", "RUNNING", "ACTIVE", "FAILED"]] = None
    graph_index_status: Optional[Literal["PENDING", "RUNNING", "ACTIVE", "FAILED"]] = None
    summary_index_status: Optional[Literal["PENDING", "RUNNING", "ACTIVE", "FAILED"]] = None
    vision_index_status: Optional[Literal["PENDING", "RUNNING", "ACTIVE", "FAILED"]] = None
    vector_index_updated: Optional[datetime] = Field(None, description="Vector index last updated time")
    fulltext_index_updated: Optional[datetime] = Field(None, description="Fulltext index last updated time")
    graph_index_updated: Optional[datetime] = Field(None, description="Graph index last updated time")
    summary_index_updated: Optional[datetime] = Field(None, description="Summary index last updated time")
    vision_index_updated: Optional[datetime] = Field(None, description="Vision index last updated time")
    summary: Optional[str] = Field(None, description="Summary of the document")
    size: Optional[float] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class CollectionView(BaseModel):
    """
    Lightweight collection information for lists, MCP and agents
    """

    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    status: Optional[Literal["ACTIVE", "INACTIVE", "DELETED"]] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    is_published: Optional[bool] = False
    published_at: Optional[datetime] = Field(None, description="Publication time, null when not published")
    owner_user_id: Optional[str] = Field(None, description="Collection owner user ID")
    owner_username: Optional[str] = Field(None, description="Collection owner username")
    subscription_id: Optional[str] = Field(
        None,
        description="Subscription ID if this is a subscribed collection, null for owned collections",
    )
    subscribed_at: Optional[datetime] = Field(None, description="Subscription time, null for owned collections")


class CollectionViewList(BaseModel):
    """
    A list of collection views
    """

    items: Optional[list[CollectionView]] = None
    pageResult: Optional[PageResult] = None


class CollectionCreate(BaseModel):
    title: Optional[str] = None
    config: Optional[CollectionConfig] = None
    type: Optional[str] = None
    description: Optional[str] = None


class CollectionUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    config: Optional[CollectionConfig] = None


class DocumentList(PaginatedResponse):
    """
    A list of documents with pagination
    """

    items: Optional[list[Document]] = None


class RebuildIndexesRequest(BaseModel):
    index_types: list[Literal["VECTOR", "FULLTEXT", "GRAPH", "SUMMARY", "VISION"]] = Field(
        ..., description="Types of indexes to rebuild", min_length=1
    )


class RebuildIndexesResponse(BaseModel):
    code: str = Field(..., description="Result code", examples=["200"])
    message: str = Field(..., description="Human-readable rebuild status")
    affected_documents: Optional[conint(ge=0)] = Field(
        None,
        description="Number of documents affected by a collection-level rebuild",
    )


class DocumentPreview(BaseModel):
    doc_object_path: Optional[str] = Field(None, description="The path to the document object.")
    doc_filename: Optional[str] = Field(None, description="The name of the document.")
    converted_pdf_object_path: Optional[str] = Field(None, description="The path to the converted PDF object.")
    markdown_content: Optional[str] = Field(None, description="The markdown content of the document.")
    chunks: Optional[list[Chunk]] = None
    vision_chunks: Optional[list[VisionChunk]] = None


# Wave 10 §K.13: replacement envelope for the new
# ``/collections/{id}/summary/regen`` + ``/description/regen``
# endpoints. Returned with ``202 Accepted`` to signal that regen
# was scheduled (lease acquired, async task launched). Lease busy
# is surfaced as ``409 Conflict`` (no envelope; standard FastAPI
# detail body).
class CollectionRegenTriggerResponse(BaseModel):
    """``202 Accepted`` envelope for collection regen triggers."""

    collection_id: str = Field(..., description="Collection id whose regen was triggered")
    stage: Literal["summary", "description"] = Field(
        ...,
        description="Which stage of the regen pipeline was triggered",
    )
    task_id: str = Field(..., description="Opaque tracking token for the dispatched async task")
    estimated_completion_seconds: int = Field(
        ...,
        description="Best-effort latency estimate (Stage 1 ~60s agent explore, Stage 2 ~10s LLM derive)",
    )


# ---------- Document upload / confirm / fetch-url envelopes (Step 5b3) ---------- #
#
# The six schemas below were carved out of ``aperag.schema.view_models``
# in Phase 3 Step 5b3 so that the KB ``document_service`` can build its
# response envelopes without binding to the banned aggregate.
# ``ConfirmDocumentsRequest`` and ``FetchUrlRequest`` stay in
# ``view_models`` for now — they are consumed by FastAPI route handlers
# (``aperag.views.documents_v2``) and will migrate in Step 5a when
# routes move to ``aperag/domains/knowledge_base/api/routes.py``.


class UploadDocumentResponse(BaseModel):
    document_id: str = Field(..., description="ID of the uploaded document")
    filename: str = Field(..., description="Name of the uploaded file")
    size: int = Field(..., description="Size of the uploaded file in bytes")
    status: Literal["UPLOADED", "PENDING", "RUNNING", "COMPLETE", "FAILED", "DELETED", "EXPIRED"] = Field(
        ...,
        description="Status of the document (UPLOADED for new uploads, or existing status for duplicate files)",
    )


class FailedDocument(BaseModel):
    document_id: Optional[str] = None
    name: Optional[str] = Field(None, description="Name of the document")
    error: Optional[str] = None


class ConfirmDocumentsResponse(BaseModel):
    confirmed_count: int = Field(..., description="Number of documents successfully confirmed")
    failed_count: int = Field(..., description="Number of documents that failed to confirm")
    failed_documents: Optional[list[FailedDocument]] = Field(None, description="Details of failed confirmations")


class FetchUrlResultItem(BaseModel):
    url: str = Field(..., description="The source URL")
    fetch_status: Literal["success", "error"] = Field(..., description="Whether the URL was fetched successfully")
    document_id: Optional[str] = Field(None, description="ID of the created document (only present on success)")
    filename: Optional[str] = Field(None, description="Filename of the created document (only present on success)")
    size: Optional[int] = Field(
        None,
        description="Size of the created document in bytes (only present on success)",
    )
    status: Optional[str] = Field(None, description="Document status (only present on success)")
    error: Optional[str] = Field(None, description="Error message (only present on failure)")


class FetchUrlResponse(BaseModel):
    results: list[FetchUrlResultItem] = Field(..., description="Results for each URL")
    total: int = Field(..., description="Total number of URLs processed")
    succeeded: int = Field(..., description="Number of URLs successfully fetched")
    failed: int = Field(..., description="Number of URLs that failed")


class StagedDocumentsResponse(BaseModel):
    documents: list[UploadDocumentResponse] = Field(
        ..., description="List of staged (UPLOADED) documents awaiting confirmation"
    )
    total: int = Field(..., description="Total number of staged documents")


# ---------- Step 5a: KB route request-bodies + sharing + mineru ---------- #


class ConfirmDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(..., description="List of document IDs to confirm", min_length=1)


class FetchUrlRequest(BaseModel):
    urls: list[AnyUrl] = Field(
        ...,
        description="List of URLs to fetch and import (max 10)",
        examples=[["https://example.com/article1", "https://example.com/article2"]],
    )


class DeleteDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(..., description="Document IDs to delete", min_length=1)


class DeleteDocumentsResponse(BaseModel):
    deleted_ids: list[str] = Field(..., description="Document IDs accepted for deletion")
    status: Literal["success"] = Field(..., description="Batch deletion status")


class SharingStatusResponse(BaseModel):
    """Marketplace sharing status envelope for a KB collection.

    Phase 4 may re-home this class to a marketplace schemas module when
    that domain materializes; for now it lives in KB because the owning
    route (``GET /collections/{id}/sharing``) is a KB route.
    """

    is_published: bool = Field(..., description="Whether published to marketplace")
    published_at: Optional[datetime] = Field(None, description="Publication time, null when not published")


class MineruTokenTestRequest(BaseModel):
    """Request body for POST /collections/test-mineru-token."""

    token: str = Field(..., description="MinerU API token to validate")


class MineruTokenTestResponse(BaseModel):
    """Response envelope for POST /collections/test-mineru-token.

    `status_code` is the HTTP status returned by MinerU's upstream test endpoint;
    `data` is the passthrough body echoed to the caller.
    """

    status_code: int = Field(..., description="HTTP status code returned by MinerU upstream")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Passthrough body from MinerU upstream",
    )


# ---------- Phase 8 #47 G1: export envelope ---------- #
#
# Carved from ``aperag.schema.view_models.ExportTaskResponse`` so the
# KB domain owns the response shape used by its export routes
# (``POST /collections/{id}/export`` + ``GET /export-tasks/*``).
# ``aperag.schema.view_models`` continues to re-export this name during
# the cleanup window.


class ExportTaskResponse(BaseModel):
    export_task_id: str = Field(..., description="Unique ID of the export task")
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED", "EXPIRED"] = Field(
        ..., description="Current status of the export task"
    )
    progress: Optional[conint(ge=0, le=100)] = Field(None, description="Progress percentage (0-100)")
    message: Optional[str] = Field(None, description="Human-readable status message")
    error_message: Optional[str] = Field(None, description="Error detail when status is FAILED")
    download_url: Optional[str] = Field(
        None,
        description="URL to download the ZIP file (only set when status is COMPLETED)",
    )
    file_size: Optional[int] = Field(None, description="Size of the ZIP file in bytes")
    gmt_created: Optional[datetime] = None
    gmt_completed: Optional[datetime] = None
    gmt_expires: Optional[datetime] = Field(
        None,
        description="Time when the export file will be automatically deleted (7 days after creation)",
    )
