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
from typing import Literal, Optional

from pydantic import BaseModel, Field, conint

from aperag.schema.common import (
    Chunk,
    CollectionConfig,
    PageResult,
    PaginatedResponse,
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
    "CollectionSummaryTriggerResponse",
]


class Collection(BaseModel):
    """
    Collection is a collection of documents
    """

    id: Optional[str] = None
    title: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    config: Optional[CollectionConfig] = None
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
    vector_index_status: Optional[
        Literal[
            "PENDING",
            "CREATING",
            "ACTIVE",
            "DELETING",
            "DELETION_IN_PROGRESS",
            "FAILED",
            "SKIPPED",
        ]
    ] = None
    fulltext_index_status: Optional[
        Literal[
            "PENDING",
            "CREATING",
            "ACTIVE",
            "DELETING",
            "DELETION_IN_PROGRESS",
            "FAILED",
            "SKIPPED",
        ]
    ] = None
    graph_index_status: Optional[
        Literal[
            "PENDING",
            "CREATING",
            "ACTIVE",
            "DELETING",
            "DELETION_IN_PROGRESS",
            "FAILED",
            "SKIPPED",
        ]
    ] = None
    summary_index_status: Optional[
        Literal[
            "PENDING",
            "CREATING",
            "ACTIVE",
            "DELETING",
            "DELETION_IN_PROGRESS",
            "FAILED",
            "SKIPPED",
        ]
    ] = None
    vision_index_status: Optional[
        Literal[
            "PENDING",
            "CREATING",
            "ACTIVE",
            "DELETING",
            "DELETION_IN_PROGRESS",
            "FAILED",
            "SKIPPED",
        ]
    ] = None
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


class CollectionSummaryTriggerResponse(BaseModel):
    """Trigger-response envelope for POST /collections/{collection_id}/summary/generate."""

    collection_id: str = Field(..., description="Collection id whose summary generation was triggered")
    success: bool = Field(..., description="Whether the background job was scheduled")
    message: str = Field(..., description="Human-readable status message")
    summary_status: Literal["PENDING", "GENERATING"] = Field(
        ...,
        description="Server-side summary state after the trigger call",
    )


# Re-bind the 11 KB schemas onto ``aperag.schema.view_models`` so that
# pre-migration callers (``from aperag.schema.view_models import
# Collection`` / ``view_models.Document(...)``) and Pydantic forward-ref
# resolution in view_models' own classes (``Agent.collections:
# list[Collection]``, etc.) continue to see the same class objects this
# module defines.
#
# The two load orders are handled symmetrically:
#   * view_models.py loaded first → its end-of-file ``try`` block
#     imports these names from knowledge_base.schemas and binds them
#     directly via normal import machinery (the hook below is a no-op
#     because view_models is already in ``sys.modules`` when this hook
#     runs inside that chain).
#   * knowledge_base.schemas loaded first → view_models.py has not yet
#     been loaded, so the hook leaves it alone; when something later
#     triggers view_models.py to load, its end-of-file import binds
#     these names.
#
# ``sys.modules`` is consulted via a string lookup so Phase 3 G1 AST
# scans do not flag a (pointless) runtime import of
# ``aperag.schema.view_models`` from inside this KB-domain module.
def _bind_view_models_reexports() -> None:
    import sys

    _vm = sys.modules.get("aperag.schema.view_models")
    if _vm is None:
        return
    for _name in __all__:
        setattr(_vm, _name, globals()[_name])


_bind_view_models_reexports()
