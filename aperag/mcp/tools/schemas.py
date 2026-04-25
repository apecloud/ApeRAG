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

"""Pydantic response shapes for D10 read primitives.

Per ``docs/modularization/d10-design-pack.md`` §A — the typed return
envelopes for the 8 read primitive surface tools. These shapes are the
cross-lane contract that D10.d (search primitives) / D10.e (cursor
contract) / D10.g (cache layer) import for handle linkage and pagination
plumbing.

This module contains shapes only — no business logic. Function signatures
live in the per-primitive modules (``list_collections.py`` etc.). Stable
handle types (``ChunkId`` / ``SectionPath`` / ``HeadingAnchor``) live in
``handles.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from aperag.mcp.tools.handles import ChunkId, HeadingAnchor, SectionPath


class ByteRange(BaseModel):
    """Optional byte range for ``read_document`` (best-effort, parse-version-bound).

    Per §A.5 + §A.9: byte-range is optional / best-effort and may become
    invalid across ``parse_version`` boundaries when the parsed markdown is
    re-derived.
    """

    start: int = Field(..., ge=0, description="Inclusive start byte offset.")
    end: int = Field(..., ge=0, description="Exclusive end byte offset.")


class OutlineHeading(BaseModel):
    """A single heading node in a document's outline tree (§A.6).

    Field names are LOCKED per §A.9 — do not rename without
    ``[D10 spec amendment]`` thread.
    """

    level: int = Field(..., ge=1, le=6, description="Heading level (1-6).")
    text: str = Field(..., description="Rendered heading text.")
    section_path: SectionPath = Field(
        ...,
        description="Primary stable handle (e.g., '1/2.3/4') per R1 Lock.",
    )
    heading_anchor: HeadingAnchor = Field(
        ...,
        description="Slug-style anchor (e.g., '#chapter-2-implementation').",
    )
    chunk_id: Optional[ChunkId] = Field(
        None,
        description="Corresponding indexing chunk id, when mapped.",
    )
    children: list["OutlineHeading"] = Field(
        default_factory=list,
        description="Nested child headings.",
    )


class CollectionMetadata(BaseModel):
    """Lightweight collection metadata for ``list_collections`` items (§A.1)."""

    collection_id: str
    title: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CollectionDetailMetadata(BaseModel):
    """Full collection metadata returned by ``get_collection_metadata`` (§A.4)."""

    collection_id: str
    title: str
    description: Optional[str] = None
    document_count: int
    index_modes_available: list[str] = Field(
        default_factory=list,
        description="Index modes enabled for this collection (e.g., vector / fulltext / graph).",
    )
    permission_model: Optional[str] = Field(
        None, description="Permission model identifier (owner / subscriber / etc.)."
    )
    created_at: datetime
    updated_at: datetime


class CollectionList(BaseModel):
    """Paginated list envelope for ``list_collections`` (§A.1).

    ``next_cursor`` is opaque base64 (R2 cursor contract — D10.e owns the
    codec). ``total_count`` is optional and may be omitted for performance.
    """

    items: list[CollectionMetadata]
    next_cursor: Optional[str] = None
    total_count: Optional[int] = None


class DocumentMetadata(BaseModel):
    """Metadata for a single document (§A.3)."""

    document_id: str
    collection_id: str
    title: str
    media_type: str
    size_bytes: int
    indexed_chunks_count: int
    indexing_status: Literal["pending", "indexing", "complete", "failed"]
    failure_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    outline_summary: Optional[list[OutlineHeading]] = Field(
        None,
        description="Top 2 levels of the outline for navigation hints.",
    )


class DocumentList(BaseModel):
    """Paginated list envelope for ``list_documents`` (§A.2)."""

    items: list[DocumentMetadata]
    next_cursor: Optional[str] = None
    total_count: Optional[int] = None


class DocumentContent(BaseModel):
    """Full parsed-markdown content for ``read_document`` (§A.5).

    ``parse_version`` is the composite cache key (parser pipeline hash +
    document md5) — see §E.2. Clients usually do not consume it directly.
    """

    document_id: str
    collection_id: str
    parsed_markdown: str
    parse_version: str
    truncated: bool = False
    truncation_reason: Optional[str] = None


class DocumentOutline(BaseModel):
    """Heading-tree envelope for ``read_document_outline`` (§A.6)."""

    document_id: str
    headings: list[OutlineHeading]
    parse_version: str


class DocumentSection(BaseModel):
    """Section content envelope for ``read_document_section`` (§A.7)."""

    document_id: str
    collection_id: str
    section_path: SectionPath
    heading_anchor: HeadingAnchor
    heading_text: str
    parsed_markdown: str
    parse_version: str
    parent_section_path: Optional[SectionPath] = None
    sibling_count: int = Field(..., ge=0, description="Number of sibling sections at the same level.")


class DocumentChunk(BaseModel):
    """Chunk content envelope for ``read_document_chunk`` (§A.8)."""

    chunk_id: ChunkId
    document_id: str
    collection_id: str
    parsed_markdown: str
    section_path: Optional[SectionPath] = None
    chunk_index: int = Field(..., ge=0, description="Zero-based ordering within document.")
    chunk_total: int = Field(..., ge=0, description="Total chunks in the document.")
    parse_version: str


# Resolve forward references for the recursive OutlineHeading.children field.
OutlineHeading.model_rebuild()


__all__ = [
    "ByteRange",
    "CollectionDetailMetadata",
    "CollectionList",
    "CollectionMetadata",
    "DocumentChunk",
    "DocumentContent",
    "DocumentList",
    "DocumentMetadata",
    "DocumentOutline",
    "DocumentSection",
    "OutlineHeading",
]
