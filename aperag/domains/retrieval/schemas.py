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

"""Pydantic view models owned by the ``retrieval`` canonical domain.

Relocated from ``aperag.schema.view_models`` by the Phase 2 hard-cut.
OpenAPI component names and shapes are preserved byte-for-byte so the
existing FE types keep resolving 1-to-1 from the new source.

``aperag/schema/view_models.py`` keeps re-export lines at the bottom
pointing here, so legacy callers of
``aperag.schema.view_models.SearchRequest`` etc. still work until the
Phase 3 DB-split PR deletes the legacy aggregate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, confloat, field_validator


class VectorSearchParams(BaseModel):
    topk: Optional[int] = Field(None, description="Top K results")
    similarity: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Similarity threshold")


class FulltextSearchParams(BaseModel):
    topk: Optional[int] = Field(None, description="Top K results")
    keywords: Optional[list[str]] = Field(None, description="Custom keywords to use for fulltext search")


class GraphSearchParams(BaseModel):
    topk: Optional[int] = Field(None, description="Top K results")


class SummarySearchParams(BaseModel):
    topk: Optional[int] = Field(None, description="Top K results")
    similarity: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Similarity threshold")


class VisionSearchParams(BaseModel):
    topk: Optional[int] = Field(None, description="Top K results")
    similarity: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Similarity threshold")


IndexStateValue = Literal["ACTIVE", "FAILED", "NOT_ENABLED", "INDEXING"]
"""Per-modality index state advertised on search result metadata (§G.5).

* ``ACTIVE`` — the modality finished indexing and is currently serving.
* ``FAILED`` — the modality reached the retry budget without succeeding.
* ``NOT_ENABLED`` — the modality is disabled at the collection level.
* ``INDEXING`` — the modality is in flight (PENDING / RUNNING).

Clients use this to reason about the §F.4 short inconsistency window:
a vector hit that ships ``index_state_per_modality["fulltext"] ==
"INDEXING"`` tells the agent layer "fulltext for this document is
behind; we may have stale recall and should re-issue once it lands".
"""


IndexerModality = Literal["vector", "fulltext", "graph", "summary", "vision"]
"""The indexer modality that served a hit (§G.5).

Distinct from the ``modality`` field below, which keeps its D10.h-
locked ``"text" | "image"`` content-shape semantic. Per design pack
§G.5 the indexer modality disambiguator lives at metadata level so
agents can correlate hits across modalities without descending into
``SearchResultItem.recall_type`` (which carries the same info but at
the parent envelope).
"""


class SearchResultMetadata(BaseModel):
    """
    Public metadata carried by search result items.

    This intentionally allow-lists fields needed by clients and excludes raw
    index/storage metadata such as indexer, index_method, chat_id, object_path,
    and embedded node payloads.
    """

    model_config = ConfigDict(extra="forbid")

    source: Optional[str] = Field(None, description="Display source for the result")
    title: Optional[str] = Field(None, description="Human-readable title when available")
    collection_id: Optional[str] = Field(None, description="Collection identifier for client follow-up actions")
    document_id: Optional[str] = Field(None, description="Document identifier for client follow-up actions")
    chunk_id: Optional[str] = Field(
        None,
        description="Indexing-layer chunk identifier for follow-up `read_document_chunk` calls (D10 §A.9 R1 LOCKED handle).",
    )
    section_path: Optional[str] = Field(
        None,
        description="Slash-separated section path for follow-up `read_document_section` calls (D10 §A.9 R1 LOCKED handle).",
    )
    heading_anchor: Optional[str] = Field(
        None,
        description="Slug-style heading anchor (e.g. '#chapter-2-implementation') alternative to section_path (D10 §A.9 R1 LOCKED handle).",
    )
    asset_id: Optional[str] = Field(None, description="Asset identifier for image or binary references")
    mimetype: Optional[str] = Field(None, description="Asset MIME type when the result references an asset")
    page_idx: Optional[int] = Field(None, description="Zero-based page index when available")
    url: Optional[str] = Field(None, description="External source URL when available")
    modality: Optional[Literal["text", "image"]] = Field(None, description="Public content modality")

    # ---- §G.5 amendments (Wave 3 T3.2): per-modality independent
    # visibility under the §F.3 cutover model. -----------------------

    parse_version: Optional[str] = Field(
        None,
        description=(
            "Indexing-layer parse_version that produced this hit (§G.5). "
            "Lets agents detect mixed-version results across modalities in "
            "one response and reason about the §F.4 inconsistency window."
        ),
    )
    index_modality: Optional[IndexerModality] = Field(
        None,
        description=(
            "Indexer modality that served this hit (§G.5). Named "
            "``index_modality`` to disambiguate from the D10.h-locked "
            "content-shape ``modality`` field above; the design pack §G.5 "
            "uses bare ``modality`` but the existing public surface "
            "already binds that name to the content shape."
        ),
    )
    index_state_per_modality: Optional[dict[str, IndexStateValue]] = Field(
        None,
        description=(
            "Per-modality index state for the document this hit belongs "
            "to (§G.5). Keys: ``vector`` / ``fulltext`` / ``graph`` / "
            "``summary`` / ``vision``. Values: ``ACTIVE`` / ``FAILED`` / "
            "``NOT_ENABLED`` / ``INDEXING``. Clients can skip a modality "
            "currently FAILED or decide whether to wait + retry vs "
            "proceed with partial coverage."
        ),
    )

    @classmethod
    def from_raw(cls, metadata: Optional[dict[str, Any]]) -> Optional["SearchResultMetadata"]:
        if not isinstance(metadata, dict) or not metadata:
            return None

        def public_str(*keys: str) -> Optional[str]:
            for key in keys:
                value = metadata.get(key)
                if isinstance(value, str) and value:
                    return value
            return None

        page_idx = metadata.get("page_idx")
        if isinstance(page_idx, str) and page_idx.isdigit():
            page_idx = int(page_idx)
        if not isinstance(page_idx, int):
            page_idx = None

        modality = "image" if metadata.get("indexer") == "vision" else None
        if modality is None and any(metadata.get(key) for key in ("asset_id", "mimetype")):
            modality = "image"

        # §G.5 — extract per-modality index state if the upstream
        # search pipeline attached it. The value comes through as a
        # plain dict; we shallow-validate it here and pass through
        # only entries whose value matches the locked enum so a
        # malformed upstream cannot leak unknown values to clients.
        raw_index_state = metadata.get("index_state_per_modality")
        index_state_per_modality: Optional[dict[str, IndexStateValue]] = None
        if isinstance(raw_index_state, dict) and raw_index_state:
            allowed = ("ACTIVE", "FAILED", "NOT_ENABLED", "INDEXING")
            sanitized: dict[str, IndexStateValue] = {}
            for key, value in raw_index_state.items():
                if isinstance(key, str) and isinstance(value, str) and value in allowed:
                    sanitized[key] = value  # type: ignore[assignment]
            if sanitized:
                index_state_per_modality = sanitized

        # §G.5 — indexer modality (separate from D10.h content
        # ``modality``). Accept either ``index_modality`` or legacy
        # ``indexer`` key from the raw payload.
        raw_index_modality = metadata.get("index_modality") or metadata.get("indexer")
        index_modality: Optional[IndexerModality] = None
        if isinstance(raw_index_modality, str) and raw_index_modality in (
            "vector",
            "fulltext",
            "graph",
            "summary",
            "vision",
        ):
            index_modality = raw_index_modality  # type: ignore[assignment]

        data = {
            "source": public_str("source", "name"),
            "title": public_str("title"),
            "collection_id": public_str("collection_id"),
            "document_id": public_str("document_id"),
            "chunk_id": public_str("chunk_id"),
            "section_path": public_str("section_path"),
            "heading_anchor": public_str("heading_anchor"),
            "asset_id": public_str("asset_id"),
            "mimetype": public_str("mimetype"),
            "page_idx": page_idx,
            "url": public_str("url"),
            "modality": modality,
            "parse_version": public_str("parse_version"),
            "index_modality": index_modality,
            "index_state_per_modality": index_state_per_modality,
        }
        public_data = {key: value for key, value in data.items() if value is not None}
        return cls(**public_data) if public_data else None


class SearchResultItem(BaseModel):
    rank: Optional[int] = Field(None, description="Result rank")
    score: Optional[float] = Field(None, description="Result score")
    content: Optional[str] = Field(None, description="Result content")
    source: Optional[str] = Field(None, description="Source document or metadata")
    recall_type: Optional[
        Literal[
            "vector_search",
            "graph_search",
            "fulltext_search",
            "summary_search",
            "vision_search",
        ]
    ] = Field(None, description="Recall type")
    metadata: Optional[SearchResultMetadata] = Field(None, description="Public metadata of the result")

    @field_validator("metadata", mode="before")
    @classmethod
    def sanitize_metadata(cls, value):
        if value is None or isinstance(value, SearchResultMetadata):
            return value
        if isinstance(value, dict):
            return SearchResultMetadata.from_raw(value)
        return value


class SearchResult(BaseModel):
    id: Optional[str] = Field(None, description="The id of the search result")
    query: Optional[str] = None
    vector_search: Optional[VectorSearchParams] = None
    fulltext_search: Optional[FulltextSearchParams] = None
    graph_search: Optional[GraphSearchParams] = None
    summary_search: Optional[SummarySearchParams] = None
    vision_search: Optional[VisionSearchParams] = None
    items: Optional[list[SearchResultItem]] = None
    created: Optional[datetime] = Field(None, description="The creation time of the search result")


class SearchResultList(BaseModel):
    """
    A list of search results
    """

    items: Optional[list[SearchResult]] = None


class SearchRequest(BaseModel):
    """
    Search request
    """

    query: Optional[str] = None
    vector_search: Optional[VectorSearchParams] = None
    fulltext_search: Optional[FulltextSearchParams] = None
    graph_search: Optional[GraphSearchParams] = None
    summary_search: Optional[SummarySearchParams] = None
    vision_search: Optional[VisionSearchParams] = None
    save_to_history: Optional[bool] = Field(
        False,
        description="Whether to save search result to database history",
        examples=[True],
    )
    rerank: Optional[bool] = Field(
        False,
        description="Whether to enable rerank for search results",
        examples=[True],
    )


__all__ = [
    "VectorSearchParams",
    "FulltextSearchParams",
    "GraphSearchParams",
    "SummarySearchParams",
    "VisionSearchParams",
    "SearchResultMetadata",
    "SearchResultItem",
    "SearchResult",
    "SearchResultList",
    "SearchRequest",
]
