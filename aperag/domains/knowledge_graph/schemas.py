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

"""Pydantic view models owned by the ``knowledge_graph`` canonical domain.

These classes were previously defined in ``aperag.schema.view_models``;
they were relocated here by the Phase 2 hard-cut so the
``knowledge_graph`` domain no longer reaches back into the legacy
aggregate module. OpenAPI component names and shapes are intentionally
preserved byte-for-byte so the frontend types regenerate 1-to-1 from
the new source.

``aperag/schema/view_models.py`` keeps re-export lines at the bottom
pointing here, so legacy callers of
``aperag.schema.view_models.GraphLabelsResponse`` etc. still work until
the Phase 3 DB-split PR deletes the legacy aggregate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, confloat, conint, field_validator


class GraphLabelsResponse(BaseModel):
    """
    Response containing available graph labels
    """

    labels: list[str] = Field(
        ...,
        description="List of available node labels in the knowledge graph",
        examples=[["墨香居", "李明华", "林晓雯", "深夜读书会"]],
    )


class GraphNodeProperties(BaseModel):
    """
    Public node properties for graph visualization.
    """

    model_config = ConfigDict(
        extra="forbid",
    )
    entity_id: Optional[str] = Field(None, description="Entity identifier", examples=["墨香居"])
    entity_name: Optional[str] = Field(None, description="Entity display name", examples=["墨香居"])
    entity_type: Optional[str] = Field(None, description="Type of the entity", examples=["organization"])
    description: Optional[str] = Field(
        None,
        description="Description of the entity",
        examples=["墨香居是这条老巷子里唯一的旧书店，经营着各种书籍，承载了老板李明华的情怀。"],
    )
    source_chunk_count: Optional[conint(ge=0)] = Field(
        None,
        description="Number of source chunks supporting this entity; raw chunk IDs are not exposed",
        examples=[3],
    )


class GraphNode(BaseModel):
    """
    Knowledge graph node representing an entity
    """

    id: str = Field(
        ...,
        description="Unique identifier for the node (entity name)",
        examples=["墨香居"],
    )
    labels: list[str] = Field(..., description="Labels associated with the node", examples=[["墨香居"]])
    properties: GraphNodeProperties = Field(..., description="Public node properties")


class GraphEdgeProperties(BaseModel):
    """
    Public edge properties for graph visualization.
    """

    model_config = ConfigDict(
        extra="forbid",
    )
    weight: Optional[float] = Field(None, description="Relationship weight/strength", examples=[9])
    description: Optional[str] = Field(
        None,
        description="Description of the relationship",
        examples=["深夜读书会是墨香居的新活动，旨在提升书店的活力和吸引顾客。"],
    )
    keywords: Optional[str] = Field(
        None,
        description="Keywords associated with the relationship",
        examples=["书店活力,活动"],
    )
    source_chunk_count: Optional[conint(ge=0)] = Field(
        None,
        description="Number of source chunks supporting this relationship; raw chunk IDs are not exposed",
        examples=[2],
    )


class GraphEdge(BaseModel):
    """
    Knowledge graph edge representing a relationship
    """

    id: str = Field(
        ...,
        description="Unique identifier for the edge",
        examples=["墨香居-深夜读书会"],
    )
    type: Optional[str] = Field("DIRECTED", description="Type of the relationship", examples=["DIRECTED"])
    source: str = Field(..., description="Source node ID", examples=["墨香居"])
    target: str = Field(..., description="Target node ID", examples=["深夜读书会"])
    properties: GraphEdgeProperties = Field(..., description="Public edge properties")


class KnowledgeGraph(BaseModel):
    """
    Knowledge graph containing nodes and edges
    """

    nodes: list[GraphNode] = Field(..., description="List of nodes in the knowledge graph")
    edges: list[GraphEdge] = Field(..., description="List of edges in the knowledge graph")
    is_truncated: bool = Field(
        ...,
        description="Whether the graph was truncated due to size limits",
        examples=[False],
    )


class MergeSuggestionsRequest(BaseModel):
    """
    Start a new graph-curation analysis run.
    """

    model_config = ConfigDict(
        extra="forbid",
    )


class GraphCurationRunSummary(BaseModel):
    """
    Summary of an asynchronous graph-curation run.
    """

    id: str = Field(..., description="Run ID", examples=["gcr_abcd1234efgh5678"])
    collection_id: str = Field(..., description="Collection ID", examples=["col123"])
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] = Field(
        ..., description="Run lifecycle status", examples=["COMPLETED"]
    )
    stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Best-effort execution stats for the latest run",
    )
    error_message: Optional[str] = Field(
        None,
        description="Failure message if the run failed",
        examples=["LLM adjudication timed out"],
    )
    created: Optional[datetime] = Field(None, description="Creation timestamp", examples=["2026-04-23T00:00:00Z"])
    updated: Optional[datetime] = Field(None, description="Last update timestamp", examples=["2026-04-23T00:02:00Z"])
    started: Optional[datetime] = Field(None, description="Run start timestamp", examples=["2026-04-23T00:00:05Z"])
    finished: Optional[datetime] = Field(
        None,
        description="Run finish timestamp",
        examples=["2026-04-23T00:02:00Z"],
    )


class GraphMergeSuggestionEntity(BaseModel):
    """
    Snapshot of an entity at suggestion-generation time.
    """

    entity_id: str = Field(..., description="Entity ID", examples=["e_moxiangju"])
    entity_name: str = Field(..., description="Entity name", examples=["墨香居"])
    entity_type: str = Field(..., description="Entity type", examples=["ORGANIZATION"])
    description: str = Field(..., description="Entity description", examples=["这条老巷子里唯一的旧书店"])
    source_chunk_count: conint(ge=0) = Field(
        ...,
        description="Number of source chunks referenced by this entity",
        examples=[3],
    )


class GraphMergeSuggestionItem(BaseModel):
    """
    Persisted merge suggestion produced by graph curation.
    """

    id: str = Field(..., description="Suggestion ID", examples=["gcs_abcd1234efgh5678"])
    run_id: str = Field(..., description="Run that produced this suggestion", examples=["gcr_abcd1234efgh5678"])
    collection_id: str = Field(..., description="Collection ID", examples=["col123"])
    status: Literal["PENDING", "ACCEPTED", "REJECTED", "EXPIRED", "SUPERSEDED"] = Field(
        ..., description="Suggestion lifecycle status", examples=["PENDING"]
    )
    entity_ids: list[str] = Field(
        ...,
        description="Entity IDs in the candidate merge set",
        examples=[["e_moxiangju", "e_old_bookstore"]],
    )
    entities: list[GraphMergeSuggestionEntity] = Field(
        ...,
        description="Entity snapshots captured when the suggestion was generated",
    )
    target_entity_id: str = Field(
        ...,
        description="Recommended surviving entity ID if the suggestion is accepted",
        examples=["e_moxiangju"],
    )
    confidence_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Aggregated confidence score for this suggestion",
        examples=[0.91],
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation from pairwise LLM adjudication",
        examples=["两个实体都在描述同一家旧书店，名称和上下文高度重合。"],
    )
    evidence: Optional[dict[str, Any]] = Field(
        None,
        description="Structured supporting evidence used to generate the suggestion",
    )
    resolution_note: Optional[str] = Field(
        None,
        description="System note explaining why the suggestion left pending state",
        examples=["superseded_by:gcs_other"],
    )
    created: Optional[datetime] = Field(None, description="Creation timestamp", examples=["2026-04-23T00:02:00Z"])
    updated: Optional[datetime] = Field(None, description="Last update timestamp", examples=["2026-04-23T00:05:00Z"])
    operated_at: Optional[datetime] = Field(
        None,
        description="User-operation timestamp for accepted/rejected suggestions",
        examples=["2026-04-23T00:05:00Z"],
    )


class MergeSuggestionsRunResponse(BaseModel):
    """
    Response returned when starting a graph-curation run.
    """

    run: GraphCurationRunSummary
    started: bool = Field(..., description="Whether a new run was actually scheduled", examples=[True])
    message: str = Field(
        ...,
        description="Human-readable status message",
        examples=["Graph curation run started"],
    )


class MergeSuggestionsResponse(BaseModel):
    """
    Latest persisted graph-curation run and its suggestions.
    """

    run: Optional[GraphCurationRunSummary] = Field(
        None,
        description="Latest graph-curation run for this collection, if any",
    )
    suggestions: list[GraphMergeSuggestionItem] = Field(
        ...,
        description="Suggestions from the latest run, ordered by confidence score",
    )


class SuggestionActionRequest(BaseModel):
    """
    Request to take action on a merge suggestion
    """

    model_config = ConfigDict(
        extra="forbid",
    )
    action: Literal["accept", "reject"] = Field(
        ...,
        description="Action to take on the suggestion (case-insensitive, e.g., 'Accept', 'REJECT', 'accept')",
        examples=["accept"],
    )

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class SuggestionActionMergeResult(BaseModel):
    """
    Merge result returned when accepting a graph-curation suggestion.
    """

    target_entity_id: str = Field(..., description="Surviving entity ID after merge", examples=["e_moxiangju"])
    merged_source_ids: list[str] = Field(..., description="Merged-away entity IDs", examples=[["e_old_bookstore"]])
    description: str = Field(..., description="Merged description of the surviving entity")
    source_chunk_ids: list[str] = Field(
        ...,
        description="Source chunk IDs preserved on the surviving entity",
        examples=[["chunk_1", "chunk_8"]],
    )
    edges_redirected: conint(ge=0) = Field(..., description="Number of redirected edges", examples=[5])
    edges_collapsed: conint(ge=0) = Field(..., description="Number of duplicate edges collapsed", examples=[2])


class SuggestionActionResponse(BaseModel):
    """
    Response containing suggestion action results
    """

    status: Literal["success", "error"] = Field(..., description="Status of the action operation", examples=["success"])
    message: str = Field(
        ...,
        description="Detailed message about the action operation",
        examples=["Suggestion msug123 has been accepted and merge completed"],
    )
    suggestion_id: str = Field(..., description="The suggestion ID that was processed", examples=["msug123"])
    action: Literal["accept", "reject"] = Field(
        ...,
        description="The action that was performed (normalized to lowercase)",
        examples=["accept"],
    )
    suggestion_status: Literal["ACCEPTED", "REJECTED"] = Field(
        ...,
        description="Suggestion status after action processing",
        examples=["ACCEPTED"],
    )
    merge_result: Optional[SuggestionActionMergeResult] = Field(
        None,
        description="Merge operation result (only present when action is 'accept')",
    )


# ----------------------------------------------------------------------
# Wave 7 §K.12.6/§K.12.7 — internal-only graph search/expand/detail shapes
# ----------------------------------------------------------------------
#
# These models back the three MCP-facing endpoints under
# ``/collections/{id}/graphs/entities/...`` and ``/graphs/subgraph/...``.
# They are explicitly NOT user-facing — frontend has no consumer for
# them — but they appear in OpenAPI regen so SDK callers can bind to
# them statically.
#
# Per §K.12 invariant #12 (grep-zero LightRAG) the names are
# ``Graph*`` only; per invariant #11 (D-3 candidate detection writes
# only) these are read-only shapes — no mutation surface.


class GraphSearchEntity(BaseModel):
    """Entity returned by ``/graphs/entities/search`` and
    ``/graphs/entities/{name}``.

    Mirrors :class:`aperag.indexing.graph.EntityWithLineage` projected
    to the public-facing fields. The ``description`` field collapses
    ``compacted_description`` (when present) and ``description_parts``
    using the same rule as :func:`GraphSearchService.compose_context`,
    so MCP callers see one stable text — independent of whether the
    GraphIndexCompactor (task #2) has run yet.
    """

    name: str = Field(..., description="Canonical entity name", examples=["墨香居"])
    entity_type: str = Field(..., description="Entity type", examples=["organization"])
    description: str = Field(
        ...,
        description="Compacted description if available, otherwise joined description parts",
        examples=["墨香居是这条老巷子里唯一的旧书店。"],
    )
    source_chunk_count: int = Field(
        0,
        description="Number of source chunks supporting this entity",
        examples=[3],
        ge=0,
    )


class GraphRelationView(BaseModel):
    """Relation projection used by subgraph expansion responses."""

    source: str = Field(..., description="Source entity name", examples=["李明华"])
    target: str = Field(..., description="Target entity name", examples=["墨香居"])
    description: str = Field(
        ...,
        description="Compacted description if available, otherwise joined description parts",
        examples=["李明华经营墨香居"],
    )


class GraphEntitiesSearchResponse(BaseModel):
    """Response for ``GET /graphs/entities/search``."""

    entities: list[GraphSearchEntity] = Field(
        ...,
        description="Entities returned by vector recall, ordered by descending similarity score",
    )


class GraphSubgraphExpandRequest(BaseModel):
    """Request body for ``POST /graphs/subgraph/expand``."""

    model_config = ConfigDict(extra="forbid")

    entity_names: list[str] = Field(
        ...,
        description="Anchor entity names to expand from",
        examples=[["墨香居", "李明华"]],
        min_length=1,
    )
    hops: int = Field(
        1,
        description="Number of hops to expand (bounded by backend implementation)",
        examples=[1],
        ge=1,
        le=3,
    )


class GraphSubgraphExpandResponse(BaseModel):
    """Response for ``POST /graphs/subgraph/expand``."""

    entities: list[GraphSearchEntity] = Field(
        ...,
        description="Entities reachable from the anchor names within ``hops``",
    )
    relations: list[GraphRelationView] = Field(
        ...,
        description="Relations connecting the returned entities",
    )


__all__ = [
    "GraphLabelsResponse",
    "GraphNodeProperties",
    "GraphNode",
    "GraphEdgeProperties",
    "GraphEdge",
    "KnowledgeGraph",
    "MergeSuggestionsRequest",
    "GraphCurationRunSummary",
    "GraphMergeSuggestionEntity",
    "GraphMergeSuggestionItem",
    "MergeSuggestionsRunResponse",
    "MergeSuggestionsResponse",
    "SuggestionActionRequest",
    "SuggestionActionMergeResult",
    "SuggestionActionResponse",
    "GraphSearchEntity",
    "GraphRelationView",
    "GraphEntitiesSearchResponse",
    "GraphSubgraphExpandRequest",
    "GraphSubgraphExpandResponse",
]
