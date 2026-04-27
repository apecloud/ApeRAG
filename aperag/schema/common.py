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

"""Shared Pydantic view model primitives reused across multiple domains.

Phase 3 Step 4b (msg=1505044c) extracts the collection- and document-
shaped schemas into :mod:`aperag.domains.knowledge_base.schemas`. Those
KB schemas transitively depend on a handful of low-level shapes
(``CollectionConfig``, pagination envelopes, chunk payloads) that are
also consumed by retrieval / bots / source / views code outside the
KB domain. Keeping those helpers in ``aperag.schema.view_models`` is
not an option because Phase 3 G1 bans any
``aperag/domains/**`` module from importing ``aperag.schema.view_models``.

This module is the shared home for those primitives. It is
intentionally **not** in the G1 legacy aggregate ban list so
``aperag.domains.<d>.schemas`` can depend on it directly.
``aperag.schema.view_models`` re-exports everything here for
back-compat with pre-extraction callers.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, confloat, conint, model_validator

_log = logging.getLogger(__name__)


class ModelSpec(BaseModel):
    """Inline model spec embedded in collection / bot config blobs.

    Post-#1697 the canonical shape is ``{model_id}``. Pre-#1697 callers
    (collections / bots already in the DB, plus any external clients
    that hand-build the JSON) wrote ``{model, model_service_provider,
    custom_llm_provider}`` instead. Pydantic silently dropped those
    extras after the schema cut, so existing rows would parse with
    ``model_id=None`` and the runtime resolver would 404. Per Weston
    msg=80e873c1 / Blocker C we keep parsing both shapes — when the
    legacy triple is present, it is stashed on the private
    ``__legacy_*__`` slots and a ``ModelPlatformService`` lookup at
    runtime resolves it to a real ``model_id`` (see
    :func:`aperag.schema.utils.resolve_model_spec_legacy`).
    """

    model_config = ConfigDict(protected_namespaces=(), extra="ignore")

    model_id: Optional[str] = Field(
        None,
        description="ApeRAG model id to use",
        examples=["mdl_default_chat"],
    )
    temperature: Optional[confloat(ge=0.0, le=2.0)] = Field(
        0.1,
        description="Controls randomness in the output. Values between 0 and 2. Lower values make output more focused and deterministic",
        examples=[0.1],
    )
    max_tokens: Optional[conint(ge=1)] = Field(
        None, description="Maximum number of tokens to generate", examples=[4096]
    )
    max_completion_tokens: Optional[conint(ge=1)] = Field(
        None,
        description="Upper bound for generated completion tokens, including visible and reasoning tokens",
        examples=[4096],
    )
    timeout: Optional[conint(ge=1)] = Field(None, description="Maximum execution time in seconds for the API request")
    top_n: Optional[conint(ge=1)] = Field(None, description="Number of top results to return when reranking documents")
    tags: Optional[list[str]] = Field(
        [],
        description="Tags for model categorization",
        examples=[["free", "recommend"]],
    )
    # Internal-only stash for the legacy triple. Not a public field
    # (excluded from ``model_dump``) — the runtime resolver fills
    # ``model_id`` from these when needed and that value is what
    # downstream code reads.
    legacy_model: Optional[str] = Field(default=None, exclude=True, repr=False)
    legacy_provider: Optional[str] = Field(default=None, exclude=True, repr=False)
    legacy_custom_llm_provider: Optional[str] = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="before")
    @classmethod
    def _hoist_legacy_triple(cls, data):
        if not isinstance(data, dict):
            return data
        # Don't clobber a real ``model_id`` — the new shape always wins.
        if data.get("model_id"):
            return data
        legacy_model = data.get("model")
        legacy_provider = data.get("model_service_provider")
        legacy_custom = data.get("custom_llm_provider")
        if legacy_model or legacy_provider:
            data = {**data}
            data["legacy_model"] = legacy_model
            data["legacy_provider"] = legacy_provider
            data["legacy_custom_llm_provider"] = legacy_custom
            _log.debug(
                "ModelSpec parsed legacy shape; stashed for runtime resolve",
                extra={"legacy_model": legacy_model, "legacy_provider": legacy_provider},
            )
        return data

    @property
    def model(self) -> Optional[str]:
        return self.model_id

    def has_legacy_triple(self) -> bool:
        """``True`` when this spec was parsed from the pre-#1697 shape
        and still needs ``ModelPlatformService.resolve_legacy_model_id``
        before downstream code uses ``model_id``.
        """
        return bool(self.legacy_model or self.legacy_provider)


class KnowledgeGraphConfig(BaseModel):
    """
    Configuration for knowledge graph generation
    """

    entity_types: Optional[list[str]] = Field(
        [
            "organization",
            "person",
            "geo",
            "event",
            "product",
            "technology",
            "date",
            "category",
        ],
        description="List of entity types to extract during graph indexing",
        examples=[["organization", "person", "geo", "event"]],
    )


class IndexPrompts(BaseModel):
    """
    Custom prompts for various index types
    """

    graph: Optional[str] = Field(None, description="Custom prompt for graph/entity extraction")
    summary: Optional[str] = Field(None, description="Custom prompt for document summarization")
    vision: Optional[str] = Field(None, description="Custom prompt for image analysis")


class CollectionConfig(BaseModel):
    source: Optional[str] = Field(
        "system",
        description="Source system identifier. Only `system` is supported.",
        examples=["system"],
    )
    enable_vector: Optional[bool] = Field(True, description="Whether to enable vector index")
    enable_fulltext: Optional[bool] = Field(True, description="Whether to enable fulltext index")
    # Wave 3 ships graph modality structurally implemented but gated
    # in :class:`aperag.indexing.worker_factory.ProductionWorkerFactory`
    # because the §D.3.6 Nebula / Postgres ``LineageGraphStore`` adapter
    # + LightRAG-style LLM extractor are Wave 4 scope. Default flipped
    # to False so new collections do not opt into the placeholder path
    # by accident; Wave 4 release flips it back to True. Per architect
    # msg=c79e9a3f.
    enable_knowledge_graph: Optional[bool] = Field(False, description="Whether to enable knowledge graph index")
    # Wave 4 T8 chunk 4b: lineage graph backend dispatch.
    # ``postgres`` is the §D.3.5 reference adapter (no extra
    # infrastructure needed beyond the application's own PostgreSQL);
    # ``neo4j`` and ``nebula`` are the production graph databases for
    # deployments that already run them. ``worker_factory`` reads this
    # field to construct the right :class:`LineageGraphStore` per
    # collection. New collections that opt into knowledge graph default
    # to ``postgres`` since it shares the existing app DB.
    graph_backend_type: Optional[Literal["postgres", "neo4j", "nebula"]] = Field(
        "postgres",
        description="Lineage graph backend for knowledge graph indexing",
    )
    enable_summary: Optional[bool] = Field(False, description="Whether to enable summary index")
    enable_vision: Optional[bool] = Field(False, description="Whether to enable vision index")
    knowledge_graph_config: Optional[KnowledgeGraphConfig] = Field(
        default_factory=lambda: KnowledgeGraphConfig.model_validate(
            {
                "entity_types": [
                    "organization",
                    "person",
                    "geo",
                    "event",
                    "product",
                    "technology",
                    "date",
                    "category",
                ]
            }
        )
    )
    index_prompts: Optional[IndexPrompts] = None
    language: Optional[Literal["zh-CN", "en-US", "ja-JP", "ko-KR"]] = Field(
        "zh-CN",
        description="Language for the collection content and processing",
        examples=["zh-CN"],
    )
    embedding: Optional[ModelSpec] = None
    completion: Optional[ModelSpec] = None


class PageResult(BaseModel):
    """
    PageResult info (deprecated, use paginatedResponse instead)
    """

    page_number: Optional[int] = Field(None, description="The page number")
    page_size: Optional[int] = Field(None, description="The page size")
    count: Optional[int] = Field(None, description="The total count of items")


class PaginatedResponse(BaseModel):
    total: Optional[conint(ge=0)] = Field(None, description="Total number of items", examples=[100])
    page: Optional[conint(ge=1)] = Field(None, description="Current page number", examples=[1])
    page_size: Optional[conint(ge=1)] = Field(None, description="Number of items per page", examples=[10])
    total_pages: Optional[conint(ge=1)] = Field(None, description="Total number of pages", examples=[10])
    has_next: Optional[bool] = Field(None, description="Whether there is a next page", examples=[True])
    has_prev: Optional[bool] = Field(None, description="Whether there is a previous page", examples=[False])


class VisionChunk(BaseModel):
    id: Optional[str] = None
    asset_id: Optional[str] = None
    text: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class Chunk(BaseModel):
    id: Optional[str] = None
    text: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


__all__ = [
    "ModelSpec",
    "KnowledgeGraphConfig",
    "IndexPrompts",
    "CollectionConfig",
    "PageResult",
    "PaginatedResponse",
    "VisionChunk",
    "Chunk",
]
