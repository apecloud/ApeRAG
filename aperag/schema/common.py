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

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, confloat, conint


class ModelSpec(BaseModel):
    model: Optional[str] = Field(
        None,
        description="The name of the language model to use",
        examples=["gpt-4o-mini"],
    )
    model_service_provider: Optional[str] = Field(
        None,
        description="Used for querying auth information (api_key/api_base/...) for a model service provider.",
        examples=["openai"],
    )
    custom_llm_provider: Optional[str] = Field(
        None,
        description="Used for Non-OpenAI LLMs (e.g. 'bedrock' for amazon.titan-tg1-large)",
        examples=["openai"],
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
    enable_knowledge_graph: Optional[bool] = Field(True, description="Whether to enable knowledge graph index")
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
