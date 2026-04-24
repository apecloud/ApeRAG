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

"""Canonical Pydantic view models for the ``model_platform`` domain.

Phase 4 Step 4-S3d (largest sub-step of 4-S3) carves the LLM-provider
and embedding / rerank API-shape schemas out of
``aperag.schema.view_models``. Dual-hook symmetric re-export keeps
pre-migration callers working.

26 schemas grouped by concern:

* Model filtering (``TagFilterCondition`` / ``TagFilterRequest``)
* Provider-aggregate views (``ModelConfig`` / ``ModelConfigList``)
* Per-scenario defaults (``DefaultModelConfig`` / ``DefaultModelsResponse``
  / ``DefaultModelsUpdateRequest``)
* Provider + provider-model CRUD (``LlmProvider`` / ``LlmProviderModel``
  / ``LlmConfigurationResponse`` / ``LlmProviderCreateWithApiKey`` /
  ``LlmProviderUpdateWithApiKey`` / ``LlmProviderModelList`` /
  ``LlmProviderModelCreate`` / ``LlmProviderModelCreateRequest`` /
  ``LlmProviderModelUpdate``)
* Embedding API shapes (``EmbeddingRequest`` / ``EmbeddingData`` /
  ``EmbeddingUsage`` / ``EmbeddingResponse``)
* Rerank API shapes (``Document1`` / ``RerankRequest`` / ``Document2``
  / ``RerankDocument`` / ``RerankUsage`` / ``RerankResponse``)

``aperag.llm.*`` stays as shared infrastructure (Phase 4 canonical
msg=d47fa490 Section 7); the embedding / rerank schemas here are the
HTTP-public contract shape, not the ``aperag.llm.*`` runtime wrappers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, confloat, conint

from aperag.schema.common import ModelSpec, PageResult

__all__ = [
    "TagFilterCondition",
    "TagFilterRequest",
    "ModelConfig",
    "ModelConfigList",
    "DefaultModelConfig",
    "DefaultModelsResponse",
    "DefaultModelsUpdateRequest",
    "LlmProvider",
    "LlmProviderModel",
    "LlmConfigurationResponse",
    "LlmProviderCreateWithApiKey",
    "LlmProviderUpdateWithApiKey",
    "LlmProviderModelList",
    "LlmProviderModelCreate",
    "LlmProviderModelCreateRequest",
    "LlmProviderModelUpdate",
    "EmbeddingRequest",
    "EmbeddingData",
    "EmbeddingUsage",
    "EmbeddingResponse",
    "Document1",
    "RerankRequest",
    "Document2",
    "RerankDocument",
    "RerankUsage",
    "RerankResponse",
]


class TagFilterCondition(BaseModel):
    operation: Literal["AND", "OR"] = Field(
        ...,
        description="Logical operation for tags in this condition",
        examples=["AND"],
    )
    tags: list[str] = Field(
        ...,
        description="List of tags for this condition",
        examples=[["free", "recommend"]],
    )


class TagFilterRequest(BaseModel):
    """
    Tag filtering request. Empty request body or empty tag_filters returns recommend models by default.
    """

    tag_filters: Optional[list[TagFilterCondition]] = Field(
        None,
        description='List of tag filter conditions (OR relationship between conditions). If not provided or empty, returns models with "recommend" tag by default.',
        examples=[
            [
                {"operation": "AND", "tags": ["free", "recommend"]},
                {"operation": "OR", "tags": ["openai", "gpt"]},
            ]
        ],
    )


class ModelConfig(BaseModel):
    name: Optional[str] = None
    completion_dialect: Optional[str] = None
    embedding_dialect: Optional[str] = None
    rerank_dialect: Optional[str] = None
    label: Optional[str] = None
    allow_custom_base_url: Optional[bool] = None
    base_url: Optional[str] = None
    embedding: Optional[list[ModelSpec]] = None
    completion: Optional[list[ModelSpec]] = None
    rerank: Optional[list[ModelSpec]] = None


class ModelConfigList(BaseModel):
    items: Optional[list[ModelConfig]] = None
    pageResult: Optional[PageResult] = None


class DefaultModelConfig(BaseModel):
    scenario: Literal[
        "default_for_collection_completion",
        "default_for_agent_completion",
        "default_for_embedding",
        "default_for_rerank",
        "default_for_background_task",
    ] = Field(
        ...,
        description="The scenario for which this default model is configured",
        examples=["default_for_embedding"],
    )
    custom_llm_provider: Optional[str] = None
    provider_name: Optional[str] = Field(None, description="The name of the model provider", examples=["openai"])
    model: Optional[str] = Field(None, description="The name of the model", examples=["text-embedding-3-small"])


class DefaultModelsResponse(BaseModel):
    items: list[DefaultModelConfig] = Field(
        ..., description="List of default model configurations for different scenarios"
    )


class DefaultModelsUpdateRequest(BaseModel):
    defaults: list[DefaultModelConfig] = Field(
        ...,
        description="List of default model configurations to update",
        examples=[
            [
                {
                    "scenario": "default_for_embedding",
                    "provider_name": "openai",
                    "model": "text-embedding-3-small",
                },
                {
                    "scenario": "default_for_collection_completion",
                    "provider_name": "openai",
                    "model": "gpt-4o-mini",
                },
            ]
        ],
    )


class LlmProvider(BaseModel):
    name: str = Field(..., description="Unique provider name identifier", examples=["openai"])
    user_id: str = Field(
        ...,
        description='User ID of the provider owner, "public" for system providers',
        examples=["public"],
    )
    label: str = Field(..., description="Human-readable provider display name", examples=["OpenAI"])
    completion_dialect: Optional[str] = Field(
        "openai",
        description="API dialect for completion/chat APIs",
        examples=["openai"],
    )
    embedding_dialect: Optional[str] = Field(
        "openai", description="API dialect for embedding APIs", examples=["openai"]
    )
    rerank_dialect: Optional[str] = Field("jina_ai", description="API dialect for rerank APIs", examples=["jina_ai"])
    allow_custom_base_url: Optional[bool] = Field(False, description="Whether custom base URLs are allowed")
    base_url: str = Field(
        ...,
        description="Default API base URL for this provider",
        examples=["https://api.openai.com/v1"],
    )
    extra: Optional[str] = Field(None, description="Additional configuration data in JSON format")
    api_key: Optional[str] = Field(None, description="API key for this provider (if configured by user)")
    created: Optional[datetime] = Field(None, description="Creation timestamp")
    updated: Optional[datetime] = Field(None, description="Last update timestamp")


class LlmProviderModel(BaseModel):
    provider_name: str = Field(..., description="Reference to LLMProvider.name", examples=["openai"])
    api: Literal["completion", "embedding", "rerank"] = Field(
        ..., description="API type for this model", examples=["completion"]
    )
    model: str = Field(..., description="Model name/identifier", examples=["gpt-4o-mini"])
    custom_llm_provider: str = Field(..., description="Custom LLM provider implementation", examples=["openai"])
    context_window: Optional[int] = Field(None, description="Context window size (total tokens)", examples=[128000])
    max_input_tokens: Optional[int] = Field(None, description="Maximum input tokens", examples=[120000])
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens", examples=[8000])
    tags: Optional[list[str]] = Field(
        [],
        description="Tags for model categorization",
        examples=[["free", "recommend"]],
    )
    created: Optional[datetime] = Field(None, description="Creation timestamp")
    updated: Optional[datetime] = Field(None, description="Last update timestamp")


class LlmConfigurationResponse(BaseModel):
    providers: list[LlmProvider] = Field(..., description="List of LLM providers")
    models: list[LlmProviderModel] = Field(..., description="List of LLM provider models")


class LlmProviderCreateWithApiKey(BaseModel):
    name: Optional[str] = Field(
        None,
        description="Unique provider name identifier (auto-generated if not provided)",
    )
    label: str = Field(..., description="Human-readable provider display name")
    completion_dialect: Optional[str] = Field("openai", description="API dialect for completion/chat APIs")
    embedding_dialect: Optional[str] = Field("openai", description="API dialect for embedding APIs")
    rerank_dialect: Optional[str] = Field("jina_ai", description="API dialect for rerank APIs")
    allow_custom_base_url: Optional[bool] = Field(False, description="Whether custom base URLs are allowed")
    base_url: str = Field(..., description="Default API base URL for this provider")
    extra: Optional[str] = Field(None, description="Additional configuration data in JSON format")
    api_key: Optional[str] = Field(None, description="Optional API key for this provider")
    status: Optional[Literal["enable", "disable"]] = Field(
        None,
        description="Provider status - enable to create/update API key, disable to remove API key",
    )


class LlmProviderUpdateWithApiKey(BaseModel):
    label: Optional[str] = Field(None, description="Human-readable provider display name")
    completion_dialect: Optional[str] = Field(None, description="API dialect for completion/chat APIs")
    embedding_dialect: Optional[str] = Field(None, description="API dialect for embedding APIs")
    rerank_dialect: Optional[str] = Field(None, description="API dialect for rerank APIs")
    allow_custom_base_url: Optional[bool] = Field(None, description="Whether custom base URLs are allowed")
    base_url: Optional[str] = Field(None, description="Default API base URL for this provider")
    extra: Optional[str] = Field(None, description="Additional configuration data in JSON format")
    api_key: Optional[str] = Field(None, description="Optional API key for this provider")
    status: Optional[Literal["enable", "disable"]] = Field(
        None,
        description="Provider status - enable to create/update API key, disable to remove API key",
    )


class LlmProviderModelList(BaseModel):
    items: Optional[list[LlmProviderModel]] = None
    pageResult: Optional[PageResult] = None


class LlmProviderModelCreate(BaseModel):
    provider_name: str = Field(..., description="Reference to LLMProvider.name")
    api: Literal["completion", "embedding", "rerank"] = Field(..., description="API type for this model")
    model: str = Field(..., description="Model name/identifier")
    custom_llm_provider: str = Field(..., description="Custom LLM provider implementation")
    context_window: Optional[int] = Field(None, description="Context window size (total tokens)", examples=[128000])
    max_input_tokens: Optional[int] = Field(None, description="Maximum input tokens", examples=[120000])
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens", examples=[8000])
    tags: Optional[list[str]] = Field([], description="Tags for model categorization")


class LlmProviderModelCreateRequest(BaseModel):
    api: Literal["completion", "embedding", "rerank"] = Field(..., description="API type for this model")
    model: str = Field(..., description="Model name/identifier")
    custom_llm_provider: str = Field(..., description="Custom LLM provider implementation")
    context_window: Optional[int] = Field(None, description="Context window size (total tokens)", examples=[128000])
    max_input_tokens: Optional[int] = Field(None, description="Maximum input tokens", examples=[120000])
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens", examples=[8000])
    tags: Optional[list[str]] = Field([], description="Tags for model categorization")


class LlmProviderModelUpdate(BaseModel):
    custom_llm_provider: Optional[str] = Field(None, description="Custom LLM provider implementation")
    context_window: Optional[int] = Field(None, description="Context window size (total tokens)", examples=[128000])
    max_input_tokens: Optional[int] = Field(None, description="Maximum input tokens", examples=[120000])
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens", examples=[8000])
    tags: Optional[list[str]] = Field(None, description="Tags for model categorization")


class EmbeddingRequest(BaseModel):
    """
    Request to generate embeddings for text inputs
    """

    provider: str = Field(
        ...,
        description="LLM provider name (e.g., openai, anthropic)",
        examples=["openai"],
    )
    model: str = Field(
        ...,
        description="Model name for embedding generation",
        examples=["text-embedding-3-small"],
    )
    input: Union[str, list[str]]


class EmbeddingData(BaseModel):
    """
    Individual embedding result
    """

    object: str = Field(..., description="Object type identifier", examples=["embedding"])
    embedding: list[float] = Field(
        ...,
        description="The embedding vector as a list of floats",
        examples=[[0.0023064255, -0.009327292, 0.015797421, 0.0012345678]],
    )
    index: int = Field(
        ...,
        description="Index of the input text corresponding to this embedding",
        examples=[0],
    )


class EmbeddingUsage(BaseModel):
    """
    Token usage information for the embedding request
    """

    prompt_tokens: int = Field(..., description="Number of tokens in the input text(s)", examples=[16])
    total_tokens: int = Field(
        ...,
        description="Total number of tokens used (same as prompt_tokens for embeddings)",
        examples=[16],
    )


class EmbeddingResponse(BaseModel):
    """
    Response containing generated embeddings in OpenAI-compatible format
    """

    object: str = Field(..., description="Object type identifier", examples=["list"])
    data: list[EmbeddingData] = Field(..., description="List of embedding results")
    model: str = Field(
        ...,
        description="Model used for embedding generation",
        examples=["text-embedding-3-small"],
    )
    usage: EmbeddingUsage


class Document1(BaseModel):
    text: str = Field(
        ...,
        description="Document text content",
        examples=["Paris is the capital of France."],
    )
    metadata: Optional[dict[str, Any]] = Field(
        None,
        description="Optional document metadata",
        examples=[{"id": "doc_123", "source": "wikipedia"}],
    )


class RerankRequest(BaseModel):
    """
    Request to rerank documents based on query relevance
    """

    provider: str = Field(
        ...,
        description="LLM provider name (e.g., cohere, jina_ai)",
        examples=["cohere"],
    )
    model: str = Field(..., description="Model name for reranking", examples=["rerank-english-v3.0"])
    query: str = Field(
        ...,
        description="Search query to rank documents against",
        examples=["What is the capital of France?"],
    )
    documents: Union[list[str], list[Document1]]
    top_k: Optional[conint(ge=1, le=1000)] = Field(
        10, description="Maximum number of top-ranked documents to return", examples=[3]
    )
    return_documents: Optional[bool] = Field(
        True,
        description="Whether to return document content in response",
        examples=[True],
    )


class Document2(BaseModel):
    """
    Document content and metadata (only present if return_documents=true)
    """

    text: str = Field(
        ...,
        description="Document text content",
        examples=["Paris is the capital of France."],
    )
    metadata: Optional[dict[str, Any]] = Field(
        None,
        description="Document metadata if provided in the request",
        examples=[{"id": "doc_123", "source": "wikipedia"}],
    )


class RerankDocument(BaseModel):
    """
    Individual reranked document result
    """

    index: int = Field(
        ...,
        description="Original index of the document in the input array",
        examples=[0],
    )
    relevance_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Relevance score between 0 and 1 (higher is more relevant)",
        examples=[0.95],
    )
    document: Optional[Document2] = Field(
        None,
        description="Document content and metadata (only present if return_documents=true)",
    )


class RerankUsage(BaseModel):
    """
    Token usage information for the rerank request
    """

    total_tokens: int = Field(
        ...,
        description="Total number of tokens processed (query + all documents)",
        examples=[156],
    )


class RerankResponse(BaseModel):
    """
    Response containing reranked documents in industry-standard format
    """

    object: str = Field(..., description="Object type identifier", examples=["list"])
    data: list[RerankDocument] = Field(
        ...,
        description="List of reranked documents ordered by relevance (highest first)",
    )
    model: str = Field(..., description="Model used for reranking", examples=["rerank-english-v3.0"])
    usage: RerankUsage


def _bind_view_models_reexports() -> None:
    """Phase 3 / Phase 4 dual-hook pattern — see identity/schemas.py
    for the full symmetric-load-order explanation."""

    import sys

    _vm = sys.modules.get("aperag.schema.view_models")
    if _vm is None:
        return
    for _name in __all__:
        setattr(_vm, _name, globals()[_name])


_bind_view_models_reexports()
