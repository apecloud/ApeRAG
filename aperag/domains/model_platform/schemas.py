"""Pydantic schemas for the model platform product model."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, confloat, conint

from aperag.schema.common import PageResult


class ModelCapability(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class ModelUseScenario(str, Enum):
    AGENT_CHAT = "agent_chat"
    COLLECTION_COMPLETION = "collection_completion"
    COLLECTION_EMBEDDING = "collection_embedding"
    RETRIEVAL_RERANK = "retrieval_rerank"
    BACKGROUND_TASK = "background_task"


class ModelUseStrategy(str, Enum):
    SINGLE = "single"
    FALLBACK = "fallback"


class ModelProvider(BaseModel):
    id: Optional[str] = None
    provider_type: str
    display_name: str
    description: Optional[str] = None
    supported_capabilities: list[ModelCapability] = Field(default_factory=list)
    account_schema: dict[str, Any] = Field(default_factory=dict)
    default_base_url: Optional[str] = None
    enabled: bool = True
    sort_order: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class ModelProviderList(BaseModel):
    items: list[ModelProvider] = Field(default_factory=list)
    pageResult: Optional[PageResult] = None


class ModelAccount(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    provider_type: str
    name: str
    display_name: str
    base_url: str
    status: Literal["ACTIVE", "INACTIVE"] = "ACTIVE"
    auth_config: dict[str, Any] = Field(default_factory=dict)
    last_validated_at: Optional[datetime] = None
    validation_error: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class ModelAccountCreate(BaseModel):
    provider_type: str
    name: str
    display_name: str
    base_url: str
    api_key: str
    auth_config: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelAccountUpdate(BaseModel):
    name: Optional[str] = None
    display_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    auth_config: Optional[dict[str, Any]] = None
    status: Optional[Literal["ACTIVE", "INACTIVE"]] = None
    extra: Optional[dict[str, Any]] = None


class ModelAccountList(BaseModel):
    items: list[ModelAccount] = Field(default_factory=list)
    pageResult: Optional[PageResult] = None


class Model(BaseModel):
    id: Optional[str] = None
    account_id: str
    provider_model_id: str
    display_name: str
    capability: ModelCapability
    runner_type: str
    runner_config: dict[str, Any] = Field(default_factory=dict)
    context_window: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    embedding_dimensions: Optional[int] = None
    supports_vision: bool = False
    supports_tool_calling: bool = False
    status: Literal["ACTIVE", "INACTIVE"] = "ACTIVE"
    extra: dict[str, Any] = Field(default_factory=dict)
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class ModelCreate(BaseModel):
    account_id: str
    provider_model_id: str
    display_name: str
    capability: ModelCapability
    runner_type: Optional[str] = None
    runner_config: dict[str, Any] = Field(default_factory=dict)
    context_window: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    embedding_dimensions: Optional[int] = None
    supports_vision: bool = False
    supports_tool_calling: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelUpdate(BaseModel):
    display_name: Optional[str] = None
    capability: Optional[ModelCapability] = None
    runner_type: Optional[str] = None
    runner_config: Optional[dict[str, Any]] = None
    context_window: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    embedding_dimensions: Optional[int] = None
    supports_vision: Optional[bool] = None
    supports_tool_calling: Optional[bool] = None
    status: Optional[Literal["ACTIVE", "INACTIVE"]] = None
    extra: Optional[dict[str, Any]] = None


class ModelList(BaseModel):
    items: list[Model] = Field(default_factory=list)
    pageResult: Optional[PageResult] = None


class ModelUse(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    scenario: ModelUseScenario
    capability: ModelCapability
    strategy: ModelUseStrategy = ModelUseStrategy.SINGLE
    primary_model_id: Optional[str] = None
    fallback_model_ids: list[str] = Field(default_factory=list)
    enabled: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class ModelUseUpdate(BaseModel):
    primary_model_id: Optional[str] = None
    fallback_model_ids: list[str] = Field(default_factory=list)
    strategy: ModelUseStrategy = ModelUseStrategy.SINGLE
    enabled: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelUseList(BaseModel):
    items: list[ModelUse] = Field(default_factory=list)
    pageResult: Optional[PageResult] = None


class ModelValidationResponse(BaseModel):
    ok: bool
    message: Optional[str] = None


class EmbeddingRequest(BaseModel):
    model_id: str = Field(..., description="ApeRAG model id")
    input: Union[str, list[str]]


class EmbeddingData(BaseModel):
    object: str = Field(..., examples=["embedding"])
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class Document1(BaseModel):
    text: str
    metadata: Optional[dict[str, Any]] = None


class RerankRequest(BaseModel):
    model_id: str = Field(..., description="ApeRAG rerank model id")
    query: str
    documents: Union[list[str], list[Document1]]
    top_k: Optional[conint(ge=1, le=1000)] = 10
    return_documents: Optional[bool] = True


class Document2(BaseModel):
    text: str
    metadata: Optional[dict[str, Any]] = None


class RerankDocument(BaseModel):
    index: int
    relevance_score: confloat(ge=0.0, le=1.0)
    document: Optional[Document2] = None


class RerankUsage(BaseModel):
    total_tokens: int


class RerankResponse(BaseModel):
    object: str
    data: list[RerankDocument]
    model: str
    usage: RerankUsage


__all__ = [
    "Document1",
    "Document2",
    "EmbeddingData",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "Model",
    "ModelAccount",
    "ModelAccountCreate",
    "ModelAccountList",
    "ModelAccountUpdate",
    "ModelCapability",
    "ModelCreate",
    "ModelList",
    "ModelProvider",
    "ModelProviderList",
    "ModelUpdate",
    "ModelUse",
    "ModelUseList",
    "ModelUseScenario",
    "ModelUseStrategy",
    "ModelUseUpdate",
    "ModelValidationResponse",
    "RerankDocument",
    "RerankRequest",
    "RerankResponse",
    "RerankUsage",
]
