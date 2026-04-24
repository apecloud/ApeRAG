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

# Pydantic view models used by FastAPI routes. These models are now part of
# the code-first OpenAPI source, exported by scripts/export_openapi.py.

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, confloat, conint

# Phase 3 Step 4b (msg=1505044c): shared primitives moved to
# `aperag.schema.common` so the `knowledge_base` domain can depend on
# them without tripping Phase 3 G1 (``aperag.schema.view_models`` is on
# the legacy-aggregate ban list, ``aperag.schema.common`` is not).
from aperag.schema.common import (  # noqa: F401  re-export for back-compat
    Chunk,
    CollectionConfig,
    IndexPrompts,
    KnowledgeGraphConfig,
    ModelSpec,
    PageResult,
    PaginatedResponse,
    VisionChunk,
)


class Agent(BaseModel):
    completion: Optional[ModelSpec] = None
    system_prompt_template: Optional[str] = None
    query_prompt_template: Optional[str] = None
    collections: Optional[list[Collection]] = None


class BotConfig(BaseModel):
    agent: Optional[Agent] = None


class Bot(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[Literal["knowledge", "common", "agent"]] = Field(
        None, description="The type of bot", examples=["agent"]
    )
    config: Optional[BotConfig] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class BotList(BaseModel):
    """
    A list of bots
    """

    items: Optional[list[Bot]] = None
    pageResult: Optional[PageResult] = None


class FailResponse(BaseModel):
    code: Optional[str] = Field(None, description="Error code", examples=["400"])
    message: Optional[str] = Field(None, description="Error message", examples=["Invalid request"])


class BotCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[Literal["agent"]] = Field(None, description="The supported bot type", examples=["agent"])
    config: Optional[BotConfig] = None


class BotUpdate(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    config: Optional[BotConfig] = None


class BotUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    config: Optional[BotConfig] = None


class Chat(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    bot_id: Optional[str] = None
    peer_id: Optional[str] = None
    peer_type: Optional[Literal["system", "feishu", "weixin", "weixin_official", "web", "dingtalk"]] = None
    status: Optional[Literal["active", "archived"]] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class ChatList(PaginatedResponse):
    """
    A list of chats with pagination
    """

    items: Optional[list[Chat]] = None


class ChatCreate(BaseModel):
    title: Optional[str] = None


class Reference(BaseModel):
    score: Optional[float] = None
    text: Optional[str] = None
    image_uri: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class File(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None


class ChatMessage(BaseModel):
    id: Optional[str] = None
    part_id: Optional[str] = None
    type: Optional[
        Literal[
            "welcome",
            "message",
            "start",
            "stop",
            "error",
            "tool_call_result",
            "thinking",
            "references",
        ]
    ] = None
    timestamp: Optional[float] = None
    role: Optional[Literal["human", "ai"]] = None
    data: Optional[str] = None
    references: Optional[list[Reference]] = None
    urls: Optional[list[str]] = None
    files: Optional[list[File]] = None


class ChatDetails(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    bot_id: Optional[str] = None
    peer_id: Optional[str] = None
    peer_type: Optional[Literal["system", "feishu", "weixin", "weixin_official", "web", "dingtalk"]] = None
    history: Optional[list[list[ChatMessage]]] = Field(
        None,
        description="Array of conversation turns, where each turn is an array of message parts",
    )
    status: Optional[Literal["active", "archived"]] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class ChatUpdate(BaseModel):
    title: Optional[str] = None


class TitleGenerateRequest(BaseModel):
    max_length: Optional[conint(ge=6, le=50)] = Field(20, description="Maximum length of the generated title")
    language: Optional[Literal["zh-CN", "en-US", "ja-JP", "ko-KR"]] = Field(
        "zh-CN", description="Language for the title generation (IETF BCP 47 tag)"
    )
    turns: Optional[conint(ge=1)] = Field(1, description="Number of most recent conversation turns to consider")


class TitleGenerateResponse(BaseModel):
    title: str = Field(..., description="Generated title string")


class TurnFeedback(BaseModel):
    turn_id: str
    type: Literal["good", "bad"]
    tag: Optional[Literal["Harmful", "Unsafe", "Fake", "Unhelpful", "Other"]] = None
    message: Optional[str] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class TurnFeedbackList(BaseModel):
    items: list[TurnFeedback]


class Feedback(BaseModel):
    type: Optional[Literal["good", "bad"]] = None
    tag: Optional[Literal["Harmful", "Unsafe", "Fake", "Unhelpful", "Other"]] = None
    message: Optional[str] = None


TurnFeedbackWrite = Feedback


# ConfirmDocumentsRequest / FetchUrlRequest / DeleteDocumentsRequest /
# DeleteDocumentsResponse were carved to
# ``aperag.domains.knowledge_base.schemas`` in Phase 3 Step 5a
# alongside the KB route move; the end-of-file try block re-imports
# them. UploadDocumentResponse / FailedDocument / ConfirmDocumentsResponse /
# FetchUrlResultItem / FetchUrlResponse / StagedDocumentsResponse were
# carved out in Phase 3 Step 5b3. The end-of-file try block re-imports
# them so pre-migration callers (``from aperag.schema.view_models
# import UploadDocumentResponse``)
# keep resolving the same class objects.


class Settings(BaseModel):
    use_mineru: Optional[bool] = Field(None, description="Whether to use MinerU")
    mineru_api_token: Optional[str] = Field(None, description="API token for MinerU")
    use_markitdown: Optional[bool] = Field(None, description="Whether to use MarkItDown")


class ParserHealthItem(BaseModel):
    key: str
    label: str
    status: Literal["ok", "warning", "error", "disabled"]
    detail: str


class ParserSupportTier(BaseModel):
    key: str
    label: str
    category: Literal["official", "conditional", "enhanced", "optional"]
    parser: str
    formats: list[str]
    status: Literal["available", "limited", "unavailable", "disabled"]
    detail: str
    requirements: list[str]


class ParserHealthReport(BaseModel):
    default_parser: str
    parser_order: list[str]
    available_extensions: list[str]
    dependencies: list[ParserHealthItem]
    services: list[ParserHealthItem]
    support_tiers: list[ParserSupportTier]
    warnings: list[str]
    recommendations: list[str]


class PromptDetail(BaseModel):
    """
    Detailed prompt information with source and customization status
    """

    content: Optional[str] = Field(None, description="Actual prompt content (resolved with priority)")
    source: Optional[Literal["user", "system", "hardcoded"]] = Field(None, description="Source of the prompt")
    customized: Optional[bool] = Field(None, description="Whether user has customized this prompt")
    description: Optional[str] = Field(None, description="Optional description")


class UserPromptsResponse(BaseModel):
    """
    User's prompt configuration with all types
    """

    agent_system: Optional[PromptDetail] = None
    agent_query: Optional[PromptDetail] = None
    index_graph: Optional[PromptDetail] = None
    index_summary: Optional[PromptDetail] = None
    index_vision: Optional[PromptDetail] = None


class Prompts(BaseModel):
    """
    Prompts to update (all fields are optional, only provided fields will be updated)
    """

    agent_system: Optional[str] = Field(None, description="Agent system prompt (persona definition)")
    agent_query: Optional[str] = Field(None, description="Agent query prompt template")
    index_graph: Optional[str] = Field(None, description="Graph index prompt for entity/relation extraction")
    index_summary: Optional[str] = Field(None, description="Summary index prompt for document summarization")
    index_vision: Optional[str] = Field(None, description="Vision index prompt for image content extraction")


class UpdateUserPromptsRequest(BaseModel):
    prompts: Prompts = Field(
        ...,
        description="Prompts to update (all fields are optional, only provided fields will be updated)",
        examples=[
            {
                "agent_system": "You are a professional technical support assistant...",
                "index_graph": "Extract entities from medical text...",
            }
        ],
    )


class UpdateUserPromptsResponse(BaseModel):
    message: Optional[str] = None
    updated: Optional[list[str]] = None


class DeleteUserPromptResponse(BaseModel):
    message: Optional[str] = None
    type: Optional[str] = None
    new_content: Optional[str] = None
    source: Optional[Literal["system", "hardcoded"]] = None


class ResetPromptsRequest(BaseModel):
    types: Optional[list[str]] = Field(None, description="Prompt types to reset, omit to reset all")


class ResetPromptsResponse(BaseModel):
    message: Optional[str] = None
    reset: Optional[list[str]] = None


class SystemPromptDetail(BaseModel):
    type: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None


class SystemPromptsResponse1(BaseModel):
    content: Optional[str] = None
    description: Optional[str] = None


class SystemPromptsResponse(RootModel[Optional[dict[str, SystemPromptsResponse1]]]):
    """
    System default prompts
    """

    root: Optional[dict[str, SystemPromptsResponse1]] = None


class PreviewRequest(BaseModel):
    template: str
    variables: Optional[dict[str, Any]] = None


class PreviewResponse(BaseModel):
    rendered: Optional[str] = None


class ValidateRequest(BaseModel):
    type: Literal["agent_system", "agent_query", "index_graph", "index_summary", "index_vision"]
    template: str


class ValidateResponse(BaseModel):
    valid: Optional[bool] = None
    errors: Optional[list[str]] = None
    warnings: Optional[list[str]] = None


class TargetEntityDataRequest(BaseModel):
    """
    Optional target entity configuration. If not specified, auto-select entity with highest degree.
    """

    entity_name: Optional[str] = Field(
        None,
        description="Target entity name. If not specified, auto-select entity with highest degree",
    )
    entity_type: Optional[str] = Field(None, description="Entity type for the target entity")
    description: Optional[str] = Field(None, description="Description for the target entity")
    source_id: Optional[str] = Field(None, description="Source ID for the target entity")
    file_path: Optional[str] = Field(None, description="File path for the target entity")


class NodeMergeRequest(BaseModel):
    """
    Request to merge multiple graph nodes directly using entity IDs.

    """

    model_config = ConfigDict(
        extra="forbid",
    )
    entity_ids: list[str] = Field(
        ...,
        description="List of entity IDs to merge directly",
        examples=[["墨香居", "书店", "旧书店"]],
        min_length=1,
    )
    target_entity_data: Optional[TargetEntityDataRequest] = None


class TargetEntityDataResponse(BaseModel):
    """
    Complete data of the target entity after merge
    """

    entity_name: str = Field(
        ...,
        description="The entity name that was kept (merge target)",
        examples=["墨香居"],
    )
    entity_type: str = Field(..., description="Entity type of the target entity", examples=["ORGANIZATION"])
    description: str = Field(
        ...,
        description="Merged description of the target entity",
        examples=["墨香居是这条老巷子里唯一的旧书店，经营着各种书籍，承载了老板李明华的情怀。"],
    )
    source_id: Optional[str] = Field(None, description="Source ID information", examples=["chunk-001,chunk-002"])
    file_path: Optional[str] = Field(None, description="File path information", examples=["story.txt,book.txt"])


class NodeMergeResponse(BaseModel):
    """
    Response containing node merge results
    """

    status: Literal["success", "error"] = Field(..., description="Status of the merge operation", examples=["success"])
    message: str = Field(
        ...,
        description="Detailed message about the merge operation",
        examples=["Successfully merged 2 entities into 墨香居"],
    )
    entity_ids: list[str] = Field(
        ...,
        description="Entity IDs that were merged",
        examples=[["墨香居", "书店", "旧书店"]],
    )
    target_entity_data: TargetEntityDataResponse
    source_entities: list[str] = Field(
        ...,
        description="List of entities that were merged into the target",
        examples=[["书店", "旧书店"]],
    )
    redirected_edges: conint(ge=0) = Field(
        ...,
        description="Number of edges that were redirected during merge",
        examples=[12],
    )
    merged_description_length: conint(ge=0) = Field(..., description="Length of the merged description", examples=[512])
    suggestion_id: Optional[str] = Field(
        None,
        description="Suggestion ID if this merge was based on a suggestion",
        examples=["msug123"],
    )


# SharingStatusResponse, MineruTokenTestRequest, MineruTokenTestResponse
# were carved to ``aperag.domains.knowledge_base.schemas`` in Phase 3
# Step 5a. They remain importable from here through the end-of-file
# try block re-export.


class SharedCollectionConfig(BaseModel):
    """
    Configuration settings for shared collection features
    """

    enable_vector: bool = Field(..., description="Whether vector search is enabled")
    enable_fulltext: bool = Field(..., description="Whether fulltext search is enabled")
    enable_knowledge_graph: bool = Field(..., description="Whether knowledge graph is enabled")
    enable_summary: bool = Field(..., description="Whether summary generation is enabled")
    enable_vision: bool = Field(..., description="Whether vision processing is enabled")


class SharedCollection(BaseModel):
    """
    Shared Collection information for marketplace users
    """

    id: str = Field(..., description="Collection ID")
    title: str = Field(..., description="Collection title")
    description: Optional[str] = Field(None, description="Collection description")
    owner_user_id: str = Field(..., description="Original owner user ID")
    owner_username: Optional[str] = Field(None, description="Original owner username")
    subscription_id: Optional[str] = Field(
        None,
        description="Subscription record ID (has value if subscribed, null if not subscribed)",
    )
    gmt_subscribed: Optional[datetime] = Field(None, description="Subscription time (only has value when subscribed)")
    subscription_count: int = Field(..., description="Total number of subscriptions")
    config: SharedCollectionConfig = Field(..., description="Collection configuration settings")


class SharedCollectionList(BaseModel):
    """
    Shared Collection list response
    """

    items: list[SharedCollection] = Field(..., description="List of shared Collections")
    total: int = Field(..., description="Total count (for pagination)")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")


class ApiKey(BaseModel):
    id: Optional[str] = None
    key: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class ApiKeyList(BaseModel):
    """
    A list of API keys
    """

    items: Optional[list[ApiKey]] = None
    pageResult: Optional[PageResult] = None


class ApiKeyCreate(BaseModel):
    description: Optional[str] = None


class ApiKeyUpdate(BaseModel):
    description: Optional[str] = None


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


# Auth0 / Authing / Logto / Auth / Config moved to
# ``aperag.domains.identity.schemas`` in Phase 4 Step 4-S3a; the
# end-of-file try block re-imports them so pre-migration callers
# (auth routes, OpenAPI spec builders) keep working.


class AuditLog(BaseModel):
    """
    Audit log entry
    """

    id: Optional[str] = Field(None, description="Audit log ID")
    user_id: Optional[str] = Field(None, description="User ID who performed the action")
    username: Optional[str] = Field(None, description="Username for display")
    resource_type: Optional[
        Literal[
            "collection",
            "document",
            "bot",
            "chat",
            "message",
            "api_key",
            "llm",
            "llm_provider",
            "llm_provider_model",
            "model_service_provider",
            "user",
            "flow",
            "search",
            "index",
        ]
    ] = Field(None, description="Type of resource")
    resource_id: Optional[str] = Field(None, description="ID of the resource (extracted at query time)")
    api_name: Optional[str] = Field(None, description="API operation name")
    http_method: Optional[str] = Field(None, description="HTTP method (POST, PUT, DELETE)")
    path: Optional[str] = Field(None, description="API path")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    start_time: Optional[int] = Field(None, description="Request start time (milliseconds since epoch)")
    end_time: Optional[int] = Field(None, description="Request end time (milliseconds since epoch)")
    duration_ms: Optional[int] = Field(None, description="Request duration in milliseconds (calculated)")
    request_data: Optional[str] = Field(None, description="Request data (JSON string)")
    response_data: Optional[str] = Field(None, description="Response data (JSON string)")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    created: Optional[datetime] = Field(None, description="Created timestamp")


class AuditLogList(PaginatedResponse):
    """
    List of audit logs with pagination
    """

    items: Optional[list[AuditLog]] = Field(None, description="Audit log entries")


# InvitationCreate / Invitation / InvitationList / Register / User /
# Login / UserList / ChangePassword moved to
# ``aperag.domains.identity.schemas`` in Phase 4 Step 4-S3a; the
# end-of-file try block re-imports them.


class QuotaInfo(BaseModel):
    """
    Quota information for a specific quota type
    """

    quota_type: str = Field(..., description="Type of quota", examples=["max_collection_count"])
    quota_limit: int = Field(..., description="Maximum allowed usage", examples=[10])
    current_usage: int = Field(..., description="Current usage count", examples=[3])
    remaining: int = Field(..., description="Remaining quota available", examples=[7])


class UserQuotaInfo(BaseModel):
    """
    Complete quota information for a user
    """

    user_id: str = Field(..., description="User ID", examples=["user123"])
    username: Optional[str] = Field(None, description="Username", examples=["john_doe"])
    email: Optional[str] = Field(None, description="User email", examples=["john@example.com"])
    role: str = Field(..., description="User role", examples=["rw"])
    quotas: list[QuotaInfo] = Field(..., description="List of quota information")


class UserQuotaList(BaseModel):
    """
    List of user quota information (admin view)
    """

    items: list[UserQuotaInfo] = Field(..., description="List of user quota information")


class QuotaUpdateRequest(BaseModel):
    """
    Request to update user quotas (supports both single and batch updates)
    """

    max_collection_count: Optional[conint(ge=0)] = Field(None, description="New limit for collection count")
    max_document_count: Optional[conint(ge=0)] = Field(None, description="New limit for document count")
    max_document_count_per_collection: Optional[conint(ge=0)] = Field(
        None, description="New limit for documents per collection"
    )
    max_bot_count: Optional[conint(ge=0)] = Field(None, description="New limit for bot count")


class UpdatedQuota(BaseModel):
    quota_type: str = Field(
        ...,
        description="Type of quota that was updated",
        examples=["max_collection_count"],
    )
    old_limit: int = Field(..., description="Previous quota limit", examples=[10])
    new_limit: int = Field(..., description="New quota limit", examples=[20])


class QuotaUpdateResponse(BaseModel):
    """
    Response after updating user quotas (supports both single and batch updates)
    """

    success: bool = Field(..., description="Whether the update was successful", examples=[True])
    message: str = Field(..., description="Status message", examples=["Quotas updated successfully"])
    user_id: str = Field(..., description="User ID that was updated", examples=["user123"])
    updated_quotas: list[UpdatedQuota] = Field(..., description="List of updated quotas")


class SystemDefaultQuotas(BaseModel):
    """
    System default quota configuration
    """

    max_collection_count: conint(ge=0) = Field(..., description="Default maximum collection count", examples=[10])
    max_document_count: conint(ge=0) = Field(..., description="Default maximum document count", examples=[1000])
    max_document_count_per_collection: conint(ge=0) = Field(
        ..., description="Default maximum documents per collection", examples=[100]
    )
    max_bot_count: conint(ge=0) = Field(..., description="Default maximum bot count", examples=[5])


class SystemDefaultQuotasResponse(BaseModel):
    """
    Response containing system default quotas
    """

    quotas: SystemDefaultQuotas


class SystemDefaultQuotasUpdateRequest(BaseModel):
    """
    Request to update system default quotas
    """

    quotas: SystemDefaultQuotas


class SystemDefaultQuotasUpdateResponse(BaseModel):
    """
    Response after updating system default quotas
    """

    success: bool = Field(..., description="Whether the update was successful", examples=[True])
    message: str = Field(
        ...,
        description="Status message",
        examples=["System default quotas updated successfully"],
    )
    quotas: SystemDefaultQuotas


class QuestionSet(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    collection_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    gmt_created: Optional[datetime] = None
    gmt_updated: Optional[datetime] = None


class QuestionSetList(BaseModel):
    items: Optional[list[QuestionSet]] = None
    total: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None


class QuestionType(RootModel[Literal["FACTUAL", "INFERENTIAL", "USER_DEFINED"]]):
    root: Literal["FACTUAL", "INFERENTIAL", "USER_DEFINED"] = Field(..., description="Question type enumeration")


class Question(BaseModel):
    id: Optional[str] = None
    question_set_id: Optional[str] = None
    question_type: Optional[QuestionType] = None
    question_text: Optional[str] = None
    ground_truth: Optional[str] = None
    gmt_created: Optional[datetime] = None
    gmt_updated: Optional[datetime] = None


class QuestionSetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    collection_id: Optional[str] = None
    questions: Optional[list[Question]] = Field(
        None, description="A list of questions. Maximum 1000 questions are allowed."
    )


class LLMConfig(BaseModel):
    model_name: Optional[str] = None
    model_service_provider: Optional[str] = None
    custom_llm_provider: Optional[str] = None


class QuestionSetGenerate(BaseModel):
    collection_id: str
    llm_config: Optional[LLMConfig] = None
    question_count: Optional[int] = None
    prompt: Optional[str] = None


class QuestionSetDetail(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    collection_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    questions: Optional[list[Question]] = None
    gmt_created: Optional[datetime] = None
    gmt_updated: Optional[datetime] = None


class QuestionSetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class Question1(BaseModel):
    question_text: str
    ground_truth: str
    question_type: Optional[QuestionType] = None


class QuestionsAdd(BaseModel):
    questions: list[Question1]


class QuestionUpdate(BaseModel):
    question_text: Optional[str] = None
    ground_truth: Optional[str] = None
    question_type: Optional[QuestionType] = None


class EvaluationStatus(RootModel[Literal["PENDING", "RUNNING", "PAUSED", "COMPLETED", "FAILED"]]):
    root: Literal["PENDING", "RUNNING", "PAUSED", "COMPLETED", "FAILED"] = Field(
        ..., description="Evaluation task lifecycle status"
    )


class Evaluation(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    name: Optional[str] = None
    collection_id: Optional[str] = None
    question_set_id: Optional[str] = None
    agent_llm_config: Optional[LLMConfig] = None
    judge_llm_config: Optional[LLMConfig] = None
    status: Optional[EvaluationStatus] = None
    error_message: Optional[str] = None
    total_questions: Optional[int] = None
    completed_questions: Optional[int] = None
    average_score: Optional[float] = None
    gmt_created: Optional[datetime] = None
    gmt_updated: Optional[datetime] = None


class EvaluationList(BaseModel):
    items: Optional[list[Evaluation]] = None
    total: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None


class EvaluationCreate(BaseModel):
    name: str
    collection_id: str
    question_set_id: str
    agent_llm_config: LLMConfig
    judge_llm_config: LLMConfig


class EvaluationItemStatus(RootModel[Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]]):
    root: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] = Field(
        ..., description="Evaluation item lifecycle status"
    )


class EvaluationItem(BaseModel):
    id: Optional[str] = None
    evaluation_id: Optional[str] = None
    question_id: Optional[str] = None
    status: Optional[EvaluationItemStatus] = None
    question_text: Optional[str] = None
    ground_truth: Optional[str] = None
    rag_answer: Optional[str] = None
    rag_answer_details: Optional[dict[str, Any]] = None
    llm_judge_score: Optional[int] = None
    llm_judge_reasoning: Optional[str] = None
    gmt_created: Optional[datetime] = None
    gmt_updated: Optional[datetime] = None


class Config1(BaseModel):
    collection_id: Optional[str] = None
    question_set_id: Optional[str] = None
    agent_llm_config: Optional[dict[str, Any]] = None
    judge_llm_config: Optional[dict[str, Any]] = None


class EvaluationDetail(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    collection_name: Optional[str] = None
    question_set_name: Optional[str] = None
    status: Optional[EvaluationStatus] = None
    average_score: Optional[float] = None
    config: Optional[Config1] = None
    items: Optional[list[EvaluationItem]] = None
    gmt_created: Optional[datetime] = None


class EvaluationChatWithAgentRequest(BaseModel):
    collection_id: str
    agent_llm_config: LLMConfig
    question_text: str
    language: Optional[str] = None


class ChatSuccessResponse(BaseModel):
    messages: Optional[list[ChatMessage]] = None


class AgentErrorResponse(BaseModel):
    type: Optional[Literal["error"]] = Field(None, description="The type of the response, must be 'error'.")
    id: Optional[str] = None
    data: Optional[str] = Field(None, description="Error message")
    timestamp: Optional[int] = None


class EvaluationChatWithAgentResponse(RootModel[Union[ChatSuccessResponse, AgentErrorResponse]]):
    root: Union[ChatSuccessResponse, AgentErrorResponse]


class AgentMessage(BaseModel):
    """
    Message format for agent-type bots with additional capabilities
    """

    query: str = Field(..., description="User query", examples=["Tell me about ApeRAG features"])
    collections: list[Collection] = Field(
        ...,
        description="List of collection objects to search in",
        examples=[
            [
                {"id": "col_123", "title": "Example Collection"},
                {"id": "col_456", "title": "Another Collection"},
            ]
        ],
    )
    completion: Optional[ModelSpec] = Field(
        None,
        description="Model specification for completion including provider and model details",
    )
    web_search_enabled: Optional[bool] = Field(False, description="Whether to enable web search", examples=[True])
    language: Optional[
        Literal[
            "en-US",
            "zh-CN",
            "zh-TW",
            "ja-JP",
            "ko-KR",
            "fr-FR",
            "de-DE",
            "es-ES",
            "it-IT",
            "pt-BR",
            "ru-RU",
        ]
    ] = Field("en-US", description="Language preference for the response", examples=["en-US"])
    files: Optional[list[File]] = None


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


# ---------------------------------------------------------------------------
# Phase 2 compatibility re-exports
# ---------------------------------------------------------------------------
#
# The retrieval + knowledge_graph Pydantic view models were relocated
# to ``aperag.domains.retrieval.schemas`` and
# ``aperag.domains.knowledge_graph.schemas`` by the Phase 2 hard-cut.
# Their class definitions above were deleted. These re-export lines
# keep legacy consumers working — e.g. ``aperag.schema.view_models.SearchRequest``
# still imports the canonical class — until the Phase 3 DB-split PR
# retires ``aperag.schema.view_models`` itself. Domain code must import
# directly from the canonical modules; consumers outside
# ``aperag/domains/**`` may continue to use either path during the
# transition window.
from aperag.domains.knowledge_graph.schemas import (  # noqa: E402,F401
    GraphCurationRunSummary,
    GraphEdge,
    GraphEdgeProperties,
    GraphLabelsResponse,
    GraphMergeSuggestionEntity,
    GraphMergeSuggestionItem,
    GraphNode,
    GraphNodeProperties,
    KnowledgeGraph,
    MergeSuggestionsRequest,
    MergeSuggestionsResponse,
    MergeSuggestionsRunResponse,
    SuggestionActionMergeResult,
    SuggestionActionRequest,
    SuggestionActionResponse,
)
from aperag.domains.retrieval.schemas import (  # noqa: E402,F401
    FulltextSearchParams,
    GraphSearchParams,
    SearchRequest,
    SearchResult,
    SearchResultItem,
    SearchResultList,
    SearchResultMetadata,
    SummarySearchParams,
    VectorSearchParams,
    VisionSearchParams,
)

# Phase 3 Step 4b back-compat shim (msg=1505044c): the 11 KB-domain
# schemas below were extracted to ``aperag.domains.knowledge_base.schemas``
# and are re-imported here so pre-migration callers
# (``from aperag.schema.view_models import Collection``), Pydantic
# forward-ref resolution in ``Agent.collections`` etc., and FastAPI
# response_model bindings continue to see the canonical class objects
# from the KB domain module. When this module loads first the import
# below succeeds; when ``aperag.domains.knowledge_base.schemas`` loads
# first, the import raises ``ImportError`` (circular) and the KB
# module's end-of-file ``_bind_view_models_reexports`` hook completes
# the binding. Phase 6 cleanup will remove this shim after every caller
# is rewritten to use the KB path.
try:
    from aperag.domains.knowledge_base.schemas import (  # noqa: E402, F401
        Collection,
        CollectionCreate,
        CollectionSummaryTriggerResponse,
        CollectionUpdate,
        CollectionView,
        CollectionViewList,
        ConfirmDocumentsRequest,
        ConfirmDocumentsResponse,
        DeleteDocumentsRequest,
        DeleteDocumentsResponse,
        Document,
        DocumentList,
        DocumentPreview,
        FailedDocument,
        FetchUrlRequest,
        FetchUrlResponse,
        FetchUrlResultItem,
        MineruTokenTestRequest,
        MineruTokenTestResponse,
        RebuildIndexesRequest,
        RebuildIndexesResponse,
        SharingStatusResponse,
        StagedDocumentsResponse,
        UploadDocumentResponse,
    )
except ImportError:
    # Circular import window: KB schemas module is still loading the
    # shared helpers above (CollectionConfig / PageResult / etc.). The
    # KB module's ``_bind_view_models_reexports`` hook will set the
    # same attributes on this module once it finishes. Callers that
    # imported view_models first will see them bound at module-load
    # completion; callers that imported KB first will see them bound
    # via the KB hook.
    pass

# Phase 4 Step 4-S3a identity domain schemas dual-hook re-export.
# Same symmetric pattern as the KB block above.
try:
    from aperag.domains.identity.schemas import (  # noqa: E402, F401
        Auth,
        Auth0,
        Authing,
        ChangePassword,
        Config,
        Invitation,
        InvitationCreate,
        InvitationList,
        Login,
        Logto,
        Register,
        User,
        UserList,
    )
except ImportError:
    pass
