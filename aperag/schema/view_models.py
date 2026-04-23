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

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, RootModel, confloat, conint, field_validator


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


class PageResult(BaseModel):
    """
    PageResult info (deprecated, use paginatedResponse instead)
    """

    page_number: Optional[int] = Field(None, description="The page number")
    page_size: Optional[int] = Field(None, description="The page size")
    count: Optional[int] = Field(None, description="The total count of items")


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


class Chat(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    bot_id: Optional[str] = None
    peer_id: Optional[str] = None
    peer_type: Optional[Literal["system", "feishu", "weixin", "weixin_official", "web", "dingtalk"]] = None
    status: Optional[Literal["active", "archived"]] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


class PaginatedResponse(BaseModel):
    total: Optional[conint(ge=0)] = Field(None, description="Total number of items", examples=[100])
    page: Optional[conint(ge=1)] = Field(None, description="Current page number", examples=[1])
    page_size: Optional[conint(ge=1)] = Field(None, description="Number of items per page", examples=[10])
    total_pages: Optional[conint(ge=1)] = Field(None, description="Total number of pages", examples=[10])
    has_next: Optional[bool] = Field(None, description="Whether there is a next page", examples=[True])
    has_prev: Optional[bool] = Field(None, description="Whether there is a previous page", examples=[False])


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


class DeleteDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(..., description="Document IDs to delete", min_length=1)


class DeleteDocumentsResponse(BaseModel):
    deleted_ids: list[str] = Field(..., description="Document IDs accepted for deletion")
    status: Literal["success"] = Field(..., description="Batch deletion status")


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


class VisionChunk(BaseModel):
    id: Optional[str] = None
    asset_id: Optional[str] = None
    text: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class Chunk(BaseModel):
    id: Optional[str] = None
    text: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class DocumentPreview(BaseModel):
    doc_object_path: Optional[str] = Field(None, description="The path to the document object.")
    doc_filename: Optional[str] = Field(None, description="The name of the document.")
    converted_pdf_object_path: Optional[str] = Field(None, description="The path to the converted PDF object.")
    markdown_content: Optional[str] = Field(None, description="The markdown content of the document.")
    chunks: Optional[list[Chunk]] = None
    vision_chunks: Optional[list[VisionChunk]] = None


class UploadDocumentResponse(BaseModel):
    document_id: str = Field(..., description="ID of the uploaded document")
    filename: str = Field(..., description="Name of the uploaded file")
    size: int = Field(..., description="Size of the uploaded file in bytes")
    status: Literal["UPLOADED", "PENDING", "RUNNING", "COMPLETE", "FAILED", "DELETED", "EXPIRED"] = Field(
        ...,
        description="Status of the document (UPLOADED for new uploads, or existing status for duplicate files)",
    )


class ConfirmDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(..., description="List of document IDs to confirm", min_length=1)


class FailedDocument(BaseModel):
    document_id: Optional[str] = None
    name: Optional[str] = Field(None, description="Name of the document")
    error: Optional[str] = None


class ConfirmDocumentsResponse(BaseModel):
    confirmed_count: int = Field(..., description="Number of documents successfully confirmed")
    failed_count: int = Field(..., description="Number of documents that failed to confirm")
    failed_documents: Optional[list[FailedDocument]] = Field(None, description="Details of failed confirmations")


class FetchUrlRequest(BaseModel):
    urls: list[AnyUrl] = Field(
        ...,
        description="List of URLs to fetch and import (max 10)",
        examples=[["https://example.com/article1", "https://example.com/article2"]],
    )


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
    asset_id: Optional[str] = Field(None, description="Asset identifier for image or binary references")
    mimetype: Optional[str] = Field(None, description="Asset MIME type when the result references an asset")
    page_idx: Optional[int] = Field(None, description="Zero-based page index when available")
    url: Optional[str] = Field(None, description="External source URL when available")
    modality: Optional[Literal["text", "image"]] = Field(None, description="Public content modality")

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

        data = {
            "source": public_str("source", "name"),
            "title": public_str("title"),
            "collection_id": public_str("collection_id"),
            "document_id": public_str("document_id"),
            "asset_id": public_str("asset_id"),
            "mimetype": public_str("mimetype"),
            "page_idx": page_idx,
            "url": public_str("url"),
            "modality": modality,
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


class SharingStatusResponse(BaseModel):
    """
    Simple sharing status response
    """

    is_published: bool = Field(..., description="Whether published to marketplace")
    published_at: Optional[datetime] = Field(None, description="Publication time, null when not published")


class CollectionSummaryTriggerResponse(BaseModel):
    """Trigger-response envelope for POST /collections/{collection_id}/summary/generate."""

    collection_id: str = Field(..., description="Collection id whose summary generation was triggered")
    success: bool = Field(..., description="Whether the background job was scheduled")
    message: str = Field(..., description="Human-readable status message")
    summary_status: Literal["PENDING", "GENERATING"] = Field(
        ...,
        description="Server-side summary state after the trigger call",
    )


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


class Auth0(BaseModel):
    auth_domain: Optional[str] = None
    auth_app_id: Optional[str] = None


class Authing(BaseModel):
    auth_domain: Optional[str] = None
    auth_app_id: Optional[str] = None


class Logto(BaseModel):
    auth_domain: Optional[str] = None
    auth_app_id: Optional[str] = None


class Auth(BaseModel):
    type: Optional[Literal["none", "auth0", "authing", "logto", "cookie"]] = None
    auth0: Optional[Auth0] = None
    authing: Optional[Authing] = None
    logto: Optional[Logto] = None


class Config(BaseModel):
    admin_user_exists: Optional[bool] = Field(None, description="Whether the admin user exists")
    auth: Optional[Auth] = None
    login_methods: Optional[list[str]] = Field(
        None,
        description="Available login methods",
        examples=[["local", "google", "github"]],
    )


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


class InvitationCreate(BaseModel):
    username: Optional[str] = Field(None, description="The username of the user")
    email: Optional[str] = Field(None, description="The email of the user")
    role: Optional[Literal["admin", "rw", "ro"]] = Field(None, description="The role of the user (admin, rw, ro)")


class Invitation(BaseModel):
    email: Optional[str] = Field(None, description="The email of the user")
    token: Optional[str] = Field(None, description="The token of the invitation")
    created_by: Optional[str] = Field(None, description="The ID of the user who created the invitation")
    created_at: Optional[str] = Field(None, description="The date and time the invitation was created")
    is_valid: Optional[bool] = Field(None, description="Whether the invitation is valid")
    used_at: Optional[str] = Field(None, description="The date and time the invitation was used")
    role: Optional[Literal["admin", "rw", "ro"]] = Field(None, description="The role of the user (admin, rw, ro)")
    expires_at: Optional[str] = Field(None, description="The date and time the invitation will expire")


class InvitationList(BaseModel):
    """
    A list of invitations
    """

    items: Optional[list[Invitation]] = None
    pageResult: Optional[PageResult] = None


class Register(BaseModel):
    """
    The email of the user
    """

    token: Optional[str] = Field(None, description="The invitation token")
    email: Optional[str] = Field(None, description="The email of the user")
    username: Optional[str] = Field(None, description="The username of the user")
    password: Optional[str] = Field(None, description="The password of the user")


class User(BaseModel):
    id: Optional[str] = Field(None, description="The ID of the user")
    username: Optional[str] = Field(None, description="The username of the user")
    email: Optional[str] = Field(None, description="The email of the user")
    role: Optional[str] = Field(None, description="The role of the user")
    is_active: Optional[bool] = Field(None, description="Whether the user is active")
    date_joined: Optional[str] = Field(None, description="The date and time the user joined the system")
    registration_source: Optional[str] = Field(
        None,
        description="The registration source of the user (local, google, github, etc.)",
    )


class Login(BaseModel):
    username: Optional[str] = Field(None, description="The username of the user")
    password: Optional[str] = Field(None, description="The password of the user")


class UserList(BaseModel):
    """
    A list of users
    """

    items: Optional[list[User]] = None
    pageResult: Optional[PageResult] = None


class ChangePassword(BaseModel):
    username: Optional[str] = Field(None, description="The username of the user")
    old_password: Optional[str] = Field(None, description="The old password of the user")
    new_password: Optional[str] = Field(None, description="The new password of the user")


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


class WebSearchRequest(BaseModel):
    """
    Web search request
    """

    query: Optional[str] = Field(
        None,
        description="Search query for web search. Optional when using source-only site browsing.",
        examples=["ApeRAG 2025年最新发展"],
    )
    max_results: Optional[int] = Field(5, description="Maximum number of results to return", examples=[5])
    timeout: Optional[int] = Field(30, description="Request timeout in seconds", examples=[30])
    locale: Optional[str] = Field("en-US", description="Browser locale", examples=["en-US"])
    source: Optional[str] = Field(
        None,
        description="Domain or URL for site-specific filtering. When provided with query, limits search results to this domain (e.g., 'site:vercel.com query').",
        examples=["vercel.com"],
    )


class WebSearchResultItem(BaseModel):
    """
    Individual web search result
    """

    rank: int = Field(..., description="Result rank", examples=[1])
    title: str = Field(..., description="Page title", examples=["ApeRAG 2025年技术路线图"])
    url: str = Field(
        ...,
        description="Page URL",
        examples=["https://example.com/aperag-2025-roadmap"],
    )
    snippet: str = Field(..., description="Page snippet", examples=["ApeRAG在2025年将重点发展..."])
    domain: str = Field(..., description="Domain name", examples=["example.com"])
    timestamp: Optional[datetime] = Field(None, description="Result timestamp", examples=["2025-01-01T00:00:00Z"])


class WebSearchMeta(BaseModel):
    """
    Lightweight execution diagnostics for web search.
    """

    search_status: Literal["ok", "empty", "unavailable", "disabled"] = Field(
        ...,
        description="Overall search outcome: successful, empty, unavailable, or disabled.",
        examples=["ok"],
    )
    provider_used: list[str] = Field(
        default_factory=list,
        description="Search providers attempted during this request.",
        examples=[["jina", "duckduckgo"]],
    )
    backend_used: list[str] = Field(
        default_factory=list,
        description="Concrete backend path used for this request when known.",
        examples=[["duckduckgo:auto"]],
    )
    fallback_used: bool = Field(
        False,
        description="Whether the request had to fall back from its preferred path.",
        examples=[True],
    )
    error_code: Optional[str] = Field(
        None,
        description="Machine-readable error code when the search path is unavailable.",
        examples=["search_provider_unavailable"],
    )


class WebSearchResponse(BaseModel):
    """
    Web search response
    """

    query: str = Field(..., description="Original search query")
    results: list[WebSearchResultItem] = Field(..., description="List of search results")
    total_results: Optional[int] = Field(None, description="Total number of results found")
    search_time: Optional[float] = Field(None, description="Search time in seconds")
    meta: Optional[WebSearchMeta] = Field(
        None,
        description="Lightweight execution diagnostics for status, provider selection, and fallback behavior.",
    )


class WebReadRequest(BaseModel):
    """
    Web content reading request
    """

    url_list: list[str] = Field(
        ...,
        description="List of URLs to read (for single URL, use array with one element)",
        examples=[["https://example.com/article"]],
    )
    timeout: Optional[int] = Field(30, description="Request timeout in seconds", examples=[30])
    locale: Optional[str] = Field("en-US", description="Browser locale", examples=["en-US"])
    max_concurrent: Optional[int] = Field(3, description="Maximum concurrent requests for multiple URLs", examples=[3])


class WebReadResultItem(BaseModel):
    """
    Individual web content reading result
    """

    url: str = Field(..., description="Requested URL")
    status: Literal["success", "error"] = Field(..., description="Processing status")
    title: Optional[str] = Field(None, description="Page title")
    content: Optional[str] = Field(None, description="Extracted content in Markdown format")
    extracted_at: Optional[datetime] = Field(None, description="Content extraction timestamp")
    word_count: Optional[int] = Field(None, description="Word count of content")
    token_count: Optional[int] = Field(None, description="Estimated token count")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code if failed")


class WebReadResponse(BaseModel):
    """
    Web content reading response
    """

    results: list[WebReadResultItem] = Field(..., description="List of reading results")
    total_urls: int = Field(..., description="Total number of URLs processed")
    successful: int = Field(..., description="Number of successful extractions")
    failed: int = Field(..., description="Number of failed extractions")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")


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
