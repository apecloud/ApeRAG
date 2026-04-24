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

from pydantic import BaseModel, ConfigDict, Field, RootModel, conint

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


class FailResponse(BaseModel):
    code: Optional[str] = Field(None, description="Error code", examples=["400"])
    message: Optional[str] = Field(None, description="Error message", examples=["Invalid request"])


# Bot / BotConfig / BotList / BotCreate / BotUpdate / BotUpdateRequest,
# Chat / ChatList / ChatCreate / ChatDetails / ChatUpdate / ChatMessage,
# Reference / File, TitleGenerateRequest / TitleGenerateResponse,
# TurnFeedback / TurnFeedbackList / Feedback / TurnFeedbackWrite, Agent
# were carved to ``aperag.domains.conversation.schemas`` in Phase 5 step
# 5-S3. The end-of-file ``try`` block re-imports them so pre-migration
# callers (``from aperag.schema.view_models import Bot``) keep
# resolving the same class objects.


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


# SharedCollectionConfig / SharedCollection / SharedCollectionList
# moved to ``aperag.domains.marketplace.schemas`` in Phase 4 Step
# 4-S3c; end-of-file try block re-imports them.


# ApiKey / ApiKeyList / ApiKeyCreate / ApiKeyUpdate moved to
# ``aperag.domains.governance.schemas`` in Phase 4 Step 4-S3b;
# end-of-file try block re-imports them.


# TagFilterCondition / TagFilterRequest / ModelConfig / ModelConfigList /
# DefaultModelConfig / DefaultModelsResponse / DefaultModelsUpdateRequest /
# LlmProvider / LlmProviderModel / LlmConfigurationResponse /
# LlmProviderCreateWithApiKey / LlmProviderUpdateWithApiKey /
# LlmProviderModelList / LlmProviderModelCreate /
# LlmProviderModelCreateRequest / LlmProviderModelUpdate /
# EmbeddingRequest / EmbeddingData / EmbeddingUsage / EmbeddingResponse /
# Document1 / RerankRequest / Document2 / RerankDocument / RerankUsage /
# RerankResponse moved to ``aperag.domains.model_platform.schemas``
# in Phase 4 Step 4-S3d; end-of-file try block re-imports them.


# Auth0 / Authing / Logto / Auth / Config moved to
# ``aperag.domains.identity.schemas`` in Phase 4 Step 4-S3a; the
# end-of-file try block re-imports them so pre-migration callers
# (auth routes, OpenAPI spec builders) keep working.


# AuditLog / AuditLogList moved to
# ``aperag.domains.governance.schemas`` in Phase 4 Step 4-S3b;
# end-of-file try block re-imports them.


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


# ``AgentMessage`` was carved to
# ``aperag.domains.agent_runtime.schemas`` in Phase 5 step 5-S5a in
# preparation for the ``aperag.agent_runtime`` top-level rename landing
# in 5-S5b. The end-of-file ``try`` block re-imports it so
# pre-migration callers (``from aperag.schema.view_models import
# AgentMessage``) keep resolving the same class object.


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

# Phase 4 Step 4-S3b governance domain schemas dual-hook re-export.
try:
    from aperag.domains.governance.schemas import (  # noqa: E402, F401
        ApiKey,
        ApiKeyCreate,
        ApiKeyList,
        ApiKeyUpdate,
        AuditLog,
        AuditLogList,
    )
except ImportError:
    pass

# Phase 4 Step 4-S3c marketplace domain schemas dual-hook re-export.
try:
    from aperag.domains.marketplace.schemas import (  # noqa: E402, F401
        SharedCollection,
        SharedCollectionConfig,
        SharedCollectionList,
    )
except ImportError:
    pass

# Phase 4 Step 4-S3d model_platform domain schemas dual-hook re-export.
try:
    from aperag.domains.model_platform.schemas import (  # noqa: E402, F401
        DefaultModelConfig,
        DefaultModelsResponse,
        DefaultModelsUpdateRequest,
        Document1,
        Document2,
        EmbeddingData,
        EmbeddingRequest,
        EmbeddingResponse,
        EmbeddingUsage,
        LlmConfigurationResponse,
        LlmProvider,
        LlmProviderCreateWithApiKey,
        LlmProviderModel,
        LlmProviderModelCreate,
        LlmProviderModelCreateRequest,
        LlmProviderModelList,
        LlmProviderModelUpdate,
        LlmProviderUpdateWithApiKey,
        ModelConfig,
        ModelConfigList,
        RerankDocument,
        RerankRequest,
        RerankResponse,
        RerankUsage,
        TagFilterCondition,
        TagFilterRequest,
    )
except ImportError:
    pass

# Phase 5 step 5-S3 back-compat shim (msg=92f5788d / msg=4a93c97e Q2):
# the 21 conversation-domain schemas below were extracted to
# ``aperag.domains.conversation.schemas`` and are re-imported here so
# pre-migration callers (``from aperag.schema.view_models import Bot``)
# and FastAPI response_model bindings continue to see the canonical
# class objects from the conversation domain module. Dual-hook pattern
# mirrors the KB step 4b (cuiwenbo msg=f894cca4): if conversation
# loads first, its ``_bind_view_models_reexports`` binds these names
# onto view_models; if view_models loads first this ``try`` block
# succeeds. Phase 6 cleanup removes the shim once every caller has
# migrated to the conversation path.
try:
    from aperag.domains.conversation.schemas import (  # noqa: E402, F401
        Agent,
        Bot,
        BotConfig,
        BotCreate,
        BotList,
        BotUpdate,
        BotUpdateRequest,
        Chat,
        ChatCreate,
        ChatDetails,
        ChatList,
        ChatMessage,
        ChatUpdate,
        Feedback,
        File,
        Reference,
        TitleGenerateRequest,
        TitleGenerateResponse,
        TurnFeedback,
        TurnFeedbackList,
        TurnFeedbackWrite,
    )
except ImportError:
    # Conversation module is still loading the shared common helpers
    # above; its own ``_bind_view_models_reexports`` will complete the
    # binding from the other side.
    pass


# Phase 5 step 5-S5a dual-hook back-compat for ``AgentMessage``; the
# agent_runtime domain schema module re-binds the class on this
# namespace via its own ``_bind_view_models_reexports`` hook in the
# reverse load order. Same pattern as the Phase 3 / Phase 5 blocks
# above.
try:
    from aperag.domains.agent_runtime.schemas import AgentMessage  # noqa: E402, F401
except ImportError:
    # agent_runtime schema module is still loading the shared common
    # helpers above; its own ``_bind_view_models_reexports`` will
    # complete the binding from the other side.
    pass
