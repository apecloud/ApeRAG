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

"""Pydantic schemas for the evaluation-v2 orchestration API.

These types mirror the frozen contract (`contract freeze v1`) used by
`aperag/views/evaluation_v2.py` and by the frontend client. Keep the wire
shape here stable; runtime data access DTOs live next to the repositories.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from aperag.db.models import (
    BenchmarkDatasetSourceType,
    BenchmarkDatasetVersionStatus,
    EvaluationDatasetSourceType,
    EvaluationJudgeMode,
    EvaluationRunItemAttemptStatus,
    EvaluationRunItemStatus,
    EvaluationRunStatus,
)

# ---------------------------------------------------------------------------
# BenchmarkDataset
# ---------------------------------------------------------------------------


class BenchmarkDatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    collection_id: Optional[str] = None
    source_type: BenchmarkDatasetSourceType = BenchmarkDatasetSourceType.MANUAL
    schema_hint: Optional[dict[str, Any]] = None


class BenchmarkDatasetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class BenchmarkDatasetEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    user_id: str
    collection_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    source_type: BenchmarkDatasetSourceType
    schema_hint: Optional[dict[str, Any]] = None
    created_at: datetime = Field(validation_alias="gmt_created")
    updated_at: datetime = Field(validation_alias="gmt_updated")
    latest_version: Optional["BenchmarkDatasetVersionEnvelope"] = None


class EvaluationPagination(BaseModel):
    total: int = 0
    offset: int = 0
    limit: int = 20


class BenchmarkDatasetListResponse(BaseModel):
    items: list[BenchmarkDatasetEnvelope] = Field(default_factory=list)
    pagination: EvaluationPagination = Field(default_factory=EvaluationPagination)


# ---------------------------------------------------------------------------
# BenchmarkDatasetVersion + BenchmarkCase
# ---------------------------------------------------------------------------


class BenchmarkCaseCreate(BaseModel):
    case_key: Optional[str] = Field(
        None,
        max_length=128,
        description="Stable user-facing key. Auto-generated when omitted.",
    )
    input_message: str = Field(..., min_length=1)
    expected_answer: Optional[str] = None
    reference_context: Optional[str] = None
    tags: Optional[list[str]] = None
    case_metadata: Optional[dict[str, Any]] = None
    sort_key: int = 0


class BenchmarkDatasetVersionCreate(BaseModel):
    version_name: Optional[str] = Field(None, max_length=255)
    source_snapshot: Optional[dict[str, Any]] = None
    cases: list[BenchmarkCaseCreate] = Field(default_factory=list)


class BenchmarkCaseEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    dataset_version_id: str
    case_key: str
    input_message: str
    expected_answer: Optional[str] = None
    reference_context: Optional[str] = None
    tags: Optional[list[str]] = None
    case_metadata: Optional[dict[str, Any]] = None
    sort_key: int
    created_at: datetime = Field(validation_alias="gmt_created")


class BenchmarkDatasetVersionEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    dataset_id: str
    version: int
    version_name: Optional[str] = None
    status: BenchmarkDatasetVersionStatus
    case_count: int
    source_snapshot: Optional[dict[str, Any]] = None
    created_at: datetime = Field(validation_alias="gmt_created")
    updated_at: datetime = Field(validation_alias="gmt_updated")
    published_at: Optional[datetime] = Field(default=None, validation_alias="gmt_published")


class BenchmarkDatasetVersionListResponse(BaseModel):
    items: list[BenchmarkDatasetVersionEnvelope] = Field(default_factory=list)
    pagination: EvaluationPagination = Field(default_factory=EvaluationPagination)


class BenchmarkCaseListResponse(BaseModel):
    items: list[BenchmarkCaseEnvelope] = Field(default_factory=list)
    pagination: EvaluationPagination = Field(default_factory=EvaluationPagination)


# ---------------------------------------------------------------------------
# EvaluationRun lifecycle
# ---------------------------------------------------------------------------


class JudgeConfig(BaseModel):
    mode: EvaluationJudgeMode = EvaluationJudgeMode.EXACT_MATCH
    model: Optional[str] = None
    model_service_provider: Optional[str] = None
    prompt_template: Optional[str] = None
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    params: Optional[dict[str, Any]] = None


class EvaluationRunCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: Optional[str] = Field(None, max_length=255)
    bot_id: str
    dataset_version_id: str
    judge: Optional[JudgeConfig] = None
    bot_config_snapshot: Optional[dict[str, Any]] = None
    model_config_snapshot: Optional[dict[str, Any]] = None


class EvaluationRunSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total: int = Field(default=0, validation_alias="total_cases")
    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    avg_score: Optional[float] = Field(default=None, validation_alias="average_score")
    pass_rate: Optional[float] = None


class EvaluationRunProgress(BaseModel):
    percent: Optional[int] = None
    eta_ms: Optional[int] = None


class EvaluationRunEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True, protected_namespaces=())

    id: str
    user_id: str
    bot_id: str
    dataset_version_id: str
    name: Optional[str] = None
    status: EvaluationRunStatus
    summary: Optional[EvaluationRunSummary] = None
    judge_config: Optional[JudgeConfig] = None
    bot_config_snapshot: Optional[dict[str, Any]] = None
    model_config_snapshot: Optional[dict[str, Any]] = None
    error: Optional[str] = Field(default=None, validation_alias="error_message")
    created_at: datetime = Field(validation_alias="gmt_created")
    updated_at: datetime = Field(validation_alias="gmt_updated")
    started_at: Optional[datetime] = Field(default=None, validation_alias="gmt_started")
    finished_at: Optional[datetime] = Field(default=None, validation_alias="gmt_finished")


class EvaluationRunListResponse(BaseModel):
    items: list[EvaluationRunEnvelope] = Field(default_factory=list)
    pagination: EvaluationPagination = Field(default_factory=EvaluationPagination)


class EvaluationRunItemEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    run_id: str
    case_id: str
    case_key: str
    status: EvaluationRunItemStatus
    best_score: Optional[Decimal] = None
    latest_attempt_id: Optional[str] = None
    latest_attempt: Optional["EvaluationRunItemAttemptEnvelope"] = None
    attempt_count: int
    error: Optional[str] = Field(default=None, validation_alias="error_message")
    created_at: datetime = Field(validation_alias="gmt_created")
    updated_at: datetime = Field(validation_alias="gmt_updated")


class EvaluationRunItemListResponse(BaseModel):
    items: list[EvaluationRunItemEnvelope] = Field(default_factory=list)
    pagination: EvaluationPagination = Field(default_factory=EvaluationPagination)


class EvaluationRunItemAttemptEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    run_item_id: str
    run_id: str
    attempt_no: int
    status: EvaluationRunItemAttemptStatus
    agent_chat_id: Optional[str] = None
    agent_turn_id: Optional[str] = None
    answer_text: Optional[str] = None
    judge_result: Optional[dict[str, Any]] = None
    score: Optional[Decimal] = None
    latency_ms: Optional[int] = None
    token_usage: Optional[dict[str, Any]] = None
    error: Optional[str] = Field(default=None, validation_alias="error_message")
    retry_reason: Optional[str] = None
    created_at: datetime = Field(validation_alias="gmt_created")
    started_at: Optional[datetime] = Field(default=None, validation_alias="gmt_started")
    finished_at: Optional[datetime] = Field(default=None, validation_alias="gmt_finished")


class EvaluationRunItemAttemptList(BaseModel):
    items: list[EvaluationRunItemAttemptEnvelope] = Field(default_factory=list)


class EvaluationRunDetailResponse(BaseModel):
    run: EvaluationRunEnvelope
    summary: Optional[EvaluationRunSummary] = None
    progress: Optional[EvaluationRunProgress] = None


class CancelRunResponse(BaseModel):
    run_id: str
    status: EvaluationRunStatus


class RetryRunItemResponse(BaseModel):
    item: EvaluationRunItemEnvelope


BenchmarkDatasetEnvelope.model_rebuild()
EvaluationRunItemEnvelope.model_rebuild()


# ---------------------------------------------------------------------------
# EvaluationDataset (simplified evaluation model — evaluation-v3 Phase 1)
# ---------------------------------------------------------------------------
#
# These schemas are added alongside the Benchmark* set so the new
# ``EvaluationDataset`` / ``EvaluationDatasetItem`` tables can be exercised by
# repository tests and by the Phase 2 service/view switch without touching the
# still-live benchmark contract. The public API is NOT re-wired in Phase 1.


class EvaluationDatasetItemCreate(BaseModel):
    case_key: Optional[str] = Field(
        None,
        max_length=128,
        description="Stable user-facing key. Auto-generated when omitted.",
    )
    input_message: str = Field(..., min_length=1)
    expected_answer: Optional[str] = None
    reference_context: Optional[str] = None
    tags: Optional[list[str]] = None
    case_metadata: Optional[dict[str, Any]] = None
    sort_key: int = 0


class EvaluationDatasetItemUpdate(BaseModel):
    case_key: Optional[str] = Field(None, max_length=128)
    input_message: Optional[str] = Field(None, min_length=1)
    expected_answer: Optional[str] = None
    reference_context: Optional[str] = None
    tags: Optional[list[str]] = None
    case_metadata: Optional[dict[str, Any]] = None
    sort_key: Optional[int] = None


class EvaluationDatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    collection_id: Optional[str] = None
    source_type: EvaluationDatasetSourceType = EvaluationDatasetSourceType.MANUAL
    schema_hint: Optional[dict[str, Any]] = None
    items: Optional[list[EvaluationDatasetItemCreate]] = None


class EvaluationDatasetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class EvaluationDatasetItemEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    dataset_id: str
    case_key: str
    input_message: str
    expected_answer: Optional[str] = None
    reference_context: Optional[str] = None
    tags: Optional[list[str]] = None
    case_metadata: Optional[dict[str, Any]] = None
    sort_key: int
    created_at: datetime = Field(validation_alias="gmt_created")
    updated_at: datetime = Field(validation_alias="gmt_updated")


class EvaluationDatasetEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    user_id: str
    collection_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    source_type: EvaluationDatasetSourceType
    schema_hint: Optional[dict[str, Any]] = None
    item_count: int = 0
    created_at: datetime = Field(validation_alias="gmt_created")
    updated_at: datetime = Field(validation_alias="gmt_updated")


class EvaluationDatasetListResponse(BaseModel):
    items: list[EvaluationDatasetEnvelope] = Field(default_factory=list)
    pagination: EvaluationPagination = Field(default_factory=EvaluationPagination)


class EvaluationDatasetItemListResponse(BaseModel):
    items: list[EvaluationDatasetItemEnvelope] = Field(default_factory=list)
    pagination: EvaluationPagination = Field(default_factory=EvaluationPagination)


class EvaluationDatasetItemsAppendRequest(BaseModel):
    items: list[EvaluationDatasetItemCreate] = Field(default_factory=list)


class EvaluationDatasetItemsAppendResponse(BaseModel):
    items: list[EvaluationDatasetItemEnvelope] = Field(default_factory=list)
