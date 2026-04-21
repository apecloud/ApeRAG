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
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from aperag.db.models import (
    BenchmarkDatasetSourceType,
    BenchmarkDatasetVersionStatus,
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
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    collection_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    source_type: BenchmarkDatasetSourceType
    schema_hint: Optional[dict[str, Any]] = None
    gmt_created: datetime
    gmt_updated: datetime
    latest_version: Optional["BenchmarkDatasetVersionEnvelope"] = None


class BenchmarkDatasetList(BaseModel):
    items: list[BenchmarkDatasetEnvelope] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


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
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_version_id: str
    case_key: str
    input_message: str
    expected_answer: Optional[str] = None
    reference_context: Optional[str] = None
    tags: Optional[list[str]] = None
    case_metadata: Optional[dict[str, Any]] = None
    sort_key: int
    gmt_created: datetime


class BenchmarkDatasetVersionEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    version: int
    version_name: Optional[str] = None
    status: BenchmarkDatasetVersionStatus
    case_count: int
    source_snapshot: Optional[dict[str, Any]] = None
    gmt_created: datetime
    gmt_updated: datetime
    gmt_published: Optional[datetime] = None


class BenchmarkDatasetVersionList(BaseModel):
    items: list[BenchmarkDatasetVersionEnvelope] = Field(default_factory=list)
    total: int = 0


class BenchmarkCaseList(BaseModel):
    items: list[BenchmarkCaseEnvelope] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 100


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
    total_cases: int = 0
    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    average_score: Optional[float] = None
    pass_rate: Optional[float] = None


class EvaluationRunEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

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
    error_message: Optional[str] = None
    gmt_created: datetime
    gmt_updated: datetime
    gmt_started: Optional[datetime] = None
    gmt_finished: Optional[datetime] = None


class EvaluationRunList(BaseModel):
    items: list[EvaluationRunEnvelope] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


class EvaluationRunItemEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    case_id: str
    case_key: str
    status: EvaluationRunItemStatus
    best_score: Optional[Decimal] = None
    latest_attempt_id: Optional[str] = None
    attempt_count: int
    error_message: Optional[str] = None
    gmt_created: datetime
    gmt_updated: datetime


class EvaluationRunItemList(BaseModel):
    items: list[EvaluationRunItemEnvelope] = Field(default_factory=list)
    total: int = 0


class EvaluationRunItemAttemptEnvelope(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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
    error_message: Optional[str] = None
    retry_reason: Optional[str] = None
    gmt_created: datetime
    gmt_started: Optional[datetime] = None
    gmt_finished: Optional[datetime] = None


class EvaluationRunItemAttemptList(BaseModel):
    items: list[EvaluationRunItemAttemptEnvelope] = Field(default_factory=list)


class CancelRunResponse(BaseModel):
    run_id: str
    status: EvaluationRunStatus


class RetryRunResponse(BaseModel):
    run_id: str
    status: EvaluationRunStatus
    retried_count: int


RetryScope = Literal["failed", "all"]


BenchmarkDatasetEnvelope.model_rebuild()
