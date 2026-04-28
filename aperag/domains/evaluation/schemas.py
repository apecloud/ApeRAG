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

"""Pydantic schemas for the simplified evaluation API.

Public contract surface is two resource families only:

* ``/api/v2/evaluation-datasets*`` — dataset + per-item CRUD.
* ``/api/v2/evaluation-runs*``     — run lifecycle with snapshot-on-run items.

There is no Benchmark / Dataset Version concept in this layer. See the
``#evaluation #20`` design report for the full rationale and validation
checklist. Run items are self-contained snapshots; read paths must never
join back to the mutable ``evaluation_dataset_items`` rows.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from aperag.domains.evaluation.db.models import (
    EvaluationJudgeMode,
    EvaluationRunItemAttemptStatus,
    EvaluationRunItemStatus,
    EvaluationRunStatus,
)

# ---------------------------------------------------------------------------
# EvaluationDataset + items
# ---------------------------------------------------------------------------


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
    schema_hint: Optional[dict[str, Any]] = None
    item_count: int = 0
    created_at: datetime = Field(validation_alias="gmt_created")
    updated_at: datetime = Field(validation_alias="gmt_updated")


class EvaluationPagination(BaseModel):
    total: int = 0
    offset: int = 0
    limit: int = 20


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


# ---------------------------------------------------------------------------
# AI auto-generate QA pairs (preview-then-confirm flow). Per architect lock
# msg=05c3ec83 / earayu2 msg=6e354b8e:
#
#   * BE walks the collection's chunks.jsonl, picks ``count`` substantive
#     chunks, calls the configured LLM once per chunk, parses each
#     ``{question, expected_answer}`` pair.
#   * BE returns each preview item with ``reference_context`` populated
#     from the source chunk (FE keeps it hidden in the main list and
#     surfaces it under a per-row "source" disclosure; bulk-create
#     transparently persists all three fields).
#   * No DB write here — the caller picks/edits and POSTs to the existing
#     ``/items`` append endpoint.
# ---------------------------------------------------------------------------


_GENERATE_PREVIEW_DEFAULT_COUNT = 10
_GENERATE_PREVIEW_MAX_COUNT = 100


class EvaluationDatasetGeneratePreviewRequest(BaseModel):
    collection_id: str = Field(..., min_length=1)
    count: int = Field(
        default=_GENERATE_PREVIEW_DEFAULT_COUNT,
        ge=1,
        le=_GENERATE_PREVIEW_MAX_COUNT,
        description="Number of QA pairs to generate (default 10, max 100).",
    )
    language: Optional[str] = Field(
        None,
        description=(
            "ISO locale (zh-CN / en-US / ja-JP / ko-KR). When omitted the "
            "BE falls back to ``Collection.config.language``."
        ),
    )
    prompt_template: Optional[str] = Field(
        None,
        description=(
            "Override the default per-chunk QA generation prompt. The "
            "default ships an explicit instruction to emit JSON with "
            "``question`` and ``expected_answer`` fields in the resolved "
            "language."
        ),
    )


class GeneratedDatasetItem(BaseModel):
    question: str
    expected_answer: str
    reference_context: Optional[str] = None


class EvaluationDatasetGeneratePreviewResponse(BaseModel):
    items: list[GeneratedDatasetItem] = Field(default_factory=list)
    requested_count: int = 0
    delivered_count: int = 0
    language: Optional[str] = None


# ---------------------------------------------------------------------------
# EvaluationRun lifecycle
# ---------------------------------------------------------------------------


class JudgeConfig(BaseModel):
    mode: EvaluationJudgeMode = EvaluationJudgeMode.EXACT_MATCH
    model_id: Optional[str] = None
    prompt_template: Optional[str] = None
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    params: Optional[dict[str, Any]] = None


class EvaluationRunCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    dataset_id: str = Field(..., description="Required dataset id for this run")
    bot_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional explicit bot id. When omitted the service resolves the "
            "user's default bot (active bot with title "
            "'Default Agent Bot' first, otherwise oldest active bot)."
        ),
    )
    name: Optional[str] = Field(None, max_length=255)
    # Architect spec ``msg=2424afe2``: per-run model overrides. When
    # ``None`` the worker falls back to ``Collection.config.completion``
    # for both phases.
    answer_model: Optional[str] = Field(
        None,
        max_length=64,
        description="Override the LLM used to generate the answer (RAG / agent). Defaults to the collection's completion model.",
    )
    judge_model: Optional[str] = Field(
        None,
        max_length=64,
        description="Override the LLM used by LLM-as-judge for scoring. Defaults to the collection's completion model.",
    )
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
    dataset_id: str
    collection_id: Optional[str] = None
    dataset_name: Optional[str] = None
    name: Optional[str] = None
    status: EvaluationRunStatus
    summary: Optional[EvaluationRunSummary] = None
    answer_model: Optional[str] = None
    judge_model: Optional[str] = None
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
    source_dataset_item_id: Optional[str] = None
    case_key: str
    sort_key: int
    input_message: str
    expected_answer: Optional[str] = None
    reference_context: Optional[str] = None
    tags: Optional[list[str]] = None
    case_metadata: Optional[dict[str, Any]] = None
    status: EvaluationRunItemStatus
    best_score: Optional[Decimal] = None
    latest_attempt_id: Optional[str] = None
    latest_attempt: Optional["EvaluationRunItemAttemptEnvelope"] = None
    attempt_count: int
    # Architect spec ``msg=2424afe2`` — Phase 5 forward-compat: the
    # multi-dim judge writes
    # ``{"correctness": int, "completeness": int|null,
    #   "faithfulness": int|null, "relevance": int|null}`` here. The
    # MVP single-dim Correctness leaves three of the four ``null``;
    # the FE renders only the populated dimensions in the run-item
    # detail panel.
    judge_breakdown: Optional[dict[str, Any]] = None
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


EvaluationRunItemEnvelope.model_rebuild()
