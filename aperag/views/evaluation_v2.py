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

import asyncio

from fastapi import APIRouter, Depends, Query, Response

from aperag.db.models import User
from aperag.evaluation_v2.schemas import (
    BenchmarkCaseEnvelope,
    BenchmarkCaseListResponse,
    BenchmarkDatasetCreate,
    BenchmarkDatasetEnvelope,
    BenchmarkDatasetListResponse,
    BenchmarkDatasetUpdate,
    BenchmarkDatasetVersionCreate,
    BenchmarkDatasetVersionEnvelope,
    BenchmarkDatasetVersionListResponse,
    CancelRunResponse,
    EvaluationPagination,
    EvaluationRunCreate,
    EvaluationRunDetailResponse,
    EvaluationRunEnvelope,
    EvaluationRunItemAttemptList,
    EvaluationRunItemEnvelope,
    EvaluationRunItemListResponse,
    EvaluationRunListResponse,
)
from aperag.evaluation_v2.services import benchmark_dataset_service, evaluation_run_service
from aperag.views.auth import required_user

router = APIRouter(tags=["evaluation-v2"])


def _pagination(total: int, page: int, page_size: int) -> EvaluationPagination:
    return EvaluationPagination(
        total=total,
        offset=max(page - 1, 0) * page_size,
        limit=page_size,
    )


async def _enrich_run_item(user_id: str, run_id: str, item: EvaluationRunItemEnvelope) -> EvaluationRunItemEnvelope:
    if not item.id or (item.attempt_count <= 0 and not item.latest_attempt_id):
        return item

    attempts = await evaluation_run_service.list_run_item_attempts(user_id, run_id, item.id)
    latest_attempt = attempts[-1] if attempts else None
    return item.model_copy(update={"latest_attempt": latest_attempt})


async def _enrich_run_items(
    user_id: str, run_id: str, items: list[EvaluationRunItemEnvelope]
) -> list[EvaluationRunItemEnvelope]:
    if not items:
        return items

    return list(await asyncio.gather(*[_enrich_run_item(user_id, run_id, item) for item in items]))


@router.post("/benchmark-datasets", response_model=BenchmarkDatasetEnvelope)
async def create_benchmark_dataset_view(
    body: BenchmarkDatasetCreate,
    user: User = Depends(required_user),
) -> BenchmarkDatasetEnvelope:
    return await benchmark_dataset_service.create_dataset(str(user.id), body)


@router.get("/benchmark-datasets", response_model=BenchmarkDatasetListResponse)
async def list_benchmark_datasets_view(
    collection_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    user: User = Depends(required_user),
) -> BenchmarkDatasetListResponse:
    items, total = await benchmark_dataset_service.list_datasets(str(user.id), collection_id, page, page_size)
    return BenchmarkDatasetListResponse(items=items, pagination=_pagination(total, page, page_size))


@router.get("/benchmark-datasets/{dataset_id}", response_model=BenchmarkDatasetEnvelope)
async def get_benchmark_dataset_view(
    dataset_id: str,
    user: User = Depends(required_user),
) -> BenchmarkDatasetEnvelope:
    return await benchmark_dataset_service.get_dataset(str(user.id), dataset_id)


@router.put("/benchmark-datasets/{dataset_id}", response_model=BenchmarkDatasetEnvelope)
async def update_benchmark_dataset_view(
    dataset_id: str,
    body: BenchmarkDatasetUpdate,
    user: User = Depends(required_user),
) -> BenchmarkDatasetEnvelope:
    return await benchmark_dataset_service.update_dataset(str(user.id), dataset_id, body)


@router.delete("/benchmark-datasets/{dataset_id}", status_code=204)
async def delete_benchmark_dataset_view(
    dataset_id: str,
    user: User = Depends(required_user),
) -> Response:
    await benchmark_dataset_service.delete_dataset(str(user.id), dataset_id)
    return Response(status_code=204)


@router.post(
    "/benchmark-datasets/{dataset_id}/versions",
    response_model=BenchmarkDatasetVersionEnvelope,
)
async def create_benchmark_dataset_version_view(
    dataset_id: str,
    body: BenchmarkDatasetVersionCreate,
    user: User = Depends(required_user),
) -> BenchmarkDatasetVersionEnvelope:
    return await benchmark_dataset_service.create_version(str(user.id), dataset_id, body)


@router.get(
    "/benchmark-datasets/{dataset_id}/versions",
    response_model=BenchmarkDatasetVersionListResponse,
)
async def list_benchmark_dataset_versions_view(
    dataset_id: str,
    user: User = Depends(required_user),
) -> BenchmarkDatasetVersionListResponse:
    items = await benchmark_dataset_service.list_versions(str(user.id), dataset_id)
    return BenchmarkDatasetVersionListResponse(
        items=items,
        pagination=EvaluationPagination(total=len(items), offset=0, limit=len(items)),
    )


@router.get(
    "/benchmark-datasets/{dataset_id}/versions/{version_id}/cases",
    response_model=BenchmarkCaseListResponse,
)
async def list_benchmark_cases_view(
    dataset_id: str,
    version_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    user: User = Depends(required_user),
) -> BenchmarkCaseListResponse:
    version = await benchmark_dataset_service.get_version(str(user.id), version_id)
    if version.dataset_id != dataset_id:
        return BenchmarkCaseListResponse(items=[], pagination=_pagination(0, page, page_size))

    rows, total = await benchmark_dataset_service.db_ops.list_cases_for_version(version_id, page, page_size)
    items = [BenchmarkCaseEnvelope.model_validate(row) for row in rows]
    return BenchmarkCaseListResponse(items=items, pagination=_pagination(total, page, page_size))


@router.post("/evaluation-runs", response_model=EvaluationRunEnvelope)
async def create_evaluation_run_view(
    body: EvaluationRunCreate,
    user: User = Depends(required_user),
) -> EvaluationRunEnvelope:
    return await evaluation_run_service.create_run(str(user.id), body)


@router.get("/evaluation-runs", response_model=EvaluationRunListResponse)
async def list_evaluation_runs_view(
    bot_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    user: User = Depends(required_user),
) -> EvaluationRunListResponse:
    items, total = await evaluation_run_service.list_runs(str(user.id), bot_id, page, page_size)
    return EvaluationRunListResponse(items=items, pagination=_pagination(total, page, page_size))


@router.get("/evaluation-runs/{run_id}", response_model=EvaluationRunDetailResponse)
async def get_evaluation_run_view(
    run_id: str,
    user: User = Depends(required_user),
) -> EvaluationRunDetailResponse:
    return await evaluation_run_service.get_run_detail(str(user.id), run_id)


@router.post("/evaluation-runs/{run_id}/cancel", response_model=CancelRunResponse)
async def cancel_evaluation_run_view(
    run_id: str,
    user: User = Depends(required_user),
) -> CancelRunResponse:
    run = await evaluation_run_service.cancel_run(str(user.id), run_id)
    return CancelRunResponse(run_id=run.id, status=run.status)


@router.get("/evaluation-runs/{run_id}/items", response_model=EvaluationRunItemListResponse)
async def list_evaluation_run_items_view(
    run_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    user: User = Depends(required_user),
) -> EvaluationRunItemListResponse:
    items, total = await evaluation_run_service.list_run_items(str(user.id), run_id, page, page_size)
    enriched_items = await _enrich_run_items(str(user.id), run_id, items)
    return EvaluationRunItemListResponse(
        items=enriched_items,
        pagination=_pagination(total, page, page_size),
    )


@router.get(
    "/evaluation-runs/{run_id}/items/{item_id}/attempts",
    response_model=EvaluationRunItemAttemptList,
)
async def list_evaluation_run_item_attempts_view(
    run_id: str,
    item_id: str,
    user: User = Depends(required_user),
) -> EvaluationRunItemAttemptList:
    items = await evaluation_run_service.list_run_item_attempts(str(user.id), run_id, item_id)
    return EvaluationRunItemAttemptList(items=items)


@router.post(
    "/evaluation-runs/{run_id}/items/{item_id}/retry",
    response_model=EvaluationRunItemEnvelope,
)
async def retry_evaluation_run_item_view(
    run_id: str,
    item_id: str,
    user: User = Depends(required_user),
) -> EvaluationRunItemEnvelope:
    item = await evaluation_run_service.retry_run_item(str(user.id), run_id, item_id)
    return await _enrich_run_item(str(user.id), run_id, item)
