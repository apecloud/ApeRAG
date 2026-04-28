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
from typing import Protocol

from fastapi import APIRouter, Depends, Query, Response

from aperag.domains.evaluation.schemas import (
    CancelRunResponse,
    EvaluationDatasetCreate,
    EvaluationDatasetEnvelope,
    EvaluationDatasetGeneratePreviewRequest,
    EvaluationDatasetGeneratePreviewResponse,
    EvaluationDatasetItemEnvelope,
    EvaluationDatasetItemListResponse,
    EvaluationDatasetItemsAppendRequest,
    EvaluationDatasetItemsAppendResponse,
    EvaluationDatasetItemUpdate,
    EvaluationDatasetListResponse,
    EvaluationDatasetUpdate,
    EvaluationPagination,
    EvaluationRunCreate,
    EvaluationRunDetailResponse,
    EvaluationRunEnvelope,
    EvaluationRunItemAttemptList,
    EvaluationRunItemEnvelope,
    EvaluationRunItemListResponse,
    EvaluationRunListResponse,
)
from aperag.domains.evaluation.services import evaluation_dataset_service, evaluation_run_service
from aperag.domains.identity.service.auth_dependencies import required_user


class AuthenticatedUser(Protocol):
    """Per-domain auth context for evaluation routes (G16 compliance)."""

    id: object


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


# ---------------------------------------------------------------------------
# Evaluation datasets
# ---------------------------------------------------------------------------


@router.post("/evaluation-datasets", response_model=EvaluationDatasetEnvelope)
async def create_evaluation_dataset_view(
    body: EvaluationDatasetCreate,
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationDatasetEnvelope:
    return await evaluation_dataset_service.create_dataset(str(user.id), body)


@router.get("/evaluation-datasets", response_model=EvaluationDatasetListResponse)
async def list_evaluation_datasets_view(
    collection_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationDatasetListResponse:
    items, total = await evaluation_dataset_service.list_datasets(str(user.id), collection_id, page, page_size)
    return EvaluationDatasetListResponse(items=items, pagination=_pagination(total, page, page_size))


@router.get("/evaluation-datasets/{dataset_id}", response_model=EvaluationDatasetEnvelope)
async def get_evaluation_dataset_view(
    dataset_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationDatasetEnvelope:
    return await evaluation_dataset_service.get_dataset(str(user.id), dataset_id)


@router.put("/evaluation-datasets/{dataset_id}", response_model=EvaluationDatasetEnvelope)
async def update_evaluation_dataset_view(
    dataset_id: str,
    body: EvaluationDatasetUpdate,
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationDatasetEnvelope:
    return await evaluation_dataset_service.update_dataset(str(user.id), dataset_id, body)


@router.delete("/evaluation-datasets/{dataset_id}", status_code=204)
async def delete_evaluation_dataset_view(
    dataset_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Response:
    await evaluation_dataset_service.delete_dataset(str(user.id), dataset_id)
    return Response(status_code=204)


# Items -------------------------------------------------------------------


@router.get(
    "/evaluation-datasets/{dataset_id}/items",
    response_model=EvaluationDatasetItemListResponse,
)
async def list_evaluation_dataset_items_view(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationDatasetItemListResponse:
    items, total = await evaluation_dataset_service.list_items(str(user.id), dataset_id, page, page_size)
    return EvaluationDatasetItemListResponse(items=items, pagination=_pagination(total, page, page_size))


@router.post(
    "/evaluation-datasets/{dataset_id}/items",
    response_model=EvaluationDatasetItemsAppendResponse,
)
async def append_evaluation_dataset_items_view(
    dataset_id: str,
    body: EvaluationDatasetItemsAppendRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationDatasetItemsAppendResponse:
    created = await evaluation_dataset_service.append_items(str(user.id), dataset_id, list(body.items))
    return EvaluationDatasetItemsAppendResponse(items=created)


@router.post(
    "/evaluation-datasets/{dataset_id}/items/generate-preview",
    response_model=EvaluationDatasetGeneratePreviewResponse,
)
async def generate_evaluation_dataset_items_preview_view(
    dataset_id: str,
    body: EvaluationDatasetGeneratePreviewRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationDatasetGeneratePreviewResponse:
    """AI auto-generate QA pairs for the dataset (preview-only).

    Walks the collection's serving chunks, fires one LLM call per
    substantive chunk, and returns the produced ``{question,
    expected_answer, reference_context}`` items without writing to the
    dataset. The caller (FE) lets the user select / edit and POSTs the
    chosen rows to the existing ``/items`` append endpoint.

    Architect lock ``msg=05c3ec83`` + ``msg=a9fb7efd``; ``count``
    defaults to 10 and is capped at 100; ``language`` falls back to
    ``Collection.config.language``.
    """
    user_id = str(user.id)
    # Ensure the caller owns the dataset before we burn LLM credits.
    await evaluation_dataset_service._require_dataset(user_id, dataset_id)

    from aperag.db.ops import async_db_ops
    from aperag.domains.evaluation.dataset_generator import generate_preview_items
    from aperag.exceptions import ResourceNotFoundException

    collection = await async_db_ops.query_collection(user_id, body.collection_id)
    if collection is None:
        raise ResourceNotFoundException("Collection", body.collection_id)

    items, resolved_language = await generate_preview_items(
        collection=collection,
        count=body.count,
        language=body.language,
        prompt_template=body.prompt_template,
    )
    return EvaluationDatasetGeneratePreviewResponse(
        items=items,
        requested_count=body.count,
        delivered_count=len(items),
        language=resolved_language,
    )


@router.put(
    "/evaluation-datasets/{dataset_id}/items/{item_id}",
    response_model=EvaluationDatasetItemEnvelope,
)
async def update_evaluation_dataset_item_view(
    dataset_id: str,
    item_id: str,
    body: EvaluationDatasetItemUpdate,
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationDatasetItemEnvelope:
    return await evaluation_dataset_service.update_item(str(user.id), dataset_id, item_id, body)


@router.delete(
    "/evaluation-datasets/{dataset_id}/items/{item_id}",
    status_code=204,
)
async def delete_evaluation_dataset_item_view(
    dataset_id: str,
    item_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Response:
    await evaluation_dataset_service.delete_item(str(user.id), dataset_id, item_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Evaluation runs
# ---------------------------------------------------------------------------


@router.post("/evaluation-runs", response_model=EvaluationRunEnvelope)
async def create_evaluation_run_view(
    body: EvaluationRunCreate,
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationRunEnvelope:
    return await evaluation_run_service.create_run(str(user.id), body)


@router.get("/evaluation-runs", response_model=EvaluationRunListResponse)
async def list_evaluation_runs_view(
    bot_id: str | None = Query(default=None),
    dataset_id: str | None = Query(default=None),
    collection_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationRunListResponse:
    items, total = await evaluation_run_service.list_runs(
        str(user.id), bot_id, dataset_id, collection_id, page, page_size
    )
    return EvaluationRunListResponse(items=items, pagination=_pagination(total, page, page_size))


@router.get("/evaluation-runs/{run_id}", response_model=EvaluationRunDetailResponse)
async def get_evaluation_run_view(
    run_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationRunDetailResponse:
    return await evaluation_run_service.get_run_detail(str(user.id), run_id)


@router.post("/evaluation-runs/{run_id}/cancel", response_model=CancelRunResponse)
async def cancel_evaluation_run_view(
    run_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> CancelRunResponse:
    run = await evaluation_run_service.cancel_run(str(user.id), run_id)
    return CancelRunResponse(run_id=run.id, status=run.status)


@router.get("/evaluation-runs/{run_id}/items", response_model=EvaluationRunItemListResponse)
async def list_evaluation_run_items_view(
    run_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    user: AuthenticatedUser = Depends(required_user),
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
    user: AuthenticatedUser = Depends(required_user),
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
    user: AuthenticatedUser = Depends(required_user),
) -> EvaluationRunItemEnvelope:
    item = await evaluation_run_service.retry_run_item(str(user.id), run_id, item_id)
    return await _enrich_run_item(str(user.id), run_id, item)
