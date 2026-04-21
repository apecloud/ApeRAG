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

"""Orchestration services for evaluation-v2.

These services sit between the FastAPI view layer and the async repository
mixin on `AsyncDatabaseOps`. They own the write-time invariants (version
numbering, run item creation, snapshot capture) while keeping the view code
thin and the repository layer pure data access.

Phase 1 only ships the synchronous CRUD path. The async worker pipeline
(dispatching Runtime V3 turns, judging, aggregating summaries) lands in
Phase 3 and will plug into `run_service.launch_run`.
"""

from typing import Optional

from aperag.db.models import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDatasetVersion,
    BenchmarkDatasetVersionStatus,
    EvaluationRun,
    EvaluationRunItem,
    EvaluationRunItemStatus,
    EvaluationRunStatus,
)
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.evaluation_v2.schemas import (
    BenchmarkDatasetCreate,
    BenchmarkDatasetEnvelope,
    BenchmarkDatasetUpdate,
    BenchmarkDatasetVersionCreate,
    BenchmarkDatasetVersionEnvelope,
    EvaluationRunCreate,
    EvaluationRunDetailResponse,
    EvaluationRunEnvelope,
    EvaluationRunItemAttemptEnvelope,
    EvaluationRunItemEnvelope,
    EvaluationRunProgress,
    EvaluationRunSummary,
    JudgeConfig,
)
from aperag.exceptions import ResourceNotFoundException, ValidationException
from aperag.utils.utils import utc_now


def _to_dataset_envelope(
    dataset: BenchmarkDataset,
    latest_version: Optional[BenchmarkDatasetVersion] = None,
) -> BenchmarkDatasetEnvelope:
    envelope = BenchmarkDatasetEnvelope.model_validate(dataset)
    if latest_version is not None:
        envelope.latest_version = BenchmarkDatasetVersionEnvelope.model_validate(latest_version)
    return envelope


def _to_run_envelope(run: EvaluationRun) -> EvaluationRunEnvelope:
    envelope = EvaluationRunEnvelope.model_validate(run)
    if run.summary:
        try:
            envelope.summary = EvaluationRunSummary.model_validate(run.summary)
        except Exception:
            envelope.summary = None
    if run.judge_config:
        try:
            envelope.judge_config = JudgeConfig.model_validate(run.judge_config)
        except Exception:
            envelope.judge_config = None
    return envelope


def _to_run_progress(summary: Optional[EvaluationRunSummary]) -> EvaluationRunProgress:
    if not summary or summary.total <= 0:
        return EvaluationRunProgress(percent=0)

    resolved = summary.completed + summary.failed + summary.cancelled
    return EvaluationRunProgress(percent=round((resolved / summary.total) * 100))


class BenchmarkDatasetService:
    """Dataset + dataset-version CRUD."""

    def __init__(self, db_ops: Optional[AsyncDatabaseOps] = None):
        self.db_ops = db_ops or async_db_ops

    async def create_dataset(self, user_id: str, request: BenchmarkDatasetCreate) -> BenchmarkDatasetEnvelope:
        dataset = BenchmarkDataset(
            user_id=user_id,
            collection_id=request.collection_id,
            name=request.name,
            description=request.description,
            source_type=request.source_type,
            schema_hint=request.schema_hint,
        )
        created = await self.db_ops.create_benchmark_dataset(dataset)
        return _to_dataset_envelope(created)

    async def list_datasets(
        self,
        user_id: str,
        collection_id: Optional[str],
        page: int,
        page_size: int,
    ) -> tuple[list[BenchmarkDatasetEnvelope], int]:
        rows, total = await self.db_ops.list_benchmark_datasets(
            user_id=user_id, collection_id=collection_id, page=page, page_size=page_size
        )
        envelopes: list[BenchmarkDatasetEnvelope] = []
        for dataset in rows:
            latest = await self.db_ops.latest_published_version(dataset.id)
            envelopes.append(_to_dataset_envelope(dataset, latest))
        return envelopes, total

    async def get_dataset(self, user_id: str, dataset_id: str) -> BenchmarkDatasetEnvelope:
        dataset = await self.db_ops.get_benchmark_dataset(user_id, dataset_id)
        if not dataset:
            raise ResourceNotFoundException("BenchmarkDataset", dataset_id)
        latest = await self.db_ops.latest_published_version(dataset.id)
        return _to_dataset_envelope(dataset, latest)

    async def update_dataset(
        self,
        user_id: str,
        dataset_id: str,
        request: BenchmarkDatasetUpdate,
    ) -> BenchmarkDatasetEnvelope:
        updated = await self.db_ops.update_benchmark_dataset(
            user_id,
            dataset_id,
            name=request.name,
            description=request.description,
        )
        if not updated:
            raise ResourceNotFoundException("BenchmarkDataset", dataset_id)
        return _to_dataset_envelope(updated)

    async def delete_dataset(self, user_id: str, dataset_id: str) -> None:
        ok = await self.db_ops.soft_delete_benchmark_dataset(user_id, dataset_id)
        if not ok:
            raise ResourceNotFoundException("BenchmarkDataset", dataset_id)

    # Versions ------------------------------------------------------------

    async def create_version(
        self,
        user_id: str,
        dataset_id: str,
        request: BenchmarkDatasetVersionCreate,
    ) -> BenchmarkDatasetVersionEnvelope:
        dataset = await self.db_ops.get_benchmark_dataset(user_id, dataset_id)
        if not dataset:
            raise ResourceNotFoundException("BenchmarkDataset", dataset_id)
        if not request.cases:
            raise ValidationException("A dataset version must contain at least one case")

        now = utc_now()
        version_no = await self.db_ops.next_dataset_version_number(dataset_id)
        version = BenchmarkDatasetVersion(
            dataset_id=dataset_id,
            version=version_no,
            version_name=request.version_name,
            status=BenchmarkDatasetVersionStatus.PUBLISHED,
            case_count=0,
            source_snapshot=request.source_snapshot,
            gmt_published=now,
        )

        seen_keys: set[str] = set()
        cases: list[BenchmarkCase] = []
        for idx, payload in enumerate(request.cases):
            key = (payload.case_key or f"case_{idx + 1:04d}").strip()
            if key in seen_keys:
                raise ValidationException(f"Duplicate case_key within version payload: {key}")
            seen_keys.add(key)
            cases.append(
                BenchmarkCase(
                    case_key=key,
                    input_message=payload.input_message,
                    expected_answer=payload.expected_answer,
                    reference_context=payload.reference_context,
                    tags=payload.tags,
                    case_metadata=payload.case_metadata,
                    sort_key=payload.sort_key or idx,
                )
            )

        created = await self.db_ops.create_benchmark_dataset_version(dataset_id, version, cases)
        return BenchmarkDatasetVersionEnvelope.model_validate(created)

    async def list_versions(self, user_id: str, dataset_id: str) -> list[BenchmarkDatasetVersionEnvelope]:
        dataset = await self.db_ops.get_benchmark_dataset(user_id, dataset_id)
        if not dataset:
            raise ResourceNotFoundException("BenchmarkDataset", dataset_id)
        rows = await self.db_ops.list_dataset_versions(dataset_id)
        return [BenchmarkDatasetVersionEnvelope.model_validate(r) for r in rows]

    async def get_version(self, user_id: str, version_id: str) -> BenchmarkDatasetVersion:
        version = await self.db_ops.get_dataset_version(version_id)
        if not version:
            raise ResourceNotFoundException("BenchmarkDatasetVersion", version_id)
        dataset = await self.db_ops.get_benchmark_dataset(user_id, version.dataset_id)
        if not dataset:
            raise ResourceNotFoundException("BenchmarkDatasetVersion", version_id)
        return version


class EvaluationRunService:
    """Evaluation run lifecycle: create / list / get / cancel / retry.

    The worker pipeline that actually runs cases against Runtime V3 lives in
    `aperag.evaluation_v2.tasks` and is wired in Phase 3. `launch_run` is the
    single hand-off point the view layer uses, so Phase 3 can swap in the
    Celery producer without touching the view.
    """

    def __init__(
        self,
        db_ops: Optional[AsyncDatabaseOps] = None,
        dataset_service: Optional[BenchmarkDatasetService] = None,
    ):
        self.db_ops = db_ops or async_db_ops
        self.dataset_service = dataset_service or BenchmarkDatasetService(self.db_ops)

    async def create_run(self, user_id: str, request: EvaluationRunCreate) -> EvaluationRunEnvelope:
        version = await self.dataset_service.get_version(user_id, request.dataset_version_id)
        if version.status != BenchmarkDatasetVersionStatus.PUBLISHED:
            raise ValidationException("Only published dataset versions can be evaluated")
        bot = await self.db_ops.query_bot(user_id, request.bot_id)
        if not bot:
            raise ResourceNotFoundException("Bot", request.bot_id)

        cases = await self.db_ops.list_all_cases_for_version(version.id)
        if not cases:
            raise ValidationException("Dataset version has no cases to evaluate")

        judge_payload = request.judge.model_dump() if request.judge else None
        run = EvaluationRun(
            user_id=user_id,
            bot_id=request.bot_id,
            dataset_version_id=version.id,
            name=request.name,
            bot_config_snapshot=request.bot_config_snapshot,
            model_config_snapshot=request.model_config_snapshot,
            judge_config=judge_payload,
            status=EvaluationRunStatus.QUEUED,
            summary=EvaluationRunSummary(total=len(cases), pending=len(cases)).model_dump(),
        )
        items = [
            EvaluationRunItem(
                case_id=case.id,
                case_key=case.case_key,
            )
            for case in cases
        ]

        created = await self.db_ops.create_evaluation_run(run, items)
        await self.launch_run(created.id)
        return _to_run_envelope(created)

    async def launch_run(self, run_id: str) -> None:
        """Hand the run off to the async worker pipeline.

        Phase 3 will replace this stub with a Celery producer call. Keeping
        the seam here means the view layer and tests do not need to change.
        """
        return None

    async def list_runs(
        self,
        user_id: str,
        bot_id: Optional[str],
        page: int,
        page_size: int,
    ) -> tuple[list[EvaluationRunEnvelope], int]:
        rows, total = await self.db_ops.list_evaluation_runs(
            user_id=user_id, bot_id=bot_id, page=page, page_size=page_size
        )
        return [_to_run_envelope(r) for r in rows], total

    async def get_run(self, user_id: str, run_id: str) -> EvaluationRunEnvelope:
        run = await self._require_run(user_id, run_id)
        return _to_run_envelope(run)

    async def get_run_detail(self, user_id: str, run_id: str) -> EvaluationRunDetailResponse:
        run = await self._require_run(user_id, run_id)
        envelope = _to_run_envelope(run)
        return EvaluationRunDetailResponse(
            run=envelope,
            summary=envelope.summary,
            progress=_to_run_progress(envelope.summary),
        )

    async def cancel_run(self, user_id: str, run_id: str) -> EvaluationRunEnvelope:
        run = await self._require_run(user_id, run_id)
        if run.status in (
            EvaluationRunStatus.COMPLETED,
            EvaluationRunStatus.FAILED,
            EvaluationRunStatus.CANCELLED,
        ):
            return _to_run_envelope(run)

        updated = await self.db_ops.update_run_status(run_id, EvaluationRunStatus.CANCELLED)
        return _to_run_envelope(updated)

    async def retry_run_item(self, user_id: str, run_id: str, item_id: str) -> EvaluationRunItemEnvelope:
        """Re-queue a single run item for another attempt.

        Per frozen contract the only retry surface is run-scoped item retry:
        `POST /api/v2/evaluation-runs/{runId}/items/{itemId}/retry`. Phase 3
        worker pipeline will pick the PENDING item and produce a new attempt.
        """
        run = await self._require_run(user_id, run_id)
        item = await self.db_ops.get_run_item(item_id)
        if not item or item.run_id != run_id:
            raise ResourceNotFoundException("EvaluationRunItem", item_id)
        if item.status not in (
            EvaluationRunItemStatus.FAILED,
            EvaluationRunItemStatus.CANCELLED,
        ):
            raise ValidationException("Only failed or cancelled run items can be retried")

        if run.status in (
            EvaluationRunStatus.CANCELLED,
            EvaluationRunStatus.COMPLETED,
            EvaluationRunStatus.FAILED,
        ):
            await self.db_ops.update_run_status(
                run_id,
                EvaluationRunStatus.QUEUED,
                error_message=None,
            )

        item = await self.db_ops.requeue_run_item(run_id, item_id)
        await self.launch_run(run_id)
        return EvaluationRunItemEnvelope.model_validate(item)

    async def list_run_items(
        self, user_id: str, run_id: str, page: int, page_size: int
    ) -> tuple[list[EvaluationRunItemEnvelope], int]:
        await self._require_run(user_id, run_id)
        rows, total = await self.db_ops.list_run_items(run_id, page, page_size)
        return [EvaluationRunItemEnvelope.model_validate(r) for r in rows], total

    async def list_run_item_attempts(
        self, user_id: str, run_id: str, item_id: str
    ) -> list[EvaluationRunItemAttemptEnvelope]:
        item = await self.db_ops.get_run_item(item_id)
        if not item or item.run_id != run_id:
            raise ResourceNotFoundException("EvaluationRunItem", item_id)
        run = await self.db_ops.get_evaluation_run(user_id, item.run_id)
        if not run:
            raise ResourceNotFoundException("EvaluationRun", item.run_id)
        attempts = await self.db_ops.list_attempts_for_item(item_id)
        return [EvaluationRunItemAttemptEnvelope.model_validate(a) for a in attempts]

    async def _require_run(self, user_id: str, run_id: str) -> EvaluationRun:
        run = await self.db_ops.get_evaluation_run(user_id, run_id)
        if not run:
            raise ResourceNotFoundException("EvaluationRun", run_id)
        return run


benchmark_dataset_service = BenchmarkDatasetService()
evaluation_run_service = EvaluationRunService(dataset_service=benchmark_dataset_service)
