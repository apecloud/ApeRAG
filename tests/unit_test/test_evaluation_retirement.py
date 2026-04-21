from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import aperag.service.evaluation_service as evaluation_service_module
import aperag.views.evaluation as evaluation_view


@pytest.mark.asyncio
async def test_create_resume_retry_raise_when_evaluation_feature_disabled():
    service = evaluation_service_module.EvaluationService()

    with pytest.raises(evaluation_service_module.EvaluationDisabledError):
        await service.create_evaluation(object(), "user-1")

    with pytest.raises(evaluation_service_module.EvaluationDisabledError):
        await service.resume_evaluation("eval-1", "user-1")

    with pytest.raises(evaluation_service_module.EvaluationDisabledError):
        await service.retry_evaluation("eval-1", "user-1", "all")


@pytest.mark.asyncio
async def test_schedule_evaluations_bulk_disables_active_runs_when_retired(monkeypatch):
    fake_session = object()
    disable_mock = AsyncMock(return_value=(2, 3))

    async def _fake_get_async_session(_engine):
        yield fake_session

    class _FakeDbOps:
        def __init__(self, session):
            assert session is fake_session

        disable_active_evaluations = disable_mock

    monkeypatch.setattr(evaluation_service_module, "get_async_session", _fake_get_async_session)
    monkeypatch.setattr(evaluation_service_module, "AsyncDatabaseOps", _FakeDbOps)

    executor = evaluation_service_module.EvaluationExecutor(engine=None)
    await executor.schedule_evaluations()

    disable_mock.assert_awaited_once_with(evaluation_service_module.EVALUATION_DISABLED_MESSAGE)


@pytest.mark.asyncio
async def test_evaluation_routes_raise_gone_when_feature_retired():
    with pytest.raises(HTTPException) as exc_info:
        await evaluation_view.evaluation_feature_disabled()

    assert exc_info.value.status_code == 410
    assert exc_info.value.detail == evaluation_service_module.EVALUATION_DISABLED_MESSAGE
