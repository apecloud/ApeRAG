from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from aperag.db.models import (
    CollectionSummary,
    CollectionSummaryStatus,
    DocumentIndex,
    DocumentIndexStatus,
    DocumentIndexType,
)
from aperag.tasks import reconciler as reconciler_module
from aperag.tasks.reconciler import CollectionSummaryReconciler, DocumentIndexReconciler
from aperag.utils.constant import IndexAction
from aperag.utils.utils import utc_now
from config.celery_tasks import collection_summary_task


class FakeSession:
    def __init__(self):
        self.committed = False
        self.commit_count = 0
        self.rollback_called = False

    def commit(self):
        self.committed = True
        self.commit_count += 1

    def rollback(self):
        self.rollback_called = True


@pytest.fixture
def sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    DocumentIndex.__table__.create(engine)
    CollectionSummary.__table__.create(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    with session_factory() as session:
        yield session


class TestDocumentIndexReconciler:
    def test_document_claim_is_committed_before_dispatch(self, monkeypatch):
        fake_session = FakeSession()
        reconciler = DocumentIndexReconciler(task_scheduler=MagicMock())
        claimed_indexes = [
            {
                "index_id": 1,
                "document_id": "doc1",
                "index_type": DocumentIndexType.VECTOR.value,
                "action": IndexAction.CREATE,
                "target_version": 1,
            }
        ]

        monkeypatch.setattr(reconciler_module, "get_sync_session", lambda: iter([fake_session]))
        monkeypatch.setattr(
            reconciler,
            "_claim_document_indexes",
            lambda session, document_id, indexes_to_claim: claimed_indexes,
        )

        dispatch_state = {}

        def fake_dispatch(document_id, action, claimed):
            dispatch_state["committed_before_dispatch"] = fake_session.committed
            dispatch_state["document_id"] = document_id
            dispatch_state["action"] = action
            dispatch_state["claimed"] = claimed

        monkeypatch.setattr(reconciler, "_dispatch_claimed_indexes", fake_dispatch)

        operations = {
            IndexAction.CREATE: [SimpleNamespace(id=1, index_type=DocumentIndexType.VECTOR.value)],
            IndexAction.UPDATE: [],
            IndexAction.DELETE: [],
        }

        reconciler._reconcile_single_document("doc1", operations)

        assert fake_session.commit_count == 1
        assert dispatch_state["committed_before_dispatch"] is True
        assert dispatch_state["document_id"] == "doc1"
        assert dispatch_state["action"] == IndexAction.CREATE
        assert dispatch_state["claimed"] == claimed_indexes

    def test_document_dispatch_failure_triggers_claim_rollback(self, monkeypatch):
        fake_session = FakeSession()
        reconciler = DocumentIndexReconciler(task_scheduler=MagicMock())
        claimed_indexes = [
            {
                "index_id": 1,
                "document_id": "doc1",
                "index_type": DocumentIndexType.VECTOR.value,
                "action": IndexAction.CREATE,
                "target_version": 1,
            }
        ]

        monkeypatch.setattr(reconciler_module, "get_sync_session", lambda: iter([fake_session]))
        monkeypatch.setattr(
            reconciler,
            "_claim_document_indexes",
            lambda session, document_id, indexes_to_claim: claimed_indexes,
        )

        rollback_calls = []

        def fake_dispatch(document_id, action, claimed):
            assert fake_session.committed is True
            raise RuntimeError("broker unavailable")

        def fake_rollback(document_id, claimed, error_message):
            rollback_calls.append((document_id, claimed, error_message))

        monkeypatch.setattr(reconciler, "_dispatch_claimed_indexes", fake_dispatch)
        monkeypatch.setattr(reconciler, "_rollback_claimed_indexes", fake_rollback)

        operations = {
            IndexAction.CREATE: [SimpleNamespace(id=1, index_type=DocumentIndexType.VECTOR.value)],
            IndexAction.UPDATE: [],
            IndexAction.DELETE: [],
        }

        with pytest.raises(RuntimeError, match="broker unavailable"):
            reconciler._reconcile_single_document("doc1", operations)

        assert fake_session.commit_count == 1
        assert rollback_calls == [("doc1", claimed_indexes, "broker unavailable")]

    def test_reclaim_stale_indexes_returns_retryable_states(self, sqlite_session):
        stale_time = utc_now() - timedelta(minutes=10)
        fresh_time = utc_now()

        stale_create = DocumentIndex(
            document_id="doc-create",
            index_type=DocumentIndexType.VECTOR,
            status=DocumentIndexStatus.CREATING,
            version=1,
            observed_version=0,
            gmt_last_reconciled=stale_time,
        )
        stale_delete = DocumentIndex(
            document_id="doc-delete",
            index_type=DocumentIndexType.FULLTEXT,
            status=DocumentIndexStatus.DELETION_IN_PROGRESS,
            version=1,
            observed_version=1,
            gmt_last_reconciled=stale_time,
        )
        fresh_create = DocumentIndex(
            document_id="doc-fresh",
            index_type=DocumentIndexType.GRAPH,
            status=DocumentIndexStatus.CREATING,
            version=2,
            observed_version=1,
            gmt_last_reconciled=fresh_time,
        )
        sqlite_session.add_all([stale_create, stale_delete, fresh_create])
        sqlite_session.commit()

        reconciler = DocumentIndexReconciler(task_scheduler=MagicMock(), stale_reclaim_timeout_seconds=60)

        reclaimed_count = reconciler._reclaim_stale_indexes(sqlite_session)
        sqlite_session.commit()

        refreshed_rows = (
            sqlite_session.execute(select(DocumentIndex).order_by(DocumentIndex.document_id)).scalars().all()
        )
        status_by_document = {row.document_id: row.status for row in refreshed_rows}

        assert reclaimed_count == 2
        assert status_by_document["doc-create"] == DocumentIndexStatus.PENDING
        assert status_by_document["doc-delete"] == DocumentIndexStatus.DELETING
        assert status_by_document["doc-fresh"] == DocumentIndexStatus.CREATING


class TestCollectionSummaryReconciler:
    def test_summary_claim_is_committed_before_dispatch_and_rolls_back_on_failure(self, monkeypatch):
        fake_session = FakeSession()
        reconciler = CollectionSummaryReconciler(stale_reclaim_timeout_seconds=60)
        summary = SimpleNamespace(id="sum1", collection_id="col1", version=7)

        monkeypatch.setattr(reconciler, "_claim_summary_for_processing", lambda session, summary_id, version: True)

        rollback_calls = []
        dispatch_state = {}

        def fake_schedule(summary_id, collection_id, target_version):
            dispatch_state["committed_before_dispatch"] = fake_session.committed
            raise RuntimeError("dispatch failed")

        def fake_rollback(summary_id, target_version, error_message):
            rollback_calls.append((summary_id, target_version, error_message))

        monkeypatch.setattr(reconciler, "_schedule_summary_generation", fake_schedule)
        monkeypatch.setattr(reconciler, "_rollback_summary_claim", fake_rollback)

        with pytest.raises(RuntimeError, match="dispatch failed"):
            reconciler._reconcile_single_summary(fake_session, summary)

        assert fake_session.commit_count == 1
        assert dispatch_state["committed_before_dispatch"] is True
        assert rollback_calls == [("sum1", 7, "dispatch failed")]

    def test_reclaim_stale_summaries_returns_pending(self, sqlite_session):
        stale_time = utc_now() - timedelta(minutes=10)
        fresh_time = utc_now()

        stale_summary = CollectionSummary(
            id="sum-stale",
            collection_id="col-stale",
            status=CollectionSummaryStatus.GENERATING,
            version=2,
            observed_version=1,
            gmt_last_reconciled=stale_time,
        )
        fresh_summary = CollectionSummary(
            id="sum-fresh",
            collection_id="col-fresh",
            status=CollectionSummaryStatus.GENERATING,
            version=2,
            observed_version=1,
            gmt_last_reconciled=fresh_time,
        )
        sqlite_session.add_all([stale_summary, fresh_summary])
        sqlite_session.commit()

        reconciler = CollectionSummaryReconciler(stale_reclaim_timeout_seconds=60)

        reclaimed_count = reconciler._reclaim_stale_summaries(sqlite_session)
        sqlite_session.commit()

        refreshed_rows = (
            sqlite_session.execute(select(CollectionSummary).order_by(CollectionSummary.id)).scalars().all()
        )
        status_by_summary = {row.id: row.status for row in refreshed_rows}

        assert reclaimed_count == 1
        assert status_by_summary["sum-stale"] == CollectionSummaryStatus.PENDING
        assert status_by_summary["sum-fresh"] == CollectionSummaryStatus.GENERATING


class TestCollectionSummaryTask:
    def test_retry_exhausted_calls_failure_callback_with_correct_arguments(self, monkeypatch):
        callback_calls = []

        class RetryTriggered(Exception):
            pass

        def fake_generate(summary_id, collection_id, target_version):
            raise RuntimeError("summary failed")

        def fake_on_summary_failed(summary_id, error_message, target_version):
            callback_calls.append((summary_id, error_message, target_version))

        monkeypatch.setattr(
            "aperag.service.collection_summary_service.collection_summary_service.generate_collection_summary_task",
            fake_generate,
        )
        monkeypatch.setattr(
            "aperag.tasks.reconciler.collection_summary_callbacks.on_summary_failed",
            fake_on_summary_failed,
        )
        monkeypatch.setattr(collection_summary_task, "retry", lambda **kwargs: RetryTriggered("retry scheduled"))

        collection_summary_task.push_request(retries=collection_summary_task.max_retries)
        try:
            with pytest.raises(RetryTriggered, match="retry scheduled"):
                collection_summary_task.run("sum1", "col1", 9)
        finally:
            collection_summary_task.pop_request()

        assert callback_calls == [("sum1", "summary failed", 9)]
