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
from aperag.tasks.models import LocalDocumentInfo, ParsedDocumentData
from aperag.tasks.reconciler import (
    CollectionSummaryReconciler,
    DocumentIndexReconciler,
    collection_summary_callbacks,
    index_task_callbacks,
)
from aperag.utils.constant import IndexAction
from aperag.utils.utils import utc_now
from config.celery_tasks import collection_summary_task, create_index_task


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


class FakeRenewer:
    def __init__(self, ownership_lost=False):
        self.ownership_lost = ownership_lost
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


@pytest.fixture
def sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    DocumentIndex.__table__.create(engine)
    CollectionSummary.__table__.create(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    with session_factory() as session:
        yield session


@pytest.fixture
def parsed_document_payload():
    parsed_data = ParsedDocumentData(
        document_id="doc1",
        collection_id="col1",
        content="hello",
        doc_parts=[],
        file_path="/tmp/doc1.txt",
        local_doc_info=LocalDocumentInfo(path="/tmp/doc1.txt"),
    )
    return parsed_data.to_dict()


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
                "processing_token": "tok-1",
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
                "processing_token": "tok-1",
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

    def test_stale_reclaim_only_reclaims_expired_tokenized_rows(self, sqlite_session, monkeypatch):
        now = utc_now()
        expired = now - timedelta(minutes=5)
        future = now + timedelta(minutes=5)

        expired_create = DocumentIndex(
            id=1,
            document_id="doc1",
            index_type=DocumentIndexType.VECTOR,
            status=DocumentIndexStatus.CREATING,
            version=1,
            observed_version=0,
            processing_token="tok-expired-create",
            lease_expires_at=expired,
        )
        live_create = DocumentIndex(
            id=2,
            document_id="doc2",
            index_type=DocumentIndexType.FULLTEXT,
            status=DocumentIndexStatus.CREATING,
            version=1,
            observed_version=0,
            processing_token="tok-live-create",
            lease_expires_at=future,
        )
        missing_token = DocumentIndex(
            id=3,
            document_id="doc3",
            index_type=DocumentIndexType.GRAPH,
            status=DocumentIndexStatus.CREATING,
            version=1,
            observed_version=0,
            processing_token=None,
            lease_expires_at=expired,
        )
        expired_delete = DocumentIndex(
            id=4,
            document_id="doc4",
            index_type=DocumentIndexType.VECTOR,
            status=DocumentIndexStatus.DELETION_IN_PROGRESS,
            version=2,
            observed_version=1,
            processing_token="tok-expired-delete",
            lease_expires_at=expired,
        )
        sqlite_session.add_all([expired_create, live_create, missing_token, expired_delete])
        sqlite_session.commit()

        reconciler = DocumentIndexReconciler(task_scheduler=MagicMock())
        reclaimed = reconciler._reclaim_stale_indexes(sqlite_session)
        sqlite_session.commit()

        assert reclaimed == 2

        refreshed = {
            row.id: row
            for row in sqlite_session.execute(select(DocumentIndex).order_by(DocumentIndex.id)).scalars().all()
        }

        assert refreshed[1].status == DocumentIndexStatus.PENDING
        assert refreshed[1].processing_token is None
        assert refreshed[1].lease_expires_at is None
        assert refreshed[1].error_message == "stale lease reclaimed"

        assert refreshed[2].status == DocumentIndexStatus.CREATING
        assert refreshed[2].processing_token == "tok-live-create"
        assert refreshed[2].lease_expires_at == future

        assert refreshed[3].status == DocumentIndexStatus.CREATING
        assert refreshed[3].processing_token is None
        assert refreshed[3].lease_expires_at == expired

        assert refreshed[4].status == DocumentIndexStatus.DELETING
        assert refreshed[4].processing_token is None
        assert refreshed[4].lease_expires_at is None

    def test_old_index_callback_token_is_ignored(self, sqlite_session, monkeypatch):
        sqlite_session.add(
            DocumentIndex(
                id=1,
                document_id="doc1",
                index_type=DocumentIndexType.VECTOR,
                status=DocumentIndexStatus.CREATING,
                version=3,
                observed_version=2,
                processing_token="tok-current",
                lease_expires_at=utc_now() + timedelta(minutes=5),
            )
        )
        sqlite_session.commit()

        monkeypatch.setattr(reconciler_module, "get_sync_session", lambda: iter([sqlite_session]))

        index_task_callbacks.on_index_created("doc1", DocumentIndexType.VECTOR.value, 3, "tok-stale", "{}")

        refreshed = sqlite_session.get(DocumentIndex, 1)
        assert refreshed.status == DocumentIndexStatus.CREATING
        assert refreshed.processing_token == "tok-current"
        assert refreshed.observed_version == 2


class TestCollectionSummaryReconciler:
    def test_summary_claim_is_committed_before_dispatch_and_rolls_back_on_failure(self, monkeypatch):
        fake_session = FakeSession()
        reconciler = CollectionSummaryReconciler()
        summary = SimpleNamespace(id="sum1", collection_id="col1", version=7)

        monkeypatch.setattr(
            reconciler,
            "_claim_summary_for_processing",
            lambda session, summary_id, version: "tok-sum1",
        )

        rollback_calls = []
        dispatch_state = {}

        def fake_schedule(summary_id, collection_id, target_version, processing_token):
            dispatch_state["committed_before_dispatch"] = fake_session.committed
            raise RuntimeError("dispatch failed")

        def fake_rollback(summary_id, target_version, processing_token, error_message):
            rollback_calls.append((summary_id, target_version, processing_token, error_message))

        monkeypatch.setattr(reconciler, "_schedule_summary_generation", fake_schedule)
        monkeypatch.setattr(reconciler, "_rollback_summary_claim", fake_rollback)

        with pytest.raises(RuntimeError, match="dispatch failed"):
            reconciler._reconcile_single_summary(fake_session, summary)

        assert fake_session.commit_count == 1
        assert dispatch_state["committed_before_dispatch"] is True
        assert rollback_calls == [("sum1", 7, "tok-sum1", "dispatch failed")]

    def test_stale_reclaim_only_reclaims_expired_tokenized_rows(self, sqlite_session):
        now = utc_now()
        expired = now - timedelta(minutes=5)
        future = now + timedelta(minutes=5)

        expired_summary = CollectionSummary(
            id="sum-expired",
            collection_id="col1",
            status=CollectionSummaryStatus.GENERATING,
            version=2,
            observed_version=1,
            processing_token="tok-expired",
            lease_expires_at=expired,
        )
        live_summary = CollectionSummary(
            id="sum-live",
            collection_id="col2",
            status=CollectionSummaryStatus.GENERATING,
            version=3,
            observed_version=2,
            processing_token="tok-live",
            lease_expires_at=future,
        )
        missing_token = CollectionSummary(
            id="sum-missing-token",
            collection_id="col3",
            status=CollectionSummaryStatus.GENERATING,
            version=4,
            observed_version=3,
            processing_token=None,
            lease_expires_at=expired,
        )
        sqlite_session.add_all([expired_summary, live_summary, missing_token])
        sqlite_session.commit()

        reconciler = CollectionSummaryReconciler()
        reclaimed = reconciler._reclaim_stale_summaries(sqlite_session)
        sqlite_session.commit()

        assert reclaimed == 1

        refreshed = {
            row.id: row
            for row in sqlite_session.execute(select(CollectionSummary).order_by(CollectionSummary.id)).scalars().all()
        }

        assert refreshed["sum-expired"].status == CollectionSummaryStatus.PENDING
        assert refreshed["sum-expired"].processing_token is None
        assert refreshed["sum-expired"].lease_expires_at is None
        assert refreshed["sum-expired"].error_message == "stale lease reclaimed"

        assert refreshed["sum-live"].status == CollectionSummaryStatus.GENERATING
        assert refreshed["sum-live"].processing_token == "tok-live"
        assert refreshed["sum-live"].lease_expires_at == future

        assert refreshed["sum-missing-token"].status == CollectionSummaryStatus.GENERATING
        assert refreshed["sum-missing-token"].processing_token is None
        assert refreshed["sum-missing-token"].lease_expires_at == expired

    def test_old_summary_failure_callback_token_is_ignored(self, sqlite_session, monkeypatch):
        sqlite_session.add(
            CollectionSummary(
                id="sum1",
                collection_id="col1",
                status=CollectionSummaryStatus.GENERATING,
                version=5,
                observed_version=4,
                processing_token="tok-current",
                lease_expires_at=utc_now() + timedelta(minutes=5),
            )
        )
        sqlite_session.commit()

        monkeypatch.setattr(reconciler_module, "get_sync_session", lambda: iter([sqlite_session]))

        collection_summary_callbacks.on_summary_failed("sum1", "boom", 5, "tok-stale")

        refreshed = sqlite_session.get(CollectionSummary, "sum1")
        assert refreshed.status == CollectionSummaryStatus.GENERATING
        assert refreshed.processing_token == "tok-current"
        assert refreshed.error_message is None


class TestCollectionSummaryTask:
    def test_retry_exhausted_calls_failure_callback_with_correct_arguments(self, monkeypatch):
        callback_calls = []

        class RetryTriggered(Exception):
            pass

        def fake_generate(summary_id, collection_id, target_version, processing_token, callback_allowed=None):
            raise RuntimeError("summary failed")

        def fake_on_summary_failed(summary_id, error_message, target_version, processing_token):
            callback_calls.append((summary_id, error_message, target_version, processing_token))

        monkeypatch.setattr(
            "config.celery_tasks._validate_collection_summary_relevance",
            lambda summary_id, target_version, processing_token: None,
        )
        monkeypatch.setattr(
            "config.celery_tasks._make_collection_summary_lease_renewer",
            lambda summary_id, target_version, processing_token: FakeRenewer(),
        )
        monkeypatch.setattr(
            "aperag.domains.knowledge_base.service.collection_summary_service.collection_summary_service.generate_collection_summary_task",
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
                collection_summary_task.run("sum1", "col1", 9, "tok-9")
        finally:
            collection_summary_task.pop_request()

        assert callback_calls == [("sum1", "summary failed", 9, "tok-9")]

    def test_collection_summary_task_suppresses_failure_callback_after_ownership_lost(self, monkeypatch):
        callback_calls = []

        def fake_generate(summary_id, collection_id, target_version, processing_token, callback_allowed=None):
            raise RuntimeError("summary failed after owner lost")

        monkeypatch.setattr(
            "config.celery_tasks._validate_collection_summary_relevance",
            lambda summary_id, target_version, processing_token: None,
        )
        monkeypatch.setattr(
            "config.celery_tasks._make_collection_summary_lease_renewer",
            lambda summary_id, target_version, processing_token: FakeRenewer(ownership_lost=True),
        )
        monkeypatch.setattr(
            "aperag.domains.knowledge_base.service.collection_summary_service.collection_summary_service.generate_collection_summary_task",
            fake_generate,
        )
        monkeypatch.setattr(
            "aperag.tasks.reconciler.collection_summary_callbacks.on_summary_failed",
            lambda *args, **kwargs: callback_calls.append((args, kwargs)),
        )
        monkeypatch.setattr(
            collection_summary_task,
            "retry",
            lambda **kwargs: pytest.fail("retry should not be scheduled after ownership loss"),
        )

        collection_summary_task.push_request(retries=collection_summary_task.max_retries)
        try:
            result = collection_summary_task.run("sum1", "col1", 9, "tok-9")
        finally:
            collection_summary_task.pop_request()

        assert result["status"] == "skipped"
        assert result["reason"] == "ownership_lost"
        assert callback_calls == []


class TestIndexTaskOwnership:
    def test_create_index_task_returns_skipped_when_ownership_lost(self, monkeypatch, parsed_document_payload):
        callback_calls = []

        result_stub = SimpleNamespace(
            success=True,
            error=None,
            data={"index": "ok"},
            to_dict=lambda: {"success": True},
        )

        monkeypatch.setattr(
            "config.celery_tasks._validate_task_relevance",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "config.celery_tasks._make_document_index_lease_renewer",
            lambda targets, description: FakeRenewer(ownership_lost=True),
        )
        monkeypatch.setattr(
            "aperag.tasks.document.document_index_task.create_index",
            lambda document_id, index_type, parsed_data: result_stub,
        )
        monkeypatch.setattr(
            create_index_task,
            "_handle_index_success",
            lambda *args, **kwargs: callback_calls.append((args, kwargs)),
        )

        context = {
            "VECTOR_version": 2,
            "VECTOR_processing_token": "tok-2",
            "VECTOR_index_id": 11,
        }

        create_index_task.push_request(retries=0)
        try:
            result = create_index_task.run("doc1", DocumentIndexType.VECTOR.value, parsed_document_payload, context)
        finally:
            create_index_task.pop_request()

        assert result["status"] == "skipped"
        assert result["reason"] == "ownership_lost"
        assert result["document_id"] == "doc1"
        assert result["index_type"] == DocumentIndexType.VECTOR.value
        assert callback_calls == []
