from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.cli.indexing_worker import _install_worker_runtime
from aperag.indexing import InMemoryWorkQueue, NoopMetricsEmitter
from aperag.indexing.models import DocumentIndex, IndexStatus, Modality
from aperag.indexing.orchestrator import DispatchPayload
from aperag.indexing.quota import InMemoryQuotaBackend, QuotaPolicyRegistry
from aperag.indexing.runtime import set_runtime
from aperag.indexing.worker_factory import ProductionWorkerFactory, _resolve_tenant_scope_key


def test_cli_worker_installs_runtime_for_graph_tenant_scope_resolution():
    """Standalone indexing-worker must install runtime before graph workers build."""

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DocumentIndex.metadata.create_all(engine, tables=[DocumentIndex.__table__])
    queue = InMemoryWorkQueue()
    quota_backend = InMemoryQuotaBackend(QuotaPolicyRegistry())
    worker_factory = ProductionWorkerFactory(engine=engine, object_store=object())

    with Session(engine) as session, session.begin():
        row = DocumentIndex(
            document_id="doc-hotfix",
            parse_version="parse-v1",
            modality=Modality.GRAPH.value,
            status=IndexStatus.PENDING.value,
            tenant_scope_key="tenant:hotfix",
            collection_id="col-hotfix",
            source_path="graph.jsonl",
            is_serving=False,
        )
        session.add(row)
        session.flush()
        row_id = int(row.id)

    _install_worker_runtime(
        engine=engine,
        queue=queue,
        metrics_emitter=NoopMetricsEmitter(),
        quota_backend=quota_backend,
        worker_factory=worker_factory,
    )
    try:
        payload = DispatchPayload(
            index_id=row_id,
            document_id="doc-hotfix",
            parse_version="parse-v1",
            modality=Modality.GRAPH,
            source_path="graph.jsonl",
            collection_id="col-hotfix",
        )

        assert _resolve_tenant_scope_key(payload=payload) == "tenant:hotfix"
    finally:
        set_runtime(None)
