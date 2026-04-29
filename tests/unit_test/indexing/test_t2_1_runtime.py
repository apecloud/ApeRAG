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

"""T2.1 Worker pool / reconciler / cleanup contract tests.

Locks the §K Wave 2 acceptance gates for the runtime lane:

1. **Orchestrator claim → derive → sync → finalize cycle** — atomic
   PENDING→RUNNING claim + ACTIVE on success + retry-budgeted FAILED
   on exception (§E.2 / §I.2).
2. **§I.2 retry backoff** — 30s → 60s → 120s → 240s → 480s, 5-cap.
3. **Reconciler PENDING dispatch** — pushes payloads onto the
   per-modality queue without claiming the rows (orchestrator's job).
4. **Reconciler RUNNING reclaim** — flips stale-heartbeat RUNNING rows
   back to PENDING without burning retry budget (§E.4).
5. **Reconciler FAILED retry** — flips elapsed-backoff FAILED rows
   back to PENDING; past-budget rows stay FAILED.
6. **Worker §F.3 cutover** — three-statement TX (status=ACTIVE →
   demote prior is_serving → promote new) inside the worker session
   immediately after sync(); §F.1 partial unique invariant honoured
   (per architect ruling msg=492315e8 Ruling 1, NOT a reconciler scan).
7. **Cleanup orphan GC** (path A) — backend ``delete_by_filter`` /
   ``delete_by_query`` + DB row DELETE for superseded parse_versions
   past the cool-down (§F.5). Graph orphan GC is a backend no-op
   (§D.3.6 sync supersede already cleared lineage per amended §D.3.2
   canonical) — DB row still GC'd.
8. **Cleanup document-deletion** (path B) — caller-driven
   ``cleanup_for_deleted_documents``: flat backend delete per
   parse_version for non-graph modalities; lineage-aware cleanup
   on the graph worker's ``LineageGraphStore`` (one call per doc
   regardless of parse_version count).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import (
    Engine,
    and_,
    create_engine,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.indexing import (
    DeriveResult,
    DispatchPayload,
    InMemoryObjectStore,
    InMemoryVectorBackend,
    InMemoryWorkQueue,
    Modality,
    VectorModality,
    cleanup_for_deleted_documents,
    cleanup_orphan_parse_versions,
    drain_queue_sync,
    parse_document,
    process_one_task,
    reconcile_failed_retry,
    reconcile_graph_vectors_enqueue,
    reconcile_pending_dispatch,
    reconcile_running_reclaim,
)
from aperag.indexing.base import ModalityWorker
from aperag.indexing.models import DocumentIndex, IndexStatus
from aperag.indexing.orchestrator import (
    INITIAL_RETRY_DELAY_SECONDS,
    MAX_RETRY_COUNT,
    _retry_delay_for,
)

# ---------------------------------------------------------------------
# Test fixtures — SQLite mirror of document_index_v2 (matches the live
# alembic schema after the c2e8d5a1f3b9 migration). We use the live
# ORM ``DocumentIndex`` so the table_args (including the §F.1 partial
# unique index) line up exactly with production.
# ---------------------------------------------------------------------


@pytest.fixture
def engine() -> Engine:
    # StaticPool + check_same_thread=False so every Session(engine) call
    # reuses the same underlying SQLite in-memory connection — without
    # this each new Session would open a fresh empty DB and lose the
    # table we just created.
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DocumentIndex.metadata.create_all(eng, tables=[DocumentIndex.__table__])
    return eng


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _insert_row(
    engine: Engine,
    *,
    document_id: str,
    parse_version: str,
    modality: Modality,
    status: IndexStatus = IndexStatus.PENDING,
    source_path: str | None = "collections/c/documents/d/derived/parse_v/chunks.jsonl",
    collection_id: str | None = "c",
    is_serving: bool = False,
    retry_count: int = 0,
    retry_after: datetime | None = None,
    last_heartbeat: datetime | None = None,
    derived_artifact_path: str | None = None,
) -> int:
    """Insert a row + return its id. Defaults are "valid PENDING dispatch"."""
    with Session(engine) as session, session.begin():
        result = session.execute(
            insert(DocumentIndex)
            .values(
                document_id=document_id,
                parse_version=parse_version,
                modality=modality.value,
                status=status.value,
                tenant_scope_key="user:test",
                source_path=source_path,
                collection_id=collection_id,
                is_serving=is_serving,
                retry_count=retry_count,
                retry_after=retry_after,
                last_heartbeat=last_heartbeat,
                derived_artifact_path=derived_artifact_path,
            )
            .returning(DocumentIndex.id)
        )
        return int(result.scalar_one())


def _row(engine: Engine, row_id: int) -> DocumentIndex:
    with Session(engine) as session:
        return session.scalars(select(DocumentIndex).where(DocumentIndex.id == row_id)).one()


def _set_updated_at(engine: Engine, row_id: int, when: datetime) -> None:
    """Force-set ``updated_at`` for cool-down tests (sqlalchemy's
    ``onupdate`` would otherwise overwrite to NOW)."""
    with Session(engine) as session, session.begin():
        session.execute(update(DocumentIndex).where(DocumentIndex.id == row_id).values(updated_at=when))


# ---------------------------------------------------------------------
# (1) Orchestrator claim → derive → sync → finalize
# ---------------------------------------------------------------------


def _seed_chunks(store: InMemoryObjectStore) -> tuple[str, str, str]:
    """Helper: parse a tiny doc + return (doc_id, parse_version, chunks_path)."""
    parsed = parse_document(
        store=store,
        collection_id="c",
        document_id="doc-1",
        source_bytes=b"# T2.1\n\nOne paragraph.",
    )
    return "doc-1", parsed.parse_version, parsed.chunks_path


def test_orchestrator_claim_derive_sync_finalize_happy_path(engine):
    store = InMemoryObjectStore()
    doc_id, parse_version, chunks_path = _seed_chunks(store)
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=store)

    row_id = _insert_row(
        engine,
        document_id=doc_id,
        parse_version=parse_version,
        modality=Modality.VECTOR,
        source_path=chunks_path,
    )
    payload = DispatchPayload(
        index_id=row_id,
        document_id=doc_id,
        parse_version=parse_version,
        modality=Modality.VECTOR,
        source_path=chunks_path,
        collection_id="c",
    )

    outcome = asyncio.run(
        process_one_task(
            engine=engine,
            payload=payload,
            worker=worker,
            heartbeat_interval_seconds=0,  # disable heartbeat task in unit test
        )
    )
    assert outcome == "completed"

    row = _row(engine, row_id)
    assert row.status == IndexStatus.ACTIVE.value
    assert row.derived_artifact_path == chunks_path
    assert row.error_message is None
    assert row.retry_after is None
    # §F.3 cutover runs in the worker session — successful completion
    # must leave the row both ACTIVE and is_serving=TRUE in one TX.
    assert row.is_serving is True
    assert backend.points_for_document(doc_id, parse_version), "successful sync must populate the vector backend"


def test_orchestrator_lost_claim_skips_silently(engine):
    """Two concurrent workers must not both run derive on the same row."""
    store = InMemoryObjectStore()
    doc_id, parse_version, chunks_path = _seed_chunks(store)
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=store)

    row_id = _insert_row(
        engine,
        document_id=doc_id,
        parse_version=parse_version,
        modality=Modality.VECTOR,
        source_path=chunks_path,
        status=IndexStatus.RUNNING,  # already claimed by a phantom worker
        last_heartbeat=_utcnow(),
    )
    payload = DispatchPayload(
        index_id=row_id,
        document_id=doc_id,
        parse_version=parse_version,
        modality=Modality.VECTOR,
        source_path=chunks_path,
    )

    outcome = asyncio.run(process_one_task(engine=engine, payload=payload, worker=worker, heartbeat_interval_seconds=0))
    assert outcome == "skipped", "second worker must not double-run derive on a RUNNING row"

    row = _row(engine, row_id)
    assert row.status == IndexStatus.RUNNING.value, "phantom claim is preserved"
    assert backend.all_points() == [], "skipped task must not populate backend"


def test_orchestrator_failure_writes_failed_with_backoff(engine):
    """A modality exception must mark the row FAILED with retry_after.

    The first failure schedules ``retry_after = now + 30s`` per §I.2.
    """

    class _ExplodingWorker(ModalityWorker):
        modality = Modality.VECTOR

        async def derive(self, *, document_id, parse_version, source_path):
            raise RuntimeError("simulated derive blow-up")

        async def sync(self, *, document_id, parse_version, derived_artifact_path):
            pass  # pragma: no cover — never reached

    row_id = _insert_row(
        engine,
        document_id="doc-x",
        parse_version="aaaaaaaaaaaaaaaa",
        modality=Modality.VECTOR,
    )
    payload = DispatchPayload(
        index_id=row_id,
        document_id="doc-x",
        parse_version="aaaaaaaaaaaaaaaa",
        modality=Modality.VECTOR,
        source_path="ignored",
    )

    before = _utcnow()
    outcome = asyncio.run(
        process_one_task(
            engine=engine,
            payload=payload,
            worker=_ExplodingWorker(),
            heartbeat_interval_seconds=0,
        )
    )
    assert outcome == "failed"

    row = _row(engine, row_id)
    assert row.status == IndexStatus.FAILED.value
    assert row.retry_count == 1
    assert "simulated derive blow-up" in (row.error_message or "")
    assert row.retry_after is not None
    # SQLite reads DateTime back as tz-naive; normalize before subtracting.
    retry_after = row.retry_after
    if retry_after.tzinfo is None:
        retry_after = retry_after.replace(tzinfo=timezone.utc)
    delta = retry_after - before
    # First failure → 30s backoff per §I.2 (allow ±5s slack for test scheduler jitter).
    assert timedelta(seconds=25) < delta < timedelta(seconds=40)


def test_orchestrator_empty_derive_reschedules_without_retry_burn(engine):
    """§C.7: empty derive path means upstream not ready — reschedule, do not retry-burn."""

    class _RescheduleWorker(ModalityWorker):
        modality = Modality.VECTOR

        async def derive(self, *, document_id, parse_version, source_path):
            return DeriveResult(derived_artifact_path="")

        async def sync(self, *, document_id, parse_version, derived_artifact_path):
            pytest.fail("sync must not be called when derive returned empty")

    row_id = _insert_row(
        engine,
        document_id="doc-r",
        parse_version="bbbbbbbbbbbbbbbb",
        modality=Modality.VECTOR,
    )
    payload = DispatchPayload(
        index_id=row_id,
        document_id="doc-r",
        parse_version="bbbbbbbbbbbbbbbb",
        modality=Modality.VECTOR,
        source_path="ignored",
    )
    outcome = asyncio.run(
        process_one_task(
            engine=engine,
            payload=payload,
            worker=_RescheduleWorker(),
            heartbeat_interval_seconds=0,
        )
    )
    assert outcome == "rescheduled"

    row = _row(engine, row_id)
    assert row.status == IndexStatus.PENDING.value, (
        "empty-derive path must put the row back to PENDING for next reconciler cycle"
    )
    assert row.retry_count == 0, "empty-derive must NOT consume the retry budget (§C.7)"


# ---------------------------------------------------------------------
# (2) §I.2 backoff schedule — 30s → 60s → 120s → 240s → 480s, capped at 5
# ---------------------------------------------------------------------


def test_retry_delay_matches_section_i2_schedule():
    expected = [30, 60, 120, 240, 480]
    actual = [_retry_delay_for(n) for n in (1, 2, 3, 4, 5)]
    assert actual == expected
    # Cap: any retry past 5 stays at the max delay (sequence does not overflow).
    assert _retry_delay_for(MAX_RETRY_COUNT + 1) == 480
    assert INITIAL_RETRY_DELAY_SECONDS == 30


# ---------------------------------------------------------------------
# (3) Reconciler PENDING dispatch
# ---------------------------------------------------------------------


def test_reconciler_pending_dispatch_pushes_to_per_modality_queues(engine):
    queue = InMemoryWorkQueue()
    vec_id = _insert_row(
        engine,
        document_id="doc-1",
        parse_version="aaaaaaaaaaaaaaaa",
        modality=Modality.VECTOR,
    )
    ft_id = _insert_row(
        engine,
        document_id="doc-1",
        parse_version="aaaaaaaaaaaaaaaa",
        modality=Modality.FULLTEXT,
    )

    pushed = asyncio.run(reconcile_pending_dispatch(engine=engine, queue=queue))
    assert pushed == 2

    vec_payloads = drain_queue_sync(queue, Modality.VECTOR)
    ft_payloads = drain_queue_sync(queue, Modality.FULLTEXT)
    assert len(vec_payloads) == 1 and vec_payloads[0]["index_id"] == vec_id
    assert len(ft_payloads) == 1 and ft_payloads[0]["index_id"] == ft_id

    # Status stays PENDING — orchestrator (not reconciler) does the
    # atomic claim. Re-running dispatch is harmless.
    assert _row(engine, vec_id).status == IndexStatus.PENDING.value
    pushed_again = asyncio.run(reconcile_pending_dispatch(engine=engine, queue=queue))
    assert pushed_again == 2, "PENDING dispatch is idempotent across cycles"


# Wave 3 T3.1 (alembic d0f4c1b9a8e2 + model NOT-NULL flip): the
# ``test_reconciler_skips_pending_rows_missing_source_path`` test was
# deleted alongside the ``source_path`` NULL → NOT NULL promotion. The
# scenario it exercised (a PENDING row with ``source_path IS NULL``)
# is now impossible at the schema layer, so the test fixture's
# ``_insert_row(... source_path=None)`` raises an ``IntegrityError``
# before the reconciler is even called. The defensive ``if not row.
# source_path`` branch in ``reconcile_pending_dispatch`` is kept as a
# zero-cost guard against malformed rows but is no longer reachable
# from a clean schema.


# ---------------------------------------------------------------------
# (4) Reconciler RUNNING reclaim (stale heartbeat)
# ---------------------------------------------------------------------


def test_reconciler_reclaims_stale_running_to_pending_without_burning_retry(engine):
    stale = _utcnow() - timedelta(seconds=120)
    fresh = _utcnow() - timedelta(seconds=10)
    stale_id = _insert_row(
        engine,
        document_id="doc-stale",
        parse_version="dddddddddddddddd",
        modality=Modality.VECTOR,
        status=IndexStatus.RUNNING,
        last_heartbeat=stale,
        retry_count=2,
    )
    fresh_id = _insert_row(
        engine,
        document_id="doc-fresh",
        parse_version="eeeeeeeeeeeeeeee",
        modality=Modality.VECTOR,
        status=IndexStatus.RUNNING,
        last_heartbeat=fresh,
        retry_count=2,
    )

    reclaimed = reconcile_running_reclaim(engine=engine, stale_seconds=60)
    assert reclaimed == 1, "only the stale-heartbeat row is reclaimed"

    stale_row = _row(engine, stale_id)
    fresh_row = _row(engine, fresh_id)

    assert stale_row.status == IndexStatus.PENDING.value
    assert stale_row.last_heartbeat is None
    assert stale_row.retry_count == 2, "stale-heartbeat reclaim must NOT burn retry budget"

    assert fresh_row.status == IndexStatus.RUNNING.value, (
        "fresh-heartbeat row stays RUNNING — only stale rows are reclaimed"
    )


# ---------------------------------------------------------------------
# (5) Reconciler FAILED retry
# ---------------------------------------------------------------------


def test_reconciler_failed_retry_flips_elapsed_backoff_only(engine):
    past = _utcnow() - timedelta(seconds=10)
    future = _utcnow() + timedelta(seconds=600)
    past_id = _insert_row(
        engine,
        document_id="doc-fp",
        parse_version="ffffffffffffffff",
        modality=Modality.VECTOR,
        status=IndexStatus.FAILED,
        retry_count=1,
        retry_after=past,
    )
    future_id = _insert_row(
        engine,
        document_id="doc-ff",
        parse_version="aaaaaaaaaaaaaaa1",
        modality=Modality.VECTOR,
        status=IndexStatus.FAILED,
        retry_count=1,
        retry_after=future,
    )
    overbudget_id = _insert_row(
        engine,
        document_id="doc-ob",
        parse_version="aaaaaaaaaaaaaaa2",
        modality=Modality.VECTOR,
        status=IndexStatus.FAILED,
        retry_count=MAX_RETRY_COUNT + 1,
        retry_after=None,
    )

    flipped = reconcile_failed_retry(engine=engine)
    assert flipped == 1, "only the elapsed-backoff retryable row flips"

    assert _row(engine, past_id).status == IndexStatus.PENDING.value
    assert _row(engine, future_id).status == IndexStatus.FAILED.value, "row whose backoff has not elapsed stays FAILED"
    assert _row(engine, overbudget_id).status == IndexStatus.FAILED.value, (
        "row past the §I.2 5-retry cap stays FAILED until operator intervention"
    )


# ---------------------------------------------------------------------
# (5b) Reconciler graph_vectors enqueue — 任务 #5 设计文档 §4.4
# ---------------------------------------------------------------------


def test_reconciler_graph_vectors_enqueue_inserts_pending_row_after_facts_active(engine):
    """事实层 ACTIVE 之后, reconciler 应该 INSERT 一行 graph_vectors PENDING,
    source_path 复用 facts 行的 derived_artifact_path.
    """
    facts_artifact = "collections/c/documents/d/derived/parse_v1/kg.jsonl"
    facts_id = _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_FACTS,
        status=IndexStatus.ACTIVE,
        derived_artifact_path=facts_artifact,
    )

    inserted = reconcile_graph_vectors_enqueue(engine=engine)
    assert inserted == 1

    # vectors 行应该被 INSERT 进来
    with Session(engine) as session:
        vectors_rows = list(
            session.scalars(
                select(DocumentIndex).where(
                    DocumentIndex.modality == Modality.GRAPH_VECTORS.value,
                )
            )
        )
    assert len(vectors_rows) == 1
    v = vectors_rows[0]
    assert v.document_id == "doc-1"
    assert v.parse_version == "v1"
    assert v.status == IndexStatus.PENDING.value
    assert v.source_path == facts_artifact
    assert v.is_serving is False
    # facts 行不变
    assert _row(engine, facts_id).status == IndexStatus.ACTIVE.value


def test_reconciler_graph_vectors_enqueue_idempotent(engine):
    """已经存在 graph_vectors 行的 (doc, parse_v) 跳过, 不重复 INSERT."""
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_FACTS,
        status=IndexStatus.ACTIVE,
        derived_artifact_path="kg.jsonl",
    )
    # 跑一次 — INSERT 1 行
    assert reconcile_graph_vectors_enqueue(engine=engine) == 1
    # 跑第二次 — 应该 0 行
    assert reconcile_graph_vectors_enqueue(engine=engine) == 0


def test_reconciler_graph_vectors_enqueue_skips_non_active_facts(engine):
    """事实层不在 ACTIVE 状态 (PENDING / RUNNING / FAILED) 不入队向量层."""
    for status in (IndexStatus.PENDING, IndexStatus.RUNNING, IndexStatus.FAILED):
        _insert_row(
            engine,
            document_id=f"doc-{status.value}",
            parse_version="v1",
            modality=Modality.GRAPH_FACTS,
            status=status,
            derived_artifact_path="kg.jsonl",
        )
    assert reconcile_graph_vectors_enqueue(engine=engine) == 0


def test_reconciler_graph_vectors_enqueue_skips_facts_without_artifact(engine):
    """事实层没设 derived_artifact_path 时跳过 (理论上不应该发生, 但容错)."""
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_FACTS,
        status=IndexStatus.ACTIVE,
        derived_artifact_path=None,
    )
    assert reconcile_graph_vectors_enqueue(engine=engine) == 0


# ---------------------------------------------------------------------
# (5b') Graph vectors stale detection — task #15
# (Planetegg msg=3322c22b surface + 5 方 align + architect ratify msg=39c4ece8 +
#  earayu2 拍板 msg=efc218ce. 设计文档 §4.4 conservative serial scheduling
#  扩展第 2 类触发: facts 重跑后 vectors stale 必须重入队.)
# ---------------------------------------------------------------------


def test_reconciler_graph_vectors_re_enqueues_stale_active_when_facts_updated_at_advances(engine):
    """场景: facts 重跑 (updated_at 推进), 老 vectors ACTIVE 没漂移路径但
    时间戳落后 → reconciler 必须重置 vectors 为 PENDING.
    """
    facts_artifact = "collections/c/documents/doc-1/derived/parse_v2/kg.jsonl"
    facts_id = _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_FACTS,
        status=IndexStatus.ACTIVE,
        derived_artifact_path=facts_artifact,
    )
    # vectors ACTIVE 行, source_path 同 facts (无路径漂移), 但 updated_at 必须比 facts 早.
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_VECTORS,
        status=IndexStatus.ACTIVE,
        source_path=facts_artifact,
        is_serving=True,
    )
    # 强制 vectors.updated_at 比 facts.updated_at 早.
    earlier = _utcnow() - timedelta(minutes=5)
    with Session(engine) as session, session.begin():
        session.execute(
            update(DocumentIndex)
            .where(
                and_(
                    DocumentIndex.modality == Modality.GRAPH_VECTORS.value,
                    DocumentIndex.document_id == "doc-1",
                )
            )
            .values(updated_at=earlier)
        )

    enqueued = reconcile_graph_vectors_enqueue(engine=engine)
    assert enqueued == 1

    with Session(engine) as session:
        vectors = session.scalars(
            select(DocumentIndex).where(DocumentIndex.modality == Modality.GRAPH_VECTORS.value)
        ).one()
    assert vectors.status == IndexStatus.PENDING.value
    assert vectors.source_path == facts_artifact
    assert vectors.is_serving is False
    assert vectors.error_message is None
    assert vectors.retry_after is None
    # facts 行不变.
    assert _row(engine, facts_id).status == IndexStatus.ACTIVE.value


def test_reconciler_graph_vectors_re_enqueues_when_artifact_path_diverges(engine):
    """场景: facts 重跑后 derived_artifact_path 漂移 (新 parse_version),
    即使时间戳相同, vectors 也必须重入队 + source_path 同步.
    """
    new_artifact = "collections/c/documents/doc-1/derived/parse_v3/kg.jsonl"
    old_artifact = "collections/c/documents/doc-1/derived/parse_v1/kg.jsonl"
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_FACTS,
        status=IndexStatus.ACTIVE,
        derived_artifact_path=new_artifact,
    )
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_VECTORS,
        status=IndexStatus.ACTIVE,
        source_path=old_artifact,
        is_serving=True,
    )

    enqueued = reconcile_graph_vectors_enqueue(engine=engine)
    assert enqueued == 1

    with Session(engine) as session:
        vectors = session.scalars(
            select(DocumentIndex).where(DocumentIndex.modality == Modality.GRAPH_VECTORS.value)
        ).one()
    assert vectors.status == IndexStatus.PENDING.value
    assert vectors.source_path == new_artifact, "source_path 必须同步成 facts 当前的 artifact"


def test_reconciler_graph_vectors_boundary_facts_equal_vectors_does_not_re_enqueue(engine):
    """huangheng CR NIT-B (msg=33b1fc56): boundary 钉死 ``>`` 不是 ``>=``.
    facts.updated_at == vectors.updated_at 时不应触发 stale (相等 ≠ 落后).
    防未来 refactor 把 ``>`` 改成 ``>=`` 静默过度入队.
    """
    facts_artifact = "collections/c/documents/doc-1/derived/parse_v1/kg.jsonl"
    same_time = _utcnow()

    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_FACTS,
        status=IndexStatus.ACTIVE,
        derived_artifact_path=facts_artifact,
    )
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_VECTORS,
        status=IndexStatus.ACTIVE,
        source_path=facts_artifact,
        is_serving=True,
    )
    # 强制两边 updated_at 完全相等.
    with Session(engine) as session, session.begin():
        session.execute(
            update(DocumentIndex)
            .where(
                and_(
                    DocumentIndex.document_id == "doc-1",
                    DocumentIndex.modality.in_(
                        [Modality.GRAPH_FACTS.value, Modality.GRAPH_VECTORS.value],
                    ),
                )
            )
            .values(updated_at=same_time)
        )

    assert reconcile_graph_vectors_enqueue(engine=engine) == 0

    with Session(engine) as session:
        vectors = session.scalars(
            select(DocumentIndex).where(DocumentIndex.modality == Modality.GRAPH_VECTORS.value)
        ).one()
    assert vectors.status == IndexStatus.ACTIVE.value, "facts == vectors 时不应触发 stale"
    assert vectors.is_serving is True


def test_reconciler_graph_vectors_idempotent_when_facts_unchanged(engine):
    """幂等: facts 没改 (updated_at 不推进 + source_path 一致), vectors ACTIVE
    不应被反复入队. 防止 reconciler 周期 30 秒重复打扰已 OK 的 vectors.
    """
    facts_artifact = "collections/c/documents/doc-1/derived/parse_v1/kg.jsonl"
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_FACTS,
        status=IndexStatus.ACTIVE,
        derived_artifact_path=facts_artifact,
    )
    # vectors ACTIVE, 跟 facts 同 path, updated_at 比 facts 晚.
    later = _utcnow() + timedelta(seconds=10)
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_VECTORS,
        status=IndexStatus.ACTIVE,
        source_path=facts_artifact,
        is_serving=True,
    )
    with Session(engine) as session, session.begin():
        session.execute(
            update(DocumentIndex).where(DocumentIndex.modality == Modality.GRAPH_VECTORS.value).values(updated_at=later)
        )

    # 跑 3 次都 0 (不重复打扰).
    for _ in range(3):
        assert reconcile_graph_vectors_enqueue(engine=engine) == 0

    with Session(engine) as session:
        vectors = session.scalars(
            select(DocumentIndex).where(DocumentIndex.modality == Modality.GRAPH_VECTORS.value)
        ).one()
    assert vectors.status == IndexStatus.ACTIVE.value
    assert vectors.is_serving is True


def test_reconciler_graph_vectors_re_enqueues_failed_when_facts_updated(engine):
    """FAILED 的 vectors 行也算 stale 候选 — facts 重跑后, 老 FAILED vectors
    应该重置成 PENDING 重新尝试 (不消耗 retry_count, 因为根因是 facts 更新).
    """
    facts_artifact = "collections/c/documents/doc-1/derived/parse_v2/kg.jsonl"
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_FACTS,
        status=IndexStatus.ACTIVE,
        derived_artifact_path=facts_artifact,
    )
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_VECTORS,
        status=IndexStatus.FAILED,
        source_path="old/path/kg.jsonl",
    )

    enqueued = reconcile_graph_vectors_enqueue(engine=engine)
    assert enqueued == 1

    with Session(engine) as session:
        vectors = session.scalars(
            select(DocumentIndex).where(DocumentIndex.modality == Modality.GRAPH_VECTORS.value)
        ).one()
    assert vectors.status == IndexStatus.PENDING.value
    assert vectors.source_path == facts_artifact


def test_reconciler_graph_vectors_skips_in_flight_pending_or_running(engine):
    """PENDING / RUNNING 状态的 vectors 已在被处理 — 不应被打扰. 即使路径
    漂移, worker 当前在跑 (RUNNING) 或排队等 worker (PENDING), 让它跑完就好;
    跑完后如果 facts 又推进, 下一轮 reconciler 会再 stale-detect.
    """
    new_artifact = "collections/c/documents/doc-1/derived/parse_v2/kg.jsonl"
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_FACTS,
        status=IndexStatus.ACTIVE,
        derived_artifact_path=new_artifact,
    )
    pending_id = _insert_row(
        engine,
        document_id="doc-1",
        parse_version="v1",
        modality=Modality.GRAPH_VECTORS,
        status=IndexStatus.PENDING,
        source_path="old/path/kg.jsonl",  # 漂移但 PENDING 不该被打扰
    )

    assert reconcile_graph_vectors_enqueue(engine=engine) == 0
    # 行不变 (没被改 source_path).
    assert _row(engine, pending_id).status == IndexStatus.PENDING.value
    assert _row(engine, pending_id).source_path == "old/path/kg.jsonl"


# ---------------------------------------------------------------------
# (5c) Graph vectors retry budget — task #7
# ---------------------------------------------------------------------
#
# graph_vectors 行复用通用的 reconcile_failed_retry 路径 (设计文档
# v3.1 §4.4 + §I.2). task #7 主要把这条复用契约钉在测试里, 防止后续
# 改造时不小心给 graph_vectors 加专属 retry 路径或绕开 MAX_RETRY_COUNT
# 上限. 跟 task #5 step 4 实现的 reconcile_graph_vectors_enqueue
# 一起, 形成「事实层 ACTIVE → 入队 → 失败可重试 → 超预算永久失败」
# 闭环.


def test_reconciler_graph_vectors_failed_retry_within_budget_flips_to_pending(engine):
    """graph_vectors FAILED + retry_count <= MAX + retry_after 已过 → flip 回 PENDING.

    钉死「graph_vectors 用通用 retry 路径」契约: 它跟 vector / fulltext / graph_facts
    走同一条 :func:`reconcile_failed_retry`, 不需要专属代码.
    """
    past = _utcnow() - timedelta(seconds=10)
    vec_id = _insert_row(
        engine,
        document_id="doc-gv-retry",
        parse_version="vvvvvvvvvvvvvvv1",
        modality=Modality.GRAPH_VECTORS,
        status=IndexStatus.FAILED,
        retry_count=2,
        retry_after=past,
    )

    flipped = reconcile_failed_retry(engine=engine)
    assert flipped == 1
    assert _row(engine, vec_id).status == IndexStatus.PENDING.value, (
        "graph_vectors row past backoff with budget remaining must rejoin PENDING queue"
    )


def test_reconciler_graph_vectors_failed_retry_overbudget_stays_permanently_failed(engine):
    """graph_vectors FAILED + retry_count > MAX_RETRY_COUNT → 永久 FAILED.

    设计文档 §I.2: ``MAX_RETRY_COUNT=5`` 上限语义是「初始尝试 + 至多
    5 次 retry = 6 次总尝试」. retry_count == MAX_RETRY_COUNT (5)
    时 ``<=`` 比较仍为 True, 那次仍然会被 reconciler flip 回 PENDING,
    给 worker 跑第 6 次. 等 _finalize_failed 把 retry_count 提到 6 (> MAX)
    才会清掉 retry_after, 进入永久 FAILED 状态.

    本测试钉死 retry_count > MAX 的永久语义: 超预算行停在 FAILED,
    reconciler 永远不再 re-enqueue. 操作员手动 reset 才能恢复.

    边界 `retry_count == MAX_RETRY_COUNT` 仍可重试由
    :func:`test_reconciler_graph_vectors_failed_retry_at_budget_boundary_still_retries`
    单独钉, 防止后续把 ``<=`` 改成 ``<`` 而把 6 次总尝试缩成 5 次.
    """
    overbudget_id = _insert_row(
        engine,
        document_id="doc-gv-ob",
        parse_version="vvvvvvvvvvvvvvv2",
        modality=Modality.GRAPH_VECTORS,
        status=IndexStatus.FAILED,
        retry_count=MAX_RETRY_COUNT + 1,
        retry_after=None,  # already cleared by orchestrator's _finalize_failed
    )

    flipped = reconcile_failed_retry(engine=engine)
    assert flipped == 0, "overbudget graph_vectors row must not be re-queued"
    assert _row(engine, overbudget_id).status == IndexStatus.FAILED.value
    # Repeated reconcile cycles must keep the same answer — no spurious flips.
    for _ in range(3):
        assert reconcile_failed_retry(engine=engine) == 0


def test_reconciler_graph_vectors_failed_retry_at_budget_boundary_still_retries(engine):
    """graph_vectors FAILED + retry_count == MAX_RETRY_COUNT → 仍 flip PENDING.

    huangheng PR #1874 NIT-C: 钉死 ``retry_count <= MAX_RETRY_COUNT``
    边界 (5 <= 5 = True) 而非 ``<``. 否则后续 refactor 改成 ``<``
    会把 5 次允许的 retry 缩成 4 次, 把总尝试 6 次缩成 5 次, 静默
    收紧 retry budget — 用户感知到的就是「最后一次重试机会被吞」.
    本 test 跟 overbudget test (`retry_count == MAX_RETRY_COUNT + 1`)
    互补, 双面 pin 上限语义.
    """
    past = _utcnow() - timedelta(seconds=10)
    boundary_id = _insert_row(
        engine,
        document_id="doc-gv-bnd",
        parse_version="vvvvvvvvvvvvvvv4",
        modality=Modality.GRAPH_VECTORS,
        status=IndexStatus.FAILED,
        retry_count=MAX_RETRY_COUNT,  # exactly at the cap — still retryable
        retry_after=past,
    )

    flipped = reconcile_failed_retry(engine=engine)
    assert flipped == 1, (
        "retry_count == MAX_RETRY_COUNT must still flip to PENDING — the cap is the last allowed retry, not the cutoff"
    )
    assert _row(engine, boundary_id).status == IndexStatus.PENDING.value


def test_reconciler_graph_vectors_failure_does_not_demote_facts_active(engine):
    """graph_vectors 永久失败时, 同 (document, parse_version) 的
    graph_facts 行不被影响.

    设计文档 §4.5: 文档级图谱可用状态由 graph_facts ACTIVE 单独决定;
    graph_vectors 只是补充检索向量, 失败时降级到精确匹配 + 别名 +
    模糊匹配 (前两层降级). reconcile_failed_retry 在 graph_vectors
    上跑时不能误碰 graph_facts 行.
    """
    facts_id = _insert_row(
        engine,
        document_id="doc-iso",
        parse_version="vvvvvvvvvvvvvvv3",
        modality=Modality.GRAPH_FACTS,
        status=IndexStatus.ACTIVE,
        is_serving=True,
        derived_artifact_path="collections/c/documents/doc-iso/derived/parse_v3/kg.jsonl",
    )
    vectors_id = _insert_row(
        engine,
        document_id="doc-iso",
        parse_version="vvvvvvvvvvvvvvv3",
        modality=Modality.GRAPH_VECTORS,
        status=IndexStatus.FAILED,
        retry_count=MAX_RETRY_COUNT + 1,
        retry_after=None,
    )

    reconcile_failed_retry(engine=engine)

    facts_row = _row(engine, facts_id)
    vectors_row = _row(engine, vectors_id)
    assert facts_row.status == IndexStatus.ACTIVE.value, (
        "graph_facts must stay ACTIVE — it does not depend on graph_vectors success"
    )
    assert facts_row.is_serving is True, "graph_facts is_serving must not flip on vector failure"
    assert vectors_row.status == IndexStatus.FAILED.value, (
        "graph_vectors stays FAILED permanently — operator-only recovery, not auto"
    )


# ---------------------------------------------------------------------
# (6) §F.3 single-TX cutover in worker (per architect ruling msg=492315e8 Ruling 1)
# ---------------------------------------------------------------------
#
# Cutover is NOT a reconciler scan — it MUST run inside the worker's
# own session immediately after sync() succeeds. Splitting status=ACTIVE
# from is_serving promotion creates an inconsistency window §F.4
# disallows. These tests assert the cutover happens atomically inside
# process_one_task() on success.


def test_orchestrator_cutover_promotes_new_and_demotes_prior_serving_in_one_tx(engine):
    """A successful sync must demote the prior is_serving row + promote
    the new row in a single TX (§F.3 three-statement contract)."""
    store = InMemoryObjectStore()
    doc_id, new_pv, chunks_path = _seed_chunks(store)
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=store)

    # Pre-existing serving row for the same (doc, modality), with a
    # different parse_version, simulating "doc was already indexed".
    old_id = _insert_row(
        engine,
        document_id=doc_id,
        parse_version="oldparseversionx"[:16],
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )
    new_id = _insert_row(
        engine,
        document_id=doc_id,
        parse_version=new_pv,
        modality=Modality.VECTOR,
        source_path=chunks_path,
    )
    payload = DispatchPayload(
        index_id=new_id,
        document_id=doc_id,
        parse_version=new_pv,
        modality=Modality.VECTOR,
        source_path=chunks_path,
    )

    outcome = asyncio.run(process_one_task(engine=engine, payload=payload, worker=worker, heartbeat_interval_seconds=0))
    assert outcome == "completed"

    old_row = _row(engine, old_id)
    new_row = _row(engine, new_id)
    assert old_row.is_serving is False, "prior serving row must be demoted by §F.3 stmt 2"
    assert new_row.is_serving is True, "new ACTIVE row must be promoted by §F.3 stmt 3"


def test_orchestrator_cutover_respects_partial_unique_invariant(engine):
    """§F.3 cutover must never leave 2 rows is_serving=TRUE for the same
    (doc, modality) — the §F.1 partial unique index guards against any
    drift even if statement 2 (demote) were skipped."""
    store = InMemoryObjectStore()
    doc_id, new_pv, chunks_path = _seed_chunks(store)
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=store)

    _insert_row(
        engine,
        document_id=doc_id,
        parse_version="oldparseversionx"[:16],
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )
    new_id = _insert_row(
        engine,
        document_id=doc_id,
        parse_version=new_pv,
        modality=Modality.VECTOR,
        source_path=chunks_path,
    )
    payload = DispatchPayload(
        index_id=new_id,
        document_id=doc_id,
        parse_version=new_pv,
        modality=Modality.VECTOR,
        source_path=chunks_path,
    )
    asyncio.run(process_one_task(engine=engine, payload=payload, worker=worker, heartbeat_interval_seconds=0))

    with Session(engine) as session:
        serving_count = session.scalar(
            select(text("COUNT(*)"))
            .select_from(DocumentIndex.__table__)
            .where(DocumentIndex.document_id == doc_id)
            .where(DocumentIndex.modality == Modality.VECTOR.value)
            .where(DocumentIndex.is_serving.is_(True))
        )
    assert serving_count == 1, (
        "post-cutover, exactly one row per (doc, modality) is serving — §F.1 partial unique invariant"
    )


def test_orchestrator_cutover_is_per_modality(engine):
    """Vector cutover must NOT affect fulltext serving for the same doc (§F.6)."""
    store = InMemoryObjectStore()
    doc_id, new_pv, chunks_path = _seed_chunks(store)
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=store)

    ft_id = _insert_row(
        engine,
        document_id=doc_id,
        parse_version="ftparsversion111"[:16],
        modality=Modality.FULLTEXT,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )
    vec_id = _insert_row(
        engine,
        document_id=doc_id,
        parse_version=new_pv,
        modality=Modality.VECTOR,
        source_path=chunks_path,
    )
    payload = DispatchPayload(
        index_id=vec_id,
        document_id=doc_id,
        parse_version=new_pv,
        modality=Modality.VECTOR,
        source_path=chunks_path,
    )
    asyncio.run(process_one_task(engine=engine, payload=payload, worker=worker, heartbeat_interval_seconds=0))

    assert _row(engine, vec_id).is_serving is True
    assert _row(engine, ft_id).is_serving is True, "fulltext serving must be untouched by vector cutover (§F.6)"


# ---------------------------------------------------------------------
# (7) Cleanup orphan parse_version GC (§F.5)
# ---------------------------------------------------------------------


def test_cleanup_deletes_superseded_parse_version_after_cooldown(engine):
    """An old parse_version with a newer sibling must be GC'd after cool-down."""
    store = InMemoryObjectStore()
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=store)

    # Stage two parse_versions for the same (doc, modality).
    old_id = _insert_row(
        engine,
        document_id="doc-1",
        parse_version="oldparseversion0"[:16],
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=False,
    )
    new_id = _insert_row(
        engine,
        document_id="doc-1",
        parse_version="newparseversion0"[:16],
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )

    # Force the old row to look "stale by 2h" so the 1h cool-down has elapsed.
    _set_updated_at(engine, old_id, _utcnow() - timedelta(hours=2))
    # Make the new row visibly newer.
    _set_updated_at(engine, new_id, _utcnow() - timedelta(seconds=10))

    # Pre-populate the backend so we can assert the cleanup deletes.
    backend.upsert_point(
        point_id="chunk-old",
        embedding=[0.1] * 16,
        payload={
            "document_id": "doc-1",
            "parse_version": "oldparseversion0"[:16],
            "modality": "vector",
            "chunk_id": "chunk-old",
            "text": "stale",
            "section_path": None,
            "heading_anchor": None,
            "page_idx": None,
        },
    )
    backend.upsert_point(
        point_id="chunk-new",
        embedding=[0.2] * 16,
        payload={
            "document_id": "doc-1",
            "parse_version": "newparseversion0"[:16],
            "modality": "vector",
            "chunk_id": "chunk-new",
            "text": "fresh",
            "section_path": None,
            "heading_anchor": None,
            "page_idx": None,
        },
    )

    counts = asyncio.run(
        cleanup_orphan_parse_versions(
            engine=engine,
            workers={Modality.VECTOR: worker},
        )
    )
    assert counts["backend_deleted"] == 1
    assert counts["rows_deleted"] == 1

    # Old backend entry is gone; new entry survives.
    surviving = backend.points_for_document("doc-1")
    surviving_chunk_ids = {p["point_id"] for p in surviving}
    assert surviving_chunk_ids == {"chunk-new"}

    # Old DB row is gone; new row still there.
    with Session(engine) as session:
        remaining = list(session.scalars(select(DocumentIndex.id).where(DocumentIndex.document_id == "doc-1")))
    assert remaining == [new_id]


def test_cleanup_respects_cooldown(engine):
    """A row whose ``updated_at`` is within the cool-down must NOT be deleted."""
    store = InMemoryObjectStore()
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=store)

    old_id = _insert_row(
        engine,
        document_id="doc-1",
        parse_version="oldparseversion0"[:16],
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=False,
    )
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="newparseversion0"[:16],
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )
    # Old row was superseded just 5 minutes ago — under the default 1h cool-down.
    _set_updated_at(engine, old_id, _utcnow() - timedelta(minutes=5))

    counts = asyncio.run(
        cleanup_orphan_parse_versions(
            engine=engine,
            workers={Modality.VECTOR: worker},
        )
    )
    assert counts["rows_deleted"] == 0, (
        "rows still inside the §F.5 cool-down must not be GC'd (cutover races may not have settled)"
    )


def test_cleanup_orphan_parse_version_for_graph_is_backend_noop(engine):
    """Per architect ruling msg=492315e8 Ruling 3: graph orphan
    parse_version GC is a backend no-op because the §D.3.6 sync
    supersede semantic already cleared old lineage members when the
    new parse_version was written. The DB row is still dropped.
    """
    store = InMemoryObjectStore()  # noqa: F841 — included for parity with non-graph tests
    graph_store = _StubLineageGraphStore()
    worker = _GraphLikeWorker(graph_store)

    old_id = _insert_row(
        engine,
        document_id="doc-1",
        parse_version="oldgraphver00000"[:16],
        modality=Modality.GRAPH,
        status=IndexStatus.ACTIVE,
        is_serving=False,
    )
    _insert_row(
        engine,
        document_id="doc-1",
        parse_version="newgraphver00000"[:16],
        modality=Modality.GRAPH,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )
    _set_updated_at(engine, old_id, _utcnow() - timedelta(hours=2))

    counts = asyncio.run(
        cleanup_orphan_parse_versions(
            engine=engine,
            workers={Modality.GRAPH: worker},
        )
    )
    assert counts["graph_noop"] == 1, "orphan parse_v GC for graph must be a counted no-op"
    assert counts["backend_deleted"] == 0
    assert counts["backend_skipped"] == 0
    assert counts["rows_deleted"] == 1, (
        "DB row is still GC'd even though the graph backend lineage was already cleared by sync"
    )
    # The graph store was never touched on the orphan path.
    assert graph_store.find_calls == 0
    assert graph_store.remove_calls == 0


# ---------------------------------------------------------------------
# (7b) cleanup_for_deleted_documents (path B — caller-driven document delete)
# ---------------------------------------------------------------------


def test_cleanup_for_deleted_documents_removes_non_graph_backend_per_parse_version(engine):
    """Document deletion path: every parse_version row's backend tombstone
    is removed via the worker's flat delete (vector / fulltext / summary
    / vision)."""
    store = InMemoryObjectStore()  # noqa: F841 — required arg shape parity
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=InMemoryObjectStore())

    pv_a = "deldocparsversA1"[:16]
    pv_b = "deldocparsversB1"[:16]
    _insert_row(
        engine,
        document_id="doc-del",
        parse_version=pv_a,
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )
    _insert_row(
        engine,
        document_id="doc-del",
        parse_version=pv_b,
        modality=Modality.VECTOR,
        status=IndexStatus.ACTIVE,
        is_serving=False,
    )
    for pv, chunk_id in ((pv_a, "chunk-a"), (pv_b, "chunk-b")):
        backend.upsert_point(
            point_id=chunk_id,
            embedding=[0.0] * 16,
            payload={
                "document_id": "doc-del",
                "parse_version": pv,
                "modality": "vector",
                "chunk_id": chunk_id,
                "text": "x",
                "section_path": None,
                "heading_anchor": None,
                "page_idx": None,
            },
        )

    counts = asyncio.run(
        cleanup_for_deleted_documents(
            engine=engine,
            workers={Modality.VECTOR: worker},
            document_ids=["doc-del"],
        )
    )
    assert counts["backend_deleted"] == 2, "every parse_version row gets a backend delete"
    assert counts["rows_deleted"] == 2
    assert backend.points_for_document("doc-del") == [], "backend tombstones gone"
    with Session(engine) as session:
        remaining = list(session.scalars(select(DocumentIndex.id).where(DocumentIndex.document_id == "doc-del")))
    assert remaining == []


def test_cleanup_for_deleted_documents_calls_graph_lineage_cleanup_once_per_doc(engine):
    """Document deletion path on graph: regardless of how many
    parse_version rows exist for a document, the lineage cleanup call
    is invoked exactly once per (document_id, graph) — the call is
    by-document, not by-parse_version (per §D.3.2 amended canonical
    PR #1725 head a0a47994)."""
    graph_store = _StubLineageGraphStore(
        entity_lineage={"doc-graph": ["entity-A", "entity-B"]},
        relation_lineage={"doc-graph": [("entity-A", "REL", "entity-B")]},
    )
    worker = _GraphLikeWorker(graph_store)

    _insert_row(
        engine,
        document_id="doc-graph",
        parse_version="graphpv0000000_a"[:16],
        modality=Modality.GRAPH,
        status=IndexStatus.ACTIVE,
        is_serving=False,
    )
    _insert_row(
        engine,
        document_id="doc-graph",
        parse_version="graphpv0000000_b"[:16],
        modality=Modality.GRAPH,
        status=IndexStatus.ACTIVE,
        is_serving=True,
    )

    counts = asyncio.run(
        cleanup_for_deleted_documents(
            engine=engine,
            workers={Modality.GRAPH: worker},
            document_ids=["doc-graph"],
        )
    )

    assert counts["graph_lineage_cleaned"] == 1, (
        "lineage cleanup runs once per document, regardless of parse_version count"
    )
    assert counts["rows_deleted"] == 2, "all parse_version rows for the document are GC'd"
    assert graph_store.remove_calls == 2, "two entities had lineage members removed"
    assert graph_store.gc_calls == 2, "each entity was checked for GC once its lineage was empty"
    assert graph_store.relation_remove_calls == 1, "the one relation was removed"


def test_cleanup_for_deleted_documents_handles_empty_input(engine):
    counts = asyncio.run(
        cleanup_for_deleted_documents(
            engine=engine,
            workers={},
            document_ids=[],
        )
    )
    assert counts == {
        "backend_deleted": 0,
        "graph_lineage_cleaned": 0,
        "rows_deleted": 0,
        "backend_skipped": 0,
    }


# ---------------------------------------------------------------------
# Stub LineageGraphStore + GraphModalityWorker for cleanup tests.
# We don't import GraphModalityWorker because constructing it requires
# extras (Nebula/Neo4j path) and an extractor; the cleanup path only
# touches `_store` + `_entity_lock` (Wave 1 conventions), so a tiny
# duck-typed stand-in covers the contract.
# ---------------------------------------------------------------------


class _StubAsyncLock:
    """Async context manager that records acquire/release sequence."""

    def __init__(self) -> None:
        self.acquired = []

    def acquire(self, entity_id: str):
        return self._Acquire(self, entity_id)

    class _Acquire:
        def __init__(self, parent, entity_id):
            self.parent = parent
            self.entity_id = entity_id

        async def __aenter__(self):
            self.parent.acquired.append(self.entity_id)
            return None

        async def __aexit__(self, *args):
            return None


class _StubLineageGraphStore:
    """Records cleanup calls so tests can assert on call shape."""

    def __init__(
        self,
        entity_lineage: dict[str, list[str]] | None = None,
        relation_lineage: dict[str, list[tuple[str, str, str]]] | None = None,
    ) -> None:
        self._entity_lineage = entity_lineage or {}
        self._relation_lineage = relation_lineage or {}
        self.find_calls = 0
        self.remove_calls = 0
        self.gc_calls = 0
        self.relation_remove_calls = 0

    async def find_entity_ids_with_lineage(self, *, document_id: str) -> list[str]:
        self.find_calls += 1
        return list(self._entity_lineage.get(document_id, []))

    async def remove_entity_lineage_member(self, *, entity_name: str, document_id: str) -> None:
        self.remove_calls += 1

    async def gc_entity_if_orphan(self, *, entity_name: str) -> None:
        self.gc_calls += 1

    async def find_relation_keys_with_lineage(self, *, document_id: str) -> list[tuple[str, str, str]]:
        return list(self._relation_lineage.get(document_id, []))

    async def remove_relation_lineage_member(self, *, relation_key: tuple[str, str, str], document_id: str) -> None:
        self.relation_remove_calls += 1


class _GraphLikeWorker(ModalityWorker):
    """Stand-in for ``GraphModalityWorker`` — exposes ``_store`` +
    ``_entity_lock`` without requiring graph extras."""

    modality = Modality.GRAPH

    def __init__(self, store: _StubLineageGraphStore | None = None) -> None:
        self._store = store or _StubLineageGraphStore()
        self._entity_lock = _StubAsyncLock()

    async def derive(self, *, document_id, parse_version, source_path):
        pass  # pragma: no cover

    async def sync(self, *, document_id, parse_version, derived_artifact_path):
        pass  # pragma: no cover


# ---------------------------------------------------------------------
# (8) End-to-end PENDING → orchestrator → ACTIVE+is_serving smoke
# ---------------------------------------------------------------------


def test_end_to_end_pending_dispatch_orchestrator_run(engine):
    """Full smoke: PENDING → reconciler dispatch → orchestrator run →
    ACTIVE + is_serving=TRUE in one TX (no separate reconciler cutover step
    per architect ruling msg=492315e8 Ruling 1)."""
    store = InMemoryObjectStore()
    doc_id, parse_version, chunks_path = _seed_chunks(store)
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=store)
    queue = InMemoryWorkQueue()

    row_id = _insert_row(
        engine,
        document_id=doc_id,
        parse_version=parse_version,
        modality=Modality.VECTOR,
        source_path=chunks_path,
    )

    # 1. Reconciler pushes PENDING to queue.
    asyncio.run(reconcile_pending_dispatch(engine=engine, queue=queue))
    assert queue.qsize(Modality.VECTOR) == 1

    # 2. Orchestrator pops + processes — both ACTIVE and is_serving=TRUE
    # land in one §F.3 transaction.
    raw = asyncio.run(queue.pop(modality=Modality.VECTOR, timeout_seconds=0.1))
    assert raw is not None
    payload = DispatchPayload.from_dict(raw)
    outcome = asyncio.run(process_one_task(engine=engine, payload=payload, worker=worker, heartbeat_interval_seconds=0))
    assert outcome == "completed"
    final = _row(engine, row_id)
    assert final.status == IndexStatus.ACTIVE.value
    assert final.is_serving is True


# ---------------------------------------------------------------------
# (9) DispatchPayload round-trip (queue serialization sanity)
# ---------------------------------------------------------------------


def test_dispatch_payload_dict_round_trip_preserves_all_fields():
    payload = DispatchPayload(
        index_id=42,
        document_id="doc-x",
        parse_version="abcdef0123456789",
        modality=Modality.SUMMARY,
        source_path="collections/c/documents/d/derived/parse_v/markdown.md",
        collection_id="c",
    )
    assert DispatchPayload.from_dict(payload.to_dict()) == payload
    # JSON round-trip — Redis BLPOP gets bytes, pop() decodes to dict.
    import json as _json

    parsed = _json.loads(payload.to_json())
    assert DispatchPayload.from_dict(parsed) == payload
