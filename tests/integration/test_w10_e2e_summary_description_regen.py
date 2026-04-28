# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 10 task #6 (Chunk G) — collection summary / description regen
end-to-end narrative validation.

Covers the Wave 10 §K.13 happy path the chunked PRs (A schema +
B legacy hard-cut + C regen service + D 2 OpenAPI endpoints +
E reconciler hook) compose into a single user journey:

  ``POST /api/v2/collections/{id}/summary/regen`` →
  ``Collection.summary`` populated →
  ``POST /api/v2/collections/{id}/description/regen`` →
  ``Collection.description`` populated →
  reconciler hook re-triggers after document edit.

Mirrors the Wave 7 task #11 pattern (service-layer + DB-direct
narrative; HTTP shape pinned by route unit tests; Layer 2 env-gated)
so a regression that survives unit-level grep is caught here while
rollback is still cheap.

Per Wave 10 design §10 the test requirements that this file covers
end-to-end (vs the unit tests that pin them in isolation):

* §10.1 lease atomic semantics — narrative step 7
* §10.2 3-tier fallback chain — narrative step 3 (agent tier alive)
* §10.4 API 400 reject when summary is NULL — narrative step 6
* §10.5 quality gate (``is_valid_summary`` / ``is_valid_description``)
  — narrative step 4
* §10.6 trigger 三场景 (add/edit/delete doc) — narrative step 7's
  doc-edit-triggers-regen flow exercises the edit case end-to-end
* §10.9 silent failure 修复 (endpoint must not return 200 when DB
  write didn't happen) — narrative step 9 failure-mode

Layer split (per Wave 4 ``test_full_indexing_pipeline.py`` precedent
+ Wave 7 ``test_w7_e2e_graph_recall_and_merge.py``):

* **Layer 2 — full narrative (gated by ``RUN_W10_E2E_NARRATIVE=1``)**:
  brings up real Postgres + Redis + the configured LLM/embedding
  provider. CI Wave 10 lane flips it on; local-dev pytest stays fast
  by default.

12-invariant table (PR body): mostly n/a — narrative-correctness is
the hard gate on this file. Material invariants validated implicitly
by the narrative.

4-pattern pre-check matrix (PR body):
  * Pattern 1 v1: ``regen_summary`` / ``regen_description`` are
    importable from ``aperag/domains/knowledge_base/service/collection_regen_service.py``
    (Chunk C).
  * Pattern 1 v2: ``Collection.summary`` / ``.description`` /
    ``.summary_updated_at`` / ``.description_updated_at`` /
    ``.regen_lease_owner`` / ``.regen_lease_expires_at`` columns exist
    (Chunk A).
  * Pattern 2: ``reconcile_collection_descriptions_hook`` wired into
    the reconciler main loop (Chunk E).
  * Pattern 3: ``POST /summary/regen`` and ``POST /description/regen``
    exposed on the FastAPI router (Chunk D).

simple-stable 4-guardrail (PR body):
  1. 不无限扩范围: one file, no production code change.
  2. 先把功能做实: real backends + real provider — narrative validates
     production behaviour, not stubbed surface.
  3. 简单稳定: one happy-path narrative + one 400-reject pin + one
     failure-mode step. Not a regression matrix.
  4. 私有化免维护: env-var-gated; CI lane flips it on, local stays
     fast.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------
# Layer 2 gate.
# ---------------------------------------------------------------------

_CHUNKS_LANDED = True  # Wave 10 chunks A / B / C / D / E all merged
# (PRs #1783 + #1786 + #1790). Three regen surfaces alive:
#   * regen_summary / regen_description (collection_regen_service)
#   * reconcile_collection_descriptions_hook (reconciler)
#   * POST /summary/regen and /description/regen (router)

_RUN_GATE_ENV = "RUN_W10_E2E_NARRATIVE"


def _layer2_skip_reason() -> str:
    return (
        f"Layer 2 e2e narrative requires {_RUN_GATE_ENV}=1 + real "
        "Postgres / Redis / configured LLM provider keys. CI Wave 10 "
        "lane flips this on; local-dev pytest stays fast by default."
    )


pytestmark = [
    pytest.mark.skipif(
        os.environ.get(_RUN_GATE_ENV) != "1",
        reason=_layer2_skip_reason(),
    ),
]


# ---------------------------------------------------------------------
# Module-scoped fixture: per-narrative collection + tenant.
#
# Like the Wave 7 task #11 narrative, this binds the entire 9-step
# narrative to one synthetic collection_id + user. Steps share state
# via the ``Collection`` row in Postgres (the production data plane),
# not Python globals — which is what makes this an *integration*
# narrative rather than a stub chain.
# ---------------------------------------------------------------------


_USER_ID = "w10-e2e-user-" + uuid.uuid4().hex[:8]
_COLLECTION_ID = "w10-e2e-" + uuid.uuid4().hex[:12]


def _resolve_engine_or_skip() -> Any:
    """Mirror ``test_full_indexing_pipeline._resolve_phase1_e2e_engine``.
    Layer 2 requires the same Postgres reachability as that suite.
    """
    from sqlalchemy import create_engine

    from aperag.config import settings

    if not settings.database_url:
        pytest.skip("Layer 2: settings.database_url empty (POSTGRES_HOST etc unset)")
    return create_engine(settings.database_url, future=True)


@pytest.fixture(scope="module")
def db_engine() -> Any:
    return _resolve_engine_or_skip()


@pytest.fixture(scope="module")
def seeded_collection(db_engine) -> Any:
    """Insert a Collection + a few Document rows into Postgres so the
    regen narrative has something to summarise. Tear down on module
    exit so the test does not leak rows across runs.
    """
    from sqlalchemy import insert
    from sqlalchemy.orm import Session

    from aperag.domains.knowledge_base.db.models import (
        Collection,
        CollectionStatus,
        CollectionType,
        Document,
        DocumentStatus,
    )

    with Session(db_engine) as session:
        session.execute(
            insert(Collection).values(
                id=_COLLECTION_ID,
                title="W10 e2e narrative collection",
                user=_USER_ID,
                status=CollectionStatus.ACTIVE.value,
                type=CollectionType.DOCUMENT.value,
                config="{}",
            )
        )
        for i in range(3):
            session.execute(
                insert(Document).values(
                    id=f"{_COLLECTION_ID}-doc{i}",
                    collection_id=_COLLECTION_ID,
                    user=_USER_ID,
                    name=f"e2e-doc{i}.md",
                    status=DocumentStatus.ACTIVE.value,
                    size=2048,
                )
            )
        session.commit()
    yield SimpleNamespace(id=_COLLECTION_ID, user=_USER_ID)
    # Best-effort cleanup — failed runs keep rows (operator triages).
    try:
        with Session(db_engine) as session:
            session.execute(
                Document.__table__.delete().where(Document.collection_id == _COLLECTION_ID)
            )
            session.execute(Collection.__table__.delete().where(Collection.id == _COLLECTION_ID))
            session.commit()
    except Exception:  # noqa: BLE001 — narrative already passed/failed
        pass


def _read_collection(db_engine, collection_id: str) -> Any:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from aperag.domains.knowledge_base.db.models import Collection

    with Session(db_engine) as session:
        return session.scalar(select(Collection).where(Collection.id == collection_id))


# ---------------------------------------------------------------------
# Step bodies — narrative validation, one or two assertions per step.
# Tests run in source order so step N+1 sees the side-effects of
# step N.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step1_collection_seeded_with_no_summary(seeded_collection, db_engine):
    """The Wave 10 hard-cut shipped ``Collection.summary`` and
    ``.description`` as nullable Text columns (Chunk A). A freshly
    created collection MUST start with both NULL — the regen service
    is the only writer (no auto-fill via constructor)."""
    row = _read_collection(db_engine, _COLLECTION_ID)
    assert row is not None
    assert row.summary is None, "fresh collection should have no summary"
    assert row.description is None, "fresh collection should have no description"
    assert row.summary_updated_at is None
    assert row.description_updated_at is None


@pytest.mark.asyncio
async def test_step2_regen_summary_writes_canonical_summary(seeded_collection, db_engine):
    """Validates Chunk C ``regen_summary`` end-to-end: it acquires the
    lease, runs the 3-tier fallback chain (Tier 1 agent / Tier 2
    chunks-fallback / Tier 3 transient skip), writes
    ``Collection.summary`` + ``.summary_updated_at`` atomically, and
    releases the lease.

    Returns ``True`` on success per the contract; the row's
    ``summary`` becomes non-NULL with length > 50 (quality gate
    ``is_valid_summary``).
    """
    from aperag.domains.knowledge_base.service.collection_regen_service import regen_summary

    success = await regen_summary(_COLLECTION_ID)
    assert success is True, "Tier 1 / Tier 2 should produce a valid summary on a seeded collection"

    row = _read_collection(db_engine, _COLLECTION_ID)
    assert row.summary is not None
    assert len(row.summary) > 50, "is_valid_summary minimum length"
    assert row.summary_updated_at is not None
    # Lease released — owner column cleared on success path.
    assert row.regen_lease_owner is None


@pytest.mark.asyncio
async def test_step3_quality_gate_rejects_short_or_refusal_text():
    """``is_valid_summary`` and ``is_valid_description`` are the
    quality gates the regen service consults before persisting. The
    happy-path step 2 already exercises a *passing* call; this step
    pins the *rejection* axis so a future relaxation of the gate
    surfaces here."""
    from aperag.domains.knowledge_base.service.collection_regen_service import (
        is_valid_description,
        is_valid_summary,
    )

    assert is_valid_summary("") is False
    assert is_valid_summary("short") is False
    # LLM refusal templates per design §6.2 quality gate.
    assert is_valid_summary("Sorry, I cannot help with that.") is False
    assert is_valid_summary("I'm sorry, but I cannot generate this.") is False
    # A sufficiently-long substantive text passes.
    valid = "ApeRAG is a Python-based RAG platform offering vector + graph + fulltext recall." * 2
    assert is_valid_summary(valid) is True
    assert is_valid_description(valid) is True


@pytest.mark.asyncio
async def test_step4_regen_description_derives_short_form(seeded_collection, db_engine):
    """Stage 2 derives ``Collection.description`` from the now-populated
    ``Collection.summary``. Cheap path — single LLM call, no agent
    multi-turn. The row's ``description`` becomes non-NULL passing
    ``is_valid_description``."""
    from aperag.domains.knowledge_base.service.collection_regen_service import regen_description

    success = await regen_description(_COLLECTION_ID)
    assert success is True

    row = _read_collection(db_engine, _COLLECTION_ID)
    assert row.description is not None
    assert len(row.description) > 50
    assert row.description_updated_at is not None
    assert row.regen_lease_owner is None


@pytest.mark.asyncio
async def test_step5_rest_summary_regen_endpoint_returns_200(seeded_collection):
    """The route layer ``POST /api/v2/collections/{id}/summary/regen``
    delegates to ``regen_summary`` (per ``regen_collection_summary_view``).
    HTTP shape: 200 + ``CollectionRegenTriggerResponse`` body.

    The ``audit`` decorator wraps the view; we exercise the inner
    function directly so the test does not need a live FastAPI app —
    HTTP shape is pinned by the route unit tests, narrative
    correctness is what we validate here.
    """
    from aperag.domains.knowledge_base.api.routes import regen_collection_summary_view
    from aperag.domains.knowledge_base.schemas import CollectionRegenTriggerResponse

    user = SimpleNamespace(id=_USER_ID)
    # ``regen_collection_summary_view`` is wrapped by ``@audit``; the
    # wrapper exposes the underlying coroutine via ``__wrapped__``.
    inner = getattr(regen_collection_summary_view, "__wrapped__", regen_collection_summary_view)
    result = await inner(_COLLECTION_ID, user=user)
    assert isinstance(result, CollectionRegenTriggerResponse)
    assert result.collection_id == _COLLECTION_ID
    assert result.stage == "summary"
    assert result.task_id  # uuid hex


@pytest.mark.asyncio
async def test_step6_description_regen_returns_400_when_summary_missing(db_engine):
    """Per design §9 + §10.4: ``POST /description/regen`` must return
    400 when ``Collection.summary IS NULL``. This is the contract
    that prevents Stage 2 from running on an un-summarised collection
    (would otherwise produce a description from no input).
    """
    from sqlalchemy import insert
    from sqlalchemy.orm import Session

    from aperag.domains.knowledge_base.api.routes import regen_collection_description_view
    from aperag.domains.knowledge_base.db.models import (
        Collection,
        CollectionStatus,
        CollectionType,
    )

    fresh_id = "w10-e2e-no-summary-" + uuid.uuid4().hex[:8]
    with Session(db_engine) as session:
        session.execute(
            insert(Collection).values(
                id=fresh_id,
                title="W10 e2e no-summary",
                user=_USER_ID,
                status=CollectionStatus.ACTIVE.value,
                type=CollectionType.DOCUMENT.value,
                config="{}",
            )
        )
        session.commit()

    user = SimpleNamespace(id=_USER_ID)
    inner = getattr(regen_collection_description_view, "__wrapped__", regen_collection_description_view)
    from fastapi import HTTPException

    try:
        with pytest.raises(HTTPException) as exc_info:
            await inner(fresh_id, user=user)
        assert exc_info.value.status_code == 400
        assert "summary" in (exc_info.value.detail or "").lower()
    finally:
        with Session(db_engine) as session:
            session.execute(Collection.__table__.delete().where(Collection.id == fresh_id))
            session.commit()


@pytest.mark.asyncio
async def test_step7_reconciler_hook_dispatches_after_document_edit(seeded_collection, db_engine):
    """Per Wave 10 §K.13 Chunk E: the reconciler hook
    ``reconcile_collection_descriptions_hook`` scans for collections
    whose ``summary_updated_at`` is older than the most recent
    ``Document.gmt_updated`` AND the latest edit is at least
    ``MIN_STALE_AGE`` old. Triggers Stage 1 regen for such collections.

    We back-date the document's ``gmt_updated`` past
    ``MIN_STALE_AGE`` so the scan picks the collection up, then call
    the hook directly. The hook dispatches via
    ``asyncio.create_task`` — we count the dispatched tasks via the
    return value rather than waiting for the regen to complete (the
    Stage 1 work itself is unit-tested elsewhere).
    """
    from sqlalchemy import update
    from sqlalchemy.orm import Session

    from aperag.domains.knowledge_base.db.models import Document
    from aperag.domains.knowledge_base.service.regen_constants import MIN_STALE_AGE
    from aperag.indexing.reconciler import reconcile_collection_descriptions_hook

    # Back-date one document so the stale-doc subquery hits.
    backdated = datetime.now(tz=timezone.utc) - MIN_STALE_AGE - timedelta(minutes=1)
    with Session(db_engine) as session:
        session.execute(
            update(Document)
            .where(Document.id == f"{_COLLECTION_ID}-doc0")
            .values(gmt_updated=backdated)
        )
        session.commit()

    # Also back-date the collection's summary_updated_at so the
    # ``Document.gmt_updated > Collection.summary_updated_at``
    # predicate hits.
    from aperag.domains.knowledge_base.db.models import Collection

    with Session(db_engine) as session:
        session.execute(
            update(Collection)
            .where(Collection.id == _COLLECTION_ID)
            .values(summary_updated_at=backdated - timedelta(minutes=1))
        )
        session.commit()

    dispatched = await reconcile_collection_descriptions_hook(engine=db_engine)
    # At least one task dispatched — our seeded collection should be
    # picked up. We do not assert exact equality because the engine
    # may have other in-flight collections from a parallel test run.
    assert dispatched >= 1


@pytest.mark.asyncio
async def test_step8_lease_busy_returns_false(seeded_collection, db_engine):
    """Per design §7 lease atomic semantics: if the lease is held by
    another worker, ``regen_summary`` returns ``False`` immediately
    without overwriting the row. We simulate the busy state by
    writing a far-future ``regen_lease_expires_at`` and a foreign
    ``regen_lease_owner`` directly to the DB.
    """
    from sqlalchemy import update
    from sqlalchemy.orm import Session

    from aperag.domains.knowledge_base.db.models import Collection
    from aperag.domains.knowledge_base.service.collection_regen_service import regen_summary

    foreign_owner = "other-worker-" + uuid.uuid4().hex[:8]
    far_future = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    with Session(db_engine) as session:
        session.execute(
            update(Collection)
            .where(Collection.id == _COLLECTION_ID)
            .values(regen_lease_owner=foreign_owner, regen_lease_expires_at=far_future)
        )
        session.commit()

    success = await regen_summary(_COLLECTION_ID)
    assert success is False, "lease busy must skip without overwriting"

    # Lease still held by the foreign owner — our call did not steal it.
    row = _read_collection(db_engine, _COLLECTION_ID)
    assert row.regen_lease_owner == foreign_owner

    # Cleanup — release the synthetic lease so subsequent module-level
    # tests can re-acquire (none after this in source order, but
    # defensive).
    with Session(db_engine) as session:
        session.execute(
            update(Collection)
            .where(Collection.id == _COLLECTION_ID)
            .values(regen_lease_owner=None, regen_lease_expires_at=None)
        )
        session.commit()


@pytest.mark.asyncio
async def test_step9_failure_mode_llm_down_returns_false_no_silent_write(seeded_collection, db_engine):
    """Failure-mode fold-in (mirror Wave 7 task #11 step 9 +
    design §10.9 silent-failure 修复):

    Patch the Stage-2 LLM call to raise. Re-trigger
    ``regen_description``. Assert:

    * Returns ``False`` — no silent ``True`` claiming success.
    * ``Collection.description`` is NOT overwritten (the prior step 4
      value remains).
    * ``description_updated_at`` is NOT advanced.

    This is the integration counterpart to the unit-level
    ``test_*_silent_failure_*`` invariants — the route layer 503 is
    pinned by route unit tests; here we pin the service-layer
    "failure → no DB write" contract end-to-end.
    """
    from aperag.domains.knowledge_base.service import collection_regen_service
    from aperag.domains.knowledge_base.service.collection_regen_service import regen_description

    pre = _read_collection(db_engine, _COLLECTION_ID)
    pre_description = pre.description
    pre_updated_at = pre.description_updated_at

    async def _fail_llm(_prompt: str) -> str:
        raise RuntimeError("LLM provider down — failure-mode injection")

    with patch.object(
        collection_regen_service,
        "_default_llm_factory",
        return_value=_fail_llm,
    ):
        success = await regen_description(_COLLECTION_ID)

    assert success is False, "LLM failure must NOT silently report success"

    post = _read_collection(db_engine, _COLLECTION_ID)
    assert post.description == pre_description, (
        "service must NOT overwrite description on LLM failure"
    )
    assert post.description_updated_at == pre_updated_at, (
        "service must NOT advance description_updated_at on LLM failure"
    )
