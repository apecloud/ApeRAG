# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 10 §K.13 Chunk E — reconciler hook tests.

Cover the three scenarios the hook must dispatch on:
  1. Collection without a summary
  2. Collection whose docs are newer than ``summary_updated_at`` AND past
     ``MIN_STALE_AGE``
  3. Summary newer than description (Stage 2 derive needed)

Plus the lease-busy skip path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.domains.knowledge_base.db.models import (
    Collection,
    CollectionStatus,
    CollectionType,
    Document,
    DocumentStatus,
)
from aperag.indexing import reconciler


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Collection.metadata.create_all(eng, tables=[Collection.__table__, Document.__table__])
    return eng


def _insert_collection(
    engine: Engine,
    *,
    collection_id: str,
    summary: str | None = None,
    summary_updated_at: datetime | None = None,
    description_updated_at: datetime | None = None,
    regen_lease_owner: str | None = None,
    regen_lease_expires_at: datetime | None = None,
    gmt_deleted: datetime | None = None,
) -> None:
    now = _utcnow()
    with Session(engine) as session, session.begin():
        session.add(
            Collection(
                id=collection_id,
                title=f"Collection {collection_id}",
                user="user_test",
                status=CollectionStatus.ACTIVE.value,
                type=CollectionType.DOCUMENT.value,
                config="{}",
                gmt_created=now,
                gmt_updated=now,
                gmt_deleted=gmt_deleted,
                summary=summary,
                summary_updated_at=summary_updated_at,
                description_updated_at=description_updated_at,
                regen_lease_owner=regen_lease_owner,
                regen_lease_expires_at=regen_lease_expires_at,
            )
        )


def _insert_document(
    engine: Engine,
    *,
    document_id: str,
    collection_id: str,
    gmt_updated: datetime,
    gmt_deleted: datetime | None = None,
) -> None:
    now = _utcnow()
    with Session(engine) as session, session.begin():
        session.add(
            Document(
                id=document_id,
                name=f"doc {document_id}",
                user="user_test",
                collection_id=collection_id,
                status=DocumentStatus.COMPLETE.value,
                size=100,
                gmt_created=now,
                gmt_updated=gmt_updated,
                gmt_deleted=gmt_deleted,
            )
        )


# ---------------------------------------------------------------------
# _select_collections_needing_regen
# ---------------------------------------------------------------------


def test_picks_collection_with_missing_summary(engine: Engine):
    _insert_collection(engine, collection_id="col_missing", summary=None)

    stage1, stage2 = reconciler._select_collections_needing_regen(engine, timedelta(minutes=10), batch_size=100)

    assert "col_missing" in stage1
    assert "col_missing" not in stage2  # no summary → not eligible for stage 2


def test_skips_collection_when_lease_held(engine: Engine):
    future = _utcnow() + timedelta(minutes=5)
    _insert_collection(
        engine,
        collection_id="col_locked",
        summary=None,
        regen_lease_owner="other-instance",
        regen_lease_expires_at=future,
    )

    stage1, _ = reconciler._select_collections_needing_regen(engine, timedelta(minutes=10), batch_size=100)

    assert "col_locked" not in stage1


def test_reclaims_collection_when_lease_expired(engine: Engine):
    expired = _utcnow() - timedelta(minutes=5)
    _insert_collection(
        engine,
        collection_id="col_stale_lease",
        summary=None,
        regen_lease_owner="dead-instance",
        regen_lease_expires_at=expired,
    )

    stage1, _ = reconciler._select_collections_needing_regen(engine, timedelta(minutes=10), batch_size=100)

    assert "col_stale_lease" in stage1


def test_skips_soft_deleted_collection(engine: Engine):
    _insert_collection(
        engine,
        collection_id="col_deleted",
        summary=None,
        gmt_deleted=_utcnow(),
    )

    stage1, stage2 = reconciler._select_collections_needing_regen(engine, timedelta(minutes=10), batch_size=100)

    assert "col_deleted" not in stage1
    assert "col_deleted" not in stage2


def test_picks_collection_when_doc_edit_postdates_summary_and_is_stale(engine: Engine):
    one_day_ago = _utcnow() - timedelta(days=1)
    twenty_min_ago = _utcnow() - timedelta(minutes=20)
    _insert_collection(
        engine,
        collection_id="col_stale",
        summary="x" * 250,
        summary_updated_at=one_day_ago,
    )
    _insert_document(
        engine,
        document_id="doc_recent",
        collection_id="col_stale",
        gmt_updated=twenty_min_ago,
    )

    stage1, _ = reconciler._select_collections_needing_regen(engine, timedelta(minutes=10), batch_size=100)

    assert "col_stale" in stage1


def test_skips_collection_when_doc_edit_too_recent(engine: Engine):
    """Within ``MIN_STALE_AGE`` of the latest doc edit, hold off — user
    might still be making changes."""
    one_day_ago = _utcnow() - timedelta(days=1)
    just_now = _utcnow() - timedelta(seconds=30)
    _insert_collection(
        engine,
        collection_id="col_active_edit",
        summary="x" * 250,
        summary_updated_at=one_day_ago,
    )
    _insert_document(
        engine,
        document_id="doc_just_edited",
        collection_id="col_active_edit",
        gmt_updated=just_now,
    )

    stage1, _ = reconciler._select_collections_needing_regen(engine, timedelta(minutes=10), batch_size=100)

    assert "col_active_edit" not in stage1


def test_picks_collection_for_stage2_when_description_stale(engine: Engine):
    one_hour_ago = _utcnow() - timedelta(hours=1)
    _insert_collection(
        engine,
        collection_id="col_stage2",
        summary="x" * 250,
        summary_updated_at=_utcnow(),
        description_updated_at=one_hour_ago,
    )

    stage1, stage2 = reconciler._select_collections_needing_regen(engine, timedelta(minutes=10), batch_size=100)

    # No doc activity → not stage 1; description older than summary →
    # picks for stage 2.
    assert "col_stage2" not in stage1
    assert "col_stage2" in stage2


def test_stage2_skipped_when_description_already_current(engine: Engine):
    now = _utcnow()
    _insert_collection(
        engine,
        collection_id="col_fresh",
        summary="x" * 250,
        summary_updated_at=now,
        description_updated_at=now + timedelta(seconds=1),
    )

    _, stage2 = reconciler._select_collections_needing_regen(engine, timedelta(minutes=10), batch_size=100)

    assert "col_fresh" not in stage2


def test_stage2_excluded_when_collection_in_stage1(engine: Engine):
    """A collection picked for Stage 1 must not also surface in Stage 2 —
    the next reconciler cycle will do Stage 2 once Stage 1 completes."""
    _insert_collection(engine, collection_id="col_first_run", summary=None)

    stage1, stage2 = reconciler._select_collections_needing_regen(engine, timedelta(minutes=10), batch_size=100)

    assert "col_first_run" in stage1
    assert "col_first_run" not in stage2
