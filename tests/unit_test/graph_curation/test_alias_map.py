# Copyright 2026 ApeCloud, Inc.
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

"""Unit tests for ``aperag.graph_curation.alias_map`` — Wave 7 task #6.

The repository talks to PostgreSQL in production; these tests exercise
the cycle-reject + transitive-flatten algorithm against an in-memory
SQLite engine so the cases that lock §K.12.10b can run without a
running Postgres. Cross-backend behaviour is covered separately by the
repository integration tests.

Pinned cases (per spec §K.12.10b + huangheng CR plan msg=22816e0d):

* Insert + read: A → B writes a row, ``resolve_canonical("A") == "B"``.
* Transitive flatten: B → C after A → B rewrites the existing
  (A, B) row to (A, C) — readers never traverse a chain.
* Cycle reject: C → A when ``resolve_canonical("A") == "C"`` raises
  :class:`AliasCycleError` instead of writing a self-loop.
* Persists across canonical GC (orphan canonical row): rows survive
  even if the canonical entity no longer exists in the lineage table.
* Per-collection isolation: rows in collection X never leak into
  collection Y.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from aperag.domains.knowledge_graph.db.models import LineageEntityAlias  # noqa: F401  (registers metadata)
from aperag.graph_curation.alias_map import AliasCycleError, AliasMapRepository


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: LineageEntityAlias.__table__.create(sync_conn))
    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


# ---------------------------------------------------------------------
# read path
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_canonical_returns_input_when_no_alias_row(session):
    repo = AliasMapRepository(session=session)
    assert await repo.resolve_canonical(collection_id="c1", name="Apple") == "Apple"


@pytest.mark.asyncio
async def test_resolve_canonical_after_simple_upsert(session):
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="Apple", target="Apple Inc.")
    assert await repo.resolve_canonical(collection_id="c1", name="Apple") == "Apple Inc."
    assert await repo.resolve_canonical(collection_id="c1", name="Apple Inc.") == "Apple Inc."


# ---------------------------------------------------------------------
# transitive flatten
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transitive_flatten_rewrites_existing_alias_rows(session):
    """§K.12.10b: ``A → B`` then ``B → C`` rewrites the (A, B) row to
    (A, C) so readers never need to walk a chain."""
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="A", target="B")
    assert await repo.resolve_canonical(collection_id="c1", name="A") == "B"

    await repo.upsert_alias(collection_id="c1", alias_name="B", target="C")
    # A still resolves to the (now-flattened) terminal canonical:
    assert await repo.resolve_canonical(collection_id="c1", name="A") == "C"
    assert await repo.resolve_canonical(collection_id="c1", name="B") == "C"


@pytest.mark.asyncio
async def test_target_is_itself_an_alias_resolves_through(session):
    """If the caller supplies ``target=B`` while ``B → C`` already
    exists, the new row points at ``C`` (1-hop guarantee)."""
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="B", target="C")

    canonical = await repo.upsert_alias(collection_id="c1", alias_name="A", target="B")
    assert canonical == "C"
    assert await repo.resolve_canonical(collection_id="c1", name="A") == "C"


# ---------------------------------------------------------------------
# cycle reject
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cycle_reject_self_loop(session):
    """``A → A`` (or anything that resolves to A when alias is A)
    raises rather than writing a self-loop."""
    repo = AliasMapRepository(session=session)
    with pytest.raises(AliasCycleError):
        await repo.upsert_alias(collection_id="c1", alias_name="A", target="A")
    assert await repo.resolve_canonical(collection_id="c1", name="A") == "A"


@pytest.mark.asyncio
async def test_cycle_reject_through_existing_chain(session):
    """``A → B`` then ``B → C`` then ``C → A`` is rejected — the third
    upsert resolves target ``A`` to the terminal canonical ``C`` (via
    transitive flatten); when that equals the new alias name (``C``)
    we raise instead of writing a self-loop."""
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="A", target="B")
    await repo.upsert_alias(collection_id="c1", alias_name="B", target="C")

    with pytest.raises(AliasCycleError):
        await repo.upsert_alias(collection_id="c1", alias_name="C", target="A")

    # No rows mutated by the failed upsert.
    assert await repo.resolve_canonical(collection_id="c1", name="C") == "C"
    assert await repo.resolve_canonical(collection_id="c1", name="A") == "C"
    assert await repo.resolve_canonical(collection_id="c1", name="B") == "C"


# ---------------------------------------------------------------------
# orphan persistence (§K.12.7)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_persists_after_canonical_gc(session):
    """Per spec §K.12.7 decision X: if the canonical entity is later
    deleted from the lineage table, the alias rows MUST stay so a
    future indexer write to the alias name still resolves correctly."""
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="A", target="C")
    # No coupling with the lineage table — alias persistence is a pure
    # property of the alias-map repo. Resolving still works whether or
    # not C currently has a lineage row.
    assert await repo.resolve_canonical(collection_id="c1", name="A") == "C"


# ---------------------------------------------------------------------
# per-collection isolation
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aliases_isolated_per_collection(session):
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="A", target="B")
    await repo.upsert_alias(collection_id="c2", alias_name="A", target="Z")

    assert await repo.resolve_canonical(collection_id="c1", name="A") == "B"
    assert await repo.resolve_canonical(collection_id="c2", name="A") == "Z"


@pytest.mark.asyncio
async def test_purge_collection_only_drops_its_own_rows(session):
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="A", target="B")
    await repo.upsert_alias(collection_id="c2", alias_name="A", target="Z")

    deleted = await repo.purge_collection("c1")
    assert deleted == 1

    assert await repo.resolve_canonical(collection_id="c1", name="A") == "A"
    assert await repo.resolve_canonical(collection_id="c2", name="A") == "Z"


@pytest.mark.asyncio
async def test_list_aliases_pointing_at_returns_alphabetical(session):
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="Zeta", target="C")
    await repo.upsert_alias(collection_id="c1", alias_name="Alpha", target="C")
    await repo.upsert_alias(collection_id="c1", alias_name="Mu", target="C")
    await repo.upsert_alias(collection_id="c1", alias_name="Other", target="D")

    pointing_at_C = await repo.list_aliases_pointing_at(collection_id="c1", canonical_name="C")
    assert pointing_at_C == ["Alpha", "Mu", "Zeta"]
