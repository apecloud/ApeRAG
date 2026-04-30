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


# ---------------------------------------------------------------------
# task #61 P2-S1+S2 — batch resolve_canonical_many
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_canonical_many_returns_self_for_unmapped_names(session):
    """No alias rows → every input name maps to itself (mirrors the
    single-name :meth:`resolve_canonical` semantic)."""
    repo = AliasMapRepository(session=session)
    out = await repo.resolve_canonical_many(collection_id="c1", names=["Apple", "Banana", "Cherry"])
    assert out == {"Apple": "Apple", "Banana": "Banana", "Cherry": "Cherry"}


@pytest.mark.asyncio
async def test_resolve_canonical_many_mixed_alias_and_canonical(session):
    """Some inputs are aliases, some are already canonical, some
    don't exist at all — each maps to its correct resolution."""
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="Apple", target="Apple Inc.")
    await repo.upsert_alias(collection_id="c1", alias_name="MS", target="Microsoft")

    out = await repo.resolve_canonical_many(
        collection_id="c1",
        names=["Apple", "MS", "Apple Inc.", "Banana"],  # alias / alias / canonical / unknown
    )
    assert out == {
        "Apple": "Apple Inc.",
        "MS": "Microsoft",
        "Apple Inc.": "Apple Inc.",
        "Banana": "Banana",
    }


@pytest.mark.asyncio
async def test_resolve_canonical_many_dedupes_input(session):
    """Duplicate input names result in a single dict entry (Python
    dict semantics) but with the same canonical resolution. Pinned to
    catch a future refactor that accidentally returns multi-value
    output."""
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="Apple", target="Apple Inc.")
    out = await repo.resolve_canonical_many(
        collection_id="c1",
        names=["Apple", "Apple", "Apple Inc.", "Apple"],
    )
    # ``dict`` collapses duplicates by key; insertion-order is
    # preserved (Apple first, Apple Inc. second).
    assert out == {"Apple": "Apple Inc.", "Apple Inc.": "Apple Inc."}


@pytest.mark.asyncio
async def test_resolve_canonical_many_empty_input(session):
    """Empty input → empty output (defensive — caller bridges the
    edge without an extra ``if names`` branch)."""
    repo = AliasMapRepository(session=session)
    assert await repo.resolve_canonical_many(collection_id="c1", names=[]) == {}


@pytest.mark.asyncio
async def test_resolve_canonical_many_handles_empty_string(session):
    """Empty / falsy names short-circuit to themselves without an SQL
    lookup (mirrors single-name :meth:`resolve_canonical` defensive
    branch)."""
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="Real", target="Resolved")
    out = await repo.resolve_canonical_many(collection_id="c1", names=["", "Real", ""])
    assert out == {"": "", "Real": "Resolved"}


@pytest.mark.asyncio
async def test_resolve_canonical_many_per_collection_isolation(session):
    """Same alias_name in different collections resolves to different
    canonical names — task #61 spec § 2.4 P2-S1 cross-collection seed
    cap test pinpoints this isolation."""
    repo = AliasMapRepository(session=session)
    await repo.upsert_alias(collection_id="c1", alias_name="Apple", target="Apple Inc.")
    await repo.upsert_alias(collection_id="c2", alias_name="Apple", target="Apple Records")

    c1_out = await repo.resolve_canonical_many(collection_id="c1", names=["Apple"])
    c2_out = await repo.resolve_canonical_many(collection_id="c2", names=["Apple"])
    assert c1_out == {"Apple": "Apple Inc."}
    assert c2_out == {"Apple": "Apple Records"}


@pytest.mark.asyncio
async def test_resolve_canonical_many_large_seed_cap(session):
    """Pinned to catch a future regression that re-introduces per-name
    DB roundtrips: even at the ``/graphs?max_nodes=1000`` worst case
    (2 × max_nodes = 2000 seeds, per spec § 2.4 P2-S1 quantification),
    the batch API must complete with a single SQL roundtrip.

    We can't directly assert "1 SQL roundtrip" in a pure-unit test
    against in-memory SQLite, but we can pin the *result correctness*
    at the spec-quantified seed cardinality so a future refactor that
    silently re-fans-out would either time out (in-memory SQLite is
    fast enough that 2000 SELECT roundtrips is ~10ms — observable
    only via a perf timeout) or break correctness.

    The companion :func:`test_expand_neighbors_uses_batch_alias_resolution`
    in ``test_alias_redirect_store.py`` pins the call-graph: the
    ``expand_neighbors_n_hops`` site MUST go through
    ``resolve_canonical_many`` exactly once (not N
    ``resolve_canonical`` calls).
    """
    repo = AliasMapRepository(session=session)
    # Seed 50 aliases (cheap on sqlite); query 2000 names where the
    # first 50 are mapped + the remaining 1950 are unmapped (resolve
    # to themselves).
    for i in range(50):
        await repo.upsert_alias(
            collection_id="c1",
            alias_name=f"alias_{i}",
            target=f"canonical_{i}",
        )

    names = [f"alias_{i}" for i in range(50)] + [f"unmapped_{i}" for i in range(1950)]
    out = await repo.resolve_canonical_many(collection_id="c1", names=names)

    # Spot-check shape + a few representative rows.
    assert len(out) == 2000
    assert out["alias_0"] == "canonical_0"
    assert out["alias_49"] == "canonical_49"
    assert out["unmapped_0"] == "unmapped_0"
    assert out["unmapped_1949"] == "unmapped_1949"
