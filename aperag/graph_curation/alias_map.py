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

"""Alias-map repository — Wave 7 §K.12.7 + §K.12.10b task #6.

Persists user-driven entity merge intent. The :class:`AliasMapRepository`
is the canonical write/read surface; downstream consumers use the two
methods :meth:`AliasMapRepository.resolve_canonical` (read,
single-name) / :meth:`AliasMapRepository.resolve_canonical_many` (read,
batch) and :meth:`AliasMapRepository.upsert_alias` (write, with cycle
reject and transitive flatten).

Design notes
------------

* Per-collection scoping is a row-level concern (column
  ``collection_id``). The repository takes ``collection_id`` on each
  call rather than at construction to keep the repo a stateless
  singleton (mirrors :class:`AsyncBaseRepository` convention).
* Cycle handling is the service-layer invariant from §K.12.10b:
  ``upsert_alias`` ALWAYS resolves the requested ``target`` through
  the existing chain first. If the resolved canonical equals the
  ``alias_name`` itself, we raise :class:`AliasCycleError` instead of
  writing a self-loop.
* Transitive flatten: when ``B → C`` is recorded, any prior
  ``A → B`` row is rewritten to ``A → C`` in the same transaction.
  Readers therefore always see at most one indirection (no chain
  walks at read time, even after multi-step merges).
"""

from __future__ import annotations

import logging
from typing import Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.repositories.base import AsyncBaseRepository
from aperag.domains.knowledge_graph.db.models import LineageEntityAlias
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)


class AliasCycleError(ValueError):
    """Raised when an alias upsert would create a self-loop —
    ``alias_name == resolve_canonical(target)``. The service layer
    aborts the merge and surfaces this to the caller so the user can
    pick a different target.
    """


class AliasMapRepository(AsyncBaseRepository):
    """Read/write surface for ``aperag_lineage_entity_alias``.

    Two write methods (:meth:`upsert_alias`, :meth:`purge_collection`)
    and two read methods (:meth:`resolve_canonical`,
    :meth:`list_aliases_pointing_at`). Everything else can be expressed
    as composition.
    """

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------

    async def resolve_canonical(self, *, collection_id: str, name: str) -> str:
        """Return the canonical name for ``name`` in ``collection_id``.

        ``name`` itself is returned when no alias row points at it
        (i.e. ``name`` is already canonical or has never been merged).
        Returns at most one indirection because :meth:`upsert_alias`
        flattens transitively at write time.

        Single-name reads are cheap; for batched (n > 1) callers prefer
        :meth:`resolve_canonical_many` which folds N lookups into one
        SQL roundtrip — see task #61 P2-S1 (Planetegg msg=db7fb085 +
        msg=1314ac59 batch alias resolution P2-HIGH).
        """
        if not name:
            return name

        async def _op(session: AsyncSession) -> str:
            row = await session.get(LineageEntityAlias, (collection_id, name))
            if row is None:
                return name
            return str(row.canonical_name)

        return await self._execute_query(_op)

    async def resolve_canonical_many(
        self,
        *,
        collection_id: str,
        names: Sequence[str],
    ) -> dict[str, str]:
        """Batch alias resolution — single SQL ``SELECT ... WHERE ... IN``
        roundtrip (task #61 P2-S1+S2).

        Returns a mapping from each input name to its canonical form.
        Names with no alias row map to themselves (mirrors
        :meth:`resolve_canonical` semantics). Empty / falsy names also
        map to themselves so callers don't have to filter input.

        Why this exists: pre-task-#61-P2 the only public API was the
        per-name :meth:`resolve_canonical`. Callers that needed to
        resolve N names did so via ``asyncio.gather`` of N parallel
        coroutines — each one acquired a separate ``AsyncSession`` /
        DB connection. On
        :meth:`LineageGraphStoreWithAliasRedirect.expand_neighbors_n_hops`
        N is the seed cap of the calling endpoint, which can be large:

        * ``GET /api/v2/collections/{id}/graphs?max_nodes=1000``
          → up to **2 × max_nodes = 2000** seeds (per Planetegg
          msg=db7fb085 + spec § 2.4 P2-S1 quantification).
        * ``GET /graphs/hybrid``: default 1000 / max 5000 seeds.

        2000 parallel ``resolve_canonical`` calls translate to 2000
        connection-pool checkouts — Singapore production observed PG
        connection saturation on the ``/graphs`` endpoint
        (Planetegg msg=4043adf4 SRE diagnostic).

        Implementation: in-place dedupe + single ``SELECT alias_name,
        canonical_name FROM aperag_lineage_entity_alias WHERE
        collection_id = ? AND alias_name IN (...)`` reads all matching
        rows in one shot. Names absent from the result set fall back
        to themselves. Total connections checked out: **1**.

        Order of the input is preserved on the dict's iteration order
        (Python ``dict`` preserves insertion order since 3.7).
        """
        # Map empty / falsy names to themselves up-front, then dedupe
        # the rest. ``dict`` insertion order preserves caller order.
        out: dict[str, str] = {}
        unique_names: list[str] = []
        seen: set[str] = set()
        for n in names:
            if not n:
                out[n] = n
                continue
            if n in seen:
                continue
            seen.add(n)
            unique_names.append(n)
        if not unique_names:
            return out

        async def _op(session: AsyncSession) -> dict[str, str]:
            stmt = select(
                LineageEntityAlias.alias_name,
                LineageEntityAlias.canonical_name,
            ).where(
                LineageEntityAlias.collection_id == collection_id,
                LineageEntityAlias.alias_name.in_(unique_names),
            )
            result = await session.execute(stmt)
            return {str(row[0]): str(row[1]) for row in result.all()}

        resolved = await self._execute_query(_op)

        # Restore caller order: every input ``name`` (in the order it
        # was passed) gets a key in the output. Names that didn't show
        # up in the SQL result map to themselves (no alias row → name
        # is already canonical).
        for n in names:
            if n in out:  # already added (empty / falsy short-circuit)
                continue
            out[n] = resolved.get(n, n)
        return out

    async def list_aliases_pointing_at(self, *, collection_id: str, canonical_name: str) -> list[str]:
        """Return every ``alias_name`` whose row points at
        ``canonical_name`` in ``collection_id``.

        Used by :meth:`upsert_alias` to perform transitive flatten and
        by tests / admin tooling. Order is alphabetical for
        determinism."""

        async def _op(session: AsyncSession) -> list[str]:
            stmt = (
                select(LineageEntityAlias.alias_name)
                .where(
                    LineageEntityAlias.collection_id == collection_id,
                    LineageEntityAlias.canonical_name == canonical_name,
                )
                .order_by(LineageEntityAlias.alias_name)
            )
            result = await session.execute(stmt)
            return [r[0] for r in result.all()]

        return await self._execute_query(_op)

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------

    async def upsert_alias(
        self,
        *,
        collection_id: str,
        alias_name: str,
        target: str,
        merged_by: str | None = None,
    ) -> str:
        """Record that ``alias_name`` should resolve to ``target`` in
        ``collection_id``.

        Returns the resolved canonical name actually written (which may
        differ from ``target`` if ``target`` itself was already an
        alias — we flatten through to the terminal canonical so readers
        never traverse a chain).

        Cycle reject: if the resolved canonical equals ``alias_name``,
        raise :class:`AliasCycleError` instead of writing a self-loop.

        Transitive flatten: any prior alias row whose ``canonical_name``
        equals the *old* row at ``(collection_id, alias_name)`` gets
        rewritten to point at the new canonical. The flatten + the
        upsert run inside one transaction so a partial flatten can
        never be observed.
        """

        async def _op(session: AsyncSession) -> str:
            # Resolve target through any existing alias row (single
            # indirection, since flatten keeps the table 1-deep).
            target_row = await session.get(LineageEntityAlias, (collection_id, target))
            canonical = str(target_row.canonical_name) if target_row is not None else target

            if canonical == alias_name:
                raise AliasCycleError(
                    f"alias upsert would create a cycle: "
                    f"alias={alias_name!r} → target={target!r} resolves to {canonical!r}"
                )

            now = utc_now()
            existing = await session.get(LineageEntityAlias, (collection_id, alias_name))
            if existing is not None:
                existing.canonical_name = canonical
                existing.merged_by = merged_by
                existing.gmt_updated = now
            else:
                session.add(
                    LineageEntityAlias(
                        collection_id=collection_id,
                        alias_name=alias_name,
                        canonical_name=canonical,
                        merged_by=merged_by,
                        gmt_created=now,
                        gmt_updated=now,
                    )
                )

            # Transitive flatten: any row pointing at ``alias_name`` now
            # needs to point at the new canonical instead. (The alias
            # name was canonical from those rows' perspective; now it is
            # itself an alias of ``canonical``.)
            await session.execute(
                update(LineageEntityAlias)
                .where(
                    LineageEntityAlias.collection_id == collection_id,
                    LineageEntityAlias.canonical_name == alias_name,
                )
                .values(
                    canonical_name=canonical,
                    gmt_updated=now,
                )
            )
            return canonical

        return await self.execute_with_transaction(_op)

    async def purge_collection(self, collection_id: str) -> int:
        """Delete all alias rows for ``collection_id``. Returns the
        number of rows deleted. Used by collection-purge / test
        teardown."""

        async def _op(session: AsyncSession) -> int:
            result = await session.execute(
                delete(LineageEntityAlias).where(LineageEntityAlias.collection_id == collection_id)
            )
            return int(result.rowcount or 0)

        return await self.execute_with_transaction(_op)


__all__ = [
    "AliasCycleError",
    "AliasMapRepository",
]
