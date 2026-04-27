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

"""Lineage graph store decorator that applies user-driven alias redirect.

Wave 7 §K.12.4 invariant #3: when the indexer writes
``upsert_entity_with_lineage(record=EntityRecord(name="A"))`` and the
user previously merged ``A → C``, the write should land on ``C``
transparently — the indexer hot path is not aware of curation merges.

Implementation strategy (architect ratify msg=cf860ae4 + huangheng
endorse msg=22816e0d / msg=93d9add1, **Option (b)**): write a thin
decorator class that wraps any concrete
:class:`aperag.indexing.graph.LineageGraphStore` plus an
:class:`aperag.graph_curation.alias_map.AliasMapRepository`, intercepts
the write methods to rewrite entity names through the alias map, and
forwards every other Protocol method unchanged. This keeps the three
backend implementations (Postgres / Neo4j / Nebula) untouched —
critical for landing task #6 in one PR without rippling into Bryce's
storage territory.

Decorator passthrough invariant (huangheng CR lock,
``test_decorator_passthrough_for_non_upsert_methods``): every method
declared on :class:`LineageGraphStore` that is NOT an ``upsert_*``
write must forward to ``_inner`` byte-for-byte — no silent behaviour
change. Tests pin this so a future Protocol method addition can't slip
past without an explicit decorator update.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from aperag.indexing.graph import (
    EntityRecord,
    EntityWithLineage,
    LineageMember,
    RelationRecord,
    RelationWithLineage,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aperag.graph_curation.alias_map import AliasMapRepository
    from aperag.indexing.graph import LineageGraphStore

logger = logging.getLogger(__name__)


class LineageGraphStoreWithAliasRedirect:
    """Wrap a :class:`LineageGraphStore` so writes go through the alias
    map.

    Constructor takes the inner store, the alias-map repository, and
    the ``collection_id`` the inner store is bound to. The decorator is
    per-collection just like the inner store — both share the same
    binding.
    """

    def __init__(
        self,
        *,
        inner: "LineageGraphStore",
        alias_repo: "AliasMapRepository",
        collection_id: str,
    ) -> None:
        self._inner = inner
        self._alias_repo = alias_repo
        self._collection_id = collection_id

    # ------------------------------------------------------------------
    # Intercepted write paths — apply alias redirect
    # ------------------------------------------------------------------

    async def upsert_entity_with_lineage(
        self,
        *,
        record: EntityRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        canonical = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=record.name)
        redirected = record if canonical == record.name else replace(record, name=canonical)
        if canonical != record.name:
            logger.debug(
                "alias_redirect: entity write %r → %r (collection=%s)",
                record.name,
                canonical,
                self._collection_id,
            )
        await self._inner.upsert_entity_with_lineage(
            record=redirected,
            lineage=lineage,
            compacted_description=compacted_description,
        )

    async def upsert_relation_with_lineage(
        self,
        *,
        record: RelationRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        # Both endpoints of a relation may have been merged; resolve
        # both. ``relation_type`` is unaffected by entity merges.
        new_source = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=record.source)
        new_target = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=record.target)
        if new_source != record.source or new_target != record.target:
            logger.debug(
                "alias_redirect: relation write (%r→%r) → (%r→%r) (collection=%s)",
                record.source,
                record.target,
                new_source,
                new_target,
                self._collection_id,
            )
            redirected = replace(record, source=new_source, target=new_target)
        else:
            redirected = record
        await self._inner.upsert_relation_with_lineage(
            record=redirected,
            lineage=lineage,
            compacted_description=compacted_description,
        )

    # ------------------------------------------------------------------
    # Passthrough — forward every non-write Protocol method unchanged.
    # Pinned by ``test_decorator_passthrough_for_non_upsert_methods``.
    # ------------------------------------------------------------------

    async def find_entity_ids_with_lineage(self, *, document_id: str) -> list[str]:
        return await self._inner.find_entity_ids_with_lineage(document_id=document_id)

    async def find_relation_keys_with_lineage(self, *, document_id: str) -> list[tuple[str, str, str]]:
        return await self._inner.find_relation_keys_with_lineage(document_id=document_id)

    async def remove_entity_lineage_member(self, *, entity_name: str, document_id: str) -> None:
        await self._inner.remove_entity_lineage_member(entity_name=entity_name, document_id=document_id)

    async def remove_relation_lineage_member(self, *, source: str, target: str, type: str, document_id: str) -> None:
        await self._inner.remove_relation_lineage_member(
            source=source, target=target, type=type, document_id=document_id
        )

    async def gc_entity_if_orphan(self, entity_name: str) -> bool:
        return await self._inner.gc_entity_if_orphan(entity_name)

    async def gc_relation_if_orphan(self, source: str, target: str, type: str) -> bool:
        return await self._inner.gc_relation_if_orphan(source, target, type)

    async def delete_entity(self, entity_name: str) -> bool:
        return await self._inner.delete_entity(entity_name)

    async def delete_relation(self, source: str, target: str, type: str) -> bool:
        return await self._inner.delete_relation(source, target, type)

    async def get_entity(self, entity_name: str) -> EntityWithLineage | None:
        return await self._inner.get_entity(entity_name)

    async def get_relation(self, source: str, target: str, type: str) -> RelationWithLineage | None:
        return await self._inner.get_relation(source, target, type)

    async def query_entities_by_keyword(self, *, query: str, top_k: int) -> list[EntityWithLineage]:
        return await self._inner.query_entities_by_keyword(query=query, top_k=top_k)

    async def expand_neighbors_n_hops(
        self, *, entity_names: list[str], hops: int = 1
    ) -> tuple[list[EntityWithLineage], list[RelationWithLineage]]:
        return await self._inner.expand_neighbors_n_hops(entity_names=entity_names, hops=hops)

    async def list_entity_labels(self) -> list[str]:
        return await self._inner.list_entity_labels()


__all__ = ["LineageGraphStoreWithAliasRedirect"]
