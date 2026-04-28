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

"""Unit tests for ``aperag.indexing.alias_redirect_store`` — Wave 7
§K.12.4 invariant #3 (write-side) + Wave 8 W8-3 task #14 (read-side).

Pinned cases:

* **Write-side redirect** (Wave 7): indexer write to an alias name
  lands on the canonical entity name in the underlying lineage store
  (``test_indexer_upsert_after_merge_redirects_to_canonical``).
  Relation writes redirect *both* endpoints when either has been
  merged.
* **Read-side redirect** (Wave 8 task #14): ``get_entity`` /
  ``get_relation`` / ``expand_neighbors_n_hops`` all rewrite anchor
  names through the alias map before calling the inner store, so MCP
  / REST / curation surfaces see canonical data even when the caller
  passed the alias.
* **Decorator passthrough** (huangheng CR lock,
  ``test_decorator_passthrough_for_non_redirected_methods``): every
  ``LineageGraphStore`` Protocol method that does NOT redirect must
  forward to ``_inner`` byte-for-byte. Pinned so a future Protocol
  method addition cannot slip past unnoticed. With an empty alias
  map, the redirected methods also passthrough byte-for-byte (since
  ``resolve_canonical`` returns the input unchanged) — this test
  verifies BOTH groups in the no-alias steady state.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aperag.indexing.alias_redirect_store import LineageGraphStoreWithAliasRedirect
from aperag.indexing.graph import (
    EntityRecord,
    LineageMember,
    RelationRecord,
)


class _FakeAliasRepo:
    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._mapping = mapping or {}

    async def resolve_canonical(self, *, collection_id: str, name: str) -> str:
        return self._mapping.get(name, name)


def _record(name: str = "Apple") -> EntityRecord:
    return EntityRecord(
        name=name,
        entity_type="organization",
        description="desc",
        source_chunk_ids=("c0",),
    )


def _member() -> LineageMember:
    return LineageMember(
        document_id="doc1",
        parse_version="v1",
        tenant_scope_key="tenant-1",
        chunk_ids=("c0",),
    )


# ---------------------------------------------------------------------
# Entity write redirect — the core invariant
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_indexer_upsert_after_merge_redirects_to_canonical():
    """Wave 7 §K.12 invariant #3: indexer writes ``record.name="A"``;
    user has merged ``A → C``; the underlying store sees the write on
    ``C``, not ``A``."""
    inner = AsyncMock()
    inner.upsert_entity_with_lineage = AsyncMock(return_value=None)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"A": "C"}),
        collection_id="col-1",
    )
    await decorator.upsert_entity_with_lineage(
        record=_record("A"),
        lineage=_member(),
        compacted_description="cd",
    )
    # Inner saw the write under the canonical name.
    inner.upsert_entity_with_lineage.assert_awaited_once()
    call_kwargs = inner.upsert_entity_with_lineage.call_args.kwargs
    assert call_kwargs["record"].name == "C"
    # Other fields untouched.
    assert call_kwargs["record"].entity_type == "organization"
    assert call_kwargs["compacted_description"] == "cd"


@pytest.mark.asyncio
async def test_indexer_upsert_no_alias_pass_through():
    inner = AsyncMock()
    inner.upsert_entity_with_lineage = AsyncMock(return_value=None)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo(),  # empty
        collection_id="col-1",
    )
    await decorator.upsert_entity_with_lineage(
        record=_record("Untouched"),
        lineage=_member(),
    )
    call_kwargs = inner.upsert_entity_with_lineage.call_args.kwargs
    assert call_kwargs["record"].name == "Untouched"


# ---------------------------------------------------------------------
# Relation write redirect — both endpoints
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relation_write_redirects_both_endpoints():
    inner = AsyncMock()
    inner.upsert_relation_with_lineage = AsyncMock(return_value=None)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"A": "C", "B": "D"}),
        collection_id="col-1",
    )
    await decorator.upsert_relation_with_lineage(
        record=RelationRecord(
            source="A",
            target="B",
            relation_type="founded",
            description="desc",
            source_chunk_ids=("c0",),
        ),
        lineage=_member(),
    )
    call_kwargs = inner.upsert_relation_with_lineage.call_args.kwargs
    assert call_kwargs["record"].source == "C"
    assert call_kwargs["record"].target == "D"
    assert call_kwargs["record"].relation_type == "founded"


@pytest.mark.asyncio
async def test_relation_write_redirects_only_one_endpoint():
    inner = AsyncMock()
    inner.upsert_relation_with_lineage = AsyncMock(return_value=None)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"A": "C"}),  # only A is aliased
        collection_id="col-1",
    )
    await decorator.upsert_relation_with_lineage(
        record=RelationRecord(
            source="A",
            target="B",
            relation_type="founded",
            description="desc",
            source_chunk_ids=("c0",),
        ),
        lineage=_member(),
    )
    call_kwargs = inner.upsert_relation_with_lineage.call_args.kwargs
    assert call_kwargs["record"].source == "C"
    assert call_kwargs["record"].target == "B"


# ---------------------------------------------------------------------
# Bulk write redirect (Wave 8 W8-2 task #13)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bulk_upsert_redirects_each_part_record_name():
    """Wave 8 W8-2: bulk path mirrors single-upsert redirect — every
    ``(record, lineage)`` tuple's ``record.name`` resolves through the
    alias map before forwarding to the inner store."""
    inner = AsyncMock()
    inner.bulk_upsert_entity_with_lineage_parts = AsyncMock(return_value=None)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"A": "C"}),
        collection_id="col-1",
    )
    parts = [
        (_record("A"), _member()),
        (_record("A"), _member()),
    ]
    await decorator.bulk_upsert_entity_with_lineage_parts(parts=parts)
    inner.bulk_upsert_entity_with_lineage_parts.assert_awaited_once()
    forwarded_parts = inner.bulk_upsert_entity_with_lineage_parts.call_args.kwargs["parts"]
    assert len(forwarded_parts) == 2
    for record, _lineage in forwarded_parts:
        assert record.name == "C"


@pytest.mark.asyncio
async def test_bulk_upsert_no_alias_pass_through():
    """Empty alias map → bulk forwards records unchanged."""
    inner = AsyncMock()
    inner.bulk_upsert_entity_with_lineage_parts = AsyncMock(return_value=None)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo(),
        collection_id="col-1",
    )
    parts = [(_record("Untouched"), _member())]
    await decorator.bulk_upsert_entity_with_lineage_parts(parts=parts)
    forwarded_parts = inner.bulk_upsert_entity_with_lineage_parts.call_args.kwargs["parts"]
    assert forwarded_parts[0][0].name == "Untouched"


@pytest.mark.asyncio
async def test_bulk_upsert_empty_parts_short_circuits():
    """Empty input is a no-op — the inner store is never called."""
    inner = AsyncMock()
    inner.bulk_upsert_entity_with_lineage_parts = AsyncMock(return_value=None)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"A": "C"}),
        collection_id="col-1",
    )
    await decorator.bulk_upsert_entity_with_lineage_parts(parts=[])
    inner.bulk_upsert_entity_with_lineage_parts.assert_not_awaited()


# ---------------------------------------------------------------------
# Read-side redirect (Wave 8 W8-3 task #14)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_entity_after_alias_returns_canonical_data():
    """Wave 8 W8-3: when user has merged ``Alicia → Alice``,
    ``get_entity('Alicia')`` returns the canonical Alice entity. This
    is the W8-3 trigger pin flipped: task #11 step 8 used to assert
    ``result is None`` (Wave 7 decorator only intercepted writes); now
    it asserts ``result.name == 'Alice'``."""
    inner = AsyncMock()
    canonical_entity = object()  # any sentinel — passthrough check
    inner.get_entity = AsyncMock(return_value=canonical_entity)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"Alicia": "Alice"}),
        collection_id="col-1",
    )
    result = await decorator.get_entity("Alicia")
    assert result is canonical_entity
    inner.get_entity.assert_awaited_once_with("Alice")


@pytest.mark.asyncio
async def test_get_entity_no_alias_passes_through_unchanged():
    inner = AsyncMock()
    inner.get_entity = AsyncMock(return_value=None)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo(),  # empty
        collection_id="col-1",
    )
    await decorator.get_entity("Untouched")
    inner.get_entity.assert_awaited_once_with("Untouched")


@pytest.mark.asyncio
async def test_get_relation_redirects_both_endpoints():
    """Mirror the write-side
    ``test_relation_write_redirects_both_endpoints`` — both endpoints
    of a relation may have been merged independently; resolve
    symmetrically before calling the inner store."""
    inner = AsyncMock()
    inner.get_relation = AsyncMock(return_value=None)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"Alicia": "Alice", "Bobby": "Bob"}),
        collection_id="col-1",
    )
    await decorator.get_relation("Alicia", "Bobby", "knows")
    inner.get_relation.assert_awaited_once_with("Alice", "Bob", "knows")


@pytest.mark.asyncio
async def test_get_relation_redirects_only_one_endpoint():
    inner = AsyncMock()
    inner.get_relation = AsyncMock(return_value=None)
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"Alicia": "Alice"}),  # only source aliased
        collection_id="col-1",
    )
    await decorator.get_relation("Alicia", "Bob", "knows")
    inner.get_relation.assert_awaited_once_with("Alice", "Bob", "knows")


@pytest.mark.asyncio
async def test_expand_neighbors_redirects_alias_anchor():
    """Anchor names in ``expand_neighbors_n_hops`` resolve through the
    alias map before traversal so a caller seeding ``["Alicia"]``
    walks the canonical Alice neighbourhood."""
    inner = AsyncMock()
    inner.expand_neighbors_n_hops = AsyncMock(return_value=([], []))
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"Alicia": "Alice", "Bobby": "Bob"}),
        collection_id="col-1",
    )
    await decorator.expand_neighbors_n_hops(entity_names=["Alicia", "Bobby", "Charlie"], hops=2)
    inner.expand_neighbors_n_hops.assert_awaited_once_with(entity_names=["Alice", "Bob", "Charlie"], hops=2)


@pytest.mark.asyncio
async def test_expand_neighbors_dedupes_alias_and_canonical_in_anchors():
    """If the caller passes both the alias and the canonical
    (``["Alicia", "Alice"]``), redirect collapses to one anchor so
    the traversal isn't double-walked."""
    inner = AsyncMock()
    inner.expand_neighbors_n_hops = AsyncMock(return_value=([], []))
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"Alicia": "Alice"}),
        collection_id="col-1",
    )
    await decorator.expand_neighbors_n_hops(entity_names=["Alicia", "Alice"], hops=1)
    inner.expand_neighbors_n_hops.assert_awaited_once_with(entity_names=["Alice"], hops=1)


@pytest.mark.asyncio
async def test_expand_neighbors_empty_seeds_passes_through():
    inner = AsyncMock()
    inner.expand_neighbors_n_hops = AsyncMock(return_value=([], []))
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo({"x": "y"}),
        collection_id="col-1",
    )
    await decorator.expand_neighbors_n_hops(entity_names=[], hops=1)
    inner.expand_neighbors_n_hops.assert_awaited_once_with(entity_names=[], hops=1)


# ---------------------------------------------------------------------
# Decorator passthrough invariant (huangheng CR lock)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decorator_passthrough_for_non_redirected_methods():
    """Every ``LineageGraphStore`` Protocol method that does NOT
    redirect must forward to ``_inner`` byte-for-byte. Spelled out one
    method at a time so a future Protocol addition can't slip past
    without an explicit decorator update.

    With an empty alias map, the redirected methods (``get_entity`` /
    ``get_relation`` / ``expand_neighbors_n_hops``) ALSO passthrough
    byte-for-byte (resolve_canonical returns input unchanged); we
    re-assert that here so the no-alias steady state is pinned across
    BOTH groups."""
    inner = AsyncMock()
    decorator = LineageGraphStoreWithAliasRedirect(
        inner=inner,
        alias_repo=_FakeAliasRepo(),
        collection_id="col-1",
    )

    # find_entity_ids_with_lineage
    inner.find_entity_ids_with_lineage = AsyncMock(return_value=["a", "b"])
    out = await decorator.find_entity_ids_with_lineage(document_id="doc1")
    assert out == ["a", "b"]
    inner.find_entity_ids_with_lineage.assert_awaited_once_with(document_id="doc1")

    # find_relation_keys_with_lineage
    inner.find_relation_keys_with_lineage = AsyncMock(return_value=[("a", "b", "x")])
    out = await decorator.find_relation_keys_with_lineage(document_id="doc1")
    assert out == [("a", "b", "x")]
    inner.find_relation_keys_with_lineage.assert_awaited_once_with(document_id="doc1")

    # remove_entity_lineage_member
    inner.remove_entity_lineage_member = AsyncMock(return_value=None)
    await decorator.remove_entity_lineage_member(entity_name="x", document_id="d")
    inner.remove_entity_lineage_member.assert_awaited_once_with(entity_name="x", document_id="d")

    # remove_relation_lineage_member
    inner.remove_relation_lineage_member = AsyncMock(return_value=None)
    await decorator.remove_relation_lineage_member(source="a", target="b", type="t", document_id="d")
    inner.remove_relation_lineage_member.assert_awaited_once_with(source="a", target="b", type="t", document_id="d")

    # gc_entity_if_orphan / gc_relation_if_orphan
    inner.gc_entity_if_orphan = AsyncMock(return_value=True)
    assert await decorator.gc_entity_if_orphan("x") is True
    inner.gc_entity_if_orphan.assert_awaited_once_with("x")

    inner.gc_relation_if_orphan = AsyncMock(return_value=False)
    assert await decorator.gc_relation_if_orphan("a", "b", "t") is False
    inner.gc_relation_if_orphan.assert_awaited_once_with("a", "b", "t")

    # delete_entity / delete_relation
    inner.delete_entity = AsyncMock(return_value=True)
    assert await decorator.delete_entity("x") is True
    inner.delete_entity.assert_awaited_once_with("x")

    inner.delete_relation = AsyncMock(return_value=True)
    assert await decorator.delete_relation("a", "b", "t") is True
    inner.delete_relation.assert_awaited_once_with("a", "b", "t")

    # get_entity / get_relation
    inner.get_entity = AsyncMock(return_value=None)
    assert await decorator.get_entity("x") is None
    inner.get_entity.assert_awaited_once_with("x")

    inner.get_relation = AsyncMock(return_value=None)
    assert await decorator.get_relation("a", "b", "t") is None
    inner.get_relation.assert_awaited_once_with("a", "b", "t")

    # query_entities_by_keyword
    inner.query_entities_by_keyword = AsyncMock(return_value=[])
    assert await decorator.query_entities_by_keyword(query="q", top_k=5) == []
    inner.query_entities_by_keyword.assert_awaited_once_with(query="q", top_k=5)

    # expand_neighbors_n_hops
    inner.expand_neighbors_n_hops = AsyncMock(return_value=([], []))
    assert await decorator.expand_neighbors_n_hops(entity_names=["x"], hops=2) == ([], [])
    inner.expand_neighbors_n_hops.assert_awaited_once_with(entity_names=["x"], hops=2)

    # list_entity_labels
    inner.list_entity_labels = AsyncMock(return_value=["a", "b"])
    assert await decorator.list_entity_labels() == ["a", "b"]
    inner.list_entity_labels.assert_awaited_once_with()
