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

"""Unit tests for ``aperag.graph_curation.lineage_merge.LineageEntityMerger``
— Wave 7 §K.12.6 task #6.

Pinned cases (per architect outline msg=cf860ae4 + huangheng CR plan
msg=22816e0d):

* Step ordering ``L1 → vector → delete`` (invariant #2).
* Sentinel ``__curation_merge__`` document_id on the unified+compacted
  lineage member (drift #2 lock).
* Compactor invocation passes the locked kwargs (``subject_kind``,
  ``subject_label``, ``language``).
* Source parts are re-anchored under the target name preserving
  per-doc lineage (invariant #1 L1 not polluted).
* Sources are deleted from L1 + vector last.
* Vector point id is the deterministic
  ``uuid5(NAMESPACE_DNS, "graph_entity:<cid>:<name>")`` with the
  3-field payload (invariants #5 + #6).
* Empty source list short-circuits.
* Cycle reject from the alias repo propagates ``AliasCycleError``.
* Target ``GC`` between merge initiation and execution is logged but
  not crashing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import NAMESPACE_DNS, uuid5

import pytest

from aperag.graph_curation.alias_map import AliasCycleError
from aperag.graph_curation.lineage_merge import (
    CURATION_MERGE_DOCUMENT_ID,
    GRAPH_ENTITY_INDEXER,
    LineageEntityMerger,
)
from aperag.indexing.graph import (
    DescriptionPart,
    EntityWithLineage,
    LineageMember,
)

# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------


def _entity(
    name: str,
    *,
    entity_type: str = "organization",
    parts: list[tuple[str, str, str, tuple[str, ...]]] | None = None,
) -> EntityWithLineage:
    """``parts`` is a list of ``(document_id, parse_version, text, chunk_ids)``."""
    parts = parts or [("doc1", "v1", f"description of {name}", ("c0",))]
    return EntityWithLineage(
        name=name,
        entity_type=entity_type,
        source_lineage=tuple(
            LineageMember(
                document_id=d,
                parse_version=v,
                tenant_scope_key="tenant-1",
                chunk_ids=tuple(cids),
            )
            for d, v, _t, cids in parts
        ),
        description_parts=tuple(DescriptionPart(document_id=d, parse_version=v, text=t) for d, v, t, _cids in parts),
    )


def _make_merger(
    *,
    target: EntityWithLineage,
    sources: dict[str, EntityWithLineage] | None = None,
    alias_resolutions: dict[str, str] | None = None,
    llm_response: str = "unified description",
    compacted: str | None = "compacted",
):
    sources = sources or {}
    alias_resolutions = alias_resolutions or {}

    store = AsyncMock()
    store.get_entity = AsyncMock(side_effect=lambda name: sources.get(name) if name != target.name else target)
    store.upsert_entity_with_lineage = AsyncMock(return_value=None)
    store.delete_entity = AsyncMock(return_value=True)

    alias_repo = AsyncMock()
    alias_repo.resolve_canonical = AsyncMock(
        side_effect=lambda *, collection_id, name: alias_resolutions.get(name, name)
    )
    alias_repo.upsert_alias = AsyncMock(return_value=target.name)

    compactor = MagicMock()
    compactor.compact_if_oversized = AsyncMock(return_value=compacted)

    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    vector_connector = MagicMock()
    vector_connector.upsert = MagicMock(return_value=["id"])
    vector_connector.delete = MagicMock(return_value=None)

    async def _llm(_prompt: str) -> str:
        return llm_response

    merger = LineageEntityMerger(
        store=store,
        alias_repo=alias_repo,
        compactor=compactor,
        vector_connector=vector_connector,
        embedder=embedder,
        llm=_llm,
        collection_id="col-1",
        language="English",
    )
    return merger, store, alias_repo, compactor, embedder, vector_connector


# ---------------------------------------------------------------------
# Empty source — short circuit
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_source_list_short_circuits():
    target = _entity("Target")
    merger, store, alias_repo, *_ = _make_merger(target=target)

    result = await merger.merge_entities(target_name="Target", source_names=[], merged_by="user1")
    assert result.final_target == "Target"
    assert result.merged_source_ids == []
    alias_repo.upsert_alias.assert_not_called()
    store.upsert_entity_with_lineage.assert_not_called()
    store.delete_entity.assert_not_called()


# ---------------------------------------------------------------------
# Step ordering: L1 → vector → delete
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_order_is_l1_then_vector_then_delete():
    """Invariant #2: L1 source-of-truth writes precede vector (derived)
    writes; deletes run last."""
    target = _entity("Apple Inc.")
    src = _entity("Apple")
    merger, store, _, _, _, vector_connector = _make_merger(target=target, sources={"Apple": src})

    call_order: list[str] = []

    async def _track_l1(*args, **kwargs):
        call_order.append("L1_upsert")

    async def _track_delete(*args, **kwargs):
        call_order.append("L1_delete")
        return True

    def _track_vec_upsert(*args, **kwargs):
        call_order.append("vector_upsert")
        return ["id"]

    def _track_vec_delete(*args, **kwargs):
        call_order.append("vector_delete")

    store.upsert_entity_with_lineage.side_effect = _track_l1
    store.delete_entity.side_effect = _track_delete
    vector_connector.upsert.side_effect = _track_vec_upsert
    vector_connector.delete.side_effect = _track_vec_delete

    await merger.merge_entities(target_name="Apple Inc.", source_names=["Apple"], merged_by="u")

    # All L1 upserts come first, then vector upsert, then delete.
    last_l1_upsert_idx = max(i for i, x in enumerate(call_order) if x == "L1_upsert")
    vec_upsert_idx = call_order.index("vector_upsert")
    l1_delete_idx = call_order.index("L1_delete")
    vec_delete_idx = call_order.index("vector_delete")
    assert last_l1_upsert_idx < vec_upsert_idx < l1_delete_idx
    assert l1_delete_idx < vec_delete_idx


# ---------------------------------------------------------------------
# Sentinel + Compactor kwargs
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unified_write_uses_curation_merge_sentinel():
    target = _entity("Apple Inc.")
    src = _entity("Apple")
    merger, store, *_ = _make_merger(target=target, sources={"Apple": src})

    await merger.merge_entities(target_name="Apple Inc.", source_names=["Apple"], merged_by="u")

    # The LAST upsert call carries the sentinel + the compacted text.
    upsert_calls = store.upsert_entity_with_lineage.call_args_list
    final_kwargs = upsert_calls[-1].kwargs
    assert final_kwargs["lineage"].document_id == CURATION_MERGE_DOCUMENT_ID
    assert final_kwargs["compacted_description"] == "compacted"
    assert final_kwargs["record"].name == "Apple Inc."


@pytest.mark.asyncio
async def test_compactor_invoked_with_locked_kwargs():
    target = _entity("Apple Inc.")
    src = _entity("Apple")
    merger, _, _, compactor, *_ = _make_merger(target=target, sources={"Apple": src})

    await merger.merge_entities(target_name="Apple Inc.", source_names=["Apple"], merged_by="u")
    compactor.compact_if_oversized.assert_awaited_once()
    call = compactor.compact_if_oversized.call_args
    assert call.kwargs == {
        "subject_kind": "entity",
        "subject_label": "Apple Inc.",
        "language": "English",
    }


# ---------------------------------------------------------------------
# Source parts re-anchored preserving per-doc lineage
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_parts_reanchored_preserving_doc_lineage():
    """Invariant #1: per-doc tracking must survive the merge — each
    source part is re-upserted under the target name with the original
    ``(document_id, parse_version, chunk_ids)`` lineage."""
    target = _entity(
        "Apple Inc.",
        parts=[("docT", "v1", "target initial", ("ct",))],
    )
    src = _entity(
        "Apple",
        parts=[
            ("docA", "v1", "fragment A", ("ca1",)),
            ("docB", "v2", "fragment B", ("cb1", "cb2")),
        ],
    )
    merger, store, *_ = _make_merger(target=target, sources={"Apple": src})

    await merger.merge_entities(target_name="Apple Inc.", source_names=["Apple"], merged_by="u")

    upsert_calls = store.upsert_entity_with_lineage.call_args_list
    # The first 2 upserts are the source parts re-anchored under the
    # target name (in order of the source's description_parts), then
    # the final unified+compacted write.
    assert len(upsert_calls) == 3
    first_call = upsert_calls[0].kwargs
    assert first_call["record"].name == "Apple Inc."
    assert first_call["record"].description == "fragment A"
    assert first_call["lineage"].document_id == "docA"
    assert first_call["lineage"].parse_version == "v1"

    second_call = upsert_calls[1].kwargs
    assert second_call["record"].name == "Apple Inc."
    assert second_call["record"].description == "fragment B"
    assert second_call["lineage"].document_id == "docB"
    assert second_call["lineage"].parse_version == "v2"

    # Final write is the unified+compacted with sentinel.
    final = upsert_calls[2].kwargs
    assert final["lineage"].document_id == CURATION_MERGE_DOCUMENT_ID


# ---------------------------------------------------------------------
# Vector payload + uuid5 pinning
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_payload_is_3_field_with_deterministic_uuid5():
    target = _entity("Apple Inc.")
    src = _entity("Apple")
    merger, _, _, _, _, vector_connector = _make_merger(target=target, sources={"Apple": src})

    await merger.merge_entities(target_name="Apple Inc.", source_names=["Apple"], merged_by="u")

    # Upsert was called once with one VectorPoint.
    vector_connector.upsert.assert_called_once()
    points = vector_connector.upsert.call_args.args[0]
    assert len(points) == 1
    point = points[0]
    expected_id = str(uuid5(NAMESPACE_DNS, f"{GRAPH_ENTITY_INDEXER}:col-1:Apple Inc."))
    assert point.id == expected_id
    # 3 fields exactly — no collection_id, no extras.
    assert set(point.payload.keys()) == {"indexer", "entity_name", "entity_type"}
    assert point.payload["indexer"] == GRAPH_ENTITY_INDEXER
    assert point.payload["entity_name"] == "Apple Inc."
    assert point.payload["entity_type"] == "organization"

    # Source vector point was deleted under the same uuid5 scheme.
    vector_connector.delete.assert_called_once()
    deleted_ids = vector_connector.delete.call_args.args[0]
    expected_src_id = str(uuid5(NAMESPACE_DNS, f"{GRAPH_ENTITY_INDEXER}:col-1:Apple"))
    assert deleted_ids == [expected_src_id]


# ---------------------------------------------------------------------
# Cycle reject propagation
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_cycle_propagates_through_merge():
    target = _entity("Target")
    merger, _, alias_repo, *_ = _make_merger(target=target)
    alias_repo.upsert_alias.side_effect = AliasCycleError("cycle")

    with pytest.raises(AliasCycleError):
        await merger.merge_entities(target_name="Target", source_names=["Source"], merged_by="u")


# ---------------------------------------------------------------------
# Target flatten through alias chain
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_target_resolves_through_alias_chain():
    """If ``target_name`` is itself already an alias, the merge body
    operates on the flattened canonical."""
    final = _entity("Canonical")
    src = _entity("Source")
    merger, store, alias_repo, *_ = _make_merger(
        target=final,
        sources={"Source": src, "AliasOfTarget": final},
        alias_resolutions={"AliasOfTarget": "Canonical"},
    )

    result = await merger.merge_entities(target_name="AliasOfTarget", source_names=["Source"], merged_by="u")
    assert result.final_target == "Canonical"
    # Sources were aliased to the canonical, not the original target.
    alias_repo.upsert_alias.assert_called_once()
    assert alias_repo.upsert_alias.call_args.kwargs["target"] == "Canonical"


# ---------------------------------------------------------------------
# Target GC tolerance
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_target_gced_between_initiation_and_execution_does_not_crash():
    """If the target entity was GC'd between merge initiation and
    execution, the merge logs a warning, writes the alias rows
    anyway (so future indexer writes still resolve), and returns
    cleanly."""
    src = _entity("Source")

    store = AsyncMock()

    async def _get(name: str):
        if name == "Source":
            return src
        return None  # target GC'd

    store.get_entity = AsyncMock(side_effect=_get)
    store.upsert_entity_with_lineage = AsyncMock()
    store.delete_entity = AsyncMock(return_value=True)

    alias_repo = AsyncMock()
    alias_repo.resolve_canonical = AsyncMock(return_value="Target")
    alias_repo.upsert_alias = AsyncMock(return_value="Target")

    compactor = MagicMock()
    compactor.compact_if_oversized = AsyncMock(return_value=None)
    embedder = MagicMock()
    vector_connector = MagicMock()

    async def _llm(_p: str) -> str:
        return ""

    merger = LineageEntityMerger(
        store=store,
        alias_repo=alias_repo,
        compactor=compactor,
        vector_connector=vector_connector,
        embedder=embedder,
        llm=_llm,
        collection_id="col-1",
    )
    result = await merger.merge_entities(target_name="Target", source_names=["Source"], merged_by="u")
    assert result.final_target == "Target"
    # Alias was still recorded.
    alias_repo.upsert_alias.assert_called_once()
    # No L1 upsert, no vector activity, no delete (nothing to consolidate).
    store.upsert_entity_with_lineage.assert_not_called()
    store.delete_entity.assert_not_called()
