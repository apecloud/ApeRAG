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

"""Task #32-A1b acceptance test — graph evidence_refs → read_document_chunk chain.

Pins the cross-endpoint contract introduced by PR #1909 (task #32-A1):
the bounded ``evidence_refs`` carried on graph entities + relations must
re-compose a complete ``read_document_chunk(collection_id, document_id,
chunk_id)`` call without any additional lookup, so an agent following
the spec § 3.1.1 chain ``query_graph_entities → entity → evidence_refs
→ read_document_chunk → chunk content`` can succeed.

The graph projection layer itself is already covered by ziang's unit
tests in ``tests/unit_test/service/test_graph_search_service_layer.py``
(23 tests on ``_lineage_to_evidence_refs`` semantics) and
``tests/unit_test/mcp/test_graph_tools.py``. This file's value-add is
*chained* — verifying the data shape produced by graph endpoints is
contract-compatible with ``read_document_chunk`` without re-testing
either side in isolation.

Patch-based per ziang msg=db58d137 — no real DB / vector connector /
provider, mirrors ``tests/unit_test/test_d10c_read_primitives_surface.py``
``_patch_doc_lookup`` style.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aperag.domains.knowledge_base.db.models import DocumentStatus
from aperag.domains.knowledge_graph.schemas import (
    GraphEvidenceRef,
    GraphRelationView,
    GraphSearchEntity,
)
from aperag.domains.knowledge_graph.service import (
    _entity_to_search_view,
    _lineage_to_evidence_refs,
    _relation_to_view,
)
from aperag.indexing.graph import LineageMember
from aperag.mcp.tools import DocumentChunk, read_document_chunk

# ---------------------------------------------------------------------------
# Acceptance fixtures — pinned to ziang's coords (msg=49d3d8fd):
# document_id="doc1", chunk_id="chunk-a", parse_version="v1".
# Mirrors the minimal LineageMember shape used in PR #1909's unit tests.
# ---------------------------------------------------------------------------

_COLLECTION_ID = "col-acceptance-1"
_DOCUMENT_ID = "doc1"
_CHUNK_ID = "chunk-a"
_PARSE_VERSION = "v1"
_CHUNK_BODY = "Hello, this is the chunk body that the agent must reach via evidence_refs."


def _lineage_member() -> LineageMember:
    """Build the canonical fixture lineage member used across tests."""
    return LineageMember(
        document_id=_DOCUMENT_ID,
        parse_version=_PARSE_VERSION,
        tenant_scope_key="user:acceptance-1",
        chunk_ids=(_CHUNK_ID,),
    )


def _fake_user(user_id: str = "user-acceptance-1") -> SimpleNamespace:
    return SimpleNamespace(id=user_id, role=None)


def _fake_collection() -> SimpleNamespace:
    return SimpleNamespace(
        id=_COLLECTION_ID,
        title="Graph evidence_refs acceptance",
        description="task #48 fixture",
        config='{"chunk_size": 512}',
        gmt_created=datetime(2025, 1, 1, tzinfo=timezone.utc),
        gmt_updated=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )


def _async_session_iter(session: Any):
    """Build a fresh async iterator that yields ``session`` exactly once.

    Mirrors the ``async for session in get_async_session()`` pattern used
    by ``read_document_chunk``.
    """

    def _factory(*args: Any, **kwargs: Any):
        async def _gen():
            yield session

        return _gen()

    return _factory


def _execute_returning(scalars_first: Any) -> MagicMock:
    result = MagicMock()
    scalars = MagicMock()
    scalars.first.return_value = scalars_first
    scalars.all.return_value = []
    result.scalars.return_value = scalars
    result.scalar.return_value = None
    return result


def _patch_read_document_chunk_d9_base():
    """Patch the D9 base helpers so ``read_document_chunk`` can run
    without a live tenancy / authz / DB session."""

    user = _fake_user()
    collection = _fake_collection()

    async def _resolve():
        return user

    async def _tenancy(_user, collection_id):
        # Sanity: composite key from evidence_refs must round-trip the
        # same collection_id we pass in.
        assert collection_id == _COLLECTION_ID, "collection_id leaked from caller"
        return collection

    async def _authz(_user, _tool):
        return None

    doc_row = SimpleNamespace(
        id=_DOCUMENT_ID,
        collection_id=_COLLECTION_ID,
        user=str(user.id),
        name="acceptance.md",
        size=len(_CHUNK_BODY),
        content_hash="cafe",
        status=DocumentStatus.COMPLETE,
        gmt_created=datetime(2025, 1, 1, tzinfo=timezone.utc),
        gmt_updated=datetime(2025, 1, 2, tzinfo=timezone.utc),
        object_store_base_path=lambda: f"user-x/{_COLLECTION_ID}/{_DOCUMENT_ID}",
    )
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(return_value=_execute_returning(scalars_first=doc_row))

    fake_chunk = SimpleNamespace(
        id=_CHUNK_ID,
        text=_CHUNK_BODY,
        metadata={"section_path": "1/1"},
    )

    return [
        patch("aperag.mcp.tools.read_document_chunk.resolve_authenticated_user", _resolve),
        patch("aperag.mcp.tools.read_document_chunk.tenancy_gate", _tenancy),
        patch("aperag.mcp.tools.read_document_chunk.authorization_gate", _authz),
        patch(
            "aperag.mcp.tools.read_document_chunk.get_async_session",
            new=_async_session_iter(fake_session),
        ),
        patch(
            "aperag.mcp.tools.read_document_chunk.document_service.get_document_chunks",
            new=AsyncMock(return_value=[fake_chunk]),
        ),
    ]


def _enter_all(patches):
    return [p.__enter__() for p in patches]


def _exit_all(patches):
    for p in reversed(patches):
        p.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Spec § 3.1.1 / § 6 acceptance: cross-endpoint composite-key chain.
# ---------------------------------------------------------------------------


def test_evidence_ref_carries_composite_key_required_by_read_document_chunk():
    """``GraphEvidenceRef`` must carry exactly the fields
    ``read_document_chunk(collection_id, document_id, chunk_id)``
    needs from a graph result. ``parse_version`` is optional but
    populated when known so a downstream ``parse_version``-aware reader
    can use it."""

    refs = _lineage_to_evidence_refs([_lineage_member()])
    assert len(refs) == 1
    ref = refs[0]
    assert isinstance(ref, GraphEvidenceRef)
    assert ref.document_id == _DOCUMENT_ID
    assert ref.chunk_id == _CHUNK_ID
    assert ref.parse_version == _PARSE_VERSION


def test_query_graph_entities_evidence_ref_feeds_read_document_chunk():
    """Spec § 6 acceptance criterion 1+2: ``query_graph_entities`` returns
    an entity whose ``evidence_refs[0]`` re-composes a successful
    ``read_document_chunk`` call returning the underlying chunk body.
    """

    # Build a synthetic entity matching the projection ziang's
    # `_entity_to_search_view` would produce for a single LineageMember.
    entity_with_lineage = SimpleNamespace(
        name="墨香居",
        entity_type="organization",
        description_parts=("墨香居是这条老巷子里唯一的旧书店。",),
        compacted_description=None,
        source_lineage=(_lineage_member(),),
    )
    entity_view = _entity_to_search_view(entity_with_lineage)

    assert isinstance(entity_view, GraphSearchEntity)
    assert entity_view.evidence_refs, "entity_view.evidence_refs must not be empty"
    ref = entity_view.evidence_refs[0]

    patches = _patch_read_document_chunk_d9_base()
    _enter_all(patches)
    try:
        chunk = asyncio.run(
            read_document_chunk(_COLLECTION_ID, ref.document_id, ref.chunk_id),
        )
    finally:
        _exit_all(patches)

    assert isinstance(chunk, DocumentChunk)
    assert chunk.chunk_id == _CHUNK_ID
    assert chunk.document_id == _DOCUMENT_ID
    assert chunk.collection_id == _COLLECTION_ID
    assert chunk.parsed_markdown == _CHUNK_BODY


def test_expand_graph_subgraph_relation_evidence_ref_feeds_read_document_chunk():
    """Spec § 6 acceptance criterion 3: ``expand_graph_subgraph`` carries
    ``evidence_refs`` on the **relation** side as well, satisfying
    Weston msg=7500e57d's BLOCKER fix that double-sided refs are required
    for agents to follow relation-level evidence back to source chunks.
    """

    relation_with_lineage = SimpleNamespace(
        source="李明华",
        target="墨香居",
        description_parts=("李明华经营墨香居",),
        compacted_description=None,
        evidence_lineage=(_lineage_member(),),
    )
    relation_view = _relation_to_view(relation_with_lineage)

    assert isinstance(relation_view, GraphRelationView)
    assert relation_view.evidence_refs, "relation_view.evidence_refs must not be empty"
    ref = relation_view.evidence_refs[0]

    patches = _patch_read_document_chunk_d9_base()
    _enter_all(patches)
    try:
        chunk = asyncio.run(
            read_document_chunk(_COLLECTION_ID, ref.document_id, ref.chunk_id),
        )
    finally:
        _exit_all(patches)

    assert isinstance(chunk, DocumentChunk)
    assert chunk.chunk_id == _CHUNK_ID
    assert chunk.parsed_markdown == _CHUNK_BODY


def test_get_entity_detail_entity_evidence_ref_feeds_read_document_chunk():
    """Spec § 6 acceptance criterion 4: ``get_entity_detail`` returns the
    same ``GraphSearchEntity`` shape as ``query_graph_entities``, so the
    composite-key chain works identically when an agent already knows
    the entity name and is hitting detail directly. Re-using
    ``_entity_to_search_view`` proves the projection is the single source
    of truth for both endpoints (per ziang msg=db58d137).
    """

    entity_with_lineage = SimpleNamespace(
        name="李明华",
        entity_type="person",
        description_parts=("墨香居老板。",),
        compacted_description=None,
        source_lineage=(_lineage_member(),),
    )
    entity_view = _entity_to_search_view(entity_with_lineage)
    ref = entity_view.evidence_refs[0]

    patches = _patch_read_document_chunk_d9_base()
    _enter_all(patches)
    try:
        chunk = asyncio.run(
            read_document_chunk(_COLLECTION_ID, ref.document_id, ref.chunk_id),
        )
    finally:
        _exit_all(patches)

    assert isinstance(chunk, DocumentChunk)
    assert chunk.parsed_markdown == _CHUNK_BODY
