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

"""Surface + behavior tests for D10.c read primitives.

Per ``docs/modularization/d10-design-pack.md`` §A + §A.9 (R1 Lock):

- The 8 primitive functions exist with their canonical names.
- Their parameter names + ordering match §A 1:1 (kw-only sentinel preserved).
- The Pydantic response shapes are importable.
- The stable handle types (ChunkId / SectionPath / HeadingAnchor) are
  importable under their LOCKED names.

Behavior tests assert the strict call sequence each primitive body MUST
follow (per cuiwenbo msg=13a4139a + 明书 msg=c5e3be15):

    tenancy gate → auth gate → [parse_version] → fetch authoritative

The body work is mocked via the ``aperag.mcp.tools._d9_base`` helpers so
the tests do not need a live database.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aperag.exceptions import CollectionNotFoundException, PermissionDeniedError
from aperag.mcp.tools import (
    ByteRange,
    ChunkId,
    CollectionDetailMetadata,
    CollectionList,
    CollectionMetadata,
    DocumentChunk,
    DocumentContent,
    DocumentList,
    DocumentMetadata,
    DocumentOutline,
    DocumentSection,
    HeadingAnchor,
    OutlineHeading,
    ParseVersionT,
    SectionPath,
    compute_parse_version,
    get_collection_metadata,
    get_document_metadata,
    list_collections,
    list_documents,
    read_document,
    read_document_chunk,
    read_document_outline,
    read_document_section,
)

# ----- Stable handle types (§A.9 LOCKED) -------------------------------------


def test_stable_handle_types_are_importable():
    """§A.9 R1 Lock: ChunkId / SectionPath / HeadingAnchor exposed under
    these exact names. Renaming requires `[D10 spec amendment]` thread.
    """
    assert ChunkId is not None
    assert SectionPath is not None
    assert HeadingAnchor is not None


# ----- Pydantic shapes -------------------------------------------------------


def test_pydantic_response_shapes_are_importable():
    """All §A response envelopes are exposed at the package surface."""
    for shape in (
        ByteRange,
        CollectionDetailMetadata,
        CollectionList,
        CollectionMetadata,
        DocumentChunk,
        DocumentContent,
        DocumentList,
        DocumentMetadata,
        DocumentOutline,
        DocumentSection,
        OutlineHeading,
    ):
        assert hasattr(shape, "model_validate"), f"{shape.__name__} should be a Pydantic BaseModel subclass"


def test_outline_heading_uses_locked_handle_field_names():
    """§A.9: OutlineHeading must expose the LOCKED handle field names."""
    fields = set(OutlineHeading.model_fields.keys())
    assert {"section_path", "heading_anchor", "chunk_id"} <= fields, (
        f"OutlineHeading is missing one or more LOCKED handle fields: {fields}"
    )


def test_document_chunk_uses_locked_handle_field_names():
    """§A.9: DocumentChunk must expose chunk_id + section_path."""
    fields = set(DocumentChunk.model_fields.keys())
    assert {"chunk_id", "section_path"} <= fields


def test_document_section_uses_locked_handle_field_names():
    """§A.9: DocumentSection must expose section_path + heading_anchor."""
    fields = set(DocumentSection.model_fields.keys())
    assert {"section_path", "heading_anchor"} <= fields


# ----- Primitive function signatures (§A 1:1) -------------------------------


def _params(fn):
    return list(inspect.signature(fn).parameters.values())


def test_list_collections_signature_matches_spec():
    """§A.1: list_collections(*, cursor, limit, sort_by, sort_order, title_filter)."""
    params = _params(list_collections)
    names = [p.name for p in params]
    assert names == ["cursor", "limit", "sort_by", "sort_order", "title_filter"]
    assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in params)


def test_list_documents_signature_matches_spec():
    """§A.2: list_documents(collection_id, *, cursor, limit, sort_by, sort_order,
    title_filter, type_filter, indexed_only)."""
    params = _params(list_documents)
    names = [p.name for p in params]
    assert names == [
        "collection_id",
        "cursor",
        "limit",
        "sort_by",
        "sort_order",
        "title_filter",
        "type_filter",
        "indexed_only",
    ]
    assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    for p in params[1:]:
        assert p.kind == inspect.Parameter.KEYWORD_ONLY


def test_get_document_metadata_signature_matches_spec():
    """§A.3: get_document_metadata(collection_id, document_id)."""
    names = [p.name for p in _params(get_document_metadata)]
    assert names == ["collection_id", "document_id"]


def test_get_collection_metadata_signature_matches_spec():
    """§A.4: get_collection_metadata(collection_id)."""
    names = [p.name for p in _params(get_collection_metadata)]
    assert names == ["collection_id"]


def test_read_document_signature_matches_spec():
    """§A.5: read_document(collection_id, document_id, *, range)."""
    params = _params(read_document)
    names = [p.name for p in params]
    assert names == ["collection_id", "document_id", "range"]
    assert params[2].kind == inspect.Parameter.KEYWORD_ONLY


def test_read_document_outline_signature_matches_spec():
    """§A.6: read_document_outline(collection_id, document_id, *, max_depth)."""
    params = _params(read_document_outline)
    names = [p.name for p in params]
    assert names == ["collection_id", "document_id", "max_depth"]
    assert params[2].kind == inspect.Parameter.KEYWORD_ONLY


def test_read_document_section_signature_matches_spec():
    """§A.7: read_document_section(collection_id, document_id, *,
    section_path, heading_anchor)."""
    params = _params(read_document_section)
    names = [p.name for p in params]
    assert names == [
        "collection_id",
        "document_id",
        "section_path",
        "heading_anchor",
    ]
    for p in params[2:]:
        assert p.kind == inspect.Parameter.KEYWORD_ONLY


def test_read_document_chunk_signature_matches_spec():
    """§A.8: read_document_chunk(collection_id, document_id, chunk_id)."""
    names = [p.name for p in _params(read_document_chunk)]
    assert names == ["collection_id", "document_id", "chunk_id"]


def test_all_primitives_are_async():
    """Per §A signatures, every read primitive is ``async def``."""
    for fn in (
        list_collections,
        list_documents,
        get_document_metadata,
        get_collection_metadata,
        read_document,
        read_document_outline,
        read_document_section,
        read_document_chunk,
    ):
        assert inspect.iscoroutinefunction(fn), f"{fn.__name__} should be an async coroutine per §A spec"


# ----- parse_version helper (§E.2) ------------------------------------------


def test_parse_version_t_is_exported():
    """ParseVersionT (Annotated[str, regex]) and compute_parse_version both
    exposed at the package surface for D10.g cache wire-in."""
    assert ParseVersionT is not None
    assert callable(compute_parse_version)


def test_compute_parse_version_is_deterministic_and_16_hex():
    """sha256 first-16-hex per architect msg=a67974b3 Q1 lock."""
    v1 = compute_parse_version(
        parser_pipeline="docparser-v1",
        document_md5="0" * 32,
        chunking_config="default",
    )
    v2 = compute_parse_version(
        parser_pipeline="docparser-v1",
        document_md5="0" * 32,
        chunking_config="default",
    )
    assert v1 == v2
    assert len(v1) == 16
    assert all(c in "0123456789abcdef" for c in v1)


def test_compute_parse_version_changes_when_inputs_change():
    """Any of the three inputs changing must yield a fresh parse_version."""
    base = compute_parse_version(
        parser_pipeline="docparser-v1",
        document_md5="abc",
        chunking_config="default",
    )
    assert base != compute_parse_version(
        parser_pipeline="docparser-v2",
        document_md5="abc",
        chunking_config="default",
    )
    assert base != compute_parse_version(
        parser_pipeline="docparser-v1",
        document_md5="def",
        chunking_config="default",
    )
    assert base != compute_parse_version(
        parser_pipeline="docparser-v1",
        document_md5="abc",
        chunking_config="other",
    )


# ----- Behavior tests --------------------------------------------------------


def _fake_user(user_id: str = "user-test-1"):
    user = SimpleNamespace(id=user_id, role=None)
    return user


def _fake_collection(collection_id: str = "col-1"):
    return SimpleNamespace(
        id=collection_id,
        title="Test Collection",
        description="desc",
        config='{"chunk_size": 512}',
        gmt_created=datetime(2025, 1, 1, tzinfo=timezone.utc),
        gmt_updated=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )


def _patch_d9_base(call_log: list[str]):
    """Patch the D9 base helpers so each primitive sees the locked
    sequence without touching a real DB. Returns a context manager."""

    fake_user = _fake_user()
    fake_collection = _fake_collection()

    async def _resolve():
        call_log.append("resolve")
        return fake_user

    async def _tenancy(user, collection_id):
        call_log.append(f"tenancy:{collection_id}")
        return fake_collection

    async def _authz(user, tool):
        call_log.append(f"authz:{tool}")
        return None

    return [
        patch("aperag.mcp.tools.list_collections.resolve_authenticated_user", _resolve),
        patch("aperag.mcp.tools.list_collections.authorization_gate", _authz),
        patch("aperag.mcp.tools.list_documents.resolve_authenticated_user", _resolve),
        patch("aperag.mcp.tools.list_documents.tenancy_gate", _tenancy),
        patch("aperag.mcp.tools.list_documents.authorization_gate", _authz),
        patch("aperag.mcp.tools.get_collection_metadata.resolve_authenticated_user", _resolve),
        patch("aperag.mcp.tools.get_collection_metadata.tenancy_gate", _tenancy),
        patch("aperag.mcp.tools.get_collection_metadata.authorization_gate", _authz),
        patch("aperag.mcp.tools.get_document_metadata.resolve_authenticated_user", _resolve),
        patch("aperag.mcp.tools.get_document_metadata.tenancy_gate", _tenancy),
        patch("aperag.mcp.tools.get_document_metadata.authorization_gate", _authz),
        patch("aperag.mcp.tools.read_document.resolve_authenticated_user", _resolve),
        patch("aperag.mcp.tools.read_document.tenancy_gate", _tenancy),
        patch("aperag.mcp.tools.read_document.authorization_gate", _authz),
        patch("aperag.mcp.tools.read_document_outline.resolve_authenticated_user", _resolve),
        patch("aperag.mcp.tools.read_document_outline.tenancy_gate", _tenancy),
        patch("aperag.mcp.tools.read_document_outline.authorization_gate", _authz),
        patch("aperag.mcp.tools.read_document_section.resolve_authenticated_user", _resolve),
        patch("aperag.mcp.tools.read_document_section.tenancy_gate", _tenancy),
        patch("aperag.mcp.tools.read_document_section.authorization_gate", _authz),
        patch("aperag.mcp.tools.read_document_chunk.resolve_authenticated_user", _resolve),
        patch("aperag.mcp.tools.read_document_chunk.tenancy_gate", _tenancy),
        patch("aperag.mcp.tools.read_document_chunk.authorization_gate", _authz),
    ]


def _enter_all(patches):
    return [p.__enter__() for p in patches]


def _exit_all(patches):
    for p in reversed(patches):
        p.__exit__(None, None, None)


def _async_session_iter(session):
    """Build a fresh async iterator that yields ``session`` exactly once.

    Mirrors the ``async for session in get_async_session()`` pattern used
    by the primitives. Unlike a coroutine, this is callable repeatedly.
    """

    def _factory(*args: Any, **kwargs: Any):
        async def _gen():
            yield session

        return _gen()

    return _factory


def _execute_returning(scalars_first=None, scalars_all=None, scalar_value=None):
    """Build a fake ``await session.execute(...)`` result object."""

    result = MagicMock()
    scalars = MagicMock()
    scalars.first.return_value = scalars_first
    scalars.all.return_value = scalars_all or []
    result.scalars.return_value = scalars
    result.scalar.return_value = scalar_value
    return result


# --- list_collections happy path --------------------------------------------


def test_list_collections_happy_path_records_call_sequence():
    """Tenancy is implicit (per-user query) — auth gate must still fire,
    then the un-cached fetch via async_db_ops.query_collections."""

    call_log: list[str] = []
    patches = _patch_d9_base(call_log)
    fake_rows = [
        SimpleNamespace(
            id="col-1",
            title="Alpha",
            description="a",
            gmt_created=datetime(2025, 1, 1, tzinfo=timezone.utc),
            gmt_updated=datetime(2025, 1, 2, tzinfo=timezone.utc),
        ),
        SimpleNamespace(
            id="col-2",
            title="Beta",
            description="b",
            gmt_created=datetime(2025, 1, 3, tzinfo=timezone.utc),
            gmt_updated=datetime(2025, 1, 4, tzinfo=timezone.utc),
        ),
    ]
    patches.append(
        patch(
            "aperag.mcp.tools.list_collections.async_db_ops.query_collections",
            new=AsyncMock(return_value=fake_rows),
        )
    )

    _enter_all(patches)
    try:
        result = asyncio.run(list_collections())
    finally:
        _exit_all(patches)

    # Strict sequence: resolve → authz → fetch (no tenancy gate for list).
    assert call_log == ["resolve", "authz:list_collections"]
    assert isinstance(result, CollectionList)
    # Default sort: created_at desc — newer first.
    assert [c.collection_id for c in result.items] == ["col-2", "col-1"]
    assert result.total_count == 2


# --- get_collection_metadata happy path -------------------------------------


def test_get_collection_metadata_strict_sequence():
    """resolve → tenancy → authz → fetch."""
    call_log: list[str] = []
    patches = _patch_d9_base(call_log)

    fake_session = MagicMock()
    fake_session.execute = AsyncMock(return_value=_execute_returning(scalar_value=3))
    patches.append(
        patch(
            "aperag.mcp.tools.get_collection_metadata.get_async_session",
            new=_async_session_iter(fake_session),
        )
    )

    _enter_all(patches)
    try:
        result = asyncio.run(get_collection_metadata("col-1"))
    finally:
        _exit_all(patches)

    assert call_log == ["resolve", "tenancy:col-1", "authz:get_collection_metadata"]
    assert isinstance(result, CollectionDetailMetadata)
    assert result.collection_id == "col-1"
    assert result.document_count == 3


# --- list_documents happy path ----------------------------------------------


def test_list_documents_strict_sequence():
    """resolve → tenancy → authz → fetch."""
    call_log: list[str] = []
    patches = _patch_d9_base(call_log)

    doc_row = SimpleNamespace(
        id="doc-1",
        collection_id="col-1",
        name="Hello.md",
        size=42,
        status=__import__(
            "aperag.domains.knowledge_base.db.models", fromlist=["DocumentStatus"]
        ).DocumentStatus.COMPLETE,
        gmt_created=datetime(2025, 1, 1, tzinfo=timezone.utc),
        gmt_updated=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    # Three execute calls: count, page, then index lookup.
    count_result = _execute_returning(scalar_value=1)
    page_result = _execute_returning(scalars_all=[doc_row])
    idx_result = _execute_returning(scalars_all=[])
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(side_effect=[count_result, page_result, idx_result])
    patches.append(
        patch(
            "aperag.mcp.tools.list_documents.get_async_session",
            new=_async_session_iter(fake_session),
        )
    )

    _enter_all(patches)
    try:
        result = asyncio.run(list_documents("col-1"))
    finally:
        _exit_all(patches)

    assert call_log == ["resolve", "tenancy:col-1", "authz:list_documents"]
    assert isinstance(result, DocumentList)
    assert len(result.items) == 1
    assert result.items[0].document_id == "doc-1"


# --- get_document_metadata happy path ---------------------------------------


def test_get_document_metadata_strict_sequence():
    call_log: list[str] = []
    patches = _patch_d9_base(call_log)

    DocumentStatus = __import__("aperag.domains.knowledge_base.db.models", fromlist=["DocumentStatus"]).DocumentStatus
    doc_row = SimpleNamespace(
        id="doc-1",
        collection_id="col-1",
        name="Hello.md",
        size=42,
        status=DocumentStatus.COMPLETE,
        gmt_created=datetime(2025, 1, 1, tzinfo=timezone.utc),
        gmt_updated=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    doc_lookup = _execute_returning(scalars_first=doc_row)
    idx_lookup = _execute_returning(scalars_all=[])
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(side_effect=[doc_lookup, idx_lookup])
    patches.append(
        patch(
            "aperag.mcp.tools.get_document_metadata.get_async_session",
            new=_async_session_iter(fake_session),
        )
    )

    _enter_all(patches)
    try:
        result = asyncio.run(get_document_metadata("col-1", "doc-1"))
    finally:
        _exit_all(patches)

    assert call_log == ["resolve", "tenancy:col-1", "authz:get_document_metadata"]
    assert isinstance(result, DocumentMetadata)
    assert result.document_id == "doc-1"


# --- read_document / outline / section / chunk happy paths ------------------


def _patch_doc_lookup(module_path: str):
    DocumentStatus = __import__("aperag.domains.knowledge_base.db.models", fromlist=["DocumentStatus"]).DocumentStatus
    doc_row = SimpleNamespace(
        id="doc-1",
        collection_id="col-1",
        user="user-test-1",
        name="Hello.md",
        size=42,
        content_hash="cafef00d",
        status=DocumentStatus.COMPLETE,
        gmt_created=datetime(2025, 1, 1, tzinfo=timezone.utc),
        gmt_updated=datetime(2025, 1, 2, tzinfo=timezone.utc),
        object_store_base_path=lambda: "user-x/col-1/doc-1",
    )
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(return_value=_execute_returning(scalars_first=doc_row))
    return patch(f"{module_path}.get_async_session", new=_async_session_iter(fake_session))


def test_read_document_strict_sequence_with_parse_version():
    call_log: list[str] = []
    patches = _patch_d9_base(call_log)
    patches.append(_patch_doc_lookup("aperag.mcp.tools.read_document"))
    patches.append(
        patch(
            "aperag.mcp.tools.read_document.read_parsed_markdown",
            new=AsyncMock(return_value="# Title\n\nbody"),
        )
    )

    _enter_all(patches)
    try:
        result = asyncio.run(read_document("col-1", "doc-1"))
    finally:
        _exit_all(patches)

    assert call_log == ["resolve", "tenancy:col-1", "authz:read_document"]
    assert isinstance(result, DocumentContent)
    assert result.document_id == "doc-1"
    assert len(result.parse_version) == 16
    assert result.parsed_markdown.startswith("# Title")


def test_read_document_outline_strict_sequence():
    call_log: list[str] = []
    patches = _patch_d9_base(call_log)
    patches.append(_patch_doc_lookup("aperag.mcp.tools.read_document_outline"))
    patches.append(
        patch(
            "aperag.mcp.tools.read_document_outline.read_parsed_markdown",
            new=AsyncMock(return_value="# Title\n\n## Sub1\n\ncontent\n\n## Sub2\n\nmore\n"),
        )
    )

    _enter_all(patches)
    try:
        result = asyncio.run(read_document_outline("col-1", "doc-1"))
    finally:
        _exit_all(patches)

    assert call_log == ["resolve", "tenancy:col-1", "authz:read_document_outline"]
    assert isinstance(result, DocumentOutline)
    assert len(result.headings) == 1  # one h1 root
    assert result.headings[0].text == "Title"
    assert len(result.headings[0].children) == 2


def test_read_document_section_strict_sequence():
    call_log: list[str] = []
    patches = _patch_d9_base(call_log)
    patches.append(_patch_doc_lookup("aperag.mcp.tools.read_document_section"))
    patches.append(
        patch(
            "aperag.mcp.tools.read_document_section.read_parsed_markdown",
            new=AsyncMock(return_value="# Title\n\n## Sub1\n\nbody1\n\n## Sub2\n\nbody2\n"),
        )
    )

    _enter_all(patches)
    try:
        result = asyncio.run(read_document_section("col-1", "doc-1", section_path="1/1"))
    finally:
        _exit_all(patches)

    assert call_log == ["resolve", "tenancy:col-1", "authz:read_document_section"]
    assert isinstance(result, DocumentSection)
    assert result.section_path == "1/1"
    assert "Sub1" in result.parsed_markdown


def test_read_document_chunk_strict_sequence():
    call_log: list[str] = []
    patches = _patch_d9_base(call_log)
    patches.append(_patch_doc_lookup("aperag.mcp.tools.read_document_chunk"))

    fake_chunk = SimpleNamespace(
        id="chunk-1",
        text="hello world",
        metadata={"section_path": "1/1"},
    )
    patches.append(
        patch(
            "aperag.mcp.tools.read_document_chunk.document_service.get_document_chunks",
            new=AsyncMock(return_value=[fake_chunk]),
        )
    )

    _enter_all(patches)
    try:
        result = asyncio.run(read_document_chunk("col-1", "doc-1", "chunk-1"))
    finally:
        _exit_all(patches)

    assert call_log == ["resolve", "tenancy:col-1", "authz:read_document_chunk"]
    assert isinstance(result, DocumentChunk)
    assert result.chunk_id == "chunk-1"
    assert result.section_path == "1/1"


# --- tenancy violation guard (§F.4) -----------------------------------------


def test_tenancy_violation_blocks_read_document():
    """If the canonical tenancy gate raises ``CollectionNotFoundException``
    (the canonical SoT response for non-owner / non-subscriber), the
    primitive must NOT proceed to fetch — even cache cannot bypass this
    per §E.7.
    """

    async def _resolve():
        return _fake_user()

    async def _tenancy_deny(user, collection_id):
        raise CollectionNotFoundException(collection_id)

    async def _authz_should_not_be_called(user, tool):
        pytest.fail("authorization_gate must NOT run after tenancy_gate raised")

    fetch_mock = AsyncMock()
    with (
        patch("aperag.mcp.tools.read_document.resolve_authenticated_user", _resolve),
        patch("aperag.mcp.tools.read_document.tenancy_gate", _tenancy_deny),
        patch(
            "aperag.mcp.tools.read_document.authorization_gate",
            _authz_should_not_be_called,
        ),
        patch("aperag.mcp.tools.read_document.read_parsed_markdown", new=fetch_mock),
    ):
        with pytest.raises(CollectionNotFoundException):
            asyncio.run(read_document("col-not-mine", "doc-1"))

    fetch_mock.assert_not_awaited()


def test_missing_api_key_raises_permission_denied():
    """Step 1 (resolve_authenticated_user) is the locked entry — any
    primitive called without an API key must surface PermissionDeniedError
    BEFORE reaching tenancy / authz / fetch.
    """

    async def _resolve_deny():
        raise PermissionDeniedError("no api key")

    fetch_mock = AsyncMock()
    with (
        patch(
            "aperag.mcp.tools.list_collections.resolve_authenticated_user",
            _resolve_deny,
        ),
        patch(
            "aperag.mcp.tools.list_collections.async_db_ops.query_collections",
            new=fetch_mock,
        ),
    ):
        with pytest.raises(PermissionDeniedError):
            asyncio.run(list_collections())

    fetch_mock.assert_not_awaited()


# --- §E.7: cache cannot skip tenancy invariant (call-site witness) ---------


def test_call_sequence_witness_for_all_parse_version_keyed_primitives():
    """Each parse_version-keyed primitive must invoke (1) resolve, (2)
    tenancy, (3) authz BEFORE any fetch step. This test asserts that the
    body's source code references the D9 base helpers in that order — a
    static guard so a future refactor cannot accidentally reorder.
    """

    import inspect as _inspect

    for primitive in (read_document, read_document_outline, read_document_section, read_document_chunk):
        src = _inspect.getsource(primitive)
        idx_resolve = src.find("resolve_authenticated_user(")
        idx_tenancy = src.find("tenancy_gate(")
        idx_authz = src.find("authorization_gate(")
        # All three calls must exist and appear in the locked order.
        assert idx_resolve != -1, f"{primitive.__name__} missing resolve_authenticated_user"
        assert idx_tenancy != -1, f"{primitive.__name__} missing tenancy_gate"
        assert idx_authz != -1, f"{primitive.__name__} missing authorization_gate"
        assert idx_resolve < idx_tenancy < idx_authz, (
            f"{primitive.__name__} D9 base call order violated (got resolve@{idx_resolve} "
            f"tenancy@{idx_tenancy} authz@{idx_authz})"
        )


# --- §C explicit-not-silent: cursor decode ----------------------------------


def _b64(payload_obj: Any) -> str:
    import base64 as _b
    import json as _j

    return _b.urlsafe_b64encode(_j.dumps(payload_obj).encode("utf-8")).decode("ascii")


def test_list_collections_decode_cursor_none_and_empty_return_zero():
    """``cursor=None`` and ``cursor=""`` are NOT malformed — they mean
    "start from the beginning" and must yield offset 0 (no exception).
    """
    from aperag.mcp.tools.list_collections import _decode_cursor as _dc_collections

    assert _dc_collections(None) == 0
    assert _dc_collections("") == 0
    # well-formed positive cursor still works
    assert _dc_collections(_b64({"offset": 50})) == 50


def test_list_documents_decode_cursor_none_and_empty_return_zero():
    from aperag.mcp.tools.list_documents import _decode_cursor as _dc_documents

    assert _dc_documents(None) == 0
    assert _dc_documents("") == 0
    assert _dc_documents(_b64({"offset": 50})) == 50


def test_list_collections_malformed_cursor_raises():
    """§C explicit-not-silent: a non-empty malformed cursor MUST raise
    ValueError instead of silently degrading to offset 0."""
    from aperag.mcp.tools.list_collections import _decode_cursor as _dc_collections

    # garbage that is not valid base64
    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_collections("!!!not-valid-base64!!!")
    # valid base64 but not JSON
    import base64 as _b

    not_json = _b.urlsafe_b64encode(b"not json at all").decode("ascii")
    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_collections(not_json)
    # JSON but not a dict / missing offset
    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_collections(_b64([1, 2, 3]))
    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_collections(_b64({"page": 1}))
    # offset is not an int
    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_collections(_b64({"offset": "ten"}))


def test_list_collections_negative_offset_cursor_raises():
    """Negative offsets are explicit corruption, not "first page"."""
    from aperag.mcp.tools.list_collections import _decode_cursor as _dc_collections

    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_collections(_b64({"offset": -1}))


def test_list_documents_malformed_cursor_raises():
    from aperag.mcp.tools.list_documents import _decode_cursor as _dc_documents

    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_documents("!!!not-valid-base64!!!")
    import base64 as _b

    not_json = _b.urlsafe_b64encode(b"not json at all").decode("ascii")
    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_documents(not_json)
    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_documents(_b64([1, 2, 3]))
    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_documents(_b64({"page": 1}))
    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_documents(_b64({"offset": "ten"}))


def test_list_documents_negative_offset_cursor_raises():
    from aperag.mcp.tools.list_documents import _decode_cursor as _dc_documents

    with pytest.raises(ValueError, match="cursor decode failed"):
        _dc_documents(_b64({"offset": -1}))


# --- type_filter applied BEFORE pagination (Weston msg=246c84d3) ------------


def _make_doc(doc_id: str, name: str, *, size: int = 100, days_offset: int = 0):
    """Helper: build a minimal Document-like SimpleNamespace for the page."""
    DocumentStatus = __import__("aperag.domains.knowledge_base.db.models", fromlist=["DocumentStatus"]).DocumentStatus
    return SimpleNamespace(
        id=doc_id,
        collection_id="col-1",
        name=name,
        size=size,
        status=DocumentStatus.COMPLETE,
        gmt_created=datetime(2025, 1, 1 + days_offset, tzinfo=timezone.utc),
        gmt_updated=datetime(2025, 1, 2 + days_offset, tzinfo=timezone.utc),
    )


def test_list_documents_type_filter_finds_match_beyond_first_page():
    """Bug guard for Weston msg=246c84d3: when the matching markdown doc
    sits at position >= limit in the sort order, the old code sliced
    BEFORE filtering and returned []. The fix must still find it.
    """

    call_log: list[str] = []
    patches = _patch_d9_base(call_log)

    # 5 PDFs sort first, then 1 markdown — limit=3 means in the OLD code
    # the markdown never enters the page slice. Order must be deterministic
    # — we feed them already-sorted to the fake session.
    docs = [
        _make_doc("doc-pdf-1", "a.pdf", days_offset=0),
        _make_doc("doc-pdf-2", "b.pdf", days_offset=1),
        _make_doc("doc-pdf-3", "c.pdf", days_offset=2),
        _make_doc("doc-pdf-4", "d.pdf", days_offset=3),
        _make_doc("doc-pdf-5", "e.pdf", days_offset=4),
        _make_doc("doc-md-1", "report.md", days_offset=5),
    ]

    # Only the full unbounded fetch is issued in the type_filter branch
    # (no count_stmt, no separate page_stmt). Then idx lookup runs once
    # if the page is non-empty.
    full_result = _execute_returning(scalars_all=docs)
    idx_result = _execute_returning(scalars_all=[])
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(side_effect=[full_result, idx_result])
    patches.append(
        patch(
            "aperag.mcp.tools.list_documents.get_async_session",
            new=_async_session_iter(fake_session),
        )
    )

    _enter_all(patches)
    try:
        result = asyncio.run(
            list_documents("col-1", limit=3, type_filter=["text/markdown"]),
        )
    finally:
        _exit_all(patches)

    assert isinstance(result, DocumentList)
    assert len(result.items) == 1, (
        "type_filter must apply BEFORE pagination — markdown doc at position 5 must still surface when limit=3"
    )
    assert result.items[0].document_id == "doc-md-1"
    assert result.items[0].media_type == "text/markdown"
    assert result.total_count == 1, "total_count must reflect the post-filter count, not the pre-filter count"
    assert result.next_cursor is None


def test_list_documents_type_filter_total_count_reflects_post_filter():
    """With 10 PDFs + 2 markdowns, type_filter=["text/markdown"] must
    yield total_count=2 (not 12)."""

    call_log: list[str] = []
    patches = _patch_d9_base(call_log)

    docs = [_make_doc(f"doc-pdf-{i}", f"f{i}.pdf", days_offset=i) for i in range(10)]
    docs.extend(
        [
            _make_doc("doc-md-1", "alpha.md", days_offset=10),
            _make_doc("doc-md-2", "beta.md", days_offset=11),
        ]
    )

    full_result = _execute_returning(scalars_all=docs)
    idx_result = _execute_returning(scalars_all=[])
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(side_effect=[full_result, idx_result])
    patches.append(
        patch(
            "aperag.mcp.tools.list_documents.get_async_session",
            new=_async_session_iter(fake_session),
        )
    )

    _enter_all(patches)
    try:
        result = asyncio.run(list_documents("col-1", type_filter=["text/markdown"]))
    finally:
        _exit_all(patches)

    assert result.total_count == 2
    assert {d.document_id for d in result.items} == {"doc-md-1", "doc-md-2"}
    assert result.next_cursor is None
