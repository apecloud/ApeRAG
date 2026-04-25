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

"""Surface-stability tests for D10.c read primitives stub.

These tests exist so that D10.d / D10.e / D10.g owners can statically
import the cross-lane surface and rely on it being shape-stable. Per
``docs/modularization/d10-design-pack.md`` §A + §A.9 (R1 Lock):

- The 8 primitive functions exist with their canonical names.
- Their parameter names + ordering match §A 1:1 (kw-only sentinel preserved).
- The Pydantic response shapes are importable.
- The stable handle types (ChunkId / SectionPath / HeadingAnchor) are
  importable under their LOCKED names.
- Each primitive raises ``NotImplementedError`` when called — the body is
  intentionally a stub, so any silent partial implementation would be a
  regression caught here.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

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
    SectionPath,
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
    # Currently ``str`` aliases; what we assert is that the names exist and
    # are usable as type annotations / aliases (truthy + not ``None``).
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
    assert {"chunk_id", "section_path"} <= fields, (
        f"DocumentChunk is missing one or more LOCKED handle fields: {fields}"
    )


def test_document_section_uses_locked_handle_field_names():
    """§A.9: DocumentSection must expose section_path + heading_anchor."""
    fields = set(DocumentSection.model_fields.keys())
    assert {"section_path", "heading_anchor"} <= fields, (
        f"DocumentSection is missing one or more LOCKED handle fields: {fields}"
    )


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


# ----- Stub-body invariant ---------------------------------------------------


@pytest.mark.parametrize(
    "fn, args, kwargs",
    [
        (list_collections, (), {}),
        (list_documents, ("col-1",), {}),
        (get_document_metadata, ("col-1", "doc-1"), {}),
        (get_collection_metadata, ("col-1",), {}),
        (read_document, ("col-1", "doc-1"), {}),
        (read_document_outline, ("col-1", "doc-1"), {}),
        (read_document_section, ("col-1", "doc-1"), {"section_path": "1/2"}),
        (read_document_chunk, ("col-1", "doc-1", "chunk-1"), {}),
    ],
)
def test_primitive_raises_not_implemented(fn, args, kwargs):
    """Stub bodies must raise ``NotImplementedError`` — guards against any
    accidental partial implementation slipping in via this stub PR.
    """

    async def _run():
        return await fn(*args, **kwargs)

    with pytest.raises(NotImplementedError, match="D10.c"):
        asyncio.run(_run())


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
