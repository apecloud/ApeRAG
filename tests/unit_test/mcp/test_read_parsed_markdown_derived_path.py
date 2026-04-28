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

"""Pin the MCP markdown-resolution contract.

``read_parsed_markdown`` must read from the canonical Wave 3+
location (``dirname(DocumentIndex.derived_artifact_path)/markdown.md``)
*first* and only fall back to the legacy ``{base}/parsed.md`` blob
when the derived row isn't available.

Without this layered resolution, MCP returns empty markdown for any
document whose parser bypassed the legacy dual-writer — earayu2 bug
msg=dec8bcff: "parsed_markdown is empty for all documents" even
though the docs were fully indexed and visible in the FE preview.

The FE goes through ``DocumentService.get_document_preview`` →
``_markdown_path_from_derived_artifact`` (document_service.py:120,
1232). The MCP layer now mirrors that exactly so both surfaces
return the same content for the same document.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aperag.mcp.tools import _parsed_doc


@dataclass
class _FakeDocument:
    """Mirrors :meth:`Document.object_store_base_path` exactly so the
    fallback path string matches what production would produce.
    Production uses ``user-{user}/{collection_id}/{id}``."""

    id: str
    user: str = "1"
    collection_id: str = "col-1"

    def object_store_base_path(self) -> str:
        return f"user-{self.user}/{self.collection_id}/{self.id}"


@pytest.mark.asyncio
async def test_read_parsed_markdown_prefers_derived_artifact_path(monkeypatch):
    """When a serving VECTOR DocumentIndex row exists, the markdown
    must be read from ``dirname(derived_artifact_path)/markdown.md``
    — the per-parse-version location the new parser writes to (and
    the only location post-Wave 3 collections have content at)."""

    derived_path = "user-1/col-1/doc-1/derived/parse_abc123/manifest.json"
    expected_md_path = "user-1/col-1/doc-1/derived/parse_abc123/markdown.md"

    async def _fake_resolve(document_id):
        assert document_id == "doc-1"
        return f"{expected_md_path}"

    reads: list[str] = []

    async def _fake_read(path):
        reads.append(path)
        if path == expected_md_path:
            return "# Hello from derived path"
        return ""

    monkeypatch.setattr(_parsed_doc, "_resolve_serving_markdown_path", _fake_resolve)
    monkeypatch.setattr(_parsed_doc, "_read_object_store_text", _fake_read)

    out = await _parsed_doc.read_parsed_markdown(_FakeDocument(id="doc-1"))
    assert out == "# Hello from derived path"
    assert reads == [expected_md_path], (
        "must hit the derived path first, never falling through to legacy when content is found"
    )
    assert "/parsed.md" not in derived_path  # sanity: derived path is NOT the legacy path


@pytest.mark.asyncio
async def test_read_parsed_markdown_falls_back_to_legacy_when_no_serving_index(monkeypatch):
    """Documents created before the derived/ layout, or whose
    serving-VECTOR row hasn't materialized yet, still resolve via
    the legacy ``{base}/parsed.md`` writer (document_parser.py:181
    dual-writes it). Removing the fallback would regress old
    collections."""

    async def _fake_resolve(document_id):
        return None  # no serving VECTOR index yet

    reads: list[str] = []

    async def _fake_read(path):
        reads.append(path)
        if path == "user-1/col-1/doc-1/parsed.md":
            return "# Legacy content"
        return ""

    monkeypatch.setattr(_parsed_doc, "_resolve_serving_markdown_path", _fake_resolve)
    monkeypatch.setattr(_parsed_doc, "_read_object_store_text", _fake_read)

    out = await _parsed_doc.read_parsed_markdown(_FakeDocument(id="doc-1"))
    assert out == "# Legacy content"
    assert reads == ["user-1/col-1/doc-1/parsed.md"], (
        "with no derived path resolved, must read straight from the legacy location"
    )


@pytest.mark.asyncio
async def test_read_parsed_markdown_falls_back_to_legacy_when_derived_blob_missing(monkeypatch):
    """A serving row exists but the derived blob itself is
    missing/unreadable (e.g. object-store transient failure or a
    half-migrated state). The legacy path is the safety net."""

    derived_md_path = "user-1/col-1/doc-1/derived/parse_abc123/markdown.md"
    legacy_path = "user-1/col-1/doc-1/parsed.md"

    async def _fake_resolve(document_id):
        return derived_md_path

    reads: list[str] = []

    async def _fake_read(path):
        reads.append(path)
        if path == legacy_path:
            return "# Legacy content (derived blob 404'd)"
        return ""  # derived path read fails

    monkeypatch.setattr(_parsed_doc, "_resolve_serving_markdown_path", _fake_resolve)
    monkeypatch.setattr(_parsed_doc, "_read_object_store_text", _fake_read)

    out = await _parsed_doc.read_parsed_markdown(_FakeDocument(id="doc-1"))
    assert out == "# Legacy content (derived blob 404'd)"
    assert reads == [derived_md_path, legacy_path], (
        "derived must be tried first, then legacy as fallback when derived returns empty"
    )


@pytest.mark.asyncio
async def test_read_parsed_markdown_returns_empty_when_both_paths_miss(monkeypatch):
    """Neither path has content — return ``""`` so callers (and the
    LLM-facing API surface) get a deterministic empty string instead
    of crashing or surfacing a stack trace. Matches the pre-fix
    contract."""

    async def _fake_resolve(document_id):
        return "user-1/col-1/doc-1/derived/parse_abc/markdown.md"

    async def _fake_read(path):
        return ""  # everything is empty

    monkeypatch.setattr(_parsed_doc, "_resolve_serving_markdown_path", _fake_resolve)
    monkeypatch.setattr(_parsed_doc, "_read_object_store_text", _fake_read)

    out = await _parsed_doc.read_parsed_markdown(_FakeDocument(id="doc-1"))
    assert out == ""
