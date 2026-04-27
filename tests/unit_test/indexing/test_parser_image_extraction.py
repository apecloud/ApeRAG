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

"""Unit tests for Wave 5 P2 chunk 2 — parser image extraction.

Pin the §G.2.5.1 spec item 2 contract: when DocParser produces
``AssetBinPart`` payloads, the parser writes each image blob to
``derived/parse_<v>/vision/images/<image_id>.<ext>`` and lands a
``vision/source.jsonl`` descriptor. The vision worker (chunk 4)
consumes the descriptor instead of the T1 simulator's synthetic
``images.json`` companion.

We don't exercise the full DocParser chain here (that needs MinerU /
MarkItDown deps); instead we monkeypatch
:func:`aperag.indexing.parser._docparser_extract_markdown` to return a
controlled ``(markdown, image_assets)`` tuple, then assert
:func:`parse_document` writes the right files at the right paths and
the right :class:`ParseResult` fields.
"""

from __future__ import annotations

import json

import pytest

from aperag.indexing import parser as parser_module
from aperag.indexing.object_store import InMemoryObjectStore, derived_dir
from aperag.indexing.parser import (
    _vision_image_extension,
    _VisionImageAsset,
    parse_document,
)


def test_vision_image_extension_maps_known_mime_types():
    assert _vision_image_extension("image/jpeg") == "jpg"
    assert _vision_image_extension("image/jpg") == "jpg"
    assert _vision_image_extension("image/png") == "png"
    assert _vision_image_extension("image/webp") == "webp"
    assert _vision_image_extension("image/gif") == "gif"
    assert _vision_image_extension("image/svg+xml") == "svg"


def test_vision_image_extension_strips_charset_param():
    """``image/jpeg; charset=binary`` and similar provider-emitted
    MIME parameters should not break the extension lookup."""
    assert _vision_image_extension("image/jpeg; charset=binary") == "jpg"
    assert _vision_image_extension("IMAGE/PNG") == "png"


def test_vision_image_extension_falls_back_to_bin_for_unknown():
    assert _vision_image_extension(None) == "bin"
    assert _vision_image_extension("") == "bin"
    assert _vision_image_extension("application/octet-stream") == "bin"
    # Newer formats not yet in the lookup table — vision worker will
    # use ``imghdr`` on the bytes themselves at embed time, so the
    # filename is informational only.
    assert _vision_image_extension("image/heic") == "bin"


def test_parse_document_persists_image_blobs_and_descriptor(monkeypatch: pytest.MonkeyPatch):
    """End-to-end: parser hands DocParser a PDF-shape input, gets back
    markdown + 2 image assets; parser persists each blob at
    ``vision/images/<image_id>.<ext>`` and a descriptor at
    ``vision/source.jsonl``."""

    img_a = _VisionImageAsset(
        image_id="aaa111",
        data=b"\xff\xd8\xff\xe0fake-jpeg-bytes",
        mime_type="image/jpeg",
        alt_text="Figure 1: architecture diagram",
        page_idx=2,
        bbox=[0.1, 0.2, 0.3, 0.4],
    )
    img_b = _VisionImageAsset(
        image_id="bbb222",
        data=b"\x89PNGfake-png-bytes",
        mime_type="image/png",
        alt_text="",
        page_idx=None,
        bbox=None,
    )

    def _fake_docparser(*, source_bytes: bytes, extension: str, parser_config):
        return ("# Title\n\nbody", [img_a, img_b])

    monkeypatch.setattr(parser_module, "_docparser_extract_markdown", _fake_docparser)

    store = InMemoryObjectStore()
    result = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-1",
        source_bytes=b"%PDF-1.4 fake",
        source_filename="report.pdf",
    )

    assert result.vision_image_count == 2
    expected_dir = derived_dir("col-1", "doc-1", result.parse_version)
    expected_descriptor = f"{expected_dir}/vision/source.jsonl"
    assert result.vision_source_path == expected_descriptor
    assert store.obj_exists(expected_descriptor)

    image_a_path = f"{expected_dir}/vision/images/aaa111.jpg"
    image_b_path = f"{expected_dir}/vision/images/bbb222.png"
    assert store.obj_exists(image_a_path)
    assert store.obj_exists(image_b_path)
    assert store.get(image_a_path).read() == img_a.data
    assert store.get(image_b_path).read() == img_b.data

    descriptor_body = store.get(expected_descriptor).read().decode("utf-8")
    rows = [json.loads(line) for line in descriptor_body.splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0] == {
        "image_id": "aaa111",
        "image_path": image_a_path,
        "mime_type": "image/jpeg",
        "alt_text": "Figure 1: architecture diagram",
        "page_idx": 2,
        "bbox": [0.1, 0.2, 0.3, 0.4],
    }
    assert rows[1] == {
        "image_id": "bbb222",
        "image_path": image_b_path,
        "mime_type": "image/png",
        "alt_text": "",
        "page_idx": None,
        "bbox": None,
    }


def test_parse_document_skips_vision_artefacts_when_no_images(monkeypatch: pytest.MonkeyPatch):
    """A parsed document with no image assets should not land any
    ``vision/`` artefacts and ``vision_source_path`` must stay empty
    so the orchestrator's vision dispatch can short-circuit cleanly.
    """

    def _fake_docparser(*, source_bytes: bytes, extension: str, parser_config):
        return ("# Title\n\nbody", [])

    monkeypatch.setattr(parser_module, "_docparser_extract_markdown", _fake_docparser)

    store = InMemoryObjectStore()
    result = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-no-images",
        source_bytes=b"fake doc",
        source_filename="doc.docx",
    )

    assert result.vision_image_count == 0
    assert result.vision_source_path == ""
    expected_dir = derived_dir("col-1", "doc-no-images", result.parse_version)
    # No vision assets should have been written.
    assert not store.obj_exists(f"{expected_dir}/vision/source.jsonl")


def test_parse_document_simulator_path_emits_no_vision_artefacts():
    """The text-only simulator path (markdown / txt) does not invoke
    DocParser so no ``AssetBinPart`` extraction happens — the
    ``ParseResult`` defaults must keep vision fields empty so existing
    pre-Wave-5 callers see no behaviour change."""

    store = InMemoryObjectStore()
    result = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-md",
        source_bytes=b"# Header\n\nBody.",
        source_filename="notes.md",
    )

    assert result.vision_image_count == 0
    assert result.vision_source_path == ""
    expected_dir = derived_dir("col-1", "doc-md", result.parse_version)
    assert not store.obj_exists(f"{expected_dir}/vision/source.jsonl")


def test_parse_document_image_paths_use_mime_extension(monkeypatch: pytest.MonkeyPatch):
    """The on-disk filename extension follows the asset's ``mime_type``
    so operators can spot-check the bytes directly without renaming."""

    img = _VisionImageAsset(
        image_id="ccc333",
        data=b"GIF89a-fake",
        mime_type="image/gif",
        alt_text="",
        page_idx=None,
        bbox=None,
    )

    def _fake_docparser(*, source_bytes: bytes, extension: str, parser_config):
        return ("# t", [img])

    monkeypatch.setattr(parser_module, "_docparser_extract_markdown", _fake_docparser)

    store = InMemoryObjectStore()
    result = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-gif",
        source_bytes=b"fake",
        source_filename="anim.gif",
    )

    expected_dir = derived_dir("col-1", "doc-gif", result.parse_version)
    assert store.obj_exists(f"{expected_dir}/vision/images/ccc333.gif")


def test_parse_document_falls_back_to_bin_for_unknown_mime(monkeypatch: pytest.MonkeyPatch):
    """Unknown MIMEs (e.g., ``image/heic``) fall back to ``.bin`` so
    the asset is still persisted — vision worker uses ``imghdr`` on
    the bytes at embed time and tolerates the generic extension."""

    img = _VisionImageAsset(
        image_id="ddd444",
        data=b"heic-fake",
        mime_type="image/heic",
        alt_text="",
        page_idx=None,
        bbox=None,
    )

    def _fake_docparser(*, source_bytes: bytes, extension: str, parser_config):
        return ("# t", [img])

    monkeypatch.setattr(parser_module, "_docparser_extract_markdown", _fake_docparser)

    store = InMemoryObjectStore()
    result = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-heic",
        source_bytes=b"fake",
        source_filename="phone.heic",
    )

    expected_dir = derived_dir("col-1", "doc-heic", result.parse_version)
    assert store.obj_exists(f"{expected_dir}/vision/images/ddd444.bin")


def test_parse_document_image_only_input_still_lands_descriptor(monkeypatch: pytest.MonkeyPatch):
    """An image-only input (e.g., a single PNG upload) produces no
    markdown but should still land its image asset + descriptor so
    the vision modality has bytes to embed. ``markdown.md`` is empty
    on this path; ``vision_source_path`` is populated."""

    img = _VisionImageAsset(
        image_id="eee555",
        data=b"single-png-bytes",
        mime_type="image/png",
        alt_text="cover photo",
        page_idx=None,
        bbox=None,
    )

    def _fake_docparser(*, source_bytes: bytes, extension: str, parser_config):
        # No markdown emitted (image-only input) but one asset landed.
        return ("", [img])

    monkeypatch.setattr(parser_module, "_docparser_extract_markdown", _fake_docparser)

    store = InMemoryObjectStore()
    result = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-single-img",
        source_bytes=b"png-bytes",
        source_filename="cover.png",
    )

    assert result.vision_image_count == 1
    expected_dir = derived_dir("col-1", "doc-single-img", result.parse_version)
    assert store.obj_exists(f"{expected_dir}/vision/images/eee555.png")
    assert store.obj_exists(f"{expected_dir}/vision/source.jsonl")
    # Markdown still written (empty) so the artifact contract holds.
    assert store.obj_exists(result.markdown_path)
    assert store.get(result.markdown_path).read() == b""
