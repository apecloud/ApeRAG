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

"""T1.4 Summary + Vision modality contract tests.

Locks the §K Wave 1 acceptance gates for the summary + vision lanes:

- Summary: derive runs the LLM placeholder over the parser markdown
  and persists ``summary.json`` atomically; sync replaces backend
  state with the §D.1 contract; idempotent on double call.
- Vision: derive reads a synthetic image-records source, persists
  ``vision/manifest.jsonl`` atomically; sync replaces backend state;
  idempotent on double call.
"""

from __future__ import annotations

import asyncio
import json

from aperag.indexing import (
    InMemoryObjectStore,
    InMemorySummaryBackend,
    InMemoryVisionBackend,
    Modality,
    SummaryModality,
    VisionModality,
    parse_document,
    write_atomic,
)

SAMPLE_MARKDOWN = (
    "# Project Beacon\n\n"
    "Beacon is a sample document used by the indexing simulator tests.\n\n"
    "## Architecture\n\n"
    "Beacon has a controller and a worker pool that share a Redis queue.\n"
)


# ---------------------------------------------------------------------
# Summary modality
# ---------------------------------------------------------------------


def _seed_parser(store, *, document_id: str = "doc-beacon"):
    return parse_document(
        store=store,
        collection_id="col-1",
        document_id=document_id,
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
    )


def test_summary_derive_persists_summary_json_atomically():
    store = InMemoryObjectStore()
    parsed = _seed_parser(store)
    modality = SummaryModality(backend=InMemorySummaryBackend(), store=store)

    result = asyncio.run(
        modality.derive(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            source_path=parsed.markdown_path,
        )
    )
    assert result.derived_artifact_path.endswith("summary.json")
    assert store.obj_exists(result.derived_artifact_path)

    body = store.get(result.derived_artifact_path).read().decode("utf-8")
    record = json.loads(body)
    assert record["summary_text"], "summarizer must produce non-empty summary"
    assert record["embedding"], "summarizer must produce a paired embedding"


def test_summary_sync_is_replace_idempotent_on_double_call():
    store = InMemoryObjectStore()
    parsed = _seed_parser(store)
    backend = InMemorySummaryBackend()
    modality = SummaryModality(backend=backend, store=store)

    derived = asyncio.run(
        modality.derive(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            source_path=parsed.markdown_path,
        )
    )
    asyncio.run(
        modality.sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=derived.derived_artifact_path,
        )
    )
    first_state = backend.points_for_document("doc-beacon", parsed.parse_version)
    assert len(first_state) == 1, "summary backend should hold exactly one point per (doc, parse_version)"

    asyncio.run(
        modality.sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=derived.derived_artifact_path,
        )
    )
    second_state = backend.points_for_document("doc-beacon", parsed.parse_version)
    assert second_state == first_state, "summary sync must be byte-equivalent on second call (§D.4)"


def test_summary_modality_enum_is_summary():
    assert SummaryModality.modality is Modality.SUMMARY


def test_summary_payload_carries_modality_discriminator():
    store = InMemoryObjectStore()
    parsed = _seed_parser(store)
    backend = InMemorySummaryBackend()
    modality = SummaryModality(backend=backend, store=store)

    derived = asyncio.run(
        modality.derive(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            source_path=parsed.markdown_path,
        )
    )
    asyncio.run(
        modality.sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=derived.derived_artifact_path,
        )
    )
    points = backend.points_for_document("doc-beacon", parsed.parse_version)
    for point in points:
        assert point["payload"]["modality"] == Modality.SUMMARY.value


def test_summary_sync_no_op_on_missing_summary_artifact():
    """§C.7 reschedule semantic — summary.sync against a missing
    summary.json must be a silent no-op so the orchestrator can
    re-queue the (doc, parse_version) for derive."""
    store = InMemoryObjectStore()
    backend = InMemorySummaryBackend()
    modality = SummaryModality(backend=backend, store=store)
    asyncio.run(
        modality.sync(
            document_id="doc-x",
            parse_version="abcdef0123456789",
            derived_artifact_path="collections/c/documents/doc-x/derived/parse_abcdef0123456789/summary.json",
        )
    )
    assert backend.points_for_document("doc-x") == []


# ---------------------------------------------------------------------
# Vision modality
# ---------------------------------------------------------------------


def _seed_vision_source(store, *, document_id: str, payload: list[dict]) -> str:
    """Stage an in-memory image-records JSON file the simulator reads."""
    source_path = f"collections/col-1/documents/{document_id}/source/images.json"
    write_atomic(store, source_path, json.dumps(payload).encode("utf-8"))
    return source_path


def test_vision_derive_persists_manifest_atomically():
    store = InMemoryObjectStore()
    images = [
        {"image_id": "img-001", "alt_text": "header banner", "page_idx": 0, "bbox": [10, 20, 100, 120]},
        {"image_id": "img-002", "alt_text": "diagram", "page_idx": 2, "bbox": None},
    ]
    source_path = _seed_vision_source(store, document_id="doc-vision", payload=images)
    modality = VisionModality(backend=InMemoryVisionBackend(), store=store)

    result = asyncio.run(
        modality.derive(
            document_id="doc-vision",
            parse_version="abcdef0123456789",
            source_path=source_path,
        )
    )
    assert result.derived_artifact_path.endswith("vision/manifest.jsonl")
    assert store.obj_exists(result.derived_artifact_path)

    body = store.get(result.derived_artifact_path).read().decode("utf-8")
    lines = [json.loads(line) for line in body.splitlines() if line.strip()]
    assert len(lines) == 2
    assert {entry["image_id"] for entry in lines} == {"img-001", "img-002"}
    for entry in lines:
        assert entry["embedding"], "vision derive must compute an embedding per image"


def test_vision_sync_is_replace_idempotent_on_double_call():
    store = InMemoryObjectStore()
    images = [
        {"image_id": "img-001", "alt_text": "banner", "page_idx": 0, "bbox": None},
        {"image_id": "img-002", "alt_text": "diagram", "page_idx": 1, "bbox": None},
    ]
    source_path = _seed_vision_source(store, document_id="doc-vision", payload=images)
    backend = InMemoryVisionBackend()
    modality = VisionModality(backend=backend, store=store)

    derived = asyncio.run(
        modality.derive(
            document_id="doc-vision",
            parse_version="abcdef0123456789",
            source_path=source_path,
        )
    )
    asyncio.run(
        modality.sync(
            document_id="doc-vision",
            parse_version="abcdef0123456789",
            derived_artifact_path=derived.derived_artifact_path,
        )
    )
    first_state = backend.points_for_document("doc-vision", "abcdef0123456789")
    assert len(first_state) == 2, "two images → two backend points"

    asyncio.run(
        modality.sync(
            document_id="doc-vision",
            parse_version="abcdef0123456789",
            derived_artifact_path=derived.derived_artifact_path,
        )
    )
    second_state = backend.points_for_document("doc-vision", "abcdef0123456789")
    assert second_state == first_state, "vision sync must be byte-equivalent on second call (§D.4)"


def test_vision_modality_enum_is_vision():
    assert VisionModality.modality is Modality.VISION


def test_vision_payload_carries_modality_discriminator_and_image_id():
    store = InMemoryObjectStore()
    images = [{"image_id": "img-only", "alt_text": "only", "page_idx": None, "bbox": None}]
    source_path = _seed_vision_source(store, document_id="doc-vision", payload=images)
    backend = InMemoryVisionBackend()
    modality = VisionModality(backend=backend, store=store)

    derived = asyncio.run(
        modality.derive(
            document_id="doc-vision",
            parse_version="abcdef0123456789",
            source_path=source_path,
        )
    )
    asyncio.run(
        modality.sync(
            document_id="doc-vision",
            parse_version="abcdef0123456789",
            derived_artifact_path=derived.derived_artifact_path,
        )
    )
    points = backend.points_for_document("doc-vision", "abcdef0123456789")
    assert len(points) == 1
    point = points[0]
    assert point["payload"]["modality"] == Modality.VISION.value
    assert point["payload"]["image_id"] == "img-only"


def test_vision_sync_no_op_on_missing_manifest():
    store = InMemoryObjectStore()
    backend = InMemoryVisionBackend()
    asyncio.run(
        VisionModality(backend=backend, store=store).sync(
            document_id="doc-x",
            parse_version="abcdef0123456789",
            derived_artifact_path=(
                "collections/c/documents/doc-x/derived/parse_abcdef0123456789/vision/manifest.jsonl"
            ),
        )
    )
    assert backend.points_for_document("doc-x") == []
