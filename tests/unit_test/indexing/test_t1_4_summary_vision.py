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


def test_vision_derive_consumes_jsonl_descriptor_with_image_path():
    """Wave 5 P2 chunk 4: when the parser writes the new descriptor
    (`vision/source.jsonl`) with an ``image_path`` per record, vision
    derive must (a) load each image's bytes from the path, (b) hand
    them to the embedder via ``image_bytes=``, (c) still emit one
    manifest line per record."""

    store = InMemoryObjectStore()

    # Stage two image blobs at canonical paths.
    image_a_path = "collections/col-1/documents/doc-vision-jsonl/derived/parse_v1/vision/images/img-a.jpg"
    image_b_path = "collections/col-1/documents/doc-vision-jsonl/derived/parse_v1/vision/images/img-b.png"
    write_atomic(store, image_a_path, b"\xff\xd8\xff\xe0fake-jpeg")
    write_atomic(store, image_b_path, b"\x89PNGfake-png")

    # Stage the JSONL descriptor pointing at the blobs.
    descriptor_path = "collections/col-1/documents/doc-vision-jsonl/derived/parse_v1/vision/source.jsonl"
    descriptor_lines = [
        json.dumps(
            {
                "image_id": "img-a",
                "image_path": image_a_path,
                "mime_type": "image/jpeg",
                "alt_text": "banner",
                "page_idx": 0,
                "bbox": [0, 0, 100, 100],
            }
        ),
        json.dumps(
            {
                "image_id": "img-b",
                "image_path": image_b_path,
                "mime_type": "image/png",
                "alt_text": "",
                "page_idx": None,
                "bbox": None,
            }
        ),
    ]
    write_atomic(store, descriptor_path, ("\n".join(descriptor_lines) + "\n").encode("utf-8"))

    # Capture what the embedder sees so we can assert image bytes were
    # actually loaded and forwarded.
    seen: list[tuple[str, str, bytes | None]] = []

    def _capturing_embedder(image_id: str, alt_text: str, image_bytes: bytes | None = None) -> list[float]:
        seen.append((image_id, alt_text, image_bytes))
        return [0.1, 0.2, 0.3]

    modality = VisionModality(backend=InMemoryVisionBackend(), store=store, embedder=_capturing_embedder)
    result = asyncio.run(
        modality.derive(
            document_id="doc-vision-jsonl",
            parse_version="v1",
            source_path=descriptor_path,
        )
    )

    assert result.derived_artifact_path.endswith("vision/manifest.jsonl")
    assert len(seen) == 2
    a_seen = next(rec for rec in seen if rec[0] == "img-a")
    b_seen = next(rec for rec in seen if rec[0] == "img-b")
    assert a_seen[2] == b"\xff\xd8\xff\xe0fake-jpeg"
    assert b_seen[2] == b"\x89PNGfake-png"


def test_vision_derive_falls_back_when_image_blob_missing():
    """If the descriptor references an ``image_path`` that does not
    exist (parser write was interrupted / object store eviction), the
    embedder still runs with ``image_bytes=None`` so the partial-derive
    state surfaces a manifest the operator can inspect rather than the
    whole derive cycle exploding."""

    store = InMemoryObjectStore()
    descriptor_path = "collections/col-1/documents/doc-broken/derived/parse_v1/vision/source.jsonl"
    record = {
        "image_id": "img-missing",
        "image_path": "collections/col-1/documents/doc-broken/derived/parse_v1/vision/images/img-missing.jpg",
        "mime_type": "image/jpeg",
        "alt_text": "stale",
        "page_idx": None,
        "bbox": None,
    }
    write_atomic(store, descriptor_path, (json.dumps(record) + "\n").encode("utf-8"))

    seen: list[bytes | None] = []

    def _capturing_embedder(image_id: str, alt_text: str, image_bytes: bytes | None = None) -> list[float]:
        seen.append(image_bytes)
        return [0.0]

    modality = VisionModality(backend=InMemoryVisionBackend(), store=store, embedder=_capturing_embedder)
    asyncio.run(
        modality.derive(
            document_id="doc-broken",
            parse_version="v1",
            source_path=descriptor_path,
        )
    )

    assert seen == [None], "missing image blob → embedder receives image_bytes=None fallback"


def test_vision_derive_legacy_simulator_format_still_works():
    """Backward-compat: a legacy single-JSON-array source file (the
    pre-Wave-5 simulator shape) should keep working. ``image_bytes``
    is None for every record (no ``image_path`` field) and the
    placeholder embedder produces a deterministic vector."""

    store = InMemoryObjectStore()
    images = [
        {"image_id": "img-1", "alt_text": "x", "page_idx": 0, "bbox": None},
        {"image_id": "img-2", "alt_text": "y", "page_idx": 1, "bbox": None},
    ]
    source_path = _seed_vision_source(store, document_id="doc-legacy", payload=images)
    modality = VisionModality(backend=InMemoryVisionBackend(), store=store)

    result = asyncio.run(
        modality.derive(
            document_id="doc-legacy",
            parse_version="v1",
            source_path=source_path,
        )
    )
    body = store.get(result.derived_artifact_path).read().decode("utf-8")
    lines = [json.loads(line) for line in body.splitlines() if line.strip()]
    assert {entry["image_id"] for entry in lines} == {"img-1", "img-2"}
    for entry in lines:
        assert entry["embedding"]


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
