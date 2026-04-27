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

"""Wave 6 #34 — chunk_id schema canonical unification contract tests.

Pins the single canonical naming for the Qdrant point identifier
across the three modalities that share the
``{delete_by_filter, upsert_point}`` backend protocol (vector /
summary / vision):

- The ``upsert_point`` keyword arg is ``point_id`` (singular canonical
  name). The pre-Wave-6 dual API (``chunk_id | point_id``) was a
  transition shim; vector callers now pass ``point_id=chunk["chunk_id"]``
  while summary/vision pass composite ids
  (``summary:<doc>:<v>`` / ``vision:<doc>:<v>:<image_id>``).
- The ``chunk_id`` field stays on the **payload** for vector — that is
  what hybrid retrieval reads to dedup against fulltext (per §C.6).
  Summary/vision payloads do **not** carry ``chunk_id`` — their points
  are not chunks.
- The shared adapter (``_QdrantPointBackend``) does **not** inject
  ``chunk_id`` into payload anymore; each modality controls its own
  payload schema.
"""

from __future__ import annotations

import inspect
import json

import pytest

from aperag.indexing.summary import SummaryBackend
from aperag.indexing.vector import InMemoryVectorBackend, VectorBackend
from aperag.indexing.vision import VisionBackend


def test_vector_backend_protocol_uses_point_id_keyword():
    """The :class:`VectorBackend` ``upsert_point`` accepts ``point_id``,
    not the pre-Wave-6 ``chunk_id`` keyword. This pins the canonical
    name for vector at the Protocol surface."""
    sig = inspect.signature(VectorBackend.upsert_point)
    params = sig.parameters
    assert "point_id" in params, "VectorBackend.upsert_point must accept point_id"
    assert "chunk_id" not in params, "VectorBackend.upsert_point must NOT accept chunk_id (Wave 6 #34 canonical rename)"


def test_summary_and_vision_backend_protocols_use_point_id_keyword():
    """Summary + vision already used ``point_id`` pre-Wave-6; pin that
    they remain on the canonical name (cross-modality alignment)."""
    for backend_cls in (SummaryBackend, VisionBackend):
        sig = inspect.signature(backend_cls.upsert_point)
        params = sig.parameters
        assert "point_id" in params, f"{backend_cls.__name__}.upsert_point must accept point_id"
        assert "chunk_id" not in params, f"{backend_cls.__name__}.upsert_point must NOT accept chunk_id"


def test_in_memory_vector_backend_round_trips_point_id_and_payload_chunk_id():
    """Vector worker passes parser ``chunk_id`` as ``point_id`` while
    keeping ``chunk_id`` in payload for hybrid-dedup. Pin both the
    record key and the payload field."""
    backend = InMemoryVectorBackend()
    backend.upsert_point(
        point_id="abc:0001",
        embedding=[0.0] * 4,
        payload={
            "document_id": "d1",
            "parse_version": "v1",
            "modality": "vector",
            "chunk_id": "abc:0001",
        },
    )
    points = backend.points_for_document("d1", "v1")
    assert len(points) == 1
    record = points[0]
    assert record["point_id"] == "abc:0001"
    assert record["payload"]["chunk_id"] == "abc:0001", (
        "vector worker must keep chunk_id in payload for hybrid dedup with fulltext (§C.6)"
    )


def test_in_memory_vector_backend_rejects_legacy_chunk_id_keyword():
    """Pin that the legacy ``chunk_id=`` keyword is gone from the
    Wave 6-canonical API — callers must pass ``point_id=``. This
    catches accidental regression to the pre-Wave-6 dual API."""
    backend = InMemoryVectorBackend()
    with pytest.raises(TypeError):
        backend.upsert_point(
            chunk_id="abc:0001",  # type: ignore[call-arg]
            embedding=[0.0] * 4,
            payload={"document_id": "d1", "parse_version": "v1"},
        )


def test_qdrant_point_backend_adapter_uses_single_point_id_keyword():
    """The shared production adapter (``_QdrantPointBackend``) now
    exposes a single ``point_id`` parameter. Pre-Wave-6 the adapter
    accepted ``chunk_id | point_id`` polymorphically as a transition
    shim; that polymorphism is removed."""
    from aperag.indexing.worker_factory import _QdrantPointBackend

    sig = inspect.signature(_QdrantPointBackend.upsert_point)
    params = sig.parameters
    assert "point_id" in params
    assert "chunk_id" not in params


def test_qdrant_point_backend_adapter_does_not_inject_chunk_id_into_payload():
    """The pre-Wave-6 adapter would ``setdefault("chunk_id", identifier)``
    into the caller's payload — leaking a misleading ``chunk_id`` field
    into summary / vision payloads (their points are not chunks). Pin
    that the Wave 6 adapter forwards the caller's payload verbatim and
    does not synthesise any field."""
    from aperag.indexing.worker_factory import _QdrantPointBackend

    captured: dict = {}

    class _StubConnector:
        def upsert(self, points):
            # Capture the payload that the adapter forwards to the
            # underlying connector so we can assert the adapter did
            # not inject anything.
            captured["points"] = list(points)

        def delete_by_filter(self, flt):  # noqa: D401
            pass

    backend = _QdrantPointBackend(connector=_StubConnector())
    backend.upsert_point(
        point_id="summary:d1:v1",
        embedding=[0.0] * 4,
        payload={
            "document_id": "d1",
            "parse_version": "v1",
            "modality": "summary",
            "summary_text": "x",
        },
    )
    forwarded_payload = captured["points"][0].payload
    assert "chunk_id" not in forwarded_payload, (
        "adapter must not inject chunk_id into summary payload — summary points are not chunks (Wave 6 #34 cleanup)"
    )
    # Caller payload preserved verbatim.
    assert forwarded_payload["modality"] == "summary"
    assert forwarded_payload["summary_text"] == "x"


def test_parser_chunks_jsonl_field_is_chunk_id():
    """The parser-emitted chunks.jsonl schema is the source of truth
    for vector + fulltext: every chunk record carries a ``chunk_id``
    field. Pin the field name on a synthetic chunks.jsonl payload —
    the canonical naming is ``chunk_id`` at the chunk-record level
    (one chunk → one Qdrant point), and that record's ``chunk_id``
    is what the worker passes as ``point_id`` at the backend layer."""
    chunks = [
        {
            "chunk_id": "h:0000",
            "text": "alpha",
            "section_path": None,
            "heading_anchor": None,
            "page_idx": None,
        },
        {
            "chunk_id": "h:0001",
            "text": "bravo",
            "section_path": None,
            "heading_anchor": None,
            "page_idx": None,
        },
    ]
    serialised = "\n".join(json.dumps(c) for c in chunks).encode("utf-8")
    decoded = [json.loads(line) for line in serialised.decode("utf-8").splitlines() if line]
    assert all("chunk_id" in c for c in decoded)
    assert {c["chunk_id"] for c in decoded} == {"h:0000", "h:0001"}
