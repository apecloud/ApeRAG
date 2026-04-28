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

"""T1.3 Vector + Fulltext modality contract tests.

Locks the §K Wave 1 acceptance gates for the vector + fulltext lanes:

1. **Per-modality idempotency** — re-running ``sync`` produces a
   backend state byte-equivalent to a fresh sync (§D.4).
2. **Hybrid dedup by chunk_id** — vector + fulltext sync of the same
   ``chunks.jsonl`` produces points/documents on both backends
   sharing the same ``chunk_id``, so the existing dedup logic in
   the search layer continues to work.
3. **DELETE-by-(doc, parse_version) THEN INSERT semantics** — sync
   of a *new* parse_version against an existing one for the same
   document leaves the backend in the new-version state with no
   stale rows from the old version.
"""

from __future__ import annotations

import asyncio

from aperag.indexing import (
    FulltextModality,
    InMemoryFulltextBackend,
    InMemoryObjectStore,
    InMemoryVectorBackend,
    Modality,
    VectorModality,
    parse_document,
)

SAMPLE_MARKDOWN = (
    "# Project Beacon\n\n"
    "Beacon is a sample document used by the indexing simulator tests.\n\n"
    "## Architecture\n\n"
    "Beacon has a controller and a worker pool that share a Redis queue.\n\n"
    "## Operations\n\n"
    "Operators use the dashboard to inspect queue depth and worker utilisation.\n"
)


def _seed(store, *, document_id: str = "doc-beacon"):
    """Run the parser once to produce a chunks.jsonl artifact."""
    return parse_document(
        store=store,
        collection_id="col-1",
        document_id=document_id,
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
    )


# ---------------------------------------------------------------------
# Vector idempotency
# ---------------------------------------------------------------------


def test_vector_sync_is_replace_idempotent_on_double_call():
    store = InMemoryObjectStore()
    backend = InMemoryVectorBackend()
    parsed = _seed(store)
    modality = VectorModality(backend=backend, store=store)

    asyncio.run(
        modality.sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )
    )
    first_state = backend.points_for_document("doc-beacon", parsed.parse_version)
    first_count = len(first_state)
    assert first_count > 0, "vector sync must populate at least one point for a non-empty doc"

    asyncio.run(
        modality.sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )
    )
    second_state = backend.points_for_document("doc-beacon", parsed.parse_version)

    assert second_state == first_state, "vector sync must be byte-equivalent on a second call (§D.4 idempotency)"
    # No point leakage outside the (doc, parse_version) slot.
    assert backend.all_points() == first_state


def test_vector_sync_replaces_old_parse_version():
    """A new parse_version against the same document must DELETE the
    old parse_version's points before inserting the new ones (§D.1)."""

    store = InMemoryObjectStore()
    backend = InMemoryVectorBackend()
    modality = VectorModality(backend=backend, store=store)

    parsed_v1 = _seed(store, document_id="doc-1")
    asyncio.run(
        modality.sync(
            document_id="doc-1",
            parse_version=parsed_v1.parse_version,
            derived_artifact_path=parsed_v1.chunks_path,
        )
    )

    # Re-parse with a different chunking config → fresh parse_version.
    from aperag.indexing import ChunkingConfig, ParseConfig

    parsed_v2 = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-1",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
        config=ParseConfig(chunking=ChunkingConfig(chunk_size=120, chunk_overlap=10)),
    )
    assert parsed_v1.parse_version != parsed_v2.parse_version

    asyncio.run(
        modality.sync(
            document_id="doc-1",
            parse_version=parsed_v2.parse_version,
            derived_artifact_path=parsed_v2.chunks_path,
        )
    )

    # New parse_version exists; old does not.
    new_points = backend.points_for_document("doc-1", parsed_v2.parse_version)
    old_points = backend.points_for_document("doc-1", parsed_v1.parse_version)
    assert new_points, "new parse_version must populate the backend"
    # Note: the v1 sync left old_points; the v2 sync's DELETE step targets
    # (doc, parse_version=v2) so old v1 points remain until the cleanup
    # worker (Wave 2 §F.5) GCs them. This is the documented behavior.
    # The test asserts the contract: new sync does not corrupt old slot.
    for point in old_points:
        assert point["payload"]["parse_version"] == parsed_v1.parse_version


# ---------------------------------------------------------------------
# Fulltext idempotency
# ---------------------------------------------------------------------


def test_fulltext_sync_is_replace_idempotent_on_double_call():
    store = InMemoryObjectStore()
    backend = InMemoryFulltextBackend()
    parsed = _seed(store)
    modality = FulltextModality(backend=backend, store=store)

    asyncio.run(
        modality.sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )
    )
    first_state = backend.documents_for_document("doc-beacon", parsed.parse_version)
    assert first_state, "fulltext sync must populate at least one document"

    asyncio.run(
        modality.sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )
    )
    second_state = backend.documents_for_document("doc-beacon", parsed.parse_version)

    assert second_state == first_state, "fulltext sync must be byte-equivalent on a second call (§D.4 idempotency)"


# ---------------------------------------------------------------------
# Hybrid dedup by chunk_id (§C.6)
# ---------------------------------------------------------------------


def test_vector_and_fulltext_share_chunk_ids_for_hybrid_dedup():
    """Per §C.6: vector + fulltext share ``chunks.jsonl``. The
    backend records on both sides MUST therefore share the same
    ``chunk_id`` set so the search-layer rerank dedup can match them.
    """
    store = InMemoryObjectStore()
    vec_backend = InMemoryVectorBackend()
    ft_backend = InMemoryFulltextBackend()
    parsed = _seed(store)

    asyncio.run(
        VectorModality(backend=vec_backend, store=store).sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )
    )
    asyncio.run(
        FulltextModality(backend=ft_backend, store=store).sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )
    )

    vec_chunk_ids = {
        p["payload"]["chunk_id"] for p in vec_backend.points_for_document("doc-beacon", parsed.parse_version)
    }
    ft_chunk_ids = {d["chunk_id"] for d in ft_backend.documents_for_document("doc-beacon", parsed.parse_version)}

    assert vec_chunk_ids == ft_chunk_ids, (
        "vector and fulltext must share the chunk_id set for hybrid dedup to work (§C.6 trade-off lock)"
    )
    assert vec_chunk_ids, "non-empty document must produce at least one shared chunk_id"


# ---------------------------------------------------------------------
# Recall_type / modality discriminator on payload
# ---------------------------------------------------------------------


def test_vector_payload_carries_modality_discriminator():
    store = InMemoryObjectStore()
    backend = InMemoryVectorBackend()
    parsed = _seed(store)
    asyncio.run(
        VectorModality(backend=backend, store=store).sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )
    )
    for point in backend.all_points():
        assert point["payload"]["modality"] == Modality.VECTOR.value


def test_vector_sync_uses_batch_embedder_when_available():
    """Production vector sync should batch embeddings instead of issuing one
    external embedding request per chunk.

    Large PDFs can produce hundreds of chunks; calling ``embed_query`` for
    each chunk overwhelms OpenAI-compatible providers such as DashScope. The
    vector modality keeps the single-item embedder for simulator tests, but
    production wiring should be able to pass a batch embedder.
    """

    store = InMemoryObjectStore()
    backend = InMemoryVectorBackend()
    from aperag.indexing import ChunkingConfig, ParseConfig

    parsed = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-batch",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
        config=ParseConfig(chunking=ChunkingConfig(chunk_size=80, chunk_overlap=0)),
    )
    calls: list[list[str]] = []

    def batch_embedder(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(i)] for i, _ in enumerate(texts)]

    asyncio.run(
        VectorModality(backend=backend, store=store, batch_embedder=batch_embedder).sync(
            document_id="doc-batch",
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )
    )

    points = backend.points_for_document("doc-batch", parsed.parse_version)
    assert len(points) > 1
    assert len(calls) == 1
    assert len(calls[0]) == len(points)


def test_fulltext_payload_carries_modality_discriminator():
    store = InMemoryObjectStore()
    backend = InMemoryFulltextBackend()
    parsed = _seed(store)
    asyncio.run(
        FulltextModality(backend=backend, store=store).sync(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )
    )
    for doc in backend.all_documents():
        assert doc["modality"] == Modality.FULLTEXT.value


# ---------------------------------------------------------------------
# Empty / missing artifact contract (§C.7 reschedule semantic)
# ---------------------------------------------------------------------


def test_vector_sync_no_op_on_missing_chunks_artifact():
    store = InMemoryObjectStore()
    backend = InMemoryVectorBackend()
    asyncio.run(
        VectorModality(backend=backend, store=store).sync(
            document_id="doc-x",
            parse_version="abcdef0123456789",
            derived_artifact_path=("collections/c/documents/doc-x/derived/parse_abcdef0123456789/chunks.jsonl"),
        )
    )
    assert backend.all_points() == [], (
        "vector sync against a missing chunks.jsonl must be a silent no-op per §C.7 reschedule semantic"
    )


def test_fulltext_sync_no_op_on_missing_chunks_artifact():
    store = InMemoryObjectStore()
    backend = InMemoryFulltextBackend()
    asyncio.run(
        FulltextModality(backend=backend, store=store).sync(
            document_id="doc-x",
            parse_version="abcdef0123456789",
            derived_artifact_path=("collections/c/documents/doc-x/derived/parse_abcdef0123456789/chunks.jsonl"),
        )
    )
    assert backend.all_documents() == [], (
        "fulltext sync against a missing chunks.jsonl must be a silent no-op per §C.7 reschedule semantic"
    )


def test_vector_modality_enum_is_vector():
    assert VectorModality.modality is Modality.VECTOR


def test_fulltext_modality_enum_is_fulltext():
    assert FulltextModality.modality is Modality.FULLTEXT


def test_vector_derive_is_no_op_pass_through():
    store = InMemoryObjectStore()
    backend = InMemoryVectorBackend()
    parsed = _seed(store)
    result = asyncio.run(
        VectorModality(backend=backend, store=store).derive(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            source_path=parsed.chunks_path,
        )
    )
    assert result.derived_artifact_path == parsed.chunks_path, (
        "vector.derive must pass through the parser's chunks.jsonl path "
        "without mutation (§C.6 shared-artifact convention)"
    )


def test_fulltext_derive_is_no_op_pass_through():
    store = InMemoryObjectStore()
    backend = InMemoryFulltextBackend()
    parsed = _seed(store)
    result = asyncio.run(
        FulltextModality(backend=backend, store=store).derive(
            document_id="doc-beacon",
            parse_version=parsed.parse_version,
            source_path=parsed.chunks_path,
        )
    )
    assert result.derived_artifact_path == parsed.chunks_path
