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

"""Vector modality — celery T1.3.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §B/§D.2:
the vector modality reads ``chunks.jsonl`` from the parser-produced
derived directory and applies the §D.1 replace-idempotent contract
against a Qdrant-shaped backend:

    sync_vector(document_id, parse_version, chunks_path) →
        1. backend.delete_by_filter({"document_id": X, "parse_version": Y})
        2. for each chunk in chunks_path: backend.upsert(point)

Vector + Fulltext share the parser's ``chunks.jsonl`` artifact
(§C.6 conscious trade-off + future shadow split point).

The T1 simulator uses a deterministic placeholder embedding
(SHA-256-derived) so tests can assert byte-equivalent backend state
without standing up a real embedding API. Production wiring (T2.x)
swaps the placeholder for the real embedding service; the §D.1
contract — DELETE-by-(doc, parse_version) THEN INSERT — does not
change.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, runtime_checkable

from aperag.indexing.base import DeriveResult, ModalityWorker
from aperag.indexing.models import Modality
from aperag.indexing.parser import read_chunks
from aperag.objectstore.base import ObjectStore as _SyncObjectStore

logger = logging.getLogger(__name__)


# Default embedding dimension for the simulator. Production will pin
# this from the embedding model config; the tests rely on the same
# dimension on both ends so the assertion of "byte-equivalent backend
# state" is well-defined.
SIMULATOR_EMBEDDING_DIM = 16


@runtime_checkable
class VectorBackend(Protocol):
    """Minimal Qdrant-shaped backend surface for the vector modality.

    Production wires to the existing Qdrant client; T1 tests inject
    :class:`InMemoryVectorBackend`. The two methods are everything the
    §D.1 contract needs.
    """

    def delete_by_filter(self, *, document_id: str, parse_version: str) -> int:
        """Delete every point matching ``(document_id, parse_version)``.

        Returns the number of deleted points (informational; the
        contract only requires the post-state to have zero points
        matching the filter).
        """

    def upsert_point(
        self,
        *,
        point_id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Idempotent point insert keyed on ``point_id`` (Qdrant point id).

        For the vector modality the caller passes the parser-emitted
        ``chunk_id`` as ``point_id`` (one chunk → one point). The
        ``chunk_id`` value is also written into ``payload["chunk_id"]``
        by the worker so retrieval can echo it back; vector + fulltext
        share that payload field for hybrid-dedup (§C.6).
        """


class InMemoryVectorBackend:
    """Process-local in-memory backend for unit tests.

    Stores points in a dict keyed by ``point_id``. Implements the
    :class:`VectorBackend` protocol so vector.sync can target it
    transparently.
    """

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}

    def delete_by_filter(self, *, document_id: str, parse_version: str) -> int:
        deleted = 0
        for point_id in list(self._points):
            payload = self._points[point_id].get("payload", {})
            if payload.get("document_id") == document_id and payload.get("parse_version") == parse_version:
                self._points.pop(point_id)
                deleted += 1
        return deleted

    def upsert_point(
        self,
        *,
        point_id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        self._points[point_id] = {
            "point_id": point_id,
            "embedding": list(embedding),
            "payload": dict(payload),
        }

    # Test inspection helpers — not part of the production protocol.

    def points_for_document(self, document_id: str, parse_version: str | None = None) -> list[dict[str, Any]]:
        out = []
        for record in self._points.values():
            payload = record["payload"]
            if payload.get("document_id") != document_id:
                continue
            if parse_version is not None and payload.get("parse_version") != parse_version:
                continue
            out.append(record)
        return sorted(out, key=lambda r: r["point_id"])

    def all_points(self) -> list[dict[str, Any]]:
        return sorted(self._points.values(), key=lambda r: r["point_id"])


def _placeholder_embedding(text: str, dim: int = SIMULATOR_EMBEDDING_DIM) -> list[float]:
    """Deterministic placeholder embedding for the T1 simulator.

    Production replaces this with the embedding service call; the
    function signature (``text -> list[float]`` of fixed dim) is the
    seam that tests can mock. The placeholder is a hash-derived
    pseudo-vector so two calls with the same text produce the same
    embedding (idempotency).
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Spread the digest bytes over ``dim`` floats in [0, 1).
    repeat = (dim + len(digest) - 1) // len(digest)
    expanded = (digest * repeat)[:dim]
    return [b / 256.0 for b in expanded]


class VectorModality(ModalityWorker):
    """Vector modality worker (Qdrant-shaped sync)."""

    modality = Modality.VECTOR

    def __init__(
        self,
        *,
        backend: VectorBackend,
        store: _SyncObjectStore,
        embedder: callable | None = None,
        batch_embedder: callable | None = None,
    ) -> None:
        self._backend = backend
        self._store = store
        # ``embedder`` lets production swap the simulator placeholder
        # for the real embedding service without changing the
        # ``ModalityWorker`` interface.
        self._embedder = embedder or _placeholder_embedding
        self._batch_embedder = batch_embedder

    async def derive(
        self,
        *,
        document_id: str,
        parse_version: str,
        source_path: str,
    ) -> DeriveResult:
        """Vector derive is a no-op pass-through (§C.6).

        ``chunks.jsonl`` is produced by the parser (T1.1) and shared
        with the fulltext modality. The vector modality does not own
        a separate derived artifact — the embedding step happens
        lazily inside ``sync`` so the placeholder embedder can be
        swapped in T2.x without rewriting derived files.
        """
        return DeriveResult(derived_artifact_path=source_path)

    async def sync(
        self,
        *,
        document_id: str,
        parse_version: str,
        derived_artifact_path: str,
    ) -> None:
        """§D.1 replace-idempotent contract against the vector backend."""
        # Step 1: delete prior state for this (doc, parse_version).
        deleted = self._backend.delete_by_filter(document_id=document_id, parse_version=parse_version)
        if deleted:
            logger.info(
                "vector.sync deleted %d existing points for document=%s parse_version=%s",
                deleted,
                document_id,
                parse_version,
            )

        # Step 2: insert chunks from chunks.jsonl. Empty / missing
        # artifact short-circuits to a no-op (per §C.7 read contract:
        # "derive 还没完成 → reschedule, 不报错"). The orchestrator
        # is responsible for re-queueing in that case.
        chunks = read_chunks(self._store, derived_artifact_path)
        if not chunks:
            logger.info(
                "vector.sync found no chunks at %s; treating as derive-incomplete and skipping insert",
                derived_artifact_path,
            )
            return

        texts = [chunk.get("text", "") for chunk in chunks]
        embeddings = self._batch_embedder(texts) if self._batch_embedder is not None else None

        for index, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            text = texts[index]
            embedding = embeddings[index] if embeddings is not None else self._embedder(text)
            payload = {
                "document_id": document_id,
                "parse_version": parse_version,
                "modality": Modality.VECTOR.value,
                "chunk_id": chunk_id,
                "text": text,
                "section_path": chunk.get("section_path"),
                "heading_anchor": chunk.get("heading_anchor"),
                "page_idx": chunk.get("page_idx"),
            }
            self._backend.upsert_point(
                point_id=chunk_id,
                embedding=embedding,
                payload=payload,
            )


__all__ = [
    "VectorModality",
    "VectorBackend",
    "InMemoryVectorBackend",
    "SIMULATOR_EMBEDDING_DIM",
]
