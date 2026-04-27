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

"""Fulltext modality — celery T1.3.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §B/§D.2:
the fulltext modality reads ``chunks.jsonl`` from the parser-produced
derived directory and applies the §D.1 replace-idempotent contract
against an Elasticsearch-shaped backend:

    sync_fulltext(document_id, parse_version, chunks_path) →
        1. backend.delete_by_query({"document_id": X, "parse_version": Y})
        2. backend.bulk_index([chunk for chunk in chunks_path])

Vector + Fulltext share the parser's ``chunks.jsonl`` artifact (§C.6
conscious trade-off). Hybrid dedup at the search layer keys on
``chunk_id`` so a chunk that hits both modalities is not double-
counted (existing rerank dedup logic — see test).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from aperag.indexing.base import DeriveResult, ModalityWorker
from aperag.indexing.models import Modality
from aperag.indexing.parser import read_chunks
from aperag.objectstore.base import ObjectStore as _SyncObjectStore

logger = logging.getLogger(__name__)


@runtime_checkable
class FulltextBackend(Protocol):
    """Minimal Elasticsearch-shaped backend surface for fulltext modality."""

    def delete_by_query(self, *, document_id: str, parse_version: str) -> int:
        """Delete every document matching ``(document_id, parse_version)``.

        Returns the number of deleted documents (informational; the
        contract only requires the post-state to have zero matches).
        """

    def bulk_index(self, *, documents: list[dict[str, Any]]) -> None:
        """Bulk index a list of documents.

        Each document MUST carry a ``chunk_id`` field that the search
        layer dedups on. Implementations are free to use ES native
        bulk API or any equivalent.
        """


class InMemoryFulltextBackend:
    """Process-local in-memory backend for unit tests."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, Any]] = {}

    def delete_by_query(self, *, document_id: str, parse_version: str) -> int:
        deleted = 0
        for chunk_id in list(self._documents):
            doc = self._documents[chunk_id]
            if doc.get("document_id") == document_id and doc.get("parse_version") == parse_version:
                self._documents.pop(chunk_id)
                deleted += 1
        return deleted

    def bulk_index(self, *, documents: list[dict[str, Any]]) -> None:
        for document in documents:
            chunk_id = document["chunk_id"]
            self._documents[chunk_id] = dict(document)

    # Test inspection helpers — not part of the production protocol.

    def documents_for_document(self, document_id: str, parse_version: str | None = None) -> list[dict[str, Any]]:
        out = []
        for doc in self._documents.values():
            if doc.get("document_id") != document_id:
                continue
            if parse_version is not None and doc.get("parse_version") != parse_version:
                continue
            out.append(doc)
        return sorted(out, key=lambda d: d["chunk_id"])

    def all_documents(self) -> list[dict[str, Any]]:
        return sorted(self._documents.values(), key=lambda d: d["chunk_id"])


class FulltextModality(ModalityWorker):
    """Fulltext modality worker (Elasticsearch-shaped sync)."""

    modality = Modality.FULLTEXT

    def __init__(
        self,
        *,
        backend: FulltextBackend,
        store: _SyncObjectStore,
        collection_id: str | None = None,
    ) -> None:
        self._backend = backend
        self._store = store
        # Wave 3 closing fix: ``aperag/domains/retrieval/pipeline.py:
        # _fulltext_search`` queries the legacy ES schema
        # (``content``/``title``/``collection_id`` filter — the shape
        # the legacy ``aperag/domains/indexing/fulltext_index.py``
        # wrote before Wave 3 hard-cut). The new modality worker
        # must keep writing the same shape so the read path is
        # symmetric, otherwise every fulltext search returns 0
        # hits silently. ``collection_id`` is the only new piece
        # of context the modality needs; the worker_factory passes
        # it at construction time. Production tests (run_chat_
        # collection_flow.sh) verify the round-trip end-to-end.
        self._collection_id = collection_id

    async def derive(
        self,
        *,
        document_id: str,
        parse_version: str,
        source_path: str,
    ) -> DeriveResult:
        """Fulltext derive is a no-op pass-through (§C.6).

        ``chunks.jsonl`` is produced by the parser (T1.1) and shared
        with the vector modality. Fulltext does not own a separate
        derived artifact.
        """
        return DeriveResult(derived_artifact_path=source_path)

    async def sync(
        self,
        *,
        document_id: str,
        parse_version: str,
        derived_artifact_path: str,
    ) -> None:
        """§D.1 replace-idempotent contract against the fulltext backend."""
        # Step 1: delete prior state.
        deleted = self._backend.delete_by_query(document_id=document_id, parse_version=parse_version)
        if deleted:
            logger.info(
                "fulltext.sync deleted %d existing documents for document=%s parse_version=%s",
                deleted,
                document_id,
                parse_version,
            )

        # Step 2: bulk index chunks. Missing/empty artifact treated
        # as derive-incomplete (per §C.7 read contract); orchestrator
        # re-queues.
        chunks = read_chunks(self._store, derived_artifact_path)
        if not chunks:
            logger.info(
                "fulltext.sync found no chunks at %s; treating as derive-incomplete and skipping bulk_index",
                derived_artifact_path,
            )
            return

        documents = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            doc: dict[str, Any] = {
                "document_id": document_id,
                "parse_version": parse_version,
                "modality": Modality.FULLTEXT.value,
                "chunk_id": chunk["chunk_id"],
                # ``content`` is the field the retrieval pipeline
                # (``_fulltext_search``) queries; ``text`` is kept
                # as an alias for the existing in-memory test
                # backends + InMemory contract tests that read it.
                "content": chunk_text,
                "text": chunk_text,
                "section_path": chunk.get("section_path"),
                "heading_anchor": chunk.get("heading_anchor"),
                "page_idx": chunk.get("page_idx"),
            }
            if self._collection_id is not None:
                doc["collection_id"] = self._collection_id
            documents.append(doc)
        self._backend.bulk_index(documents=documents)


__all__ = [
    "FulltextModality",
    "FulltextBackend",
    "InMemoryFulltextBackend",
]
