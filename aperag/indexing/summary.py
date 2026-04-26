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

"""Summary modality — celery T1.4.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §C.6 + §D.2:
the summary modality runs an LLM over the parsed markdown to produce
a single document-level summary plus a paired embedding, written to
``derived/parse_<v>/summary.json`` (CANONICAL artifact — LLM call
cost is preserved across retries) and synced into a Qdrant-shaped
backend that is keyed by ``document_id`` (one summary point per
document, not per chunk).

The §D.1 replace-idempotent contract still applies:
``DELETE-by-(document_id, parse_version) THEN INSERT``.

The T1 simulator uses a deterministic placeholder LLM summarizer
(``summary = first paragraph of markdown``) and a hash-derived
embedding so tests can assert byte-equivalent backend state without
standing up a real LLM. Production wiring (T2.x) swaps the
placeholder for the real LLM service; the §D.1 contract stays the
same.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Protocol, runtime_checkable

from aperag.indexing.base import DeriveResult, ModalityWorker
from aperag.indexing.models import Modality
from aperag.indexing.object_store import (
    derived_artifact,
    read_or_none,
    write_atomic,
)
from aperag.objectstore.base import ObjectStore as _SyncObjectStore

logger = logging.getLogger(__name__)


SIMULATOR_SUMMARY_EMBEDDING_DIM = 16


@runtime_checkable
class SummaryBackend(Protocol):
    """Minimal Qdrant-shaped backend surface for the summary modality.

    The summary backend is keyed by ``document_id`` (one summary per
    document) plus a ``parse_version`` payload field for the
    cleanup worker to dedup orphan parse_versions.
    """

    def delete_by_filter(self, *, document_id: str, parse_version: str) -> int:
        """Delete every summary point matching ``(document_id, parse_version)``."""

    def upsert_point(
        self,
        *,
        point_id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Idempotent point insert keyed on ``point_id``.

        For the summary modality the convention is
        ``point_id = f"summary:{document_id}:{parse_version}"``.
        """


class InMemorySummaryBackend:
    """Process-local in-memory summary backend for unit tests."""

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

    def points_for_document(self, document_id: str, parse_version: str | None = None) -> list[dict[str, Any]]:
        out = []
        for record in self._points.values():
            payload = record["payload"]
            if payload.get("document_id") != document_id:
                continue
            if parse_version is not None and payload.get("parse_version") != parse_version:
                continue
            out.append(record)
        return out


def _placeholder_summary(markdown: str) -> str:
    """Deterministic placeholder summarizer for the T1 simulator.

    Returns the first non-empty paragraph of the markdown body, with
    headings stripped. Production replaces with the real LLM call;
    the function signature (``markdown -> str``) is the seam tests
    can mock.
    """
    for paragraph in markdown.strip().split("\n\n"):
        body = "\n".join(
            line.strip() for line in paragraph.splitlines() if line.strip() and not line.lstrip().startswith("#")
        ).strip()
        if body:
            return body
    return ""


def _placeholder_embedding(text: str, dim: int = SIMULATOR_SUMMARY_EMBEDDING_DIM) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    repeat = (dim + len(digest) - 1) // len(digest)
    expanded = (digest * repeat)[:dim]
    return [b / 256.0 for b in expanded]


class SummaryModality(ModalityWorker):
    """Summary modality worker (LLM summarize + Qdrant insert)."""

    modality = Modality.SUMMARY

    def __init__(
        self,
        *,
        backend: SummaryBackend,
        store: _SyncObjectStore,
        summarizer: callable | None = None,
        embedder: callable | None = None,
    ) -> None:
        self._backend = backend
        self._store = store
        self._summarizer = summarizer or _placeholder_summary
        self._embedder = embedder or _placeholder_embedding

    async def derive(
        self,
        *,
        document_id: str,
        parse_version: str,
        source_path: str,
    ) -> DeriveResult:
        """Read parsed markdown, summarize via LLM, persist canonical artifact.

        Reads the parser-produced ``markdown.md`` from the
        ``derived/parse_<v>/`` directory next to ``source_path``,
        runs the summarizer, computes the paired embedding, and
        writes ``summary.json`` atomically (§C.7).

        ``source_path`` is expected to be the parser-produced
        ``markdown.md`` path (the parser emits that as the
        canonical "shared" artifact every modality reads from).
        """
        markdown_bytes = read_or_none(self._store, source_path)
        if markdown_bytes is None:
            logger.info(
                "summary.derive: parser markdown not yet present at %s; reschedule",
                source_path,
            )
            return DeriveResult(derived_artifact_path="")

        markdown = markdown_bytes.decode("utf-8")
        summary_text = self._summarizer(markdown)
        embedding = self._embedder(summary_text)

        # Derive collection_id / document_id from the source_path
        # convention (collections/<cid>/documents/<did>/derived/...).
        parts = source_path.split("/")
        try:
            collection_id = parts[parts.index("collections") + 1]
        except (ValueError, IndexError):
            raise ValueError(
                f"summary.derive expected source_path to follow the "
                f"collections/<cid>/documents/<did>/... layout, got "
                f"{source_path!r}"
            )

        summary_path = derived_artifact(
            collection_id=collection_id,
            document_id=document_id,
            parse_version=parse_version,
            filename="summary.json",
        )
        body = json.dumps(
            {
                "summary_text": summary_text,
                "embedding": embedding,
            },
            ensure_ascii=False,
        )
        write_atomic(self._store, summary_path, body.encode("utf-8"))
        return DeriveResult(derived_artifact_path=summary_path)

    async def sync(
        self,
        *,
        document_id: str,
        parse_version: str,
        derived_artifact_path: str,
    ) -> None:
        """§D.1 replace-idempotent contract against the summary backend."""
        deleted = self._backend.delete_by_filter(document_id=document_id, parse_version=parse_version)
        if deleted:
            logger.info(
                "summary.sync deleted %d existing summary points for document=%s parse_version=%s",
                deleted,
                document_id,
                parse_version,
            )

        body = read_or_none(self._store, derived_artifact_path)
        if body is None:
            logger.info(
                "summary.sync found no summary at %s; treating as derive-incomplete and skipping insert",
                derived_artifact_path,
            )
            return

        record = json.loads(body)
        summary_text = record.get("summary_text", "")
        embedding = record.get("embedding", [])
        if not embedding:
            logger.warning(
                "summary.sync read summary at %s with empty embedding; skipping insert",
                derived_artifact_path,
            )
            return

        point_id = f"summary:{document_id}:{parse_version}"
        payload = {
            "document_id": document_id,
            "parse_version": parse_version,
            "modality": Modality.SUMMARY.value,
            "summary_text": summary_text,
        }
        self._backend.upsert_point(
            point_id=point_id,
            embedding=embedding,
            payload=payload,
        )


__all__ = [
    "SummaryModality",
    "SummaryBackend",
    "InMemorySummaryBackend",
    "SIMULATOR_SUMMARY_EMBEDDING_DIM",
]
