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

"""Vision modality — celery T1.4.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §C.6 + §D.2:
the vision modality extracts images from the source document, runs a
vision-LLM (or multimodal embedding) per image, and writes:

- ``derived/parse_<v>/vision/manifest.jsonl`` (CANONICAL — one line
  per image with metadata + embedding) — the file the §D.1 sync
  reads.
- ``derived/parse_<v>/vision/images/<image_id>.<ext>`` (CANONICAL —
  the extracted image blob, expensive to re-extract from PDF).

Sync writes one Qdrant point per image, keyed by ``image_id``.

The T1 simulator's image extraction reads a synthetic
``<source>.images.json`` companion (a list of in-memory image
records with ``image_id`` + ``alt_text``) so tests can stage a
deterministic image payload without standing up a PDF parser. The
production vision pipeline (T2.x) replaces the companion read with
real PDF / OCR + vision-LLM calls, leaving the §D.1 sync contract
untouched.
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


SIMULATOR_VISION_EMBEDDING_DIM = 16


@runtime_checkable
class VisionBackend(Protocol):
    """Minimal Qdrant-shaped backend surface for the vision modality."""

    def delete_by_filter(self, *, document_id: str, parse_version: str) -> int: ...

    def upsert_point(
        self,
        *,
        point_id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None: ...


class InMemoryVisionBackend:
    """Process-local in-memory vision backend for unit tests."""

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
        return sorted(out, key=lambda r: r["point_id"])


def _placeholder_embedding(image_id: str, alt_text: str, dim: int = SIMULATOR_VISION_EMBEDDING_DIM) -> list[float]:
    digest = hashlib.sha256(f"{image_id}|{alt_text}".encode("utf-8")).digest()
    repeat = (dim + len(digest) - 1) // len(digest)
    expanded = (digest * repeat)[:dim]
    return [b / 256.0 for b in expanded]


class VisionModality(ModalityWorker):
    """Vision modality worker (image extract + vision-LLM + Qdrant)."""

    modality = Modality.VISION

    def __init__(
        self,
        *,
        backend: VisionBackend,
        store: _SyncObjectStore,
        embedder: callable | None = None,
    ) -> None:
        self._backend = backend
        self._store = store
        self._embedder = embedder or _placeholder_embedding

    async def derive(
        self,
        *,
        document_id: str,
        parse_version: str,
        source_path: str,
    ) -> DeriveResult:
        """Extract images, run vision-LLM, persist manifest atomically.

        T1 simulator contract: ``source_path`` is a JSON file in the
        object store containing ``[{"image_id": ..., "alt_text": ...,
        "page_idx": int|None, "bbox": [...]|None}, ...]``. The real
        T2.x pipeline replaces this with PDF extraction; the
        ``manifest.jsonl`` schema is the contract that must not change.
        """
        body = read_or_none(self._store, source_path)
        if body is None:
            logger.info(
                "vision.derive: source images list not yet present at %s; reschedule",
                source_path,
            )
            return DeriveResult(derived_artifact_path="")

        try:
            image_records = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"vision.derive expected JSON list at {source_path}, got non-JSON: {exc}") from exc
        if not isinstance(image_records, list):
            raise ValueError(f"vision.derive expected JSON list of image records, got {type(image_records).__name__}")

        parts = source_path.split("/")
        try:
            collection_id = parts[parts.index("collections") + 1]
        except (ValueError, IndexError):
            raise ValueError(
                f"vision.derive expected source_path to follow the "
                f"collections/<cid>/documents/<did>/... layout, got "
                f"{source_path!r}"
            )

        manifest_lines: list[str] = []
        for record in image_records:
            image_id = record["image_id"]
            alt_text = record.get("alt_text", "")
            embedding = self._embedder(image_id, alt_text)
            entry = {
                "image_id": image_id,
                "alt_text": alt_text,
                "page_idx": record.get("page_idx"),
                "bbox": record.get("bbox"),
                "embedding": embedding,
            }
            manifest_lines.append(json.dumps(entry, ensure_ascii=False))

        manifest_path = derived_artifact(
            collection_id=collection_id,
            document_id=document_id,
            parse_version=parse_version,
            filename="vision/manifest.jsonl",
        )
        write_atomic(
            self._store,
            manifest_path,
            ("\n".join(manifest_lines) + ("\n" if manifest_lines else "")).encode("utf-8"),
        )
        return DeriveResult(derived_artifact_path=manifest_path)

    async def sync(
        self,
        *,
        document_id: str,
        parse_version: str,
        derived_artifact_path: str,
    ) -> None:
        """§D.1 replace-idempotent contract against the vision backend."""
        deleted = self._backend.delete_by_filter(document_id=document_id, parse_version=parse_version)
        if deleted:
            logger.info(
                "vision.sync deleted %d existing vision points for document=%s parse_version=%s",
                deleted,
                document_id,
                parse_version,
            )

        body = read_or_none(self._store, derived_artifact_path)
        if body is None:
            logger.info(
                "vision.sync found no manifest at %s; treating as derive-incomplete and skipping insert",
                derived_artifact_path,
            )
            return

        for line in body.decode("utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            image_id = entry["image_id"]
            embedding = entry.get("embedding", [])
            if not embedding:
                continue
            point_id = f"vision:{document_id}:{parse_version}:{image_id}"
            payload = {
                "document_id": document_id,
                "parse_version": parse_version,
                "modality": Modality.VISION.value,
                "image_id": image_id,
                "alt_text": entry.get("alt_text"),
                "page_idx": entry.get("page_idx"),
                "bbox": entry.get("bbox"),
            }
            self._backend.upsert_point(
                point_id=point_id,
                embedding=embedding,
                payload=payload,
            )


__all__ = [
    "VisionModality",
    "VisionBackend",
    "InMemoryVisionBackend",
    "SIMULATOR_VISION_EMBEDDING_DIM",
]
