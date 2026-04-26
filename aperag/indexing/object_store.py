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

"""Atomic-write object store helpers — celery T1.1 Foundation.

The existing ``aperag.objectstore`` package supplies an ``ObjectStore`` /
``AsyncObjectStore`` ABC plus ``LocalObjectStore`` and ``S3ObjectStore``
implementations. The native ``put`` operation is **not** atomic for
``LocalObjectStore`` — it streams bytes straight to the destination
file, so a crashed worker can leave a half-written file that the next
``derive`` retry might mistake for a complete artifact.

Per design pack §C.7, every derived artifact write MUST appear
**atomically**: either the full contents are visible or nothing is.
This module wraps the existing object store with a ``write_atomic``
contract that guarantees that property:

- **LocalFS**: write to ``<final>.tmp.<uuid>``, ``fsync``, ``os.rename``
  (POSIX atomic same-directory rename).
- **S3 / MinIO**: rely on the single-request ``put_object`` semantic
  (the object only becomes visible after the COMPLETE request lands).
  When the underlying ``S3ObjectStore`` switches to multipart for
  large bodies, the ``CompleteMultipartUpload`` call is the atomic
  visibility gate.

A reciprocal ``read_or_none`` returns ``None`` when the object is
missing or empty so a partly-derived workflow can simply ``reschedule``
without raising (§C.7 read contract).

The path layout helpers (``derived_dir`` / ``derived_artifact``) are
the canonical way to address per-(collection, document, parse_version)
artifacts so downstream modalities never hand-format their own paths.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from aperag.objectstore.base import ObjectStore as _BaseSyncStore
from aperag.objectstore.local import Local as LocalObjectStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Path layout — design pack §C.1
# ---------------------------------------------------------------------


def derived_dir(collection_id: str, document_id: str, parse_version: str) -> str:
    """Return the canonical ``derived/`` directory for a parse_version.

    Example::

        derived_dir("col-1", "doc-7", "abc123def4567890")
        # -> "collections/col-1/documents/doc-7/derived/parse_abc123def4567890"
    """
    return f"collections/{collection_id}/documents/{document_id}/derived/parse_{parse_version}"


def derived_artifact(*, collection_id: str, document_id: str, parse_version: str, filename: str) -> str:
    """Return the canonical path of a single derived artifact file.

    ``filename`` must not contain ``..`` or absolute path components;
    callers should pass plain names like ``"chunks.jsonl"`` or
    ``"vision/manifest.jsonl"``.
    """
    if filename.startswith("/") or ".." in Path(filename).parts:
        raise ValueError(
            f"derived artifact filename must be a relative path with no parent components, got {filename!r}"
        )
    return f"{derived_dir(collection_id, document_id, parse_version)}/{filename}"


def source_artifact(*, collection_id: str, document_id: str, filename: str) -> str:
    """Return the canonical path for a source/ artifact (the original upload).

    Source files are write-once (per §C.2) so they do not need the
    atomic-write helper; this helper only standardizes the path.
    """
    if filename.startswith("/") or ".." in Path(filename).parts:
        raise ValueError(
            f"source artifact filename must be a relative path with no parent components, got {filename!r}"
        )
    return f"collections/{collection_id}/documents/{document_id}/source/{filename}"


# ---------------------------------------------------------------------
# Atomic-write helpers
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _LocalAtomicWriter:
    """LocalFS atomic write: tmp + fsync + rename (§C.7).

    Instantiated only by ``write_atomic`` for ``LocalObjectStore``
    backends. Not part of the public surface.
    """

    store: LocalObjectStore

    def write(self, path: str, data: bytes) -> None:
        full_path = self.store._resolve_object_path(path)  # type: ignore[attr-defined]
        full_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a unique tmp suffix so two concurrent writers cannot
        # clobber each other mid-rename.
        tmp_path = full_path.with_name(f"{full_path.name}.tmp.{uuid.uuid4().hex[:8]}")
        try:
            with tmp_path.open("wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            # POSIX rename is atomic within the same directory; the
            # destination either still has the previous content or
            # has the new content — never partial.
            os.replace(tmp_path, full_path)
        except OSError as exc:
            # Best-effort tmp cleanup; we deliberately do not raise
            # from the cleanup path so the original IOError surfaces.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise IOError(f"atomic write to {full_path} failed: {exc}") from exc


def _to_bytes(data: bytes | IO[bytes]) -> bytes:
    """Normalize the ``put`` input to bytes for atomic write paths.

    Both LocalFS (read-then-rename) and S3 (single-request semantic)
    benefit from a fully-materialised body so the atomic boundary is
    clean. The size of derived artifacts is bounded (per design pack
    §C.7 estimate ~1.5 MB / document at the high end), so the
    materialisation cost is negligible compared to the LLM /
    embedding cost we already paid to produce it.
    """
    if isinstance(data, bytes):
        return data
    if hasattr(data, "read"):
        chunk = data.read()
        if isinstance(chunk, bytes):
            return chunk
        raise TypeError(f"object store atomic write expected bytes, got {type(chunk).__name__}")
    raise TypeError(f"object store atomic write expected bytes or IO[bytes], got {type(data).__name__}")


def write_atomic(store: _BaseSyncStore, path: str, data: bytes | IO[bytes]) -> None:
    """Write ``data`` to ``path`` such that partial state is never visible.

    Backend-specific behaviour:

    - ``LocalObjectStore``: tmp + fsync + same-directory rename
      (POSIX atomic).
    - Anything else (S3 / MinIO via ``S3ObjectStore``): the existing
      ``put`` is a single ``PutObject`` request; S3 only exposes the
      object after the request completes, so the atomicity gate is
      already there. We just delegate to ``put`` after materialising
      bytes so the caller cannot accidentally pass a half-consumed
      stream.

    Multipart uploads are configured inside ``S3ObjectStore`` itself;
    when boto3 chooses multipart, the visibility gate is the
    ``CompleteMultipartUpload`` request — still atomic from the
    reader's perspective.
    """
    body = _to_bytes(data)
    if isinstance(store, LocalObjectStore):
        _LocalAtomicWriter(store).write(path, body)
        return
    store.put(path, body)


async def write_atomic_async(store: _BaseSyncStore, path: str, data: bytes | IO[bytes]) -> None:
    """Async wrapper around :func:`write_atomic`.

    The underlying backends are blocking; we offload to a worker
    thread so the caller's event loop is not stuck on disk / network
    syscalls.
    """
    await asyncio.to_thread(write_atomic, store, path, data)


def read_or_none(store: _BaseSyncStore, path: str) -> bytes | None:
    """Read a derived artifact, returning ``None`` if missing or empty.

    Per §C.7: "读 derived 文件时若不存在或为空 → 视作 'derive 还没完成'
    → reschedule，不报错." This helper centralises that contract so
    every modality reads the same way.
    """
    handle = store.get(path)
    if handle is None:
        return None
    try:
        body = handle.read()
    finally:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass
    return body if body else None


async def read_or_none_async(store: _BaseSyncStore, path: str) -> bytes | None:
    """Async wrapper around :func:`read_or_none`."""
    return await asyncio.to_thread(read_or_none, store, path)


# ---------------------------------------------------------------------
# Convenience: in-memory store for tests
# ---------------------------------------------------------------------


class InMemoryObjectStore(_BaseSyncStore):
    """Process-local in-memory store for unit tests.

    Implements the full :class:`aperag.objectstore.base.ObjectStore`
    contract (``put`` / ``get`` / ``get_obj_size`` / ``delete`` /
    ``delete_objects_by_prefix`` / ``obj_exists``) using a plain
    dictionary. Not intended for production use.

    Tests can pass an instance directly to :func:`write_atomic` /
    :func:`read_or_none`; they will follow the S3-style code path
    (single ``put`` is treated as atomic) since the class is not a
    ``LocalObjectStore``.
    """

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put(self, path: str, data: bytes | IO[bytes]) -> None:
        self._objects[path] = _to_bytes(data)

    def get(self, path: str) -> IO[bytes] | None:
        body = self._objects.get(path)
        if body is None:
            return None
        return io.BytesIO(body)

    def get_obj_size(self, path: str) -> int | None:
        body = self._objects.get(path)
        return None if body is None else len(body)

    def delete(self, path: str) -> None:
        self._objects.pop(path, None)

    def delete_objects_by_prefix(self, prefix: str) -> None:
        for key in list(self._objects):
            if key.startswith(prefix):
                self._objects.pop(key, None)

    def obj_exists(self, path: str) -> bool:
        return path in self._objects

    def stream_range(self, path: str, start: int, end: int | None = None) -> tuple[IO[bytes], int] | None:
        body = self._objects.get(path)
        if body is None:
            return None
        end_idx = len(body) if end is None else min(end, len(body))
        if start >= len(body) or start < 0:
            return None
        chunk = body[start:end_idx]
        return io.BytesIO(chunk), len(chunk)

    def list_objects_by_prefix(self, path_prefix: str) -> list[str]:
        return sorted(k for k in self._objects if k.startswith(path_prefix))


__all__ = [
    "derived_dir",
    "derived_artifact",
    "source_artifact",
    "write_atomic",
    "write_atomic_async",
    "read_or_none",
    "read_or_none_async",
    "InMemoryObjectStore",
]
