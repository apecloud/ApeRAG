"""Knowledge_base domain task helpers — celery T3.1 Pattern A/B/C migration.

Wave 3 hard-cut per architect msg=3890c9d7 Pattern A/B/C ruling:
the Celery decorators + ``celery`` / ``config.celery`` imports were
removed; each function is now a plain Python function the caller
invokes directly (Pattern A synchronous), or wraps in
``asyncio.create_task()`` (Pattern C fire-and-forget), or merges
into the periodic loops in ``aperag/indexing/{cleanup,reconciler}.py``
(Pattern B periodic).

Pattern map (per architect msg=3890c9d7):
- ``collection_delete_task``           — Pattern A (durability-required;
                                         caller invokes synchronously;
                                         body marks ``Collection.deleted_at``
                                         and the periodic Path-C
                                         ``cleanup_for_deleted_collections``
                                         sweep cascades the deletion)
- ``collection_init_task``             — Pattern C (idempotent;
                                         ``asyncio.create_task()`` ok;
                                         body now only flips
                                         ``Collection.status=ACTIVE`` —
                                         per-modality index provisioning
                                         is implicit lazy in the new
                                         modality-worker model)
- ``export_collection_task``           — Pattern C (resumable; user can
                                         retry on failure;
                                         ``asyncio.create_task()`` ok)
- ``cleanup_expired_documents_task``   — Pattern B (periodic; wired into
                                         the 5-min loop in
                                         ``aperag/indexing/cleanup.py``
                                         via ``cleanup_expired_documents_hook``)

Wave 10 §K.13 hard-cut: ``collection_summary_task`` /
``reconcile_collection_summaries_task`` removed alongside the
``collection_summary_service`` deletion. The Wave 10 regen pipeline
(Chunks C-E) dispatches directly from ``reconciler.py`` via Pattern C
``asyncio.create_task`` and persists its lease on the ``Collection``
row itself (``regen_lease_owner`` + ``regen_lease_expires_at``), so
no extra task wrapper is needed.
"""

import concurrent.futures
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
import zipfile
from datetime import timedelta
from typing import Callable, Optional

from aperag.utils.utils import utc_now

EXPORT_CHUNK_SIZE = 64 * 1024  # 64 KB
EXPORT_MAX_DOWNLOAD_WORKERS = 5

logger = logging.getLogger(__name__)


# ========== Processing-lease helpers (inlined per architect msg=64fd506a Part 1) ==========
# The legacy ``aperag/tasks/processing_lease.py`` will be deleted in
# commit 5 Part 2 atomic together with the rest of ``aperag/tasks/``;
# its public surface (``generate_processing_token`` +
# ``build_lease_expires_at`` + ``ProcessingLeaseRenewer``) is inlined
# here so this file (the only knowledge_base-domain caller after Bryce
# commit 4a inlined the agent_runtime caller) decouples now and the
# Part 2 deletion is a clean delete with no ImportError fallout.

DEFAULT_PROCESSING_LEASE_TTL_SECONDS = int(os.getenv("APERAG_PROCESSING_LEASE_TTL_SECONDS", "900"))
DEFAULT_PROCESSING_LEASE_RENEW_INTERVAL_SECONDS = int(os.getenv("APERAG_PROCESSING_LEASE_RENEW_INTERVAL_SECONDS", "60"))


def generate_processing_token() -> str:
    return uuid.uuid4().hex


def build_lease_expires_at(ttl_seconds: int = DEFAULT_PROCESSING_LEASE_TTL_SECONDS):
    return utc_now() + timedelta(seconds=ttl_seconds)


class ProcessingLeaseRenewer:
    """Background helper that periodically renews the current processing lease."""

    def __init__(
        self,
        renew_fn: Callable[[], bool],
        *,
        interval_seconds: int = DEFAULT_PROCESSING_LEASE_RENEW_INTERVAL_SECONDS,
        description: str,
    ):
        self._renew_fn = renew_fn
        self._interval_seconds = max(interval_seconds, 1)
        self._description = description
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.ownership_lost = False

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"lease-renewer:{self._description}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 1)

    def _run(self):
        while not self._stop_event.wait(self._interval_seconds):
            try:
                renewed = self._renew_fn()
            except Exception:
                logger.exception("Processing lease renewer failed for %s", self._description)
                continue

            if renewed:
                continue

            self.ownership_lost = True
            logger.warning("Processing lease ownership lost for %s", self._description)
            self._stop_event.set()
            return


# ========== Internal helpers ==========


def _build_skipped_payload(reason: str, **payload) -> dict:
    payload.update({"status": "skipped", "reason": reason})
    return payload


def _handle_ownership_lost(*, payload_factory: Callable[[], dict], log_message: str):
    logger.warning("%s", log_message)
    return payload_factory()


# Wave 10 §K.13: ``_renew_collection_summary_lease`` /
# ``_make_collection_summary_lease_renewer`` /
# ``_validate_collection_summary_relevance`` /
# ``reconcile_collection_summaries_task`` removed alongside the
# ``collection_summary_service`` hard-cut. The Wave 10 lease lives on
# the ``Collection`` row itself (``regen_lease_owner`` /
# ``regen_lease_expires_at``) and is managed by
# ``collection_regen_service`` (Chunk C) — no Pattern C task wrapper
# needed because Wave 10 regen is fire-and-forget per Pattern C
# directly from the reconciler hook.


# ========== Collection Tasks ==========


def collection_delete_task(collection_id: str) -> dict:
    """Pattern A: synchronous collection-deletion mark + path-C cascade.

    Per architect msg=3890c9d7 Pattern A spec: HTTP handler invokes this
    SYNCHRONOUSLY; the body marks the collection ``deleted_at = NOW()``
    so the periodic cleanup loop (5-min,
    ``aperag/indexing/cleanup.py:cleanup_for_deleted_collections``)
    picks it up and cascades through path B per-document cleanup. The
    cascade is durability-required — losing the mark = orphan rows +
    storage. The HTTP handler must NOT wrap this in
    ``asyncio.create_task()``.

    The 1-min worst-case cleanup latency is acceptable for the user
    (collection deletion is a low-frequency op + the user already saw
    the HTTP 200 by the time the periodic sweep runs).
    """
    from sqlalchemy import update

    from aperag.config import get_sync_session
    from aperag.domains.knowledge_base.db.models import Collection, CollectionStatus

    try:
        for session in get_sync_session():
            session.execute(
                update(Collection)
                .where(Collection.id == collection_id)
                .values(
                    status=CollectionStatus.DELETED,
                    gmt_deleted=utc_now(),
                    gmt_updated=utc_now(),
                )
            )
            session.commit()
        logger.info(f"Collection {collection_id} marked deleted (path-C cascade pending periodic sweep)")
        return {"success": True, "collection_id": collection_id, "status": "deleted"}

    except Exception as e:
        logger.error(f"Collection deletion failed for {collection_id}: {str(e)}")
        raise


def collection_init_task(collection_id: str, document_user_quota: int) -> dict:
    """Pattern C: fire-and-forget collection initialization.

    Wave 3 hard-cut: collection-level index initialization is implicit
    in the new modality-worker model — per-document modality dispatch
    (via ``aperag.indexing.dispatcher.dispatch_indexing()``) creates
    the ES index / Qdrant collection lazily on first sync. This task
    body now only flips ``Collection.status = ACTIVE`` so the HTTP
    UI can show the collection as ready; the actual index provisioning
    happens on the first document upload.

    ``document_user_quota`` is preserved in the signature for caller
    compatibility but is no longer load-bearing in the task body
    (quota enforcement lives in the T2.2 quota lane, not at
    collection-init time).
    """
    from sqlalchemy import update

    from aperag.config import get_sync_session
    from aperag.domains.knowledge_base.db.models import Collection, CollectionStatus

    try:
        for session in get_sync_session():
            session.execute(
                update(Collection)
                .where(Collection.id == collection_id)
                .values(status=CollectionStatus.ACTIVE, gmt_updated=utc_now())
            )
            session.commit()
        logger.info(f"Collection {collection_id} initialized (status=ACTIVE; index provisioning lazy on first upload)")
        return {
            "success": True,
            "collection_id": collection_id,
            "status": "initialized",
            "document_user_quota": document_user_quota,
        }

    except Exception as e:
        logger.error(f"Collection initialization failed for {collection_id}: {str(e)}")
        raise


# Wave 10 §K.13: ``collection_summary_task`` (Pattern C wrapper for
# the legacy ``collection_summary_service.generate_collection_summary_task``)
# removed alongside the service hard-cut. Wave 10 regen is dispatched
# directly from the reconciler hook (Chunk E) via fire-and-forget
# ``asyncio.create_task``, no extra task wrapper needed.


def cleanup_expired_documents_task() -> dict:
    """Pattern B: periodic cleanup of expired uploaded documents.

    Wave 3 hard-cut: now an inlined SQL tombstone scan + soft-delete
    of documents in ``UPLOADED`` status > 1 day old (per legacy
    ``CollectionTask.cleanup_expired_documents`` contract). The
    surrounding 5-min loop (``aperag/indexing/cleanup.py``) calls this
    via :func:`cleanup_expired_documents_hook`.
    """
    from sqlalchemy import and_, select

    from aperag.config import get_sync_session
    from aperag.domains.knowledge_base.db.models import Document, DocumentStatus
    from aperag.objectstore.base import get_object_store

    logger.info("Starting cleanup_expired_documents")

    expired_count = 0
    failed_count = 0
    total_found = 0
    obj_store = get_object_store()
    expiration_threshold = utc_now() - timedelta(days=1)

    for session in get_sync_session():
        stmt = select(Document).where(
            and_(
                Document.status == DocumentStatus.UPLOADED,
                Document.gmt_created < expiration_threshold,
            )
        )
        expired_documents = list(session.execute(stmt).scalars().all())
        total_found = len(expired_documents)

        for document in expired_documents:
            try:
                # Best-effort object-store cleanup; log + continue on failure
                # so a transient storage hiccup does not block the DB tombstone.
                try:
                    obj_store.delete_objects_by_prefix(document.object_store_base_path())
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to delete objects for expired document %s from object store: %s",
                        document.id,
                        exc,
                    )

                document.status = DocumentStatus.EXPIRED
                document.gmt_updated = utc_now()
                session.add(document)
                expired_count += 1
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                logger.error(f"Failed to cleanup expired document {document.id}: {exc}")

        session.commit()
        break  # only one yielded session per get_sync_session

    result = {
        "expired_count": expired_count,
        "failed_count": failed_count,
        "total_found": total_found,
    }
    logger.info(f"cleanup_expired_documents completed: {result}")
    return result


# ========== Collection Export ==========


def export_collection_task(export_task_id: str) -> None:
    """Pattern C: package all object-store files under a collection prefix into a ZIP.

    No longer a Celery task — caller wraps in ``asyncio.create_task()``
    + ``asyncio.to_thread()`` (the body is synchronous I/O bound). The
    DB row's ``status`` field tracks progress; users see partial state
    in the UI and can retry on failure.

    Legacy Celery soft / hard time limits (55 / 60 min) are now
    enforced by the §H.6 ``bulkhead_timeout`` async context manager
    that the caller wraps the dispatch in (T2.2 lane).
    """
    from sqlalchemy import select

    from aperag.config import get_sync_session
    from aperag.domains.knowledge_base.db.models import (
        Collection,
        Document,
        ExportTask,
        ExportTaskStatus,
    )
    from aperag.objectstore.base import get_object_store

    store = get_object_store()

    def _update_task(
        status=None, progress=None, message=None, error_message=None, object_store_path=None, file_size=None
    ):
        for session in get_sync_session():
            result = session.execute(select(ExportTask).where(ExportTask.id == export_task_id))
            task = result.scalars().first()
            if not task:
                return
            if status is not None:
                task.status = status
            if progress is not None:
                task.progress = progress
            if message is not None:
                task.message = message
            if error_message is not None:
                task.error_message = error_message
            if object_store_path is not None:
                task.object_store_path = object_store_path
            if file_size is not None:
                task.file_size = file_size
            task.gmt_updated = utc_now()
            if status == ExportTaskStatus.COMPLETED:
                task.gmt_completed = utc_now()
                task.gmt_expires = utc_now() + timedelta(days=7)
            session.commit()

    temp_dir = None
    zip_path = None

    try:
        # Phase 1: read task info and move to PROCESSING
        user_id = None
        collection_id = None
        for session in get_sync_session():
            result = session.execute(select(ExportTask).where(ExportTask.id == export_task_id))
            task = result.scalars().first()
            if not task:
                logger.error(f"ExportTask {export_task_id} not found, aborting.")
                return
            user_id = task.user
            collection_id = task.collection_id
            task.status = ExportTaskStatus.PROCESSING
            task.progress = 0
            task.message = "Starting export..."
            task.gmt_updated = utc_now()
            session.commit()

        # Phase 2: list all files under the collection prefix
        prefix = f"user-{user_id}/{collection_id}/"
        object_paths = store.list_objects_by_prefix(prefix)
        total_files = len(object_paths)
        logger.info(f"ExportTask {export_task_id}: found {total_files} files under prefix '{prefix}'")

        _update_task(progress=5, message=f"Found {total_files} files. Starting download...")

        # Phase 3: create temp dir and download files concurrently
        temp_dir = tempfile.mkdtemp(prefix=f"export_{export_task_id}_")
        downloaded_count = [0]

        def _download_one(obj_path: str):
            rel_path = obj_path[len(prefix) :]
            local_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            stream = store.get(obj_path)
            if stream is None:
                logger.warning(f"Object not found at path: {obj_path}")
                return
            with open(local_path, "wb") as f:
                while True:
                    chunk = stream.read(EXPORT_CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
            downloaded_count[0] += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=EXPORT_MAX_DOWNLOAD_WORKERS) as executor:
            executor.map(_download_one, object_paths)

        _update_task(
            progress=85,
            message=f"Downloaded {downloaded_count[0]} of {total_files} files. Generating manifest...",
        )

        # Phase 4: generate manifest.json from DB
        manifest = _build_export_manifest(collection_id, user_id, get_sync_session, Collection, Document)
        manifest_path = os.path.join(temp_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        _update_task(progress=90, message="Packaging ZIP...")

        # Phase 5: create ZIP archive
        zip_path = os.path.join(tempfile.gettempdir(), f"export_{export_task_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, _dirs, files in os.walk(temp_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zf.write(file_path, arcname)

        _update_task(progress=95, message="Uploading ZIP to storage...")

        # Phase 6: upload ZIP to object store
        zip_object_path = f"exports/user-{user_id}/export_{export_task_id}.zip"
        with open(zip_path, "rb") as f:
            store.put(zip_object_path, f)

        file_size = os.path.getsize(zip_path)
        _update_task(
            status=ExportTaskStatus.COMPLETED,
            progress=100,
            message="Export complete.",
            object_store_path=zip_object_path,
            file_size=file_size,
        )
        logger.info(f"ExportTask {export_task_id} completed. ZIP size: {file_size} bytes")

    except Exception as exc:
        logger.exception(f"ExportTask {export_task_id} failed: {exc}")
        _update_task(status=ExportTaskStatus.FAILED, error_message=str(exc))

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        if zip_path and os.path.exists(zip_path):
            try:
                os.unlink(zip_path)
            except OSError:
                pass


def _build_export_manifest(collection_id: str, user_id: str, get_sync_session, Collection, Document) -> dict:
    from sqlalchemy import and_, select

    from aperag.domains.knowledge_base.db.models import (
        CollectionStatus,
        DocumentStatus,
    )

    collection_title = collection_id
    collection_description = ""
    documents = []

    for session in get_sync_session():
        col_result = session.execute(
            select(Collection).where(
                and_(Collection.id == collection_id, Collection.status != CollectionStatus.DELETED)
            )
        )
        collection = col_result.scalars().first()
        if collection:
            collection_title = collection.title or collection_id
            collection_description = collection.description or ""

        doc_result = session.execute(
            select(Document).where(
                and_(
                    Document.collection_id == collection_id,
                    Document.status != DocumentStatus.DELETED,
                )
            )
        )
        for doc in doc_result.scalars().all():
            documents.append(
                {
                    "id": doc.id,
                    "title": doc.name,
                    "status": doc.status if doc.status else "UNKNOWN",
                }
            )

    return {
        "schema_version": "1.0",
        "collection": {
            "id": collection_id,
            "title": collection_title,
            "description": collection_description,
            "exported_at": utc_now().isoformat(),
        },
        "documents": documents,
    }
