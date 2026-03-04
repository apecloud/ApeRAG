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

import concurrent.futures
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import timedelta

from config.celery import app

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024  # 64 KB
MAX_DOWNLOAD_WORKERS = 5


@app.task(bind=True, soft_time_limit=55 * 60, time_limit=60 * 60)
def export_collection_task(self, export_task_id: str):
    """Celery task: package all object-store files under a collection prefix into a ZIP."""
    from aperag.config import get_sync_session
    from aperag.db.models import Collection, Document, ExportTask, ExportTaskStatus
    from aperag.objectstore.base import get_object_store
    from aperag.utils.utils import utc_now
    from sqlalchemy import and_, select

    store = get_object_store()

    def _update_task(status=None, progress=None, message=None, error_message=None,
                     object_store_path=None, file_size=None):
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
            rel_path = obj_path[len(prefix):]
            local_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            stream = store.get(obj_path)
            if stream is None:
                logger.warning(f"Object not found at path: {obj_path}")
                return
            with open(local_path, "wb") as f:
                while True:
                    chunk = stream.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
            downloaded_count[0] += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
            executor.map(_download_one, object_paths)

        _update_task(
            progress=85,
            message=f"Downloaded {downloaded_count[0]} of {total_files} files. Generating manifest...",
        )

        # Phase 4: generate manifest.json from DB
        manifest = _build_manifest(collection_id, user_id, get_sync_session, Collection, Document)
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


def _build_manifest(collection_id: str, user_id: str, get_sync_session, Collection, Document) -> dict:
    from aperag.db.models import CollectionStatus, DocumentStatus
    from aperag.utils.utils import utc_now
    from sqlalchemy import and_, select

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
