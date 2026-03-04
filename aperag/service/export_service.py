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

import logging
import unicodedata

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, func, select

from aperag.db.models import Collection, CollectionStatus, ExportTask, ExportTaskStatus
from aperag.db.ops import async_db_ops
from aperag.exceptions import CollectionNotFoundException, PermissionDeniedError
from aperag.objectstore.base import get_async_object_store
from aperag.schema import view_models
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)

MAX_CONCURRENT_EXPORT_TASKS = 3


class ExportService:
    def __init__(self):
        self.db_ops = async_db_ops

    async def create_export_task(self, user_id: str, collection_id: str) -> view_models.ExportTaskResponse:
        async def _create(session):
            # Verify collection exists and user is owner
            result = await session.execute(
                select(Collection).where(
                    and_(
                        Collection.id == collection_id,
                        Collection.user == user_id,
                        Collection.status != CollectionStatus.DELETED,
                    )
                )
            )
            collection = result.scalars().first()
            if collection is None:
                # Check if collection exists at all (for better error message)
                result2 = await session.execute(
                    select(Collection).where(
                        and_(
                            Collection.id == collection_id,
                            Collection.status != CollectionStatus.DELETED,
                        )
                    )
                )
                if result2.scalars().first() is None:
                    raise CollectionNotFoundException(collection_id)
                raise PermissionDeniedError(f"You don't have permission to export collection {collection_id}")

            # Check concurrent task limit
            running_count = await session.execute(
                select(func.count()).where(
                    and_(
                        ExportTask.user == user_id,
                        ExportTask.status.in_([ExportTaskStatus.PENDING, ExportTaskStatus.PROCESSING]),
                    )
                )
            )
            count = running_count.scalar()
            if count >= MAX_CONCURRENT_EXPORT_TASKS:
                raise HTTPException(
                    status_code=429,
                    detail="Too many concurrent export tasks. Please wait for existing tasks to complete.",
                )

            # Create the export task record
            task = ExportTask(
                user=user_id,
                collection_id=collection_id,
                status=ExportTaskStatus.PENDING,
                progress=0,
                message="Export task created, waiting to start...",
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            return task

        task = await self.db_ops._execute_query(_create)

        # Trigger Celery task (import here to avoid circular imports)
        from config.export_tasks import export_collection_task

        export_collection_task.delay(task.id)

        return view_models.ExportTaskResponse(
            export_task_id=task.id,
            status=task.status,
            progress=task.progress,
            message=task.message,
        )

    async def get_export_task(self, user_id: str, task_id: str) -> view_models.ExportTaskResponse:
        async def _get(session):
            result = await session.execute(
                select(ExportTask).where(and_(ExportTask.id == task_id, ExportTask.user == user_id))
            )
            return result.scalars().first()

        task = await self.db_ops._execute_query(_get)
        if task is None:
            raise HTTPException(status_code=404, detail="Export task not found")

        download_url = None
        if task.status == ExportTaskStatus.COMPLETED:
            download_url = f"/api/v1/export-tasks/{task_id}/download"

        return view_models.ExportTaskResponse(
            export_task_id=task.id,
            status=task.status,
            progress=task.progress,
            message=task.message,
            error_message=task.error_message,
            download_url=download_url,
            file_size=task.file_size,
            gmt_created=task.gmt_created,
            gmt_completed=task.gmt_completed,
            gmt_expires=task.gmt_expires,
        )

    async def download_export(self, user_id: str, task_id: str) -> StreamingResponse:
        async def _get(session):
            result = await session.execute(
                select(ExportTask).where(and_(ExportTask.id == task_id, ExportTask.user == user_id))
            )
            return result.scalars().first()

        task = await self.db_ops._execute_query(_get)
        if task is None:
            raise HTTPException(status_code=404, detail="Export task not found")

        if task.status == ExportTaskStatus.EXPIRED:
            raise HTTPException(status_code=410, detail="Export file has expired. Please create a new export task.")

        if task.status != ExportTaskStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Export task is not ready for download (status: {task.status})",
            )

        store = get_async_object_store()
        result = await store.get(task.object_store_path)
        if result is None:
            raise HTTPException(status_code=404, detail="Export file not found in storage")

        stream, size = result

        async def stream_generator():
            async for chunk in stream:
                yield chunk

        # Build a safe filename using the collection id as fallback
        collection_title = task.collection_id

        async def _get_collection(session):
            r = await session.execute(select(Collection).where(Collection.id == task.collection_id))
            return r.scalars().first()

        collection = await self.db_ops._execute_query(_get_collection)
        if collection and collection.title:
            collection_title = collection.title

        safe_title = _sanitize_filename(collection_title)
        date_str = utc_now().strftime("%Y-%m-%d")
        filename = f"{safe_title}_export_{date_str}.zip"

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(size),
        }

        return StreamingResponse(stream_generator(), media_type="application/zip", headers=headers)


def _sanitize_filename(name: str) -> str:
    """Strip non-ASCII and filesystem-unsafe characters from a filename."""
    normalized = unicodedata.normalize("NFKD", name)
    safe = "".join(c for c in normalized if not unicodedata.combining(c))
    safe = "".join(c if (c.isalnum() or c in " -_") else "_" for c in safe)
    safe = safe.strip("_").strip() or "collection"
    return safe[:64]


export_service = ExportService()
