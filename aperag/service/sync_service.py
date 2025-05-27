from datetime import datetime
from http import HTTPStatus

from celery.result import GroupResult
from django.utils import timezone

from aperag.db import models as db_models
from aperag.db.ops import (
    PagedQuery,
    query_collection,
    query_running_sync_histories,
    query_sync_histories,
    query_sync_history,
    query_user_quota,
)
from aperag.schema.utils import parseCollectionConfig
from aperag.source.base import get_source
from aperag.tasks.sync_documents_task import get_sync_progress, sync_documents
from aperag.views.utils import fail, success
from config.celery import app


async def sync_immediately(user: str, collection_id: str):
    collection = await query_collection(user, collection_id)
    source = get_source(parseCollectionConfig(collection.config))
    if not source.sync_enabled():
        return fail(HTTPStatus.BAD_REQUEST, "source type not supports sync")
    pr = await query_running_sync_histories(user, collection_id)
    async for task in pr.data:
        return fail(HTTPStatus.BAD_REQUEST, f"have running sync task {task.id}, please cancel it first")
    instance = db_models.CollectionSyncHistory(
        user=collection.user,
        start_time=timezone.now(),
        collection_id=collection_id,
        execution_time=datetime.timedelta(seconds=0),
        total_documents_to_sync=0,
        status=db_models.Collection.SyncStatus.RUNNING,
    )
    await instance.asave()
    document_user_quota = await query_user_quota(user, db_models.QuotaType.MAX_DOCUMENT_COUNT)
    sync_documents.delay(
        collection_id=collection_id, sync_history_id=instance.id, document_user_quota=document_user_quota
    )
    return success(instance.view())


async def cancel_sync(user: str, collection_id: str, collection_sync_id: str):
    sync_history = await query_sync_history(user, collection_id, collection_sync_id)
    if sync_history is None:
        return fail(HTTPStatus.NOT_FOUND, "sync history not found")
    task_context = sync_history.task_context
    if task_context is None:
        return fail(HTTPStatus.BAD_REQUEST, f"no task context in sync history {collection_sync_id}")
    scan_task_id = task_context["scan_task_id"]
    if scan_task_id is None:
        return fail(HTTPStatus.BAD_REQUEST, f"no scan task id in sync history {collection_sync_id}")
    app.AsyncResult(scan_task_id).revoke(terminate=True)
    group_id = sync_history.task_context.get("index_task_group_id", "")
    if group_id:
        group_result = GroupResult.restore(group_id, app=app)
        for task in group_result.results:
            task = app.AsyncResult(task.id)
            task.revoke(terminate=True)
    sync_history.status = db_models.CollectionSyncStatus.CANCELED
    await sync_history.asave()
    return success({})


async def list_sync_histories(user: str, collection_id: str, pq: PagedQuery):
    pr = await query_sync_histories(user, collection_id, pq)
    response = []
    async for sync_history in pr.data:
        if sync_history.status == db_models.CollectionSyncStatus.RUNNING:
            progress = get_sync_progress(sync_history)
            sync_history.failed_documents = progress.failed_documents
            sync_history.successful_documents = progress.successful_documents
            sync_history.processing_documents = progress.processing_documents
            sync_history.pending_documents = progress.pending_documents
        response.append(sync_history.view())
    return success(response, pr)


async def get_sync_history(user: str, collection_id: str, sync_history_id: str):
    sync_history = await query_sync_history(user, collection_id, sync_history_id)
    if sync_history is None:
        return fail(HTTPStatus.NOT_FOUND, "sync history not found")
    if sync_history.status == db_models.CollectionSyncStatus.RUNNING:
        progress = get_sync_progress(sync_history)
        sync_history.failed_documents = progress.failed_documents
        sync_history.successful_documents = progress.successful_documents
        sync_history.processing_documents = progress.processing_documents
        sync_history.pending_documents = progress.pending_documents
    return success(sync_history.view())
