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

import asyncio
from datetime import datetime
import json
import logging
import os
import uuid
from http import HTTPStatus
from typing import AsyncGenerator, List, Optional, Dict, Any
from urllib.parse import parse_qsl

from asgiref.sync import sync_to_async
from celery import chain, group
from celery.result import GroupResult
from django.core.files.base import ContentFile
from django.db import IntegrityError
from django.http import HttpRequest, StreamingHttpResponse
from django.shortcuts import render
from django.utils import timezone
from ninja import File, Router, Schema
from ninja.files import UploadedFile

import aperag.chat.message
from aperag.flow.base.models import Edge, InputBinding, InputSourceType, NodeInstance, FlowInstance
from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser
import aperag.views.models
from aperag.apps import QuotaType
from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.sse.base import ChatRequest, MessageProcessor, logger
from aperag.chat.sse.frontend_consumer import FrontendFormatter
from aperag.chat.utils import get_async_redis_client
from aperag.db import models as db_models
from aperag.db.models import Chat, SearchTestHistory
from aperag.db.ops import (
    build_pq,
    query_bot,
    query_bots,
    query_bots_count,
    query_chat,
    query_chat_by_peer,
    query_chats,
    query_collection,
    query_collections,
    query_collections_count,
    query_document,
    query_documents,
    query_documents_count,
    query_msp,
    query_msp_dict,
    query_msp_list,
    query_question,
    query_questions,
    query_running_sync_histories,
    query_sync_histories,
    query_sync_history,
    query_user_quota,
)
from aperag.graph.lightrag_holder import delete_lightrag_holder, reload_lightrag_holder
from aperag.llm.prompts import MULTI_ROLE_EN_PROMPT_TEMPLATES, MULTI_ROLE_ZH_PROMPT_TEMPLATES
from aperag.objectstore.base import get_object_store
from aperag.readers.base_readers import DEFAULT_FILE_READER_CLS
from aperag.schema.utils import dumpCollectionConfig, parseCollectionConfig
from aperag.source.base import get_source
from aperag.tasks.collection import delete_collection_task, init_collection_task
from aperag.tasks.crawl_web import crawl_domain
from aperag.tasks.index import (
    add_index_for_local_document,
    generate_questions,
    remove_index,
    update_collection_status,
    update_index_for_document,
    update_index_for_question,
)
from aperag.tasks.scan import delete_sync_documents_cron_job, update_sync_documents_cron_job
from aperag.tasks.sync_documents_task import get_sync_progress, sync_documents
from aperag.utils.request import get_urls, get_user
from aperag.views import models as view_models
from aperag.views.utils import (
    fail,
    query_chat_messages,
    success,
    validate_bot_config,
    validate_source_connect_config,
    validate_url,
)
from config import settings
from config.celery import app

logger = logging.getLogger(__name__)

router = Router()


@router.get("/prompt-templates")
async def list_prompt_templates(request) -> view_models.PromptTemplateList:
    language = request.headers.get('Lang', "zh-CN")
    if language == "zh-CN":
        templates = MULTI_ROLE_ZH_PROMPT_TEMPLATES
    elif language == "en-US":
        templates = MULTI_ROLE_EN_PROMPT_TEMPLATES
    else:
        return fail(HTTPStatus.BAD_REQUEST, "unsupported language of prompt templates")
    response = []
    for template in templates:
        response.append(view_models.PromptTemplate(
            name=template["name"],
            prompt=template["prompt"],
            description=template["description"],
        ))
    return success(view_models.PromptTemplateList(items=response))


@router.post("/collections/{collection_id}/sync")
async def sync_immediately(request, collection_id):
    user = get_user(request)
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
    document_user_quota = await query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
    sync_documents.delay(collection_id=collection_id,
                         sync_history_id=instance.id,
                         document_user_quota=document_user_quota)
    return success(instance.view())


@router.post("/collections/{collection_id}/cancel_sync/{collection_sync_id}")
async def cancel_sync(request, collection_id, collection_sync_id):
    """
    cancel the collection_sync_id related tasks

    Note that if using gevent/eventlet as the worker pool, the cancel operation is not work
    Please refer to https://github.com/celery/celery/issues/4019

    """
    user = get_user(request)
    sync_history = await query_sync_history(user, collection_id, collection_sync_id)
    if sync_history is None:
        return fail(HTTPStatus.NOT_FOUND, "sync history not found")
    task_context = sync_history.task_context
    if task_context is None:
        return fail(HTTPStatus.BAD_REQUEST, f"no task context in sync history {collection_sync_id}")

    # revoke the scan task
    scan_task_id = task_context["scan_task_id"]
    if scan_task_id is None:
        return fail(HTTPStatus.BAD_REQUEST, f"no scan task id in sync history {collection_sync_id}")
    app.AsyncResult(scan_task_id).revoke(terminate=True)

    # revoke the index tasks
    group_id = sync_history.task_context.get("index_task_group_id", "")
    if group_id:
        group_result = GroupResult.restore(group_id, app=app)
        for task in group_result.results:
            task = app.AsyncResult(task.id)
            task.revoke(terminate=True)
    else:
        logger.warning(f"no index task group id in sync history {collection_sync_id}")

    sync_history.status = db_models.CollectionSyncStatus.CANCELED
    await sync_history.asave()
    return success({})


@router.get("/collections/{collection_id}/sync/history")
async def list_sync_histories(request, collection_id):
    user = get_user(request)
    pr = await query_sync_histories(user, collection_id, build_pq(request))
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


@router.get("/collections/{collection_id}/sync/{sync_history_id}")
async def get_sync_history(request, collection_id, sync_history_id):
    user = get_user(request)
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


@router.post("/collections")
async def create_collection(request, collection: view_models.CollectionCreate) -> view_models.Collection:
    user = get_user(request)
    collection_config = collection.config
    if collection.type == db_models.Collection.Type.DOCUMENT:
        is_validate, error_msg = validate_source_connect_config(collection_config)
        if not is_validate:
            return fail(HTTPStatus.BAD_REQUEST, error_msg)

    if collection_config.source == "tencent":
        redis_client = get_async_redis_client()
        if await redis_client.exists("tencent_code_" + user):
            code = await redis_client.get("tencent_code_" + user)
            redirect_uri = await redis_client.get("tencent_redirect_uri_" + user)
            collection_config.code = code.decode()
            collection_config.redirect_uri = redirect_uri
            raise NotImplementedError
            collection.config = dumpCollectionConfig(collection_config)
        else:
            return fail(HTTPStatus.BAD_REQUEST, "用户未进行授权或授权已过期，请重新操作")

    # there is quota limit on collection
    if settings.MAX_COLLECTION_COUNT:
        collection_limit = await query_user_quota(user, QuotaType.MAX_COLLECTION_COUNT)
        if collection_limit is None:
            collection_limit = settings.MAX_COLLECTION_COUNT
        if collection_limit and await query_collections_count(user) >= collection_limit:
            return fail(HTTPStatus.FORBIDDEN, f"collection number has reached the limit of {collection_limit}")

    instance = db_models.Collection(
        user=user,
        type=collection.type,
        status=db_models.Collection.Status.INACTIVE,
        title=collection.title,
        description=collection.description
    )

    if collection.config is not None:
        instance.config = dumpCollectionConfig(collection_config)
    await instance.asave()
    if collection_config.enable_knowledge_graph or False:
        await reload_lightrag_holder(instance)  # LightRAG init might be slow, so we reload it once we update the collection

    if instance.type == db_models.Collection.Type.DOCUMENT:
        document_user_quota = await query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
        init_collection_task.delay(instance.id, document_user_quota)
    else:
        return fail(HTTPStatus.BAD_REQUEST, "unknown collection type")

    return success(view_models.Collection(
        id=instance.id,
        title=instance.title,
        description=instance.description,
        type=instance.type,
        config=parseCollectionConfig(instance.config),
        created=instance.gmt_created.isoformat(),
        updated=instance.gmt_updated.isoformat(),
    ))


@router.get("/collections")
async def list_collections(request) -> view_models.CollectionList:
    user = get_user(request)
    pr = await query_collections([user, settings.ADMIN_USER], build_pq(request))
    response = []
    async for collection in pr.data:
        response.append(view_models.Collection(
            id=collection.id,
            title=collection.title,
            description=collection.description,
            status=collection.status,
            type=collection.type,
            config=parseCollectionConfig(collection.config),
            created=collection.gmt_created.isoformat(),
            updated=collection.gmt_updated.isoformat(),
        ))
    return success(view_models.CollectionList(items=response), pr=pr)


@router.get("/collections/{collection_id}")
async def get_collection(request, collection_id: str) -> view_models.Collection:
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")

    return success(view_models.Collection(
        id=collection.id,
        title=collection.title,
        status=collection.status,
        description=collection.description,
        type=collection.type,
        config=parseCollectionConfig(collection.config),
        created=collection.gmt_created.isoformat(),
        updated=collection.gmt_updated.isoformat(),
    ))


@router.put("/collections/{collection_id}")
async def update_collection(request, collection_id: str, collection: view_models.CollectionUpdate) -> view_models.Collection:
    user = get_user(request)
    instance = await query_collection(user, collection_id)
    if instance is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    instance.title = collection.title
    instance.description = collection.description
    instance.config = dumpCollectionConfig(collection.config)
    await instance.asave()
    await reload_lightrag_holder(instance) # LightRAG init might be slow, so we reload it once we update the collection
    source = get_source(collection.config)
    if source.sync_enabled():
        await update_sync_documents_cron_job(instance.id)

    return success(view_models.Collection(
        id=instance.id,
        title=instance.title,
        description=instance.description,
        type=instance.type,
        config=parseCollectionConfig(instance.config),
        status=instance.status,
        created=instance.gmt_created.isoformat(),
        updated=instance.gmt_updated.isoformat(),
    ))


@router.delete("/collections/{collection_id}")
async def delete_collection(request, collection_id: str) -> view_models.Collection:
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")

    collection_bots = await collection.bots(only_ids=True)
    if len(collection_bots) > 0:
        return fail(HTTPStatus.BAD_REQUEST, f"Collection has related to bots {','.join(collection_bots)}, can not be deleted")

    await delete_sync_documents_cron_job(collection.id)
    collection.status = db_models.Collection.Status.DELETED
    collection.gmt_deleted = timezone.now()
    await collection.asave()

    await delete_lightrag_holder(collection)
    delete_collection_task.delay(collection_id)

    return success(view_models.Collection(
        id=collection.id,
        title=collection.title,
        description=collection.description,
        type=collection.type,
        config=parseCollectionConfig(collection.config),
    ))

@router.post("/collections/{collection_id}/questions")
async def create_questions(request, collection_id: str):
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    if collection.status == db_models.Collection.Status.QUESTION_PENDING:
        return fail(HTTPStatus.BAD_REQUEST, "Collection is generating questions")

    collection.status = db_models.Collection.Status.QUESTION_PENDING
    await collection.asave()

    documents = await sync_to_async(collection.document_set.exclude)(status=db_models.Document.Status.DELETED)
    generate_tasks = []
    async for document in documents:
        generate_tasks.append(generate_questions.si(document.id))
    generate_group = group(*generate_tasks)
    callback_chain = chain(generate_group, update_collection_status.s(collection.id))
    callback_chain.delay()

    return success({})

@router.put("/collections/{collection_id}/questions")
async def update_question(request, collection_id: str, question_in: view_models.QuestionUpdate) -> view_models.Question:
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")

    # ceate question
    if not question_in.id:
        question_instance = db_models.Question(
            user=collection.user,
            collection_id=collection.id,
            status=db_models.Question.Status.PENDING,
        )
        await question_instance.asave()
    else:
        question_instance = await query_question(user, question_in.id)
        if question_instance is None:
            return fail(HTTPStatus.NOT_FOUND, "Question not found")

    question_instance.question = question_in.question
    question_instance.answer = question_in.answer if question_in.answer else ""
    question_instance.status = db_models.Question.Status.PENDING
    await sync_to_async(question_instance.documents.clear)()

    if question_in.relate_documents:
        for document_id in question_in.relate_documents:
            document = await query_document(user, collection_id, document_id)
            if document is None or document.status == db_models.Document.Status.DELETED:
                return fail(HTTPStatus.NOT_FOUND, "Document not found")
            await sync_to_async(question_instance.documents.add)(document)
    else:
        question_in.relate_documents = []
    await question_instance.asave()
    update_index_for_question.delay(question_instance.id)

    return success(view_models.Question(
        id=question_instance.id,
        question=question_instance.question,
        answer=question_instance.answer,
        relate_documents=question_in.relate_documents,
    ))

@router.delete("/collections/{collection_id}/questions/{question_id}")
async def delete_question(request, collection_id: str, question_id: str) -> view_models.Question:
    user = get_user(request)

    question = await query_question(user, question_id)
    if question is None:
        return fail(HTTPStatus.NOT_FOUND, "Question not found")
    question.status = db_models.Question.Status.DELETED
    question.gmt_deleted = timezone.now()
    await question.asave()
    update_index_for_question.delay(question.id)

    docs = await sync_to_async(question.documents.exclude)(status=db_models.Document.Status.DELETED)
    doc_ids = []
    async for doc in docs:
        doc_ids.append(doc.id)
    return success(view_models.Question(
        id=question.id,
        question=question.question,
        answer=question.answer,
        relate_documents=doc_ids,
    ))

@router.get("/collections/{collection_id}/questions")
async def list_questions(request, collection_id: str) -> view_models.QuestionList:
    user = get_user(request)
    pr = await query_questions(user, collection_id, build_pq(request))
    response = []
    async for question in pr.data:
        response.append(view_models.Question(
            id=question.id,
            question=question.question,
            answer=question.answer,
            relate_documents=question.documents,
        ))
    return success(view_models.QuestionList(items=response), pr=pr)

@router.get("/collections/{collection_id}/questions/{question_id}")
async def get_question(request, collection_id: str, question_id: str) -> view_models.Question:
    user = get_user(request)
    question = await query_question(user, question_id)
    docs = await sync_to_async(question.documents.exclude)(status=db_models.Document.Status.DELETED)
    doc_ids = []
    async for doc in docs:
        doc_ids.append(doc.id)
    return success(view_models.Question(
        id=question.id,
        question=question.question,
        answer=question.answer,
        relate_documents=doc_ids,
    ))

@router.post("/collections/{collection_id}/documents")
async def create_document(request, collection_id: str, file: List[UploadedFile] = File(...)) -> List[view_models.Document]:
    if len(file) > 500:
        return fail(HTTPStatus.BAD_REQUEST, "documents are too many,add document failed")
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")

    # there is quota limit on document
    if settings.MAX_DOCUMENT_COUNT:
        document_limit = await query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
        if document_limit is None:
            document_limit = settings.MAX_DOCUMENT_COUNT
        if await query_documents_count(user, collection_id) >= document_limit:
            return fail(HTTPStatus.FORBIDDEN, f"document number has reached the limit of {document_limit}")

    response = []
    for item in file:
        file_suffix = os.path.splitext(item.name)[1].lower()
        if file_suffix not in DEFAULT_FILE_READER_CLS.keys():
            return fail(HTTPStatus.BAD_REQUEST, f"unsupported file type {file_suffix}")
        try:
            document_instance = db_models.Document(
                user=user,
                name=item.name,
                status=db_models.Document.Status.PENDING,
                size=item.size,
                collection_id=collection.id,
            )
            await document_instance.asave()

            # Upload to object store
            obj_store = get_object_store()
            upload_path = f"{document_instance.object_store_base_path()}/original{file_suffix}"
            await sync_to_async(obj_store.put)(upload_path, item)

            document_instance.object_path = upload_path
            document_instance.metadata = json.dumps({
                "object_path": upload_path,
            })
            await document_instance.asave()
            response.append(view_models.Document(
                id=document_instance.id,
                name=document_instance.name,
                status=document_instance.status,
                size=document_instance.size,
                created=document_instance.gmt_created.isoformat(),
                updated=document_instance.gmt_updated.isoformat(),
            ))
            add_index_for_local_document.delay(document_instance.id)
        except IntegrityError:
            return fail(HTTPStatus.BAD_REQUEST, f"document {item.name} already exists")
        except Exception:
            logger.exception("add document failed")
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, "add document failed")
    return success(response)


@router.post("/collections/{collection_id}/urls")
async def create_url_document(request, collection_id: str) -> List[view_models.Document]:
    user = get_user(request)
    response = {"failed_urls": []}
    collection = await query_collection(user, collection_id)
    urls = get_urls(request)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")

    # there is quota limit on document
    if settings.MAX_DOCUMENT_COUNT:
        document_limit = await query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
        if document_limit is None:
            document_limit = settings.MAX_DOCUMENT_COUNT
        if await query_documents_count(user, collection_id) >= document_limit:
            return fail(HTTPStatus.FORBIDDEN, f"document number has reached the limit of {document_limit}")

    try:

        failed_urls = []
        for url in urls:
            if not validate_url(url):
                failed_urls.append(url)
                continue
            if '.html' not in url:
                document_name = url + '.html'
            else:
                document_name = url
            document_instance = db_models.Document(
                user=user,
                name=document_name,
                status=db_models.Document.Status.PENDING,
                collection_id=collection.id,
                size=0,
            )
            await document_instance.asave()
            string_data = json.dumps(url)
            document_instance.metadata = json.dumps({
                "url": string_data,
            })
            await document_instance.asave()
            add_index_for_local_document.delay(document_instance.id)
            crawl_domain.delay(url, url, collection_id, user, max_pages=2)

    except IntegrityError as e:
        return fail(HTTPStatus.BAD_REQUEST, f"document {document_instance.name}  " + e)
    except Exception:
        logger.exception("add document failed")
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, "add document failed")
    if len(failed_urls) != 0:
        response["message"] = "Some URLs failed validation,eg. https://example.com/path?query=123#fragment"
        response["failed_urls"] = failed_urls
    return success(response)


@router.get("/collections/{collection_id}/documents")
async def list_documents(request, collection_id: str) -> view_models.DocumentList:
    user = get_user(request)
    pr = await query_documents([user, settings.ADMIN_USER], collection_id, build_pq(request))
    response = []
    async for document in pr.data:
        response.append(view_models.Document(
            id=document.id,
            name=document.name,
            status=document.status,
            size=document.size,
            created=document.gmt_created,
            updated=document.gmt_updated,
            sensitive_info=document.sensitive_info,
        ))
    return success(view_models.DocumentList(items=response), pr=pr)


@router.put("/collections/{collection_id}/documents/{document_id}")
async def update_document(
        request, collection_id: str, document_id: str, document: view_models.Document) -> view_models.Document:
    user = get_user(request)
    instance = await query_document(user, collection_id, document_id)
    if instance is None:
        return fail(HTTPStatus.NOT_FOUND, "Document not found")
    if instance.status == db_models.Document.Status.DELETING:
        return fail(HTTPStatus.BAD_REQUEST, "Document is deleting")

    if document.config:
        try:
            config = json.loads(document.config)
            metadata = json.loads(instance.metadata)
            metadata["labels"] = config["labels"]
            instance.metadata = json.dumps(metadata)
        except Exception:
            return fail(HTTPStatus.BAD_REQUEST, "invalid document config")
    await instance.asave()
    # if user add labels for a document, we need to update index
    update_index_for_document.delay(instance.id)

    related_questions = await sync_to_async(instance.question_set.exclude)(status=db_models.Question.Status.DELETED)
    async for question in related_questions:
        question.status = db_models.Question.Status.WARNING
        await question.asave()

    return success(view_models.Document(
        id=instance.id,
        name=instance.name,
        status=instance.status,
        size=instance.size,
        collection=instance.collection,
    ))


@router.delete("/collections/{collection_id}/documents/{document_id}")
async def delete_document(request, collection_id: str, document_id: str) -> view_models.Document:
    user = get_user(request)
    document = await query_document(user, collection_id, document_id)
    if document is None:
        logger.info(f"document {document_id} not found, maybe has already been deleted")
        return success({})
    if document.status == db_models.Document.Status.DELETING:
        logger.info(f"document {document_id} is deleting, ignore delete")
        return success({})

    # Delete all related objects from object store
    obj_store = get_object_store()
    await sync_to_async(obj_store.delete_objects_by_prefix)(f"{document.object_store_base_path()}/")

    document.status = db_models.Document.Status.DELETING
    document.gmt_deleted = timezone.now()
    await document.asave()

    remove_index.delay(document.id)

    related_questions = await sync_to_async(document.question_set.exclude)(status=db_models.Question.Status.DELETED)
    async for question in related_questions:
        question.documents.remove(document)
        question.status = db_models.Question.Status.WARNING
        await question.asave()

    return success(view_models.Document(
        id=document.id,
        name=document.name,
        status=document.status,
        size=document.size,
    ))


@router.delete("/collections/{collection_id}/documents")
async def delete_documents(request, collection_id: str, document_ids: List[str]):
    user = get_user(request)
    documents = await query_documents([user], collection_id, build_pq(request))
    ok = []
    failed = []
    async for document in documents.data:
        if document.id not in document_ids:
            continue
        try:
            document.status = db_models.Document.Status.DELETING
            document.gmt_deleted = timezone.now()
            await document.asave()
            remove_index.delay(document.id)

            related_questions = await sync_to_async(document.question_set.exclude)(status=db_models.Question.Status.DELETED)
            async for question in related_questions:
                question.documents.remove(document)
                question.status = db_models.Question.Status.WARNING
                await question.asave()

            ok.append(document.id)
        except Exception as e:
            logger.exception(e)
            failed.append(document.id)
    return success({"success": ok, "failed": failed})


@router.post("/bots/{bot_id}/chats")
async def create_chat(request, bot_id: str) -> view_models.Chat:
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    instance = db_models.Chat(user=user, bot_id=bot_id, peer_type=db_models.Chat.PeerType.SYSTEM, status=db_models.Chat.Status.ACTIVE)
    await instance.asave()
    return success(view_models.Chat(
        id=instance.id,
        title=instance.title,
        bot_id=instance.bot_id,
        peer_type=instance.peer_type,
        peer_id=instance.peer_id,
        created=instance.gmt_created.isoformat(),
        updated=instance.gmt_updated.isoformat(),
    ))


@router.get("/bots/{bot_id}/chats")
async def list_chats(request, bot_id: str) -> view_models.ChatList:
    user = get_user(request)
    pr = await query_chats(user, bot_id, build_pq(request))
    response = []
    async for chat in pr.data:
        response.append(view_models.Chat(
            id=chat.id,
            title=chat.title,
            bot_id=chat.bot_id,
            peer_type=chat.peer_type,
            peer_id=chat.peer_id,
            created=chat.gmt_created.isoformat(),
            updated=chat.gmt_updated.isoformat(),
        ))
    return success(view_models.ChatList(items=response), pr=pr)


@router.put("/bots/{bot_id}/chats/{chat_id}")
async def update_chat(request, bot_id: str, chat_id: str, chat_in: view_models.ChatUpdate) -> view_models.Chat:
    user = get_user(request)
    chat = await query_chat(user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    chat.title = chat_in.title
    await chat.asave()
    return success(view_models.Chat(
        id=chat.id,
        title=chat.title,
        bot_id=chat.bot_id,
        peer_type=chat.peer_type,
        peer_id=chat.peer_id,
        created=chat.gmt_created.isoformat(),
        updated=chat.gmt_updated.isoformat(),
    ))


@router.get("/bots/{bot_id}/chats/{chat_id}")
async def get_chat(request, bot_id: str, chat_id: str) -> view_models.Chat:
    user = get_user(request)
    chat = await query_chat(user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")

    messages = await query_chat_messages(user, chat_id)
    return success(view_models.ChatDetails(
        id=chat.id,
        title=chat.title,
        bot_id=chat.bot_id,
        history=messages,
        peer_type=chat.peer_type,
        peer_id=chat.peer_id,
        created=chat.gmt_created.isoformat(),
        updated=chat.gmt_updated.isoformat(),
    ))


@router.post("/bots/{bot_id}/chats/{chat_id}/messages/{message_id}")
async def feedback_message(request, bot_id: str, chat_id: str, message_id: str, msg_in: view_models.Feedback) -> None:
    user = get_user(request)
    chat = await query_chat(user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    feedback = await aperag.chat.message.feedback_message(chat.user, chat_id, message_id, msg_in.type, msg_in.tag, msg_in.message)
    return success({})


@router.delete("/bots/{bot_id}/chats/{chat_id}")
async def delete_chat(request, bot_id: str, chat_id: str) -> view_models.Chat:
    user = get_user(request)
    chat = await query_chat(user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    chat.status = db_models.Chat.Status.DELETED
    chat.gmt_deleted = timezone.now()
    await chat.asave()
    history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
    await history.clear()
    return success(view_models.Chat(
        id=chat.id,
        title=chat.title,
        bot_id=chat.bot_id,
        peer_type=chat.peer_type,
        peer_id=chat.peer_id,
    ))


@router.post("/bots")
async def create_bot(request, bot_in: view_models.BotCreate) -> view_models.Bot:
    user = get_user(request)

    # there is quota limit on bot
    if settings.MAX_BOT_COUNT:
        bot_limit = await query_user_quota(user, QuotaType.MAX_BOT_COUNT)
        if bot_limit is None:
            bot_limit = settings.MAX_BOT_COUNT
        if await query_bots_count(user) >= bot_limit:
            return fail(HTTPStatus.FORBIDDEN, f"bot number has reached the limit of {bot_limit}")

    bot = db_models.Bot(
        user=user,
        title=bot_in.title,
        type=bot_in.type,
        status=db_models.Bot.Status.ACTIVE,
        description=bot_in.description,
        config=bot_in.config,
    )
    config = json.loads(bot_in.config)
    memory = config.get("memory", False)
    model_service_provider = config.get("model_service_provider")
    model_name = config.get("model_name")
    llm_config = config.get("llm")

    msp_dict = await query_msp_dict(user)
    if model_service_provider in msp_dict:
        msp = msp_dict[model_service_provider]
        base_url = msp.base_url
        api_key = msp.api_key
        valid, msg = validate_bot_config(model_service_provider, model_name, base_url, api_key, llm_config, bot_in.type, memory)
        if not valid:
            return fail(HTTPStatus.BAD_REQUEST, msg)
    else:
        return fail(HTTPStatus.BAD_REQUEST, "Model service provider not found")

    await bot.asave()
    collection_ids = []
    if bot_in.collection_ids is not None:
        for cid in bot_in.collection_ids:
            collection = await query_collection(user, cid)
            if not collection:
                return fail(HTTPStatus.NOT_FOUND, "Collection %s not found" % cid)
            if collection.status == db_models.Collection.Status.INACTIVE:
                return fail(HTTPStatus.BAD_REQUEST, "Collection %s is inactive" % cid)
            await db_models.BotCollectionRelation.objects.acreate(bot_id=bot.id, collection_id=cid)
            collection_ids.append(cid)
    await bot.asave()
    return success(view_models.Bot(
        id=bot.id,
        title=bot.title,
        type=bot.type,
        description=bot.description,
        config=bot.config,
        system=bot.user == settings.ADMIN_USER,
        collection_ids=collection_ids,
        created=bot.gmt_created,
        updated=bot.gmt_updated,
    ))


@router.get("/bots")
async def list_bots(request) -> view_models.BotList:
    user = get_user(request)
    pr = await query_bots([user, settings.ADMIN_USER], build_pq(request))
    response = []
    async for bot in pr.data:
        bot_config = json.loads(bot.config)
        model = bot_config.get("model", None)
        # This is a temporary solution to solve the problem of model name changes
        if model in ["chatgpt-3.5", "gpt-3.5-turbo-instruct"]:
            bot_config["model"] = "gpt-3.5-turbo"
        elif model == "chatgpt-4":
            bot_config["model"] = "gpt-4"
        elif model in ["gpt-4-vision-preview", "gpt-4-32k", "gpt-4-32k-0613"]:
            bot_config["model"] = "gpt-4-1106-preview"
        bot.config = json.dumps(bot_config)
        collection_ids = await bot.collections(only_ids=True)
        response.append(view_models.Bot(
            id=bot.id,
            title=bot.title,
            description=bot.description,
            type=bot.type,
            config=bot.config,
            collection_ids=collection_ids,
            created=bot.gmt_created,
            updated=bot.gmt_updated,
        ))
    return success(view_models.BotList(items=response), pr=pr)


@router.get("/bots/{bot_id}")
async def get_bot(request, bot_id: str) -> view_models.Bot:
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    collection_ids = await bot.collections(only_ids=True)
    return success(view_models.Bot(
        id=bot.id,
        title=bot.title,
        description=bot.description,
        type=bot.type,
        config=bot.config,
        collection_ids=collection_ids,
        created=bot.gmt_created.isoformat(),
        updated=bot.gmt_updated.isoformat(),
    ))


@router.put("/bots/{bot_id}")
async def update_bot(request, bot_id: str, bot_in: view_models.BotUpdate) -> view_models.Bot:
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    new_config = json.loads(bot_in.config)
    model_service_provider = new_config.get("model_service_provider")
    model_name = new_config.get("model_name")
    memory = new_config.get("memory", False)
    llm_config = new_config.get("llm")

    msp_dict = await query_msp_dict(user)
    if model_service_provider in msp_dict:
        msp = msp_dict[model_service_provider]
        base_url = msp.base_url
        api_key = msp.api_key
        valid, msg = validate_bot_config(model_service_provider, model_name, base_url, api_key, llm_config, bot_in.type, memory)
        if not valid:
            return fail(HTTPStatus.BAD_REQUEST, msg)
    else:
        return fail(HTTPStatus.BAD_REQUEST, "Model service provider not found")

    old_config = json.loads(bot.config)
    old_config.update(new_config)
    bot.config = json.dumps(old_config)
    bot.title = bot_in.title
    bot.type = bot_in.type
    bot.description = bot_in.description
    if bot_in.collection_ids is not None:
        await db_models.BotCollectionRelation.objects.filter(bot_id=bot.id, gmt_deleted__isnull=True).aupdate(gmt_deleted=timezone.now())
        for cid in bot_in.collection_ids:
            collection = await query_collection(user, cid)
            if not collection:
                return fail(HTTPStatus.NOT_FOUND, "Collection %s not found" % cid)
            if collection.status == db_models.Collection.Status.INACTIVE:
                return fail(HTTPStatus.BAD_REQUEST, "Collection %s is inactive" % cid)
            await db_models.BotCollectionRelation.objects.acreate(bot_id=bot.id, collection_id=cid)
    await bot.asave()

    collection_ids = await bot.collections(only_ids=True)
    return success(view_models.Bot(
        id=bot.id,
        title=bot.title,
        description=bot.description,
        config=bot.config,
        type=bot.type,
        collection_ids=collection_ids,
        created=bot.gmt_created.isoformat(),
        updated=bot.gmt_updated.isoformat(),
    ))


@router.delete("/bots/{bot_id}")
async def delete_bot(request, bot_id: str) -> view_models.Bot:
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    bot.status = db_models.Bot.Status.DELETED
    bot.gmt_deleted = timezone.now()
    await bot.asave()
    await db_models.BotCollectionRelation.objects.filter(bot_id=bot.id, gmt_deleted__isnull=True).aupdate(gmt_deleted=timezone.now())
    collection_ids = await bot.collections(only_ids=True)
    return success(view_models.Bot(
        id=bot.id,
        title=bot.title,
        description=bot.description,
        type=bot.type,
        collection_ids=collection_ids,
    ))


@router.get("/supported_model_service_providers")
async def list_model_service_providers(request) -> view_models.ModelServiceProviderList:
    user = get_user(request)
    response = []
    for supported_msp in settings.MODEL_CONFIGS:
        provider = view_models.ModelServiceProvider(
            name=supported_msp["name"],
            dialect=supported_msp["dialect"],
            label=supported_msp["label"],
            allow_custom_base_url=supported_msp["allow_custom_base_url"],
            base_url=supported_msp["base_url"],
        )
        response.append(provider)
    return success(view_models.ModelServiceProviderList(items=response))


@router.get("/model_service_providers")
async def list_model_service_providers(request) -> view_models.ModelServiceProviderList:
    user = get_user(request)

    supported_msp_dict = {msp["name"]: view_models.ModelConfig(**msp)
                          for msp in settings.MODEL_CONFIGS}

    msp_list = await query_msp_list(user)
    logger.info(msp_list)

    response = []
    for msp in msp_list:
        if msp.name in supported_msp_dict:
            supported_msp = supported_msp_dict[msp.name]
            response.append(view_models.ModelServiceProvider(
                name=msp.name,
                dialect=msp.dialect,
                label=supported_msp.label,
                allow_custom_base_url=supported_msp.allow_custom_base_url,
                base_url=msp.base_url,
                api_key=msp.api_key,
            ))

    return success(view_models.ModelServiceProviderList(items=response))


class ModelServiceProviderIn(Schema):
    name: str
    dialect: Optional[str] = None
    api_key: str
    base_url: Optional[str] = None
    extra: Optional[str] = None


@router.put("/model_service_providers/{provider}")
async def update_model_service_provider(request, provider, mspIn: ModelServiceProviderIn):
    user = get_user(request)

    supported_providers = [
        view_models.ModelConfig(**item)
        for item in settings.MODEL_CONFIGS
    ]
    supported_msp_names = {provider.name for provider in supported_providers if provider.name}

    if provider not in supported_msp_names:
        return fail(HTTPStatus.BAD_REQUEST, f"unsupported model service provider {provider}")

    msp_config = next(item for item in supported_providers if item.name == provider)
    if not msp_config.allow_custom_base_url and mspIn.base_url is not None:
        return fail(HTTPStatus.BAD_REQUEST, f"model service provider {provider} does not support setting base_url")

    msp = await query_msp(user, provider, filterDeletion=False)
    if msp is None:
        msp = db_models.ModelServiceProvider(
            user=user,
            name=provider,
            dialect=mspIn.dialect or msp_config.dialect,
            api_key=mspIn.api_key,
            base_url=mspIn.base_url if msp_config.allow_custom_base_url else msp_config.base_url,
            extra=mspIn.extra,
            status=db_models.ModelServiceProvider.Status.ACTIVE,
        )
    else:
        if msp.status == db_models.ModelServiceProvider.Status.DELETED:
            msp.status = db_models.ModelServiceProvider.Status.ACTIVE
            msp.gmt_deleted = None

        msp.dialect = mspIn.dialect or msp.dialect
        msp.api_key = mspIn.api_key
        if (msp_config.allow_custom_base_url and mspIn.base_url is not None):
            msp.base_url = mspIn.base_url
        msp.extra = mspIn.extra

    await msp.asave()
    return success({})


@router.delete("/model_service_providers/{provider}")
async def delete_model_service_provider(request, provider):
    user = get_user(request)

    supported_msp_names = {item["name"] for item in settings.MODEL_CONFIGS}
    if provider not in supported_msp_names:
        return fail(HTTPStatus.BAD_REQUEST, f"unsupported model service provider {provider}")

    msp = await query_msp(user, provider)
    if msp is None:
        return fail(HTTPStatus.NOT_FOUND, f"model service provider {provider} not found")

    msp.status = db_models.ModelServiceProvider.Status.DELETED
    msp.gmt_deleted = timezone.now()

    await msp.asave()
    return success({})


@router.get("/available_models")
async def list_available_models(request) -> view_models.ModelConfigList:
    user = get_user(request)

    supported_providers = [view_models.ModelConfig(**msp) for msp in
                           settings.MODEL_CONFIGS]
    supported_msp_dict = {provider.name: provider for provider in supported_providers}

    msp_list = await query_msp_list(user)
    logger.info(msp_list)

    available_providers = []
    for msp in msp_list:
        if msp.name in supported_msp_dict:
            available_providers.append(supported_msp_dict[msp.name])

    return success(view_models.ModelConfigList(items=available_providers, pageResult=None).model_dump(exclude_none=True))


def default_page(request, exception):
    return render(request, '404.html')


def dashboard(request):
    user_count = db_models.User.objects.count()
    collection_count = db_models.Collection.objects.count()
    document_count = db_models.Document.objects.count()
    chat_count = db_models.Chat.objects.count()
    context = {'user_count': user_count, 'Collection_count': collection_count,
               'Document_count': document_count, 'Chat_count': chat_count}
    return render(request, 'aperag/dashboard.html', context)


async def stream_frontend_sse_response(generator: AsyncGenerator[str, None], formatter, msg_id: str):
    """Stream SSE response for frontend format"""
    # Send start event
    yield f"data: {json.dumps(formatter.format_stream_start(msg_id))}\n\n"

    # Send content chunks
    async for chunk in generator:
        yield f"data: {json.dumps(formatter.format_stream_content(msg_id, chunk))}\n\n"

    # Send end event with additional metadata
    yield f"data: {json.dumps(formatter.format_stream_end(msg_id))}\n\n"


@router.post("/chat/completions/frontend")
async def frontend_chat_completions(request: HttpRequest):
    try:
        # Get user ID from request
        user = get_user(request)

        # Parse request parameters
        query_params = dict(parse_qsl(request.GET.urlencode()))
        body = request.body.decode("utf-8")

        # Create chat request
        chat_request = ChatRequest(
            user=user,
            bot_id=query_params.get("bot_id", ""),
            chat_id=query_params.get("chat_id", ""),
            msg_id=query_params.get("msg_id", "") or str(uuid.uuid4()),
            stream=query_params.get("stream", "false").lower() == "true",
            message=body
        )

        # Get bot
        bot = await query_bot(chat_request.user, chat_request.bot_id)
        if not bot:
            return StreamingHttpResponse(
                json.dumps(FrontendFormatter.format_error("Bot not found")),
                content_type="application/json"
            )

        # Get or create chat
        chat = await query_chat_by_peer(bot.user, Chat.PeerType.FEISHU, chat_request.chat_id)
        if chat is None:
            chat = Chat(
                user=bot.user,
                bot_id=bot.id,
                peer_type=Chat.PeerType.FEISHU,
                peer_id=chat_request.chat_id
            )
            await chat.asave()

        # Initialize message processor with history
        history = RedisChatMessageHistory(
            session_id=str(chat.id),
            redis_client=get_async_redis_client()
        )
        processor = MessageProcessor(bot, history)
        formatter = FrontendFormatter()

        # Process message and send response based on stream mode
        if chat_request.stream:
            return StreamingHttpResponse(
                stream_frontend_sse_response(
                    processor.process_message(chat_request.message, chat_request.msg_id),
                    formatter,
                    chat_request.msg_id
                ),
                content_type="text/event-stream"
            )
        else:
            # Collect all content for non-streaming mode
            full_content = ""
            async for chunk in processor.process_message(chat_request.message, chat_request.msg_id):
                full_content += chunk

            return StreamingHttpResponse(
                json.dumps(formatter.format_complete_response(chat_request.msg_id, full_content)),
                content_type="application/json"
            )

    except Exception as e:
        logger.exception(e)
        return StreamingHttpResponse(
            json.dumps(FrontendFormatter.format_error(str(e))),
            content_type="application/json"
        )

@router.post("/collections/{collection_id}/searchTests")
async def create_search_test(request, collection_id: str, data: view_models.SearchTestRequest) -> view_models.SearchTestResult:
    """
    Search test API, dynamically build flow according to search_type and execute.
    """
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if not collection:
        return fail(404, "Collection not found")
    # Build nodes and edges according to search_type
    nodes = {}
    edges = []
    query = data.query
    flow_id = str(uuid.uuid4())
    if data.search_type == "vector":
        # Only vector_search node
        node_id = "vector_search"
        nodes[node_id] = NodeInstance(
            id=node_id,
            type="vector_search",
            vars=[
                InputBinding(name="query", source_type=InputSourceType.STATIC, value=query),
                InputBinding(name="top_k", source_type=InputSourceType.STATIC, value=(data.vector_search.topk if data.vector_search else 5)),
                InputBinding(name="similarity_threshold", source_type=InputSourceType.STATIC, value=(data.vector_search.similarity if data.vector_search else 0.7)),
                InputBinding(name="collection_ids", source_type=InputSourceType.STATIC, value=[collection_id]),
            ]
        )
        output_node = node_id
    elif data.search_type == "fulltext":
        # Only keyword_search node
        node_id = "keyword_search"
        nodes[node_id] = NodeInstance(
            id=node_id,
            type="keyword_search",
            vars=[
                InputBinding(name="query", source_type=InputSourceType.STATIC, value=query),
                InputBinding(name="top_k", source_type=InputSourceType.STATIC, value=(data.vector_search.topk if data.vector_search else 5)),
                InputBinding(name="collection_ids", source_type=InputSourceType.STATIC, value=[collection_id]),
            ]
        )
        output_node = node_id
    elif data.search_type == "hybrid":
        # vector_search -> merge
        nodes["vector_search"] = NodeInstance(
            id="vector_search",
            type="vector_search",
            vars=[
                InputBinding(name="query", source_type=InputSourceType.STATIC, value=query),
                InputBinding(name="top_k", source_type=InputSourceType.STATIC, value=(data.vector_search.topk if data.vector_search else 5)),
                InputBinding(name="similarity_threshold", source_type=InputSourceType.STATIC, value=(data.vector_search.similarity if data.vector_search else 0.7)),
                InputBinding(name="collection_ids", source_type=InputSourceType.STATIC, value=[collection_id]),
            ]
        )
        nodes["keyword_search"] = NodeInstance(
            id="keyword_search",
            type="keyword_search",
            vars=[
                InputBinding(name="query", source_type=InputSourceType.STATIC, value=query),
                InputBinding(name="top_k", source_type=InputSourceType.STATIC, value=(data.vector_search.topk if data.vector_search else 5)),
                InputBinding(name="collection_ids", source_type=InputSourceType.STATIC, value=[collection_id]),
            ]
        )
        nodes["merge"] = NodeInstance(
            id="merge",
            type="merge",
            vars=[
                InputBinding(name="merge_strategy", source_type=InputSourceType.STATIC, value="union"),
                InputBinding(name="deduplicate", source_type=InputSourceType.STATIC, value=True),
                InputBinding(name="vector_search_docs", source_type=InputSourceType.DYNAMIC, ref_node="vector_search", ref_field="vector_search_docs"),
                InputBinding(name="keyword_search_docs", source_type=InputSourceType.DYNAMIC, ref_node="keyword_search", ref_field="keyword_search_docs"),
            ]
        )
        edges = [
            Edge(source="vector_search", target="merge"),
            Edge(source="keyword_search", target="merge"),
        ]
        output_node = "merge"
    else:
        return fail(400, "Invalid search_type")

    flow = FlowInstance(
        id=flow_id,
        name=f"search_test_{data.search_type}",
        nodes=nodes,
        edges=edges,
    )
    engine = FlowEngine()
    initial_data = {
        "query": query,
        "collection": collection
    }
    result = await engine.execute_flow(flow, initial_data)
    if not result:
        return fail(400, "Failed to execute flow")

    output_nodes = engine.find_output_nodes(flow)
    if not output_nodes:
        return fail(400, "No output node found")
    # Extract docs from output node
    output_node = output_nodes[0]
    docs = []
    if data.search_type == "vector":
        docs = result.get(output_node, {}).get("vector_search_docs", [])
    elif data.search_type == "fulltext":
        docs = result.get(output_node, {}).get("keyword_search_docs", [])
    elif data.search_type == "hybrid":
        docs = result.get(output_node, {}).get("docs", [])
    # Convert docs to SearchTestResult
    items = []
    for idx, doc in enumerate(docs):
        items.append(view_models.SearchTestResultItem(
            rank=idx+1,
            score=doc.score,
            content=doc.text,
            source=doc.metadata.get("source", "")
        ))
    record = await SearchTestHistory.objects.acreate(
        user=user,
        query=data.query,
        collection_id=collection_id,
        search_type=data.search_type,
        vector_search=data.vector_search.dict() if data.vector_search else None,
        fulltext_search=data.fulltext_search.dict() if data.fulltext_search else None,
        items=[item.dict() for item in items]
    )
    result = view_models.SearchTestResult(
        id=record.id,
        query=record.query,
        search_type=record.search_type,
        vector_search=record.vector_search,
        fulltext_search=record.fulltext_search,
        items=items,
        created=record.gmt_created.isoformat(),
    )
    return success(result)


@router.delete("/collections/{collection_id}/searchTests/{search_test_id}")
async def delete_search_test(request, collection_id: str, search_test_id: str):
    user = get_user(request)
    await SearchTestHistory.objects.filter(user=user, id=search_test_id, collection_id=collection_id, gmt_deleted__isnull=True).aupdate(gmt_deleted=timezone.now())
    return success({})

@router.get("/collections/{collection_id}/searchTests")
async def list_search_tests(request, collection_id: str) -> view_models.SearchTestResultList:
    user = get_user(request)
    qs = SearchTestHistory.objects.filter(user=user, collection_id=collection_id, gmt_deleted__isnull=True).order_by("-gmt_created")[:50]
    resultList = []
    async for record in qs:
        items = []
        for item in record.items:
            items.append(view_models.SearchTestResultItem(
                rank=item["rank"],
                score=item["score"],
                content=item["content"],
                source=item["source"],
            ))
        result = view_models.SearchTestResult(
            id=record.id,
            query=record.query,
            search_type=record.search_type,
            vector_search=record.vector_search,
            fulltext_search=record.fulltext_search,
            items=items,
            created=record.gmt_created.isoformat(),
        )
        resultList.append(result)
    return success(view_models.SearchTestResultList(items=resultList))

def _convert_to_serializable(obj):
    """Convert object to JSON serializable format
    
    Args:
        obj: Object to convert, can be BaseModel, dict, list, or primitive type
        
    Returns:
        JSON serializable object
    """
    if hasattr(obj, 'model_dump'):  # For Pydantic BaseModel
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):  # For other objects with __dict__
        return _convert_to_serializable(obj.__dict__)
    return obj

async def stream_flow_events(flow_generator: AsyncGenerator[Dict[str, Any], None], flow_task: asyncio.Task, execution_id: str):
    """Stream SSE response for flow events and output generator
    
    Args:
        flow_generator: Generator for flow execution events
        flow_task: Task for flow execution
        execution_id: Flow execution ID
    """
    try:
        # First stream all flow events
        async for event in flow_generator:
            # Send event data
            try:
                serializable_event = _convert_to_serializable(event)
                yield f"data: {json.dumps(serializable_event)}\n\n"
            except Exception as e:
                logger.exception(f"Error sending event {event}")
                raise e
            
            event_type = event.get("event_type")
            # If this is a flow end event, break to handle output generator
            if event_type == "flow_end":
                break
            
            if event_type == "flow_error":
                raise Exception(str(event))
            
        # Wait for flow execution to complete
        try:
            flow_result = await flow_task
            if not flow_result:
                raise Exception("Flow execution failed")

            # Find output nodes and their generators
            output_nodes = []
            for node_id, node_result in flow_result.items():
                if "async_generator" in node_result:
                    output_nodes.append((node_id, node_result["async_generator"]))
            
            if not output_nodes:
                raise Exception("No output nodes found")

            # Stream output from each output node
            for node_id, output_gen in output_nodes:
                try:
                    async for chunk in output_gen():
                        data = {
                            'event_type': 'output_chunk',
                            'node_id': node_id,
                            'execution_id': execution_id,
                            'timestamp': datetime.now().isoformat(),
                            'data': {'chunk':  _convert_to_serializable(chunk)}
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                except Exception as e:
                    logger.exception(f"Error streaming output from node {node_id}")
                    raise e

        except Exception as e:
            logger.exception(f"Error waiting for flow execution {execution_id}")
            raise e

    except asyncio.CancelledError:
        logger.info(f"Flow event stream cancelled for execution {execution_id}")
    except Exception as e:
        logger.exception(f"Error in flow event stream for execution {execution_id}")
        raise e

@router.post("/bots/{bot_id}/flow/debug")
async def debug_flow_stream(request: HttpRequest, bot_id: str, debug: view_models.DebugFlowRequest):
    """Debug flow execution with SSE event streaming"""
    try:
        user = get_user(request)
        bot = await query_bot(user, bot_id)
        if not bot:
            return StreamingHttpResponse(
                json.dumps({"error": "Bot not found"}),
                content_type="application/json"
            )

        # Parse flow configuration
        flow_config = debug.flow
        if not flow_config:
            flow_config = json.loads(bot.config)["flow"]
        flow = FlowParser.parse_yaml(flow_config)
        
        engine = FlowEngine()
        initial_data = {
            "query": debug.query,
            "bot": bot,
            "user": user,
            "history": [],
            "message_id": ""
        }
        # Start flow execution in background
        task = asyncio.create_task(
            engine.execute_flow(flow, initial_data)
        )
        
        # Return SSE response with events
        return StreamingHttpResponse(
            stream_flow_events(engine.get_events(), task, engine.execution_id),
            content_type="text/event-stream"
        )
        
    except Exception as e:
        logger.exception("Error in debug flow stream")
        return StreamingHttpResponse(
            json.dumps({"error": str(e)}),
            content_type="application/json"
        )
