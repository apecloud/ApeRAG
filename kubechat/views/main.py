import datetime
import json
import logging
import os
from http import HTTPStatus
from typing import List, Optional

import redis.asyncio as redis
import yaml
from asgiref.sync import sync_to_async
from celery.result import GroupResult
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.db import IntegrityError
from django.shortcuts import render
from django.utils import timezone
from ninja import File, NinjaAPI, Router, Schema
from ninja.files import UploadedFile
from pydantic import BaseModel

import kubechat.chat.message
from config import settings
from config.celery import app
from kubechat.apps import QuotaType
from kubechat.auth.validator import GlobalHTTPAuth
from kubechat.chat.history.redis import RedisChatMessageHistory
from kubechat.db.models import (
    Bot,
    BotIntegration,
    BotIntegrationStatus,
    BotStatus,
    Chat,
    ChatPeer,
    ChatStatus,
    Collection,
    CollectionStatus,
    CollectionSyncHistory,
    CollectionSyncStatus,
    CollectionType,
    Document,
    DocumentStatus,
    Question,
    QuestionStatus,
    VerifyWay,
)
from kubechat.db.ops import (
    build_pq,
    query_bot,
    query_bots,
    query_bots_count,
    query_chat,
    query_chats,
    query_collection,
    query_collections,
    query_collections_count,
    query_document,
    query_documents,
    query_documents_count,
    query_integration,
    query_integrations,
    query_question,
    query_questions,
    query_running_sync_histories,
    query_sync_histories,
    query_sync_history,
    query_user_quota,
)
from kubechat.llm.base import Predictor
from kubechat.llm.prompts import (
    DEFAULT_CHINESE_PROMPT_TEMPLATE_V2,
    DEFAULT_CHINESE_PROMPT_TEMPLATE_V3,
    DEFAULT_MODEL_MEMOTY_PROMPT_TEMPLATES,
    DEFAULT_MODEL_PROMPT_TEMPLATES,
)
from kubechat.source.base import get_source
from kubechat.tasks.collection import delete_collection_task, init_collection_task
from kubechat.tasks.crawl_web import crawl_domain
from kubechat.tasks.index import (
    add_index_for_local_document,
    generate_questions,
    message_feedback,
    remove_index,
    update_index,
    update_index_for_question,
)
from kubechat.tasks.scan import delete_sync_documents_cron_job, update_sync_documents_cron_job
from kubechat.tasks.sync_documents_task import get_sync_progress, sync_documents
from kubechat.utils.request import get_urls, get_user
from kubechat.views.utils import (
    add_ssl_file,
    fail,
    query_chat_messages,
    success,
    validate_bot_config,
    validate_source_connect_config,
    validate_url,
)
from readers.base_readers import DEFAULT_FILE_READER_CLS

logger = logging.getLogger(__name__)

api = NinjaAPI(version="1.0.0", auth=GlobalHTTPAuth(), urls_namespace="collection")
router = Router()


class CollectionIn(Schema):
    title: str
    type: str
    description: Optional[str]
    config: Optional[str]


class CreateDocumentIn(Schema):
    name: str
    config: Optional[str]


class UpdateDocumentIn(Schema):
    name: str
    config: Optional[str]

class QuestionIn(Schema):
    id: Optional[str]
    question: str
    answer: str
    relate_ducuments: Optional[List[str]]

class ConnectionInfo(Schema):
    db_type: str
    host: str
    port: Optional[int]
    db_name: Optional[str]
    username: Optional[str]
    password: Optional[str]
    verify: VerifyWay
    ca_cert: Optional[str]
    client_key: Optional[str]
    client_cert: Optional[str]


class Auth(BaseModel):
    username: str = ""
    password: str = ""


class Host(BaseModel):
    source: str = "system"
    host: str = ""
    port: str = ""


class DocumentConfig(BaseModel):
    host: Host = Host
    auth: Auth = Auth
    filedir: str = ""
    filepath: str = ""


@router.get("/models")
def list_models(request):
    response = []
    model_families = yaml.safe_load(settings.MODEL_FAMILIES)
    for model_family in model_families:
        for model_server in model_family.get("models", []):
            response.append({
                "value": model_server["name"],
                "label": model_server.get("label", model_server["name"]),
                "enabled": model_server.get("enabled", "true").lower() == "true",
                "memory": model_server.get("memory", "disabled").lower() == "enabled",
                "default_token": Predictor.check_default_token(model_name=model_server["name"]),
                "prompt_template": DEFAULT_MODEL_MEMOTY_PROMPT_TEMPLATES.get(model_server["name"],
                                                                                    DEFAULT_CHINESE_PROMPT_TEMPLATE_V3),
                "context_window": model_server.get("context_window", 7500),
                "temperature": model_server.get("temperature", model_family.get("temperature", 0.01)),
                "similarity_score_threshold": model_server.get("similarity_score_threshold", 0.5),
                "similarity_topk": model_server.get("similarity_topk", 3),
                "family_name": model_family["name"],
                "family_label": model_family["label"],
            })
    response.sort(key=lambda x: x["enabled"], reverse=True)
    return success(response)


@router.post("/collections/{collection_id}/sync")
async def sync_immediately(request, collection_id):
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    source = get_source(json.loads(collection.config))
    if not source.sync_enabled():
        return fail(HTTPStatus.BAD_REQUEST, "source type not supports sync")

    pr = await query_running_sync_histories(user, collection_id)
    async for task in pr.data:
        return fail(HTTPStatus.BAD_REQUEST, f"have running sync task {task.id}, please cancel it first")

    instance = CollectionSyncHistory(
        user=collection.user,
        start_time=timezone.now(),
        collection=collection,
        execution_time=datetime.timedelta(seconds=0),
        total_documents_to_sync=0,
        status=CollectionSyncStatus.RUNNING,
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

    sync_history.status = CollectionSyncStatus.CANCELED
    await sync_history.asave()
    return success({})


@router.get("/collections/{collection_id}/sync/history")
async def list_sync_histories(request, collection_id):
    user = get_user(request)
    pr = await query_sync_histories(user, collection_id, build_pq(request))
    response = []
    async for sync_history in pr.data:
        if sync_history.status == CollectionSyncStatus.RUNNING:
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
    if sync_history.status == CollectionSyncStatus.RUNNING:
        progress = get_sync_progress(sync_history)
        sync_history.failed_documents = progress.failed_documents
        sync_history.successful_documents = progress.successful_documents
        sync_history.processing_documents = progress.processing_documents
        sync_history.pending_documents = progress.pending_documents
    return success(sync_history.view())


@router.post("/collections")
async def create_collection(request, collection: CollectionIn):
    user = get_user(request)
    config = json.loads(collection.config)
    if collection.type == CollectionType.DOCUMENT:
        is_validate, error_msg = validate_source_connect_config(config)
        if not is_validate:
            return fail(HTTPStatus.BAD_REQUEST, error_msg)

    if config.get("source") == "tencent":
        redis_client = redis.Redis.from_url(settings.MEMORY_REDIS_URL)
        if await redis_client.exists("tencent_code_" + user):
            code = await redis_client.get("tencent_code_" + user)
            redirect_uri = await redis_client.get("tencent_redirect_uri_" + user)
            config["code"] = code.decode()
            config["redirect_uri"] = redirect_uri
            collection.config = json.dumps(config)
        else:
            return fail(HTTPStatus.BAD_REQUEST, "用户未进行授权或授权已过期，请重新操作")

    # there is quota limit on collection
    if settings.MAX_COLLECTION_COUNT:
        collection_limit = await query_user_quota(user, QuotaType.MAX_COLLECTION_COUNT)
        if collection_limit is None:
            collection_limit = settings.MAX_COLLECTION_COUNT
        if collection_limit and await query_collections_count(user) >= collection_limit:
            return fail(HTTPStatus.FORBIDDEN, f"collection number has reached the limit of {collection_limit}")

    instance = Collection(
        user=user,
        type=collection.type,
        status=CollectionStatus.INACTIVE,
        title=collection.title,
        description=collection.description
    )

    if collection.config is not None:
        instance.config = collection.config
    await instance.asave()

    if instance.type == CollectionType.DATABASE:
        if config["verify"] != VerifyWay.PREFERRED:
            add_ssl_file(config, instance)
            collection.config = json.dumps(config)
            await instance.asave()
    elif instance.type == CollectionType.DOCUMENT:
        document_user_quota = await query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
        init_collection_task.delay(instance.id, document_user_quota)
    elif instance.type == CollectionType.CODE:
        chat = Chat(
            user=instance.user,
            status=ChatStatus.ACTIVE,
            collection=instance,
            summary=instance.description,
        )
        await chat.asave()
    else:
        return fail(HTTPStatus.BAD_REQUEST, "unknown collection type")

    return success(instance.view())


@router.get("/collections")
async def list_collections(request):
    user = get_user(request)
    pr = await query_collections(user, build_pq(request))
    response = []
    async for collection in pr.data:
        response.append(collection.view())
    return success(response, pr)


@router.get("/default_collections")
async def list_system_collections(request):
    pr = await query_collections(settings.ADMIN_USER, build_pq(request))
    response = []
    async for collection in pr.data:
        response.append(collection.view())
    return success(response, pr)


@router.get("/collections/{collection_id}")
async def get_collection(request, collection_id):
    user = get_user(request)
    instance = await query_collection(user, collection_id)
    if instance is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")

    bots = await sync_to_async(instance.bot_set.exclude)(status=BotStatus.DELETED)
    bot_ids = []
    async for bot in bots:
        bot_ids.append(bot.id)
    return success(instance.view(bot_ids=bot_ids))


@router.put("/collections/{collection_id}")
async def update_collection(request, collection_id, collection: CollectionIn):
    user = get_user(request)
    instance = await query_collection(user, collection_id)
    if instance is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    instance.title = collection.title
    instance.description = collection.description
    instance.config = collection.config
    await instance.asave()
    source = get_source(json.loads(collection.config))
    if source.sync_enabled():
        await update_sync_documents_cron_job(instance.id)

    bots = await sync_to_async(instance.bot_set.exclude)(status=BotStatus.DELETED)
    bot_ids = []
    async for bot in bots:
        bot_ids.append(bot.id)
           
    return success(instance.view(bot_ids=bot_ids))


@router.delete("/collections/{collection_id}")
async def delete_collection(request, collection_id):
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    await delete_sync_documents_cron_job(collection.id)
    bots = await sync_to_async(collection.bot_set.exclude)(status=BotStatus.DELETED)
    bot_ids = []
    async for bot in bots:
        bot_ids.append(bot.id)
    if len(bot_ids) > 0:
        return fail(HTTPStatus.BAD_REQUEST, f"Collection has related to bots {','.join(bot_ids)}, can not be deleted")
    collection.status = CollectionStatus.DELETED
    collection.gmt_deleted = timezone.now()
    await collection.asave()

    delete_collection_task.delay(collection_id)

    return success(collection.view())

@router.post("/collections/{collection_id}/questions")
async def create_questions(request, collection_id):
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    documents = await sync_to_async(collection.document_set.exclude)(status=DocumentStatus.DELETED)
    collection.need_generate = len(documents)
    await collection.asave()
    async for document in documents:
        generate_questions.delay(document.id)
    return success({}) 

@router.put("/collections/{collection_id}/questions")
async def update_question(request, collection_id, question_in: QuestionIn):
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    
    # ceate question
    if not question_in.id or question_in.id == "":
        question_instance = Question(
            user=collection.user,
            collection=collection,
            status=QuestionStatus.PENDING,
        )
        await question_instance.asave()
    else:
        question_instance = await query_question(user, collection_id, question_in.id)
        if question_instance is None:
            return fail(HTTPStatus.NOT_FOUND, "Question not found") 
    
    question_instance.question = question_in.question
    question_instance.answer = question_in.answer
    question_instance.status = QuestionStatus.PENDING
    question_instance.documents.clear()
    
    if question_in.relate_ducuments:
        for document_id in question_in.relate_ducuments:
            document = await query_document(user, collection_id, document_id)
            if document is None or document.status == DocumentStatus.DELETED:
                return fail(HTTPStatus.NOT_FOUND, "Document not found")
            question_instance.documents.add(document)
            
    await question_instance.asave()
    update_index_for_question.delay(question_instance.id)
    
    return success(question_instance.view()) 

@router.delete("/collections/{collection_id}/questions/{question_id}")
async def delete_question(request, collection_id, question_id):
    
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    
    question_instance = Question.objects.get(id=question_id)      
    question_instance.status = QuestionStatus.DELETED
    question_instance.gmt_deleted = timezone.now() 
    await question_instance.asave()
    update_index_for_question.delay(question_instance.id)
    
    return success(question_instance.view()) 

@router.get("/collections/{collection_id}/questions")
async def list_questions(request, collection_id):
    user = get_user(request)
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
        
    pr = await query_questions(user, collection_id, build_pq(request))
    response = []
    async for question in pr.data:
        response.append(question.view())
        
    question_status = QuestionStatus.ACTIVE
    if collection.need_generate > 0:
        question_status = QuestionStatus.PENDING
    return success(response, pr, question_status) 

@router.post("/collections/{collection_id}/documents")
async def create_document(request, collection_id, file: List[UploadedFile] = File(...)):
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
            document_instance = Document(
                user=user,
                name=item.name,
                status=DocumentStatus.PENDING,
                size=item.size,
                collection=collection,
                file=ContentFile(item.read(), item.name),
            )
            await document_instance.asave()
            document_instance.metadata = json.dumps({
                "path": document_instance.file.path,
            })
            await document_instance.asave()
            response.append(document_instance.view())
            add_index_for_local_document.delay(document_instance.id)
        except IntegrityError:
            return fail(HTTPStatus.BAD_REQUEST, f"document {item.name} already exists")
        except Exception:
            logger.exception("add document failed")
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, "add document failed")
    return success(response)


@router.post("/collections/{collection_id}/urls")
async def create_url_document(request, collection_id):
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
            document_instance = Document(
                user=user,
                name=document_name,
                status=DocumentStatus.PENDING,
                collection=collection,
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
async def list_documents(request, collection_id):
    user = get_user(request)
    pr = await query_documents(user, collection_id, build_pq(request))
    response = []
    async for document in pr.data:
        response.append(document.view())
    return success(response, pr)


@router.put("/collections/{collection_id}/documents/{document_id}")
async def update_document(
        request, collection_id, document_id, document: UpdateDocumentIn):
    user = get_user(request)
    instance = await query_document(user, collection_id, document_id)
    if instance is None:
        return fail(HTTPStatus.NOT_FOUND, "Document not found")
    if instance.status == DocumentStatus.DELETING:
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
    update_index.delay(instance.id)
    
    related_questions = await sync_to_async(document.question_set.exclude)(status=QuestionStatus.DELETED)
    async for question in related_questions:
        question.status = QuestionStatus.WARNING
        await question.asave()

    return success(instance.view())


@router.delete("/collections/{collection_id}/documents/{document_id}")
async def delete_document(request, collection_id, document_id):
    user = get_user(request)
    document = await query_document(user, collection_id, document_id)
    if document is None:
        logger.info(f"document {document_id} not found, maybe has already been deleted")
        return success({})
    if document.status == DocumentStatus.DELETING:
        logger.info(f"document {document_id} is deleting, ignore delete")
        return success({})
    document.status = DocumentStatus.DELETING
    document.gmt_deleted = timezone.now()
    await document.asave()

    remove_index.delay(document.id)
    
    related_questions = await sync_to_async(document.question_set.exclude)(status=QuestionStatus.DELETED)
    async for question in related_questions:
        question.documents.remove(document)
        question.status = QuestionStatus.WARNING
        await question.asave()
    
    return success(document.view())


@router.delete("/collections/{collection_id}/documents")
async def delete_documents(request, collection_id, document_ids: List[str]):
    user = get_user(request)
    documents = await query_documents(user, collection_id, build_pq(request))
    ok = []
    failed = []
    async for document in documents.data:
        if document.id not in document_ids:
            continue
        try:
            document.status = DocumentStatus.DELETING
            document.gmt_deleted = timezone.now()
            await document.asave()
            remove_index.delay(document.id)
            
            related_questions = await sync_to_async(document.question_set.exclude)(status=QuestionStatus.DELETED)
            async for question in related_questions:
                question.documents.remove(document)
                question.status = QuestionStatus.WARNING
                await question.asave()
                
            ok.append(document.id)
        except Exception as e:
            logger.exception(e)
            failed.append(document.id)
    return success({"success": ok, "failed": failed})


@router.post("/bots/{bot_id}/chats")
async def create_chat(request, bot_id):
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    instance = Chat(user=user, bot=bot, peer_type=ChatPeer.SYSTEM)
    await instance.asave()
    return success(instance.view(bot_id))


@router.get("/bots/{bot_id}/chats")
async def list_chats(request, bot_id):
    user = get_user(request)
    pr = await query_chats(user, bot_id, build_pq(request))
    response = []
    async for chat in pr.data:
        response.append(chat.view(bot_id))
    return success(response, pr)


@router.put("/bots/{bot_id}/chats/{chat_id}")
async def update_chat(request, bot_id, chat_id):
    user = get_user(request)
    chat = await query_chat(user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    chat.summary = ""
    await chat.asave()
    history = RedisChatMessageHistory(chat_id, settings.MEMORY_REDIS_URL)
    await history.clear()
    return success(chat.view(bot_id))


@router.get("/bots/{bot_id}/chats/{chat_id}")
async def get_chat(request, bot_id, chat_id):
    user = get_user(request)
    chat = await query_chat(user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")

    messages = await query_chat_messages(user, chat_id)
    return success(chat.view(bot_id, messages))


class MessageFeedbackIn(Schema):
    upvote: Optional[int]
    downvote: Optional[int]
    revised_answer: Optional[str]


@router.post("/bots/{bot_id}/chats/{chat_id}/messages/{message_id}")
async def feedback_message(request, bot_id, chat_id, message_id, msg_in: MessageFeedbackIn):
    user = get_user(request)
    chat = await query_chat(user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    feedback = await kubechat.chat.message.feedback_message(chat.user, chat_id, message_id, msg_in.upvote,
                                                            msg_in.downvote,
                                                            msg_in.revised_answer)

    # embedding the revised answer
    if msg_in.revised_answer is not None:
        message_feedback.delay(feedback_id=feedback.id)
    return success({})


@router.delete("/bots/{bot_id}/chats/{chat_id}")
async def delete_chat(request, bot_id, chat_id):
    user = get_user(request)
    chat = await query_chat(user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    chat.status = ChatStatus.DELETED
    chat.gmt_deleted = timezone.now()
    await chat.asave()
    history = RedisChatMessageHistory(chat_id, settings.MEMORY_REDIS_URL)
    await history.clear()
    return success(chat.view(bot_id))


class BotIn(Schema):
    title: str
    type: str
    description: Optional[str]
    config: Optional[str]
    collection_ids: Optional[List[str]]


@router.post("/bots")
async def create_bot(request, bot_in: BotIn):
    user = get_user(request)

    # there is quota limit on bot
    if settings.MAX_BOT_COUNT:
        bot_limit = await query_user_quota(user, QuotaType.MAX_BOT_COUNT)
        if bot_limit is None:
            bot_limit = settings.MAX_BOT_COUNT
        if await query_bots_count(user) >= bot_limit:
            return fail(HTTPStatus.FORBIDDEN, f"bot number has reached the limit of {bot_limit}")

    bot = Bot(
        user=user,
        title=bot_in.title,
        type=bot_in.type,
        status=BotStatus.ACTIVE,
        description=bot_in.description,
        config=bot_in.config,
    )
    config = json.loads(bot_in.config)
    memory = config.get("memory", False)
    model = config.get("model")
    llm_config = config.get("llm")
    valid, msg = validate_bot_config(model, llm_config, bot_in.type, memory)
    if not valid:
        return fail(HTTPStatus.BAD_REQUEST, msg)
    await bot.asave()
    collections = []
    if bot_in.collection_ids is not None:
        for cid in bot_in.collection_ids:
            collection = await query_collection(user, cid)
            if not collection:
                return fail(HTTPStatus.NOT_FOUND, "Collection %s not found" % cid)
            if collection.status == CollectionStatus.INACTIVE:
                return fail(HTTPStatus.BAD_REQUEST, "Collection %s is inactive" % cid)
            await sync_to_async(bot.collections.add)(collection)
            collections.append(collection.view())
    await bot.asave()
    return success(bot.view(collections))


@router.get("/bots")
async def list_bots(request):
    user = get_user(request)
    pr = await query_bots(user, build_pq(request))
    response = []
    async for bot in pr.data:
        collections = []
        async for collection in await sync_to_async(bot.collections.all)():
            collections.append(collection.view())
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
        response.append(bot.view(collections))
    return success(response, pr)


@router.get("/system_bots")
async def list_system_bots(request):
    pr = await query_bots(settings.ADMIN_USER, build_pq(request))
    response = []
    async for bot in pr.data:
        collections = []
        async for collection in await sync_to_async(bot.collections.all)():
            collections.append(collection.view())
        response.append(bot.view(collections))
    return success(response, pr)


@router.get("/bots/{bot_id}")
async def get_bot(request, bot_id):
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    collections = []
    async for collection in await sync_to_async(bot.collections.all)():
        collections.append(collection.view())
    return success(bot.view(collections))


@router.put("/bots/{bot_id}")
async def update_bot(request, bot_id, bot_in: BotIn):
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    new_config = json.loads(bot_in.config)
    model = new_config.get("model")
    memory = new_config.get("memory", False)
    llm_config = new_config.get("llm")
    valid, msg = validate_bot_config(model, llm_config, bot_in.type, memory)
    if not valid:
        return fail(HTTPStatus.BAD_REQUEST, msg)
    old_config = json.loads(bot.config)
    old_config.update(new_config)
    bot.config = json.dumps(old_config)
    bot.title = bot_in.title
    bot.type = bot_in.type
    bot.description = bot_in.description
    if bot_in.collection_ids is not None:
        collections = []
        for cid in bot_in.collection_ids:
            collection = await query_collection(user, cid)
            if not collection:
                return fail(HTTPStatus.NOT_FOUND, "Collection %s not found" % cid)
            if collection.status == CollectionStatus.INACTIVE:
                return fail(HTTPStatus.BAD_REQUEST, "Collection %s is inactive" % cid)
            collections.append(collection)
        await sync_to_async(bot.collections.set)(collections)
    await bot.asave()

    collections = []
    async for collection in await sync_to_async(bot.collections.all)():
        collections.append(collection.view())
    return success(bot.view(collections))


@router.delete("/bots/{bot_id}")
async def delete_bot(request, bot_id):
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    bot.status = BotStatus.DELETED
    bot.gmt_deleted = timezone.now()
    await bot.asave()
    return success(bot.view())


class IntegrationIn(Schema):
    type: str
    config: Optional[str]


@router.post("/bots/{bot_id}/integrations")
async def create_integration(request, bot_id, integration: IntegrationIn):
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    instance = BotIntegration(
        user=user,
        bot=bot,
        type=integration.type,
        config=integration.config,
    )
    await instance.asave()
    return success(instance.view(bot_id))


@router.get("/bots/{bot_id}/integrations")
async def list_integrations(request, bot_id):
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    pr = await query_integrations(user, bot_id, build_pq(request))
    response = []
    async for integration in pr.data:
        response.append(integration.view(bot_id))
    return success(response, pr)


@router.get("/bots/{bot_id}/integrations/{integration_id}")
async def get_integration(user, bot_id, integration_id):
    bot = await query_bot(user, bot_id)
    if bot is None:
        return None
    integration = await query_integration(user, bot_id, integration_id)
    if integration is None:
        return None
    return integration


@router.put("/bots/{bot_id}/integrations/{integration_id}")
async def update_integration(user, bot_id, integration_id, integration_in: IntegrationIn):
    bot = await query_bot(user, bot_id)
    if bot is None:
        return None
    integration = await query_integration(user, bot_id, integration_id)
    if integration is None:
        return None
    integration.type = integration_in.type
    integration.config = integration_in.config
    await integration.asave()
    return integration


@router.delete("/bots/{bot_id}/integrations/{integration_id}")
async def delete_integration(user, bot_id, integration_id):
    bot = await query_bot(user, bot_id)
    if bot is None:
        return None
    integration = await query_integration(user, bot_id, integration_id)
    if integration is None:
        return None
    integration.status = BotIntegrationStatus.DELETED
    integration.gmt_deleted = timezone.now()
    await integration.asave()
    return integration


def default_page(request, exception):
    return render(request, '404.html')


def dashboard(request):
    user_count = User.objects.count()
    collection_count = Collection.objects.count()
    document_count = Document.objects.count()
    chat_count = Chat.objects.count()
    context = {'user_count': user_count, 'Collection_count': collection_count,
               'Document_count': document_count, 'Chat_count': chat_count}
    return render(request, 'kubechat/dashboard.html', context)
