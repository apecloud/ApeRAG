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
from typing import Any, Dict, List, Optional

from asgiref.sync import sync_to_async
from django.db.models import QuerySet
from pydantic import BaseModel

from aperag.db.models import (
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
    Config,
    Document,
    DocumentStatus,
    MessageFeedback,
    Question,
    QuestionStatus,
    UserQuota,
    ApiKeyToken,
    ApiKeyStatus,
    ModelServiceProvider,
    ModelServiceProviderStatus,
)

logger = logging.getLogger(__name__)


class PagedQuery(BaseModel):
    page_number: Optional[int] = None
    page_size: Optional[int] = None
    match_key: Optional[str] = None
    match_value: Optional[str] = None
    order_by: Optional[str] = None
    order_desc: Optional[bool] = None


class PagedResult(BaseModel):
    count: int
    page_number: int = 1
    page_size: int = 10
    data: Any


def build_pq(request) -> Optional[PagedQuery]:
    page_number = request.GET.get('page_number')
    page_size = request.GET.get('page_size')
    match_key = request.GET.get('match_key', "")
    match_value = request.GET.get('match_value', "")
    order_by = request.GET.get('order_by', "")
    order_desc = request.GET.get('order_desc', "true")
    return PagedQuery(
        page_number=int(page_number) if page_number else None,
        page_size=int(page_size) if page_size else None,
        match_key=match_key,
        match_value=match_value,
        order_by=order_by,
        order_desc=order_desc.lower() == "true",
    )


def build_order_by(pq: PagedQuery) -> str:
    if not pq or not pq.order_by:
        return "gmt_created"
    return f"{'-' if pq.order_desc else ''}{pq.order_by}"


async def build_pr(pq: PagedQuery, query_set: QuerySet) -> PagedResult:
    count = await get_count(query_set)
    query_set = query_set.order_by(build_order_by(pq))
    if not pq or not pq.page_number or not pq.page_size:
        return PagedResult(count=count, page_number=0, page_size=count, data=query_set)

    offset = (pq.page_number - 1) * pq.page_size
    limit = pq.page_size
    return PagedResult(count=count, page_number=pq.page_number, page_size=pq.page_size,
                       data=query_set[offset:offset + limit])


def build_filters(pq: PagedQuery) -> Dict:
    if not pq:
        return {}
    if not pq.match_key or not pq.match_value:
        return {}
    return {
        f"{pq.match_key}__icontains": pq.match_value
    }


async def get_count(collection) -> int:
    return await collection.acount()


async def query_collection(user, collection_id: str):
    try:
        return await Collection.objects.exclude(status=CollectionStatus.DELETED).aget(
            user=user, pk=collection_id
        )
    except Collection.DoesNotExist:
        return None


async def query_collection_without_user(collection_id: str):
    try:
        return await Collection.objects.exclude(status=CollectionStatus.DELETED).aget(pk=collection_id)
    except Collection.DoesNotExist:
        return None


async def query_collections(users: List[str], pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Collection.objects.exclude(status=CollectionStatus.DELETED).filter(user__in=users, **filters)
    return await build_pr(pq, query_set)


async def query_collections_count(user, pq: PagedQuery = None):
    filters = build_filters(pq)
    count = await sync_to_async(Collection.objects.exclude(status=CollectionStatus.DELETED).filter(user=user, **filters).count)()
    return count


async def query_document(user, collection_id: str, document_id: str):
    try:
        return await Document.objects.exclude(status=DocumentStatus.DELETED).aget(
            user=user, collection_id=collection_id, pk=document_id
        )
    except Document.DoesNotExist:
        return None


async def query_documents(users, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Document.objects.exclude(status=DocumentStatus.DELETED).filter(
        user__in=users, collection_id=collection_id, **filters)
    return await build_pr(pq, query_set)


async def query_documents_count(user, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    count = await sync_to_async(Document.objects.exclude(status=DocumentStatus.DELETED).filter(
        user=user, collection_id=collection_id, **filters).count)()
    return count


async def query_questions(user, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Question.objects.exclude(status=QuestionStatus.DELETED).filter(
        user=user, collection_id=collection_id, **filters
    )
    return await build_pr(pq, query_set)


async def query_question(user, question_id: str):
    try:
        return await Question.objects.exclude(status=QuestionStatus.DELETED).aget(
            user=user, pk=question_id
        )
    except Question.DoesNotExist:
        return None

async def query_apikeys(user, pq: PagedQuery = None):
    filters = build_filters(pq)
    apikey_set = ApiKeyToken.objects.exclude(status=CollectionStatus.DELETED).filter(user=user, **filters)
    return await build_pr(pq, apikey_set)

async def query_apikey(user, apikey_id: str):
    try:
        return await ApiKeyToken.objects.exclude(status=ApiKeyStatus.DELETED).aget(
            user=user, pk=apikey_id
        )
    except ApiKeyToken.DoesNotExist:
        return None

async def query_chat(user, bot_id: str, chat_id: str):
    try:
        kwargs = {
            "pk": chat_id,
            "bot_id": bot_id,
        }
        if user:
            kwargs["user"] = user
        return await Chat.objects.exclude(status=ChatStatus.DELETED).aget(**kwargs)
    except Chat.DoesNotExist:
        return None


# use this to query chat created from web
async def query_web_chat(bot_id: str, chat_id: str):
    try:
        kwargs = {
            "pk": chat_id,
            "bot_id": bot_id,
            "peer_type": ChatPeer.WEB,
        }
        return await Chat.objects.exclude(status=ChatStatus.DELETED).aget(**kwargs)
    except Chat.DoesNotExist:
        return None


async def query_chat_feedbacks(user, chat_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    if user:
        filters["user"] = user
    query_set = MessageFeedback.objects.filter(chat_id=chat_id, **filters)
    return await build_pr(pq, query_set)


async def query_message_feedback(user, bot_id: str, chat_id: str, message_id: str):
    try:
        return await MessageFeedback.objects.exclude.aget(
            user=user, bot_id=bot_id, chat_id=chat_id, message_id=message_id)
    except MessageFeedback.DoesNotExist:
        return None


async def query_chat_by_peer(user, peer_type, peer_id: str):
    try:
        return await Chat.objects.exclude(status=ChatStatus.DELETED).aget(
            user=user, peer_type=peer_type, peer_id=peer_id)
    except Chat.DoesNotExist:
        return None


async def query_sync_history(user, collection_id: str, sync_history_id: str):
    try:
        return await CollectionSyncHistory.objects.aget(
            user=user, collection_id=collection_id, pk=sync_history_id
        )
    except CollectionSyncHistory.DoesNotExist:
        return None


async def query_sync_histories(user, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = CollectionSyncHistory.objects.filter(user=user, collection_id=collection_id, **filters)
    return await build_pr(pq, query_set)


async def query_chats(user, bot_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Chat.objects.exclude(status=ChatStatus.DELETED).filter(user=user, bot_id=bot_id, **filters)
    return await build_pr(pq, query_set)


async def query_web_chats(bot_id: str, peer_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    filters.update({
        "peer_type": ChatPeer.WEB,
        "peer_id": peer_id,
    })
    query_set = Chat.objects.exclude(status=ChatStatus.DELETED).filter(bot_id=bot_id, **filters)
    return await build_pr(pq, query_set)


async def query_running_sync_histories(user, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = CollectionSyncHistory.objects.filter(
        user=user, collection_id=collection_id, status=CollectionSyncStatus.RUNNING, **filters,
    ).order_by('-id')
    return await build_pr(pq, query_set)


async def query_integration(user, bot_id: str, integration_id: str):
    try:
        kwargs = {
            "pk": integration_id,
            "bot_id": bot_id,
        }
        if user:
            kwargs["user"] = user
        return await BotIntegration.objects.exclude(status=BotIntegrationStatus.DELETED).aget(**kwargs)
    except BotIntegration.DoesNotExist:
        return None


async def query_bot(user, bot_id: str):
    try:
        kwargs = {
            "pk": bot_id
        }
        if user:
            kwargs["user"] = user
        return await Bot.objects.exclude(status=BotStatus.DELETED).aget(**kwargs)
    except Bot.DoesNotExist:
        return None


async def query_bots(users: List[str], pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Bot.objects.exclude(status=BotStatus.DELETED).filter(user__in=users, **filters)
    return await build_pr(pq, query_set)


async def query_bots_count(user, pq: PagedQuery = None):
    filters = build_filters(pq)
    count = await sync_to_async(Bot.objects.exclude(status=BotStatus.DELETED).filter(user=user, **filters).count)()
    return count


def query_config(key):
    results = Config.objects.filter(key=key).values_list('value', flat=True).first()
    return results


async def query_user_quota(user, key):
    result = await sync_to_async(UserQuota.objects.filter(user=user, key=key).values_list('value', flat=True).first)()
    return result


async def query_integrations(user, bot_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = BotIntegration.objects.exclude(status=BotIntegrationStatus.DELETED).filter(
        user=user, bot_id=bot_id, **filters)
    return await build_pr(pq, query_set)


async def query_msp_list(user):
    def query_msp_list_sync():
        return list(ModelServiceProvider.objects.exclude(status=ModelServiceProviderStatus.DELETED).filter(user=user))
    
    return await sync_to_async(query_msp_list_sync)()

async def query_msp_dict(user):
    def query_msp_dict_sync():
        msp_list = ModelServiceProvider.objects.exclude(status=ModelServiceProviderStatus.DELETED).filter(user=user)
        return {msp.name: msp for msp in msp_list}
    
    return await sync_to_async(query_msp_dict_sync)()

async def query_msp(user, provider, filterDeletion=True):
    try:
        if filterDeletion:
            return await ModelServiceProvider.objects.exclude(status=ModelServiceProviderStatus.DELETED).aget(user=user, pk=provider)
        else:
            return await ModelServiceProvider.objects.aget(user=user, pk=provider)
    except ModelServiceProvider.DoesNotExist:
        return None