import logging
from typing import Any, Dict, Optional

from django.db.models import QuerySet
from pydantic import BaseModel
from asgiref.sync import sync_to_async

from kubechat.db.models import (
    Bot,
    Chat,
    ChatStatus,
    Collection,
    CollectionStatus,
    CollectionSyncHistory,
    Document,
    DocumentStatus,
    BotStatus, MessageFeedback, CollectionSyncStatus, BotIntegration, BotIntegrationStatus, ChatPeer, Config, UserQuota
)


logger = logging.getLogger(__name__)


class PagedQuery(BaseModel):
    page_number: Optional[int]
    page_size: Optional[int]
    match_key: Optional[str]
    match_value: Optional[str]
    order_by: Optional[str]
    order_desc: Optional[bool]


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


async def query_collections(user, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Collection.objects.exclude(status=CollectionStatus.DELETED).filter(user=user, **filters)
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


async def query_documents(user, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Document.objects.exclude(status=DocumentStatus.DELETED).filter(
        user=user, collection_id=collection_id, **filters)
    return await build_pr(pq, query_set)


async def query_documents_count(user, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    count = await sync_to_async(Document.objects.exclude(status=DocumentStatus.DELETED).filter(
        user=user, collection_id=collection_id, **filters).count)()
    return count


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


async def query_bots(user, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Bot.objects.exclude(status=BotStatus.DELETED).filter(user=user, **filters)
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


