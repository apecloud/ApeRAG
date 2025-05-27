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
from django.contrib.auth import aauthenticate, alogin, alogout
from django.contrib.auth.hashers import make_password
from django.db.models import QuerySet
from django.utils import timezone
from pydantic import BaseModel

from aperag.db.models import (
    ApiKey,
    Bot,
    Chat,
    Collection,
    CollectionSyncHistory,
    Config,
    Document,
    Invitation,
    MessageFeedback,
    ModelServiceProvider,
    Question,
    Role,
    User,
    UserQuota,
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
    page_number = request.GET.get("page_number")
    page_size = request.GET.get("page_size")
    match_key = request.GET.get("match_key", "")
    match_value = request.GET.get("match_value", "")
    order_by = request.GET.get("order_by", "")
    order_desc = request.GET.get("order_desc", "true")
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
    return PagedResult(
        count=count, page_number=pq.page_number, page_size=pq.page_size, data=query_set[offset : offset + limit]
    )


def build_filters(pq: PagedQuery) -> Dict:
    if not pq:
        return {}
    if not pq.match_key or not pq.match_value:
        return {}
    return {f"{pq.match_key}__icontains": pq.match_value}


async def get_count(collection) -> int:
    return await collection.acount()


async def query_collection(user, collection_id: str):
    try:
        return await Collection.objects.exclude(status=Collection.Status.DELETED).aget(user=user, pk=collection_id)
    except Collection.DoesNotExist:
        return None


async def query_collection_without_user(collection_id: str):
    try:
        return await Collection.objects.exclude(status=Collection.Status.DELETED).aget(pk=collection_id)
    except Collection.DoesNotExist:
        return None


async def query_collections(users: List[str], pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Collection.objects.exclude(status=Collection.Status.DELETED).filter(user__in=users, **filters)
    return await build_pr(pq, query_set)


async def query_collections_count(user, pq: PagedQuery = None):
    filters = build_filters(pq)
    count = await sync_to_async(
        Collection.objects.exclude(status=Collection.Status.DELETED).filter(user=user, **filters).count
    )()
    return count


async def query_document(user, collection_id: str, document_id: str):
    try:
        return await Document.objects.exclude(status=Document.Status.DELETED).aget(
            user=user, collection_id=collection_id, pk=document_id
        )
    except Document.DoesNotExist:
        return None


async def query_documents(users, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Document.objects.exclude(status=Document.Status.DELETED).filter(
        user__in=users, collection_id=collection_id, **filters
    )
    return await build_pr(pq, query_set)


async def query_documents_count(user, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    count = await sync_to_async(
        Document.objects.exclude(status=Document.Status.DELETED)
        .filter(user=user, collection_id=collection_id, **filters)
        .count
    )()
    return count


async def query_questions(user, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Question.objects.exclude(status=Question.Status.DELETED).filter(
        user=user, collection_id=collection_id, **filters
    )
    return await build_pr(pq, query_set)


async def query_question(user, question_id: str):
    try:
        return await Question.objects.exclude(status=Question.Status.DELETED).aget(user=user, pk=question_id)
    except Question.DoesNotExist:
        return None


async def query_apikeys(user, pq: PagedQuery = None):
    filters = build_filters(pq)
    apikey_set = ApiKey.objects.exclude(status=ApiKey.Status.DELETED).filter(user=user, **filters)
    return await build_pr(pq, apikey_set)


async def query_apikey(user, apikey_id: str):
    try:
        return await ApiKey.objects.exclude(status=ApiKey.Status.DELETED).aget(user=user, pk=apikey_id)
    except ApiKey.DoesNotExist:
        return None


async def query_chat(user, bot_id: str, chat_id: str):
    try:
        kwargs = {
            "pk": chat_id,
            "bot_id": bot_id,
        }
        if user:
            kwargs["user"] = user
        return await Chat.objects.exclude(status=Chat.Status.DELETED).aget(**kwargs)
    except Chat.DoesNotExist:
        return None


# use this to query chat created from web
async def query_web_chat(bot_id: str, chat_id: str):
    try:
        kwargs = {
            "pk": chat_id,
            "bot_id": bot_id,
            "peer_type": Chat.PeerType.WEB,
        }
        return await Chat.objects.exclude(status=Chat.Status.DELETED).aget(**kwargs)
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
            user=user, bot_id=bot_id, chat_id=chat_id, message_id=message_id
        )
    except MessageFeedback.DoesNotExist:
        return None


async def query_chat_by_peer(user, peer_type, peer_id: str):
    try:
        return await Chat.objects.exclude(status=Chat.Status.DELETED).aget(
            user=user, peer_type=peer_type, peer_id=peer_id
        )
    except Chat.DoesNotExist:
        return None


async def query_sync_history(user, collection_id: str, sync_history_id: str):
    try:
        return await CollectionSyncHistory.objects.aget(user=user, collection_id=collection_id, pk=sync_history_id)
    except CollectionSyncHistory.DoesNotExist:
        return None


async def query_sync_histories(user, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = CollectionSyncHistory.objects.filter(user=user, collection_id=collection_id, **filters)
    return await build_pr(pq, query_set)


async def query_chats(user, bot_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Chat.objects.exclude(status=Chat.Status.DELETED).filter(user=user, bot_id=bot_id, **filters)
    return await build_pr(pq, query_set)


async def query_web_chats(bot_id: str, peer_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    filters.update({
        "peer_type": Chat.PeerType.WEB,
        "peer_id": peer_id,
    })
    query_set = Chat.objects.exclude(status=Chat.Status.DELETED).filter(bot_id=bot_id, **filters)
    return await build_pr(pq, query_set)


async def query_running_sync_histories(user, collection_id: str, pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = CollectionSyncHistory.objects.filter(
        user=user,
        collection_id=collection_id,
        status=Collection.SyncStatus.RUNNING,
        **filters,
    ).order_by("-id")
    return await build_pr(pq, query_set)


async def query_bot(user: str, bot_id: str):
    try:
        kwargs = {"pk": bot_id}
        if user:
            kwargs["user"] = user
        return await Bot.objects.exclude(status=Bot.Status.DELETED).aget(**kwargs)
    except Bot.DoesNotExist:
        return None


async def query_bots(users: List[str], pq: PagedQuery = None):
    filters = build_filters(pq)
    query_set = Bot.objects.exclude(status=Bot.Status.DELETED).filter(user__in=users, **filters)
    return await build_pr(pq, query_set)


async def query_bots_count(user, pq: PagedQuery = None):
    filters = build_filters(pq)
    count = await sync_to_async(Bot.objects.exclude(status=Bot.Status.DELETED).filter(user=user, **filters).count)()
    return count


def query_config(key):
    results = Config.objects.filter(key=key).values_list("value", flat=True).first()
    return results


async def query_user_quota(user, key):
    result = await sync_to_async(UserQuota.objects.filter(user=user, key=key).values_list("value", flat=True).first)()
    return result


async def query_msp_list(user):
    def query_msp_list_sync():
        return list(ModelServiceProvider.objects.exclude(status=ModelServiceProvider.Status.DELETED).filter(user=user))

    return await sync_to_async(query_msp_list_sync)()


async def query_msp_dict(user):
    def query_msp_dict_sync():
        msp_list = ModelServiceProvider.objects.exclude(status=ModelServiceProvider.Status.DELETED).filter(user=user)
        return {msp.name: msp for msp in msp_list}

    return await sync_to_async(query_msp_dict_sync)()


async def query_msp(user, provider, filterDeletion=True):
    try:
        if filterDeletion:
            return await ModelServiceProvider.objects.exclude(status=ModelServiceProvider.Status.DELETED).aget(
                user=user, pk=provider
            )
        else:
            return await ModelServiceProvider.objects.aget(user=user, pk=provider)
    except ModelServiceProvider.DoesNotExist:
        return None


async def query_user_by_username(username: str):
    try:
        return await User.objects.aget(username=username)
    except User.DoesNotExist:
        return None


async def query_user_by_email(email: str):
    try:
        return await User.objects.aget(email=email)
    except User.DoesNotExist:
        return None


async def query_user_exists(username: str = None, email: str = None):
    if username:
        return await User.objects.filter(username=username).aexists()
    if email:
        return await User.objects.filter(email=email).aexists()
    return False


async def query_first_user_exists():
    return await User.objects.aexists()


async def create_user(username: str, email: str, password: str, role: Role):
    return await User.objects.acreate(
        username=username, email=email, password=make_password(password), role=role, is_staff=role == Role.ADMIN
    )


async def authenticate_user(username: str, password: str):
    return await aauthenticate(username=username, password=password)


async def login_user(request, user):
    await alogin(request, user)


async def logout_user(request):
    await alogout(request)


async def set_user_password(user: User, password: str):
    await sync_to_async(user.set_password)(password)
    await user.asave()


async def delete_user(user: User):
    await user.adelete()


async def query_invitation_by_token(token: str):
    try:
        return await Invitation.objects.aget(token=token)
    except Invitation.DoesNotExist:
        return None


async def create_invitation(email: str, token: str, created_by: str, role: Role):
    invitation = Invitation(email=email, token=token, created_by=created_by, role=role)
    await invitation.asave()
    return invitation


async def mark_invitation_used(invitation: Invitation):
    await invitation.use()


def query_users(pq: PagedQuery = None):
    filters = build_filters(pq)
    return User.objects.filter(**filters)


async def query_admin_count():
    return await User.objects.filter(role=Role.ADMIN).acount()


async def query_invitations():
    """Query all valid invitations"""
    return await sync_to_async(
        Invitation.objects.filter(
            is_used=False,
        )
        .order_by("-created_at")
        .all
    )()


async def list_user_api_keys(username: str) -> List[ApiKey]:
    """List all active API keys for a user"""
    return await sync_to_async(
        ApiKey.objects.filter(user=username, status=ApiKey.Status.ACTIVE, gmt_deleted__isnull=True).all
    )()


async def create_api_key(user: str, description: Optional[str] = None) -> ApiKey:
    """Create a new API key for a user"""
    return await ApiKey.objects.acreate(user=user, description=description, status=ApiKey.Status.ACTIVE)


async def delete_api_key(username: str, key_id: str) -> bool:
    """Delete an API key

    Returns:
        Tuple[bool, Optional[str]]: (success, error_message)
    """
    try:
        api_key = await ApiKey.objects.aget(
            id=key_id, user=username, status=ApiKey.Status.ACTIVE, gmt_deleted__isnull=True
        )
        api_key.status = ApiKey.Status.DELETED
        api_key.gmt_deleted = timezone.now()
        await api_key.asave()
        return True
    except ApiKey.DoesNotExist:
        return False


async def get_api_key_by_id(user: str, id: str) -> Optional[ApiKey]:
    """Get API key by id string"""
    try:
        return await ApiKey.objects.aget(user=user, id=id, status=ApiKey.Status.ACTIVE, gmt_deleted__isnull=True)
    except ApiKey.DoesNotExist:
        return None


async def get_api_key_by_key(key: str) -> Optional[ApiKey]:
    """Get API key by key string"""
    try:
        return await ApiKey.objects.aget(key=key, status=ApiKey.Status.ACTIVE, gmt_deleted__isnull=True)
    except ApiKey.DoesNotExist:
        return None
