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
from typing import Any, List, Optional, Type, TypeVar

from fastapi import Request
from pydantic import BaseModel
from sqlalchemy import asc, desc, func
from sqlmodel import SQLModel, select

from aperag.config import SessionDep
from aperag.db.models import (
    ApiKey,
    Bot,
    BotStatus,
    Chat,
    Collection,
    CollectionStatus,
    ConfigModel,
    Document,
    Invitation,
    MessageFeedback,
    ModelServiceProvider,
    ModelServiceProviderStatus,
    Role,
    User,
    UserQuota,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=SQLModel)


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


def build_pq(request: Request) -> PagedQuery:
    return PagedQuery(
        page_number=request.query_params.get("page_number", None),
        page_size=request.query_params.get("page_size", None),
        match_key=request.query_params.get("match_key", ""),
        match_value=request.query_params.get("match_value", ""),
        order_by=request.query_params.get("order_by", ""),
        order_desc=request.query_params.get("order_desc", True),
    )


def build_filters(model: Type[T], pq: PagedQuery) -> List:
    filters = []
    if pq and pq.match_key and pq.match_value:
        field = getattr(model, pq.match_key, None)
        if field is not None:
            filters.append(field.contains(pq.match_value))
    return filters


def build_order_by(model: Type[T], pq: PagedQuery):
    if not pq or not pq.order_by:
        return desc(model.gmt_created)
    field = getattr(model, pq.order_by, None)
    if field is None:
        return desc(model.gmt_created)
    return desc(field) if pq.order_desc else asc(field)


async def build_pr(session: SessionDep, model: Type[T], stmt, pq: PagedQuery) -> PagedResult:
    count = await session.scalar(select(func.count()).select_from(stmt.subquery()))
    if not pq or not pq.page_number or not pq.page_size:
        result = await session.execute(stmt)
        return PagedResult(count=count, page_number=1, page_size=count, data=result.all())
    offset = (pq.page_number - 1) * pq.page_size
    result = await session.execute(stmt.offset(offset).limit(pq.page_size))
    return PagedResult(count=count, page_number=pq.page_number, page_size=pq.page_size, data=result.all())


async def query_collection(session: SessionDep, user: str, collection_id: str):
    stmt = select(Collection).where(
        Collection.id == collection_id, Collection.user == user, Collection.status != CollectionStatus.DELETED
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_collections(session: SessionDep, users: List[str], pq: PagedQuery = None):
    filters = [Collection.user.in_(users), Collection.status != CollectionStatus.DELETED]
    filters += build_filters(Collection, pq)
    stmt = select(Collection).where(*filters).order_by(build_order_by(Collection, pq))
    return await build_pr(session, Collection, stmt, pq)


async def query_collections_count(session: SessionDep, user: str, pq: PagedQuery = None):
    filters = [Collection.user == user, Collection.status != CollectionStatus.DELETED]
    filters += build_filters(Collection, pq)
    stmt = select(func.count()).select_from(Collection).where(*filters)
    count = await session.scalar(stmt)
    return count


async def query_collection_without_user(session: SessionDep, collection_id: str):
    stmt = select(Collection).where(Collection.id == collection_id, Collection.status != CollectionStatus.DELETED)
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_document(session: SessionDep, user: str, collection_id: str, document_id: str):
    stmt = select(Document).where(
        Document.id == document_id,
        Document.collection_id == collection_id,
        Document.user == user,
        Document.status != CollectionStatus.DELETED,
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_documents(session: SessionDep, users: List[str], collection_id: str, pq: PagedQuery = None):
    filters = [
        Document.user.in_(users),
        Document.collection_id == collection_id,
        Document.status != CollectionStatus.DELETED,
    ]
    filters += build_filters(Document, pq)
    stmt = select(Document).where(*filters).order_by(build_order_by(Document, pq))
    return await build_pr(session, Document, stmt, pq)


async def query_documents_count(session: SessionDep, user: str, collection_id: str, pq: PagedQuery = None):
    filters = [
        Document.user == user,
        Document.collection_id == collection_id,
        Document.status != CollectionStatus.DELETED,
    ]
    filters += build_filters(Document, pq)
    stmt = select(func.count()).select_from(Document).where(*filters)
    count = await session.scalar(stmt)
    return count


async def query_apikeys(session: SessionDep, user: str, pq: PagedQuery = None):
    filters = [ApiKey.user == user, ApiKey.status != BotStatus.DELETED]
    filters += build_filters(ApiKey, pq)
    stmt = select(ApiKey).where(*filters).order_by(build_order_by(ApiKey, pq))
    return await build_pr(session, ApiKey, stmt, pq)


async def query_apikey(session: SessionDep, user: str, apikey_id: str):
    stmt = select(ApiKey).where(ApiKey.id == apikey_id, ApiKey.user == user, ApiKey.status != BotStatus.DELETED)
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_chat(session: SessionDep, user: str, bot_id: str, chat_id: str):
    stmt = select(Chat).where(
        Chat.id == chat_id, Chat.bot_id == bot_id, Chat.user == user, Chat.status != CollectionStatus.DELETED
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_chat_by_peer(session: SessionDep, user: str, peer_type, peer_id: str):
    stmt = select(Chat).where(
        Chat.user == user, Chat.peer_type == peer_type, Chat.peer_id == peer_id, Chat.status != CollectionStatus.DELETED
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_chats(session: SessionDep, user: str, bot_id: str, pq: PagedQuery = None):
    filters = [Chat.user == user, Chat.bot_id == bot_id, Chat.status != CollectionStatus.DELETED]
    filters += build_filters(Chat, pq)
    stmt = select(Chat).where(*filters).order_by(build_order_by(Chat, pq))
    return await build_pr(session, Chat, stmt, pq)


async def query_bot(session: SessionDep, user: str, bot_id: str):
    stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_bots(session: SessionDep, users: List[str], pq: PagedQuery = None):
    filters = [Bot.user.in_(users), Bot.status != BotStatus.DELETED]
    filters += build_filters(Bot, pq)
    stmt = select(Bot).where(*filters).order_by(build_order_by(Bot, pq))
    return await build_pr(session, Bot, stmt, pq)


async def query_bots_count(session: SessionDep, user: str, pq: PagedQuery = None):
    filters = [Bot.user == user, Bot.status != BotStatus.DELETED]
    filters += build_filters(Bot, pq)
    stmt = select(func.count()).select_from(Bot).where(*filters)
    count = await session.scalar(stmt)
    return count


async def query_config(session: SessionDep, key):
    stmt = select(ConfigModel).where(ConfigModel.key == key)
    result = await session.execute(stmt)
    results = result.scalars().first()
    return results


async def query_user_quota(session: SessionDep, user: str, key: str):
    stmt = select(UserQuota).where(UserQuota.user == user, UserQuota.key == key)
    result = await session.execute(stmt)
    uq = result.scalars().first()
    return uq.value if uq else None


async def query_msp_list(session: SessionDep, user: str):
    stmt = select(ModelServiceProvider).where(
        ModelServiceProvider.user == user, ModelServiceProvider.status != ModelServiceProviderStatus.DELETED
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def query_msp_dict(session: SessionDep, user: str):
    stmt = select(ModelServiceProvider).where(
        ModelServiceProvider.user == user, ModelServiceProvider.status != ModelServiceProviderStatus.DELETED
    )
    result = await session.execute(stmt)
    return {msp.name: msp for msp in result.scalars().all()}


async def query_msp(session: SessionDep, user: str, provider: str, filterDeletion: bool = True):
    stmt = select(ModelServiceProvider).where(ModelServiceProvider.user == user, ModelServiceProvider.name == provider)
    if filterDeletion:
        stmt = stmt.where(ModelServiceProvider.status != ModelServiceProviderStatus.DELETED)
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_user_by_username(session: SessionDep, username: str):
    stmt = select(User).where(User.username == username)
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_user_by_email(session: SessionDep, email: str):
    stmt = select(User).where(User.email == email)
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_user_exists(session: SessionDep, username: str = None, email: str = None):
    stmt = select(User)
    if username:
        stmt = stmt.where(User.username == username)
    if email:
        stmt = stmt.where(User.email == email)
    result = await session.execute(stmt)
    return result.scalars().first() is not None


async def create_user(session: SessionDep, username: str, email: str, password: str, role: Role):
    user = User(username=username, email=email, password=password, role=role, is_staff=(role == Role.ADMIN))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def set_user_password(session: SessionDep, user: User, password: str):
    user.password = password
    session.add(user)
    await session.commit()


async def delete_user(session: SessionDep, user: User):
    await session.delete(user)
    await session.commit()


async def query_invitation_by_token(session: SessionDep, token: str):
    stmt = select(Invitation).where(Invitation.token == token)
    result = await session.execute(stmt)
    return result.scalars().first()


async def create_invitation(session: SessionDep, email: str, token: str, created_by: str, role: Role):
    invitation = Invitation(email=email, token=token, created_by=created_by, role=role)
    session.add(invitation)
    await session.commit()
    await session.refresh(invitation)
    return invitation


async def mark_invitation_used(session: SessionDep, invitation: Invitation):
    await invitation.use(session)


def query_users(pq: PagedQuery = None):
    filters = build_filters(User, pq)
    return User.objects.filter(**filters)


async def query_invitations(session: SessionDep):
    """Query all valid invitations (not used), ordered by created_at descending."""
    stmt = select(Invitation).where(not Invitation.is_used).order_by(desc(Invitation.created_at))
    result = await session.execute(stmt)
    return result.scalars().all()


async def list_user_api_keys(session: SessionDep, username: str) -> List[ApiKey]:
    """List all active API keys for a user"""
    stmt = select(ApiKey).where(ApiKey.user == username, ApiKey.status == BotStatus.ACTIVE, ApiKey.gmt_deleted is None)
    result = await session.execute(stmt)
    return result.scalars().all()


async def create_api_key(session: SessionDep, user: str, description: Optional[str] = None) -> ApiKey:
    """Create a new API key for a user"""
    api_key = ApiKey(user=user, description=description, status=BotStatus.ACTIVE)
    session.add(api_key)
    await session.commit()
    await session.refresh(api_key)
    return api_key


async def delete_api_key(session: SessionDep, username: str, key_id: str) -> bool:
    """Delete an API key (soft delete)"""
    stmt = select(ApiKey).where(
        ApiKey.id == key_id, ApiKey.user == username, ApiKey.status == BotStatus.ACTIVE, ApiKey.gmt_deleted is None
    )
    result = await session.execute(stmt)
    api_key = result.scalars().first()
    if api_key:
        api_key.status = BotStatus.DELETED
        from datetime import datetime as dt

        api_key.gmt_deleted = dt.utcnow()
        session.add(api_key)
        await session.commit()
        return True
    return False


async def get_api_key_by_id(session: SessionDep, user: str, id: str) -> Optional[ApiKey]:
    """Get API key by id string"""
    stmt = select(ApiKey).where(
        ApiKey.user == user, ApiKey.id == id, ApiKey.status == BotStatus.ACTIVE, ApiKey.gmt_deleted is None
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def get_api_key_by_key(session: SessionDep, key: str) -> Optional[ApiKey]:
    """Get API key by key string"""
    stmt = select(ApiKey).where(ApiKey.key == key, ApiKey.status == BotStatus.ACTIVE, ApiKey.gmt_deleted is None)
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_chat_feedbacks(session: SessionDep, user: str, chat_id: str, pq: PagedQuery = None):
    filters = [MessageFeedback.chat_id == chat_id, MessageFeedback.gmt_deleted is None, MessageFeedback.user == user]
    filters += build_filters(MessageFeedback, pq)
    stmt = select(MessageFeedback).where(*filters).order_by(build_order_by(MessageFeedback, pq))
    return await build_pr(session, MessageFeedback, stmt, pq)


async def query_message_feedback(session: SessionDep, user: str, chat_id: str, message_id: str):
    stmt = select(MessageFeedback).where(
        MessageFeedback.chat_id == chat_id,
        MessageFeedback.message_id == message_id,
        MessageFeedback.gmt_deleted is None,
        MessageFeedback.user == user,
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_first_user_exists(session: SessionDep):
    stmt = select(User).where(User.gmt_deleted is None)
    result = await session.execute(stmt)
    return result.scalars().first() is not None


async def query_admin_count(session: SessionDep):
    stmt = select(func.count()).select_from(User).where(User.role == Role.ADMIN, User.gmt_deleted is None)
    count = await session.scalar(stmt)
    return count
