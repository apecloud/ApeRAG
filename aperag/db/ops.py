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
from typing import List, Optional

from sqlalchemy import desc, func
from sqlmodel import select

from aperag.config import SessionDep
from aperag.db.models import (
    ApiKey,
    ApiKeyStatus,
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


async def query_collection(session: SessionDep, user: str, collection_id: str):
    stmt = select(Collection).where(
        Collection.id == collection_id, Collection.user == user, Collection.status != CollectionStatus.DELETED
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_collections(session: SessionDep, users: List[str]):
    stmt = (
        select(Collection)
        .where(Collection.user.in_(users), Collection.status != CollectionStatus.DELETED)
        .order_by(desc(Collection.gmt_created))
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def query_collections_count(session: SessionDep, user: str):
    stmt = (
        select(func.count())
        .select_from(Collection)
        .where(Collection.user == user, Collection.status != CollectionStatus.DELETED)
    )
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


async def query_documents(session: SessionDep, users: List[str], collection_id: str):
    stmt = (
        select(Document)
        .where(
            Document.user.in_(users),
            Document.collection_id == collection_id,
            Document.status != CollectionStatus.DELETED,
        )
        .order_by(desc(Document.gmt_created))
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def query_documents_count(session: SessionDep, user: str, collection_id: str):
    stmt = (
        select(func.count())
        .select_from(Document)
        .where(
            Document.user == user,
            Document.collection_id == collection_id,
            Document.status != CollectionStatus.DELETED,
        )
    )
    count = await session.scalar(stmt)
    return count


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


async def query_chats(session: SessionDep, user: str, bot_id: str):
    stmt = (
        select(Chat)
        .where(Chat.user == user, Chat.bot_id == bot_id, Chat.status != CollectionStatus.DELETED)
        .order_by(desc(Chat.gmt_created))
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def query_bot(session: SessionDep, user: str, bot_id: str):
    stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_bots(session: SessionDep, users: List[str]):
    stmt = select(Bot).where(Bot.user.in_(users), Bot.status != BotStatus.DELETED).order_by(desc(Bot.gmt_created))
    result = await session.execute(stmt)
    return result.scalars().all()


async def query_bots_count(session: SessionDep, user: str):
    stmt = select(func.count()).select_from(Bot).where(Bot.user == user, Bot.status != BotStatus.DELETED)
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


async def query_invitations(session: SessionDep):
    """Query all valid invitations (not used), ordered by created_at descending."""
    stmt = select(Invitation).where(not Invitation.is_used).order_by(desc(Invitation.created_at))
    result = await session.execute(stmt)
    return result.scalars().all()


async def list_user_api_keys(session: SessionDep, user: str):
    """List all active API keys for a user"""
    stmt = select(ApiKey).where(ApiKey.user == user, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None))
    result = await session.execute(stmt)
    return result.scalars().all()


async def create_api_key(session: SessionDep, user: str, description: Optional[str] = None) -> ApiKey:
    """Create a new API key for a user"""
    api_key = ApiKey(user=user, description=description, status=ApiKeyStatus.ACTIVE)
    session.add(api_key)
    await session.commit()
    await session.refresh(api_key)
    return api_key


async def delete_api_key(session: SessionDep, user: str, key_id: str) -> bool:
    """Delete an API key (soft delete)"""
    stmt = select(ApiKey).where(
        ApiKey.id == key_id, ApiKey.user == user, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
    )
    result = await session.execute(stmt)
    api_key = result.scalars().first()
    if api_key:
        api_key.status = ApiKeyStatus.DELETED
        from datetime import datetime as dt

        api_key.gmt_deleted = dt.utcnow()
        session.add(api_key)
        await session.commit()
        return True
    return False


async def get_api_key_by_id(session: SessionDep, user: str, id: str) -> Optional[ApiKey]:
    """Get API key by id string"""
    stmt = select(ApiKey).where(
        ApiKey.user == user, ApiKey.id == id, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def get_api_key_by_key(session: SessionDep, key: str) -> Optional[ApiKey]:
    """Get API key by key string"""
    stmt = select(ApiKey).where(ApiKey.key == key, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None))
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_chat_feedbacks(session: SessionDep, user: str, chat_id: str):
    stmt = (
        select(MessageFeedback)
        .where(MessageFeedback.chat_id == chat_id, MessageFeedback.gmt_deleted.is_(None), MessageFeedback.user == user)
        .order_by(desc(MessageFeedback.gmt_created))
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def query_message_feedback(session: SessionDep, user: str, chat_id: str, message_id: str):
    stmt = select(MessageFeedback).where(
        MessageFeedback.chat_id == chat_id,
        MessageFeedback.message_id == message_id,
        MessageFeedback.gmt_deleted.is_(None),
        MessageFeedback.user == user,
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def query_first_user_exists(session: SessionDep):
    stmt = select(User).where(User.gmt_deleted.is_(None))
    result = await session.execute(stmt)
    return result.scalars().first() is not None


async def query_admin_count(session: SessionDep):
    stmt = select(func.count()).select_from(User).where(User.role == Role.ADMIN, User.gmt_deleted.is_(None))
    count = await session.scalar(stmt)
    return count
