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

import json
import logging
from datetime import datetime
from http import HTTPStatus
from typing import Any, AsyncGenerator

from fastapi.responses import StreamingResponse

from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.sse.base import ChatRequest, MessageProcessor
from aperag.chat.sse.frontend_consumer import BaseFormatter, FrontendFormatter
from aperag.chat.utils import get_async_redis_client
from aperag.config import SessionDep
from aperag.db import models as db_models
from aperag.db.ops import PagedQuery, query_bot, query_chat, query_chat_by_peer, query_chats
from aperag.schema import view_models
from aperag.schema.view_models import Chat, ChatDetails, ChatList
from aperag.views.utils import fail, success

logger = logging.getLogger(__name__)


def build_chat_response(chat: db_models.Chat) -> view_models.Chat:
    """Build Chat response object for API return."""
    return Chat(
        id=chat.id,
        title=chat.title,
        bot_id=chat.bot_id,
        peer_type=chat.peer_type,
        peer_id=chat.peer_id,
        created=chat.gmt_created.isoformat(),
        updated=chat.gmt_updated.isoformat(),
    )


async def create_chat(session: SessionDep, user: str, bot_id: str) -> view_models.Chat:
    bot = await query_bot(session, user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    instance = db_models.Chat(
        user=user, bot_id=bot_id, peer_type=db_models.ChatPeerType.SYSTEM, status=db_models.ChatStatus.ACTIVE
    )
    session.add(instance)
    await session.commit()
    await session.refresh(instance)
    return success(build_chat_response(instance))


async def list_chats(session: SessionDep, user: str, bot_id: str, pq: PagedQuery) -> view_models.ChatList:
    pr = await query_chats(session, user, bot_id, pq)
    response = []
    async for chat in pr.data:
        response.append(build_chat_response(chat))
    return success(ChatList(items=response), pr=pr)


async def get_chat(session: SessionDep, user: str, bot_id: str, chat_id: str) -> view_models.ChatDetails:
    chat = await query_chat(session, user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    from aperag.views.utils import query_chat_messages

    messages = await query_chat_messages(session, user, chat_id)
    chat_obj = build_chat_response(chat)
    return success(ChatDetails(**chat_obj.model_dump(), history=messages))


async def update_chat(
    session: SessionDep, user: str, bot_id: str, chat_id: str, chat_in: view_models.ChatUpdate
) -> view_models.Chat:
    chat = await query_chat(session, user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    chat.title = chat_in.title
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return success(build_chat_response(chat))


async def delete_chat(session: SessionDep, user: str, bot_id: str, chat_id: str) -> view_models.Chat:
    chat = await query_chat(session, user, bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    chat.status = db_models.ChatStatus.DELETED
    chat.gmt_deleted = datetime.utcnow()
    session.add(chat)
    await session.commit()
    history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
    await history.clear()
    return success(build_chat_response(chat))


def stream_frontend_sse_response(generator: AsyncGenerator[Any, Any], formatter: BaseFormatter, msg_id: str):
    """Yield SSE events for FastAPI StreamingResponse."""

    async def event_stream():
        yield f"data: {json.dumps(formatter.format_stream_start(msg_id))}\n\n"
        async for chunk in generator:
            yield f"data: {json.dumps(formatter.format_stream_content(msg_id, chunk))}\n\n"
        yield f"data: {json.dumps(formatter.format_stream_end(msg_id))}\n\n"

    return event_stream()


async def frontend_chat_completions(
    session: SessionDep, user: str, message: str, stream: bool, bot_id: str, chat_id: str, msg_id: str
) -> Any:
    try:
        chat_request = ChatRequest(
            user=user, bot_id=bot_id, chat_id=chat_id, msg_id=msg_id, stream=stream, message=message
        )
        bot = await query_bot(session, chat_request.user, chat_request.bot_id)
        if not bot:
            return success(FrontendFormatter.format_error("Bot not found"))
        chat = await query_chat_by_peer(session, bot.user, db_models.ChatPeerType.FEISHU, chat_request.chat_id)
        if chat is None:
            chat = db_models.Chat(
                user=bot.user, bot_id=bot.id, peer_type=db_models.ChatPeerType.FEISHU, peer_id=chat_request.chat_id
            )
            session.add(chat)
            await session.commit()
            await session.refresh(chat)
        history = RedisChatMessageHistory(session_id=str(chat.id), redis_client=get_async_redis_client())
        processor = MessageProcessor(bot, history)
        formatter = FrontendFormatter()
        if chat_request.stream:
            return StreamingResponse(
                stream_frontend_sse_response(
                    processor.process_message(chat_request.message, chat_request.msg_id), formatter, chat_request.msg_id
                ),
                media_type="text/event-stream",
            )
        else:
            full_content = ""
            async for chunk in processor.process_message(chat_request.message, chat_request.msg_id):
                full_content += chunk
            return success(formatter.format_complete_response(chat_request.msg_id, full_content))
    except Exception as e:
        logger.exception(e)
        return success(FrontendFormatter.format_error(str(e)))
