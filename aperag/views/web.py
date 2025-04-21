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
from http import HTTPStatus

from django.db import IntegrityError
from ninja import Router

import aperag.chat
from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.utils import get_async_redis_client
from aperag.db.models import Chat, ChatPeer
from aperag.db.ops import build_pq, query_bot, query_web_chat, query_web_chats
from aperag.views.utils import fail, query_chat_messages, success
from aperag.views.models import Feedback

logger = logging.getLogger(__name__)


router = Router()


def check_origin(request, bot):
    bot_config = json.loads(bot.config)
    host_white_list = bot_config.get("web", {}).get("host_white_list", [])
    origin = request.scheme + "://" + request.META.get("REMOTE_ADDR", "")
    logger.info(f"bot {bot.id} check origin: {origin}, {host_white_list}")
    logger.info(f"REQUEST META: {request.META}, REQUEST HEADERS: {request.headers}")
    # TODO: check origin
    return True
    # return origin in host_white_list


@router.get("/bots/{bot_id}/web-chats")
async def list_chats(request, bot_id, session_id):
    bot = await query_bot(None, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    if not check_origin(request, bot):
        return fail(HTTPStatus.FORBIDDEN, "Origin not allowed")
    pr = await query_web_chats(bot_id, session_id, build_pq(request))
    response = []
    async for chat in pr.data:
        response.append(chat.view(bot_id))
    return success(response, pr)


@router.post("/bots/{bot_id}/web-chats")
async def create_chat(request, bot_id, session_id):
    bot = await query_bot(None, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    if not check_origin(request, bot):
        return fail(HTTPStatus.FORBIDDEN, "Origin not allowed")
    try:
        instance = Chat(
            user=bot.user,
            bot=bot,
            peer_type=ChatPeer.WEB,
            peer_id=session_id
        )
        await instance.asave()
    except IntegrityError:
        return fail(HTTPStatus.BAD_REQUEST, "Only one chat is allowed for each client")
    return success(instance.view(bot_id))


@router.get("/bots/{bot_id}/web-chats/{chat_id}")
async def get_chat(request, bot_id, chat_id):
    bot = await query_bot(None, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    if not check_origin(request, bot):
        return fail(HTTPStatus.FORBIDDEN, "Origin not allowed")
    chat = await query_web_chat(bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    messages = await query_chat_messages(None, chat_id)
    return success(chat.view(bot_id, messages))


@router.post("/bots/{bot_id}/web-chats/{chat_id}/messages/{message_id}")
async def feedback_message(request, bot_id, chat_id, message_id, msg_in: Feedback):
    bot = await query_bot(None, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    if not check_origin(request, bot):
        return fail(HTTPStatus.FORBIDDEN, "Origin not allowed")
    chat = await query_web_chat(bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")

    await aperag.chat.message.feedback_message(chat.user, chat_id, message_id, msg_in.upvote, msg_in.downvote)
    return success({})


@router.put("/bots/{bot_id}/web-chats/{chat_id}")
async def update_chat(request, bot_id, chat_id):
    bot = await query_bot(None, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    if not check_origin(request, bot):
        return fail(HTTPStatus.FORBIDDEN, "Origin not allowed")
    chat = await query_web_chat(bot_id, chat_id)
    if chat is None:
        return fail(HTTPStatus.NOT_FOUND, "Chat not found")
    chat.summary = ""
    await chat.asave()
    history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
    await history.clear()
    return success(chat.view(bot_id))
