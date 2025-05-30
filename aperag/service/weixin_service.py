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

import redis.asyncio as aredis

import aperag.chat.message
from aperag.apps import QuotaType
from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.utils import get_async_redis_client
from aperag.db.models import Chat
from aperag.db.ops import query_chat_by_peer, query_user_quota
from aperag.pipeline.knowledge_pipeline import create_knowledge_pipeline
from config import settings
from config.settings import MAX_CONVERSATION_COUNT

logger = logging.getLogger(__name__)


async def weixin_text_response(client, user, bot, query, msg_id):
    chat_id = user
    chat = await query_chat_by_peer(bot.user, Chat.PeerType.WEIXIN, chat_id)
    if chat is None:
        chat = Chat(user=bot.user, bot_id=bot.id, peer_type=Chat.PeerType.WEIXIN, peer_id=chat_id)
        await chat.asave()
    history = RedisChatMessageHistory(session_id=str(chat.id), redis_client=get_async_redis_client())
    collection = (await bot.collections())[0]
    response = ""
    pipeline = await create_knowledge_pipeline(bot=bot, collection=collection, history=history)
    conversation_limit = await query_user_quota(user, QuotaType.MAX_CONVERSATION_COUNT)
    if conversation_limit is None:
        conversation_limit = MAX_CONVERSATION_COUNT
    try:
        await client.send_message("ApeRAG 正在解答中，请稍候......", user)
        async for msg in pipeline.run(query, message_id=msg_id):
            response += msg
        max_length = 2048
        for i in range(0, len(response), max_length):
            message = response[i : i + max_length]
            await client.send_message(message, user)
    except Exception as e:
        logger.exception(e)


async def weixin_feedback_response(client, user, bot, key, response_code, task_id):
    upvote = 1 if key == 1 else None
    downvote = 0 if key == 0 else None
    chat_id = user
    chat = await query_chat_by_peer(bot.user, Chat.PeerType.WEIXIN, chat_id)
    if chat is None:
        chat = Chat(user=bot.user, bot_id=bot.id, peer_type=Chat.PeerType.WEIXIN, peer_id=chat_id)
        await chat.asave()
    msg_id = await client.redis_client.get(f"{task_id}2msg_id")
    await aperag.chat.message.feedback_message(bot.user, chat.id, msg_id.decode(), upvote, downvote, "")
    message = await client.redis_client.get(f"{task_id}2message")
    await client.update_card(message.decode(), user, response_code, vote=key)


def generate_xml_response(to_user_name, from_user_name, create_time, msg_type, response):
    response = response.replace("\n", "&#xA;")
    resp = f"""<xml>\
                <ToUserName><![CDATA[{to_user_name}]]></ToUserName>\
                <FromUserName><![CDATA[{from_user_name}]]></FromUserName>\
                <CreateTime>{create_time}</CreateTime>\
                <MsgType><![CDATA[{msg_type}]]></MsgType>\
                <Content>{response}</Content>\
            </xml>"""
    return resp


async def weixin_officaccount_response(query, msg_id, to_user_name, bot):
    chat_id = to_user_name
    chat = await query_chat_by_peer(bot.user, Chat.PeerType.WEIXIN_OFFICIAL, chat_id)
    if chat is None:
        chat = Chat(user=bot.user, bot_id=bot.id, peer_type=Chat.PeerType.WEIXIN_OFFICIAL, peer_id=chat_id)
        await chat.asave()
    history = RedisChatMessageHistory(session_id=str(chat.id), redis_client=get_async_redis_client())
    collection = (await bot.collections())[0]
    pipeline = await create_knowledge_pipeline(bot=bot, collection=collection, history=history)
    redis_client = aredis.Redis.from_url(settings.MEMORY_REDIS_URL)
    response = ""
    conversation_limit = await query_user_quota(to_user_name, QuotaType.MAX_CONVERSATION_COUNT)
    if conversation_limit is None:
        conversation_limit = MAX_CONVERSATION_COUNT
    try:
        async for msg in pipeline.run(query, message_id=msg_id):
            response += msg
        logger.info(f"response:{response}")
        await redis_client.set(to_user_name + msg_id, response)
        logger.info(f"generate response success, restored in redis, key:{to_user_name + msg_id}")
    except Exception as e:
        logger.exception(e)
