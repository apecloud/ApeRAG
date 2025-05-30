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

import base64
import hashlib
import hmac
import logging

import requests

from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.utils import get_async_redis_client
from aperag.db.models import Chat
from aperag.db.ops import query_chat_by_peer
from aperag.pipeline.knowledge_pipeline import create_knowledge_pipeline

logger = logging.getLogger(__name__)


def validate_sign(timestamp, client_secret, request_sign):
    client_secret_enc = client_secret.encode("utf-8")
    string_to_sign = "{}\n{}".format(timestamp, client_secret)
    string_to_sign_enc = string_to_sign.encode("utf-8")
    hmac_code = hmac.new(client_secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = base64.b64encode(hmac_code).decode("utf-8")
    return sign == request_sign


async def dingtalk_text_response(user, bot, query, msg_id, sender_id, session_webhook):
    chat_id = user
    chat = await query_chat_by_peer(bot.user, Chat.PeerType.DINGTALK, chat_id)
    if chat is None:
        chat = Chat(user=bot.user, bot_id=bot.id, peer_type=Chat.PeerType.DINGTALK, peer_id=chat_id)
        await chat.asave()
    history = RedisChatMessageHistory(session_id=str(chat.id), redis_client=get_async_redis_client())
    collection = (await bot.collections())[0]
    response = ""
    pipeline = await create_knowledge_pipeline(bot=bot, collection=collection, history=history)
    try:
        async for msg in pipeline.run(query, message_id=msg_id):
            response += msg
        max_length = 2048
        for i in range(0, len(response), max_length):
            message = response[i : i + max_length]
            await notify_dingding(query, message, webhook=session_webhook, sender_id=sender_id)
    except Exception as e:
        logger.exception(e)


async def notify_dingding(question, answer, webhook, sender_id):
    data = {
        "msgtype": "text",
        "at": {"atUserIds": [f"{sender_id}"], "isAtAll": False},
        "text": {"content": f"Q:{question}\nA:{answer}"},
    }
    try:
        r = requests.post(webhook, json=data)
        reply = r.json()
        logger.info("dingding: " + str(reply))
    except Exception as e:
        logger.error(e)


async def send_message(msg, webhook, sender_id):
    data = {"msgtype": "text", "text": {"content": f"{msg}"}, "at": {"atUserIds": [sender_id], "isAtAll": False}}
    try:
        r = requests.post(webhook, json=data)
        reply = r.json()
        logger.info("dingding: " + str(reply))
    except Exception as e:
        logger.error(e)
