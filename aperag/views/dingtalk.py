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

import asyncio
import base64
import hashlib
import hmac
import json
import logging

import requests
from asgiref.sync import sync_to_async
from ninja import Router

from aperag.apps import QuotaType
from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.utils import check_quota_usage, get_async_redis_client, manage_quota_usage
from aperag.db.models import Chat, ChatPeer
from aperag.db.ops import query_bot, query_chat_by_peer, query_user_quota
from aperag.pipeline.knowledge_pipeline import create_knowledge_pipeline
from aperag.views.utils import fail, success

logger = logging.getLogger(__name__)

router = Router()

@router.post("/webhook/event")
async def post(request, user, bot_id):
    bot = await query_bot(user, bot_id)
    bot_config = json.loads(bot.config)
    secret = bot_config['dingtalk']['client_secret']
    if validate_sign(request.headers['Timestamp'], client_secret=secret, request_sign=request.headers['Sign']):
        data = json.loads(request.body.decode('utf-8'))
        # sender_nick = data.get('senderNick')
        message_content = data.get('text', {}).get('content')
        sender_id = data.get('senderStaffId')
        session_webhook = data.get('sessionWebhook')
        msg_id = data.get('msgId')
        if bot is None:
            logger.warning("bot not found: %s", bot_id)
            asyncio.create_task(send_message("bot not found", session_webhook, sender_id))
            return
        asyncio.create_task(send_message(f"我已经收到问题\"{message_content}\"啦，正在飞速生成回答中", session_webhook, sender_id))
        asyncio.create_task(dingtalk_text_response(user, bot, message_content, msg_id, sender_id, session_webhook))
        return success("")

    return fail(400, "validate dingtalk sign failed")


def validate_sign(timestamp, client_secret, request_sign):
    client_secret_enc = client_secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, client_secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(client_secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = base64.b64encode(hmac_code).decode('utf-8')
    return sign == request_sign

async def dingtalk_text_response(user, bot, query, msg_id, sender_id, session_webhook):
    chat_id = user
    chat = await query_chat_by_peer(bot.user, ChatPeer.DINGTALK, chat_id)
    if chat is None:
        chat = Chat(user=bot.user, bot=bot, peer_type=ChatPeer.DINGTALK, peer_id=chat_id)
        await chat.asave()

    history = RedisChatMessageHistory(session_id=str(chat.id), redis_client=get_async_redis_client())
    collection = await sync_to_async(bot.collections.first)()
    response = ""

    pipeline = await create_knowledge_pipeline(bot=bot, collection=collection, history=history)
    trial = pipeline.predictor.trial

    conversation_limit = await query_user_quota(user, QuotaType.MAX_CONVERSATION_COUNT)

    try:
        if trial and conversation_limit:
            if not await check_quota_usage(bot.user, conversation_limit):
                error = f"conversation rounds have reached to the limit of {conversation_limit}"
                await send_message(error, user, sender_id)
                return


        async for msg in pipeline.run(query, message_id=msg_id):
            response += msg

        max_length = 2048
        for i in range(0, len(response), max_length):
            message = response[i:i + max_length]
            await notify_dingding(query, message, webhook=session_webhook, sender_id=sender_id)
    except Exception as e:
        logger.exception(e)
    finally:
        if trial and conversation_limit:
            await manage_quota_usage(bot.user, conversation_limit)

async def notify_dingding(question, answer, webhook, sender_id):
    data = {
        "msgtype": "text",
        "at": {
            "atUserIds": [
                f"{sender_id}"
            ],
            "isAtAll": False
        },
        "text": {
            "content": f"Q:{question}\n"
                       f"A:{answer}"
        }

    }
    try:
        r = requests.post(webhook, json=data)
        reply = r.json()
        logger.info("dingding: " + str(reply))
    except Exception as e:
        logger.error(e)
async def send_message(msg, webhook, sender_id):
    data = {
        "msgtype": "text",
        "text": {
            "content": f"{msg}"
        },

        "at": {
            "atUserIds": [
                sender_id
            ],
            "isAtAll": False
        }
    }
    try:
        r = requests.post(webhook, json=data)
        reply = r.json()
        logger.info("dingding: " + str(reply))
    except Exception as e:
        logger.error(e)
