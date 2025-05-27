import base64
import hashlib
import hmac
import logging

import requests

from aperag.apps import QuotaType
from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.utils import check_quota_usage, get_async_redis_client, manage_quota_usage
from aperag.db.models import Chat
from aperag.db.ops import query_chat_by_peer, query_user_quota
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
            message = response[i : i + max_length]
            await notify_dingding(query, message, webhook=session_webhook, sender_id=sender_id)
    except Exception as e:
        logger.exception(e)
    finally:
        if trial and conversation_limit:
            await manage_quota_usage(bot.user, conversation_limit)


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
