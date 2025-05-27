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
import json
import logging
import time
from http import HTTPStatus

from ninja import Router

import aperag.chat.message
from aperag.auth.authentication import FeishuEventVerification
from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.utils import get_async_redis_client
from aperag.db.models import Chat
from aperag.db.ops import query_bot, query_chat_by_peer
from aperag.pipeline.knowledge_pipeline import create_knowledge_pipeline
from aperag.source.feishu.feishu import FeishuClient
from aperag.utils.utils import AESCipher
from aperag.views.utils import fail, success
from config import settings

logger = logging.getLogger(__name__)

router = Router()


@router.get("/spaces")
async def feishu_get_spaces_view(request, app_id, app_secret):
    ctx = {
        "app_id": app_id,
        "app_secret": app_secret,
    }
    result = []
    try:
        for space in FeishuClient(ctx).get_spaces():
            result.append(
                {
                    "space_id": space["id"],
                    "description": space["description"],
                    "name": space["name"],
                }
            )
    except Exception as e:
        logger.exception(e)
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
    return success(result)


# using redis to cache messages
msg_cache = {}


# TODO use redis to cache message ids
def message_handled(msg_id, msg):
    if msg_id in msg_cache:
        return True
    else:
        msg_cache[msg_id] = msg
        return False


@router.get("/user_access_token")
def get_user_access_token(request, code, redirect_uri):
    ctx = {
        "app_id": settings.FEISHU_APP_ID,
        "app_secret": settings.FEISHU_APP_SECRET,
    }
    client = FeishuClient(ctx)
    token = client.get_user_access_token(code, redirect_uri)
    return success({"token": token})


def build_card_content(chat_id, message_id, message, feedback_type=None):
    return {
        "config": {"wide_screen_mode": True},
        "elements": [
            {
                "tag": "markdown",
                "content": message,
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "赞"},
                        "type": "primary" if feedback_type == "good" else "default",
                        "value": {
                            "type": "good",
                            "message_id": f"{message_id}",
                            "chat_id": f"{chat_id}",
                        },
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "踩"},
                        "type": "primary" if feedback_type == "bad" else "default",
                        "value": {
                            "type": "bad",
                            "message_id": f"{message_id}",
                            "chat_id": f"{chat_id}",
                        },
                    },
                ],
                "layout": "bisected",
            },
        ],
    }


def build_card_data(chat_id, message_id, message, feedback_type=None):
    return {
        "msg_type": "interactive",
        "content": json.dumps(build_card_content(chat_id, message_id, message, feedback_type)),
    }


async def feishu_streaming_response(client, chat_id, bot, msg_id, msg):
    chat = await query_chat_by_peer(bot.user, Chat.PeerType.FEISHU, chat_id)
    if chat is None:
        chat = Chat(user=bot.user, bot_id=bot.id, peer_type=Chat.PeerType.FEISHU, peer_id=chat_id)
        await chat.asave()

    history = RedisChatMessageHistory(session_id=str(chat.id), redis_client=get_async_redis_client())
    response = ""
    collection = (await bot.collections())[0]
    card_id = client.reply_card_message(msg_id, build_card_data(chat_id, msg_id, response))
    last_ts = time.time()
    try:
        async for msg in await create_knowledge_pipeline(bot=bot, collection=collection, history=history).run(
            msg, message_id=msg_id
        ):
            response += msg
            now = time.time()
            if now - last_ts < 0.2:
                continue
            last_ts = now
            client.update_card_message(card_id, build_card_data(chat_id, msg_id, response))
        client.update_card_message(card_id, build_card_data(chat_id, msg_id, response))
    except Exception as e:
        logger.exception(e)
        if response:
            response += "\n"
        response += "[Oops] " + str(e)
        client.update_card_message(card_id, build_card_data(chat_id, msg_id, response))
    msg_cache[msg_id] = response


async def feishu_response_card_update(user, bot_id, data):
    action = data.get("action", {})
    if not action:
        logger.warning("Invalid card event: %s", data)
        return

    value = action["value"]
    chat_id = value["chat_id"]
    msg_id = value["message_id"]
    feedback_type = value.get("type", None)
    await aperag.chat.message.feedback_message(user, chat_id, msg_id, feedback_type, None, None)

    bot = await query_bot(user, bot_id)
    bot_config = json.loads(bot.config)
    feishu_config = bot_config.get("feishu")
    app_id = feishu_config.get("app_id", "")
    app_secret = feishu_config.get("app_secret", "")
    if not app_id or not app_secret:
        logger.warning("please properly setup the feishu app id and app secret first", user, bot_id)
        return

    ctx = {
        "app_id": app_id,
        "app_secret": app_secret,
    }
    client = FeishuClient(ctx)
    token = data["token"]
    card = build_card_content(chat_id, msg_id, msg_cache[msg_id], feedback_type)
    card["open_ids"] = [data["open_id"]]
    data = {
        "token": token,
        "card": card,
    }
    client.delay_update_card_message(data)


@router.post("/card/event")
async def feishu_card_event_view(request, user=None, bot_id=None):
    data = json.loads(request.body)
    if "challenge" in data:
        return {"challenge": data["challenge"]}
    asyncio.create_task(feishu_response_card_update(user, bot_id, data))


@router.post("/webhook/event")
async def feishu_webhook_event_view(request, user=None, bot_id=None):
    data = json.loads(request.body)
    bot = await query_bot(user, bot_id)
    if bot is None:
        logger.warning("bot not found: %s", bot_id)
        return
    bot_config = json.loads(bot.config)
    feishu_config = bot_config.get("feishu")

    encrypt_key = feishu_config.get("encrypt_key")
    if "encrypt" in data:
        cipher = AESCipher(encrypt_key)
        data = cipher.decrypt_string(data["encrypt"])
        data = json.loads(data)

    logger.info(data)
    if "challenge" in data:
        return {"challenge": data["challenge"]}

    if encrypt_key and not FeishuEventVerification(encrypt_key)(request):
        return fail(HTTPStatus.UNAUTHORIZED, "Unauthorized")

    header = data["header"]
    if header["event_type"] == "im.message.message_read_v1":
        return

    if header["event_type"] != "im.message.receive_v1":
        logger.warning("Unsupported event: %s", data)
        return

    event = data.get("event", None)
    if event is None:
        logger.warning("Unsupported event: %s", data)
        return

    # ignore duplicate messages
    msg_id = event["message"]["message_id"]
    if message_handled(msg_id, ""):
        return

    content = json.loads(event["message"]["content"])
    at = ""
    message = content["text"]
    if message.startswith("@"):
        parts = message.split(" ", 1)
        at = parts[0]
        message = parts[1]

    # if the message is sent to all users, ignore it
    if at == "@_all":
        return

    if not user:
        logger.warning("invalid event without user")
        return

    app_id = feishu_config.get("app_id", "")
    app_secret = feishu_config.get("app_secret", "")
    if not app_id or not app_secret:
        logger.warning("please properly setup the feishu app id and app secret first", user, bot_id)
        return

    ctx = {
        "app_id": app_id,
        "app_secret": app_secret,
    }
    client = FeishuClient(ctx)
    chat_id = event["message"]["chat_id"]
    asyncio.create_task(feishu_streaming_response(client, chat_id, bot, msg_id, message))
    return success({"code": 0})
