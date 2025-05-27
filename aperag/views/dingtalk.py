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

from ninja import Router

from aperag.db.ops import query_bot
from aperag.service.dingtalk_service import dingtalk_text_response, send_message, validate_sign
from aperag.views.utils import fail, success

logger = logging.getLogger(__name__)

router = Router()


@router.post("/webhook/event")
async def post_view(request, user, bot_id):
    bot = await query_bot(user, bot_id)
    if bot is None:
        logger.warning("bot not found: %s", bot_id)
        return fail(400, "bot not found")
    bot_config = json.loads(bot.config)
    secret = bot_config["dingtalk"]["client_secret"]
    if validate_sign(request.headers["Timestamp"], client_secret=secret, request_sign=request.headers["Sign"]):
        data = json.loads(request.body.decode("utf-8"))
        message_content = data.get("text", {}).get("content")
        sender_id = data.get("senderStaffId")
        session_webhook = data.get("sessionWebhook")
        msg_id = data.get("msgId")
        asyncio.create_task(
            send_message(f'我已经收到问题"{message_content}"啦，正在飞速生成回答中', session_webhook, sender_id)
        )
        asyncio.create_task(dingtalk_text_response(user, bot, message_content, msg_id, sender_id, session_webhook))
        return success("")
    return fail(400, "validate dingtalk sign failed")
