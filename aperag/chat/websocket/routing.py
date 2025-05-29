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

from django.urls import re_path

from aperag.utils.constant import KEY_BOT_ID, KEY_CHAT_ID, KEY_USER_ID
from aperag.utils.utils import extract_bot_and_chat_id


async def bot_consumer_router(scope, receive, send):
    logging.info("bot_consumer_router begin")

    from aperag.db.ops import query_bot, query_chat

    user = scope["user"].id
    scope[KEY_USER_ID] = "%d" % user

    path = scope["path"]
    bot_id, chat_id = extract_bot_and_chat_id(path)

    bot = await query_bot(user, bot_id)
    if bot is None:
        raise Exception("Bot not found")
    scope[KEY_BOT_ID] = bot_id

    chat = await query_chat(user, bot_id, chat_id)
    if chat is None:
        raise Exception("Chat not found")
    scope[KEY_CHAT_ID] = chat_id

    from aperag.chat.websocket.flow_consumer import FlowConsumer

    return await FlowConsumer.as_asgi()(scope, receive, send)


websocket_urlpatterns = [
    re_path(
        r"api/v1/bots/(?P<bot_id>\w+)/chats/(?P<chat_id>\w+)/connect$",
        bot_consumer_router,
    ),
]
