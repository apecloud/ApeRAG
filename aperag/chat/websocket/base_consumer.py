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

import ast
import json
import logging
import traceback
from abc import abstractmethod

import websockets
from channels.generic.websocket import AsyncWebsocketConsumer

import config.settings as settings
from aperag.apps import QuotaType
from aperag.auth.validator import DEFAULT_USER
from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.utils import (
    check_quota_usage,
    fail_response,
    get_async_redis_client,
    manage_quota_usage,
    start_response,
    stop_response,
    success_response,
)
from aperag.db.ops import query_bot, query_user_quota
from aperag.pipeline.base_pipeline import DOC_QA_REFERENCES, DOCUMENT_URLS
from aperag.utils.constant import KEY_BOT_ID, KEY_CHAT_ID, KEY_USER_ID, KEY_WEBSOCKET_PROTOCOL
from aperag.utils.utils import now_unix_milliseconds

logger = logging.getLogger(__name__)


class BaseConsumer(AsyncWebsocketConsumer):
    def __init__(self):
        super().__init__()
        self.bot = None
        self.user = DEFAULT_USER
        self.collection = None
        self.collection_id = None
        self.embedding_model = None
        self.vector_size = 0
        self.history = None
        self.pipeline = None
        self.free_tier = False

    async def connect(self):
        self.user = self.scope[KEY_USER_ID]
        bot_id = self.scope[KEY_BOT_ID]
        chat_id = self.scope[KEY_CHAT_ID]
        self.bot = await query_bot(self.user, bot_id)

        self.history = RedisChatMessageHistory(session_id=chat_id, redis_client=get_async_redis_client())
        self.conversation_limit = await query_user_quota(self.user, QuotaType.MAX_CONVERSATION_COUNT)
        if self.conversation_limit is None:
            self.conversation_limit = settings.MAX_CONVERSATION_COUNT

        # If using daphne, we need to put the token in the headers and include the headers in the subprotocol.
        # headers = {}
        # token = self.scope.get("Sec-Websocket-Protocol", None)
        # if token is not None:
        #     headers = {"SEC-WEBSOCKET-PROTOCOL": token}
        # await self.accept(subprotocol=(None, headers))

        # If using uvicorn, we need to put the token in the headers and include the headers in the message as uvicorn
        # by extracting the headers from the message.
        # To customize the message layout, we need to call the send method in the AsyncWebsocketConsumer class.
        headers = []
        token = self.scope.get(KEY_WEBSOCKET_PROTOCOL, None)
        if token is not None:
            headers.append((KEY_WEBSOCKET_PROTOCOL.encode("ascii"), token.encode("ascii")))
        await super(AsyncWebsocketConsumer, self).send({"type": "websocket.accept", "headers": headers})

    async def disconnect(self, close_code):
        pass

    @abstractmethod
    def predict(self, query, **kwargs):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        self.msg_type = data["type"]
        if self.msg_type == "bot_context":
            await self.pipeline.update_bot_context(data["data"])
            return

        self.response_type = "message"

        message = ""
        references = []
        urls = []
        message_id = f"{now_unix_milliseconds()}"

        try:
            # send start message
            await self.send(text_data=start_response(message_id))

            if self.free_tier and self.conversation_limit:
                if not await check_quota_usage(self.user, self.conversation_limit):
                    error = f"conversation rounds have reached to the limit of {self.conversation_limit}"
                    await self.send(text_data=fail_response(message_id, error=error))
                    return

            async for tokens in self.predict(data["data"], message_id=message_id):
                if tokens.startswith(DOC_QA_REFERENCES):
                    references = json.loads(tokens[len(DOC_QA_REFERENCES) :])
                    continue
                if tokens.startswith(DOCUMENT_URLS):
                    urls = ast.literal_eval(tokens[len(DOCUMENT_URLS) :])
                    continue

                # streaming response
                response = success_response(message_id, tokens)
                await self.send(text_data=response)

                # concat response tokens
                message += tokens

        except websockets.exceptions.ConnectionClosedError:
            logger.warning("Connection closed")
        except Exception as e:
            logger.warning("[Oops] %s: %s", str(e), traceback.format_exc())
            await self.send(text_data=fail_response(message_id, str(e)))
        finally:
            if self.free_tier and self.conversation_limit:
                await manage_quota_usage(self.user, self.conversation_limit)
            # send stop message
            await self.send(
                text_data=stop_response(
                    message_id,
                    references,
                    self.pipeline.memory_count if self.pipeline else 0,
                    urls=urls,
                )
            )
