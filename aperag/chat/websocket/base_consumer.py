import ast
import json
import logging
import random
import traceback
from abc import abstractmethod

import websockets
from channels.generic.websocket import AsyncWebsocketConsumer

import config.settings as settings
from deeprag.apps import QuotaType
from deeprag.auth.validator import DEFAULT_USER
from deeprag.chat.history.redis import RedisChatMessageHistory
from deeprag.chat.utils import (
    check_quota_usage,
    fail_response,
    get_async_redis_client,
    manage_quota_usage,
    start_response,
    stop_response,
    success_response,
    welcome_response,
)
from deeprag.db.ops import query_bot, query_user_quota
from deeprag.pipeline.base_pipeline import KUBE_CHAT_DOC_QA_REFERENCES, KUBE_CHAT_RELATED_QUESTIONS, \
    KUBE_CHAT_DOCUMENT_URLS
from deeprag.utils.constant import KEY_BOT_ID, KEY_CHAT_ID, KEY_USER_ID, KEY_WEBSOCKET_PROTOCOL
from deeprag.utils.utils import now_unix_milliseconds

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

        message_id = f"{now_unix_milliseconds()}"
        bot_config = json.loads(self.bot.config)
        self.related_question_prompt = bot_config.get("related_question_prompt", "")
        
        welcome = bot_config.get("welcome", {})
        faq = welcome.get("faq", [])
        questions = []
        for qa in faq:
            questions.append(qa["question"])
        if len(questions) < 3:
            random_questions = questions
        else:
            random_questions = random.sample(questions, 3)
        welcome_message = {"hello": welcome.get("hello", ""), "faq": random_questions}
        await self.send(text_data=welcome_response(message_id, welcome_message))

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
        related_question = []
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
                if tokens.startswith(KUBE_CHAT_DOC_QA_REFERENCES):
                    references = json.loads(tokens[len(KUBE_CHAT_DOC_QA_REFERENCES):])
                    continue
                if tokens.startswith(KUBE_CHAT_RELATED_QUESTIONS):
                    related_question = ast.literal_eval(tokens[len(KUBE_CHAT_RELATED_QUESTIONS):])
                    continue
                if tokens.startswith(KUBE_CHAT_DOCUMENT_URLS):
                    urls = ast.literal_eval(tokens[len(KUBE_CHAT_DOCUMENT_URLS):])
                    continue

                # streaming response
                response = success_response(message_id, tokens, issql=self.response_type == "sql")
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
            await self.send(text_data=stop_response(message_id, references, related_question,
                                                    self.related_question_prompt, self.pipeline.memory_count, urls=urls))


