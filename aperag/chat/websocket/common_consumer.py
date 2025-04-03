import ast
import json
import logging
import os
import traceback

import websockets

from deeprag.chat.utils import (
    check_quota_usage,
    fail_response,
    manage_quota_usage,
    start_response,
    stop_response,
    success_response,
)
from deeprag.chat.websocket.base_consumer import BaseConsumer
from deeprag.pipeline.base_pipeline import KUBE_CHAT_RELATED_QUESTIONS
from deeprag.pipeline.common_pipeline import CommonPipeline
from deeprag.readers.base_readers import DEFAULT_FILE_READER_CLS
from deeprag.source.utils import gen_temporary_file
from deeprag.utils.utils import now_unix_milliseconds

logger = logging.getLogger(__name__)


class CommonConsumer(BaseConsumer):
    async def connect(self):
        await super().connect()
        self.file = None
        self.file_name = None
        self.pipeline = CommonPipeline(bot=self.bot, collection=self.collection, history=self.history)
        self.free_tier = self.pipeline.predictor.trial

    async def predict(self, query, **kwargs):
        async for msg in self.pipeline.run(query, gen_references=True, **kwargs):
            yield msg

    async def receive(self, text_data=None, bytes_data=None):
        message = ""
        message_id = f"{now_unix_milliseconds()}"
        related_question = []

        # bytes_data: [file_name_length, file_name, file_content]
        if bytes_data:
            file_name_length = int.from_bytes(bytes_data[:4], byteorder='big')
            file_name_bytes = bytes_data[4:4 + file_name_length]
            self.file_name = file_name_bytes.decode('utf-8')
            file_content = bytes_data[4 + file_name_length:]

            file_suffix = os.path.splitext(self.file_name)[1].lower()
            if file_suffix not in DEFAULT_FILE_READER_CLS.keys():
                error = f"unsupported file type {file_suffix}"
                await self.send(text_data=fail_response(message_id, error=error))
                return

            temp_file = gen_temporary_file(self.file_name)
            temp_file.write(file_content)
            temp_file.close()

            reader = DEFAULT_FILE_READER_CLS[file_suffix]
            docs = reader.load_data(temp_file.name)
            self.file = docs[0].text
            return

        data = json.loads(text_data)
        self.msg_type = data["type"]
        self.response_type = "message"

        if self.msg_type == "cancel_upload":
            self.file = None
            self.file_name = None
            return

        try:
            # send start message
            await self.send(text_data=start_response(message_id))

            if self.free_tier and self.conversation_limit:
                if not await check_quota_usage(self.user, self.conversation_limit):
                    error = f"conversation rounds have reached to the limit of {self.conversation_limit}"
                    await self.send(text_data=fail_response(message_id, error=error))
                    return

            async for tokens in self.predict(data["data"], message_id=message_id, file=self.file):
                if tokens.startswith(KUBE_CHAT_RELATED_QUESTIONS):
                    related_question = ast.literal_eval(tokens[len(KUBE_CHAT_RELATED_QUESTIONS):])
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
            self.file = None
            self.file_name = None
            if self.free_tier and self.conversation_limit:
                await manage_quota_usage(self.user, self.conversation_limit)
            # send stop message
            await self.send(text_data=stop_response(message_id, [], related_question, self.related_question_prompt, self.pipeline.memory_count, urls=[]))
