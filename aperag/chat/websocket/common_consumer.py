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
import os
import traceback

import websockets

from aperag.chat.utils import (check_quota_usage, fail_response,
                               manage_quota_usage, start_response,
                               stop_response, success_response)
from aperag.chat.websocket.base_consumer import BaseConsumer
from aperag.pipeline.base_pipeline import RELATED_QUESTIONS
from aperag.pipeline.common_pipeline import CommonPipeline
from aperag.readers.base_readers import DEFAULT_FILE_READER_CLS
from aperag.source.utils import gen_temporary_file
from aperag.utils.utils import now_unix_milliseconds

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
            if hasattr(reader, "use_fallback_reader"):
                # Some readers use MinerU for document parsing. However, its speed is too slow
                # for document chat scenarios (chat with doc).
                # Therefore, MinerUReader is disabled here, and a faster reader with
                # lower parsing quality is used instead.
                # In the future, we can optimize this, for example, by implementing a separate
                # document parsing service to handle user-uploaded documents asynchronously.
                reader.use_fallback_reader(True)
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
                if tokens.startswith(RELATED_QUESTIONS):
                    related_question = ast.literal_eval(tokens[len(RELATED_QUESTIONS):])
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
