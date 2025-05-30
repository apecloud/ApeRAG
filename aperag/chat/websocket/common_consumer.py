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

import json
import logging
import os
import traceback
from pathlib import Path

import websockets

from aperag.chat.utils import (
    check_quota_usage,
    fail_response,
    manage_quota_usage,
    start_response,
    stop_response,
    success_response,
)
from aperag.chat.websocket.base_consumer import BaseConsumer
from aperag.docparser.base import MarkdownPart
from aperag.docparser.doc_parser import get_fast_doc_parser
from aperag.pipeline.common_pipeline import create_common_pipeline
from aperag.source.utils import gen_temporary_file
from aperag.utils.utils import now_unix_milliseconds

logger = logging.getLogger(__name__)


class CommonConsumer(BaseConsumer):
    async def connect(self):
        await super().connect()
        self.file = None
        self.file_name = None
        self.pipeline = await create_common_pipeline(bot=self.bot, collection=self.collection, history=self.history)

    async def predict(self, query, **kwargs):
        async for msg in self.pipeline.run(query, gen_references=True, **kwargs):
            yield msg

    async def receive(self, text_data=None, bytes_data=None):
        message = ""
        message_id = f"{now_unix_milliseconds()}"

        # bytes_data: [file_name_length, file_name, file_content]
        if bytes_data:
            file_name_length = int.from_bytes(bytes_data[:4], byteorder="big")
            file_name_bytes = bytes_data[4 : 4 + file_name_length]
            self.file_name = file_name_bytes.decode("utf-8")
            file_content = bytes_data[4 + file_name_length :]

            doc_parser = get_fast_doc_parser()
            supported_file_extensions = doc_parser.supported_extensions()

            file_suffix = os.path.splitext(self.file_name)[1].lower()
            if file_suffix not in supported_file_extensions:
                error = f"unsupported file type {file_suffix}"
                await self.send(text_data=fail_response(message_id, error=error))
                return

            temp_file = gen_temporary_file(self.file_name)
            temp_file.write(file_content)
            temp_file.close()

            content = None
            try:
                parts = doc_parser.parse_file(Path(temp_file.name))
                for part in parts:
                    if isinstance(part, MarkdownPart):
                        content = part.markdown
                        break
            except Exception:
                logger.exception("Failed to parse file %s", self.file_name)

            self.file = content
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

            async for tokens in self.predict(data["data"], message_id=message_id, file=self.file):
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
            self.file = None
            self.file_name = None
            # send stop message
            await self.send(text_data=stop_response(message_id, [], self.pipeline.memory_count, urls=[]))
