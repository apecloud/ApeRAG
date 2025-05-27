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
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

from channels.generic.http import AsyncHttpConsumer

from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.db.models import Bot
from aperag.pipeline.common_pipeline import CommonPipeline
from aperag.pipeline.knowledge_pipeline import create_knowledge_pipeline

logger = logging.getLogger(__name__)


@dataclass
class ChatRequest:
    """Chat request parameters for frontend chat"""

    user: str
    bot_id: str
    chat_id: str
    msg_id: str
    stream: bool
    message: str


@dataclass
class APIRequest:
    """API request parameters for direct API calls"""

    user: str
    bot_id: str
    msg_id: str
    stream: bool
    messages: List[Dict[str, str]]


class BaseFormatter:
    """Base class for response formatters"""

    @staticmethod
    def format_error(error: str) -> Dict[str, Any]:
        """Format an error response"""
        raise NotImplementedError


class MessageProcessor:
    """Handle message processing for different bot types"""

    def __init__(self, bot: Bot, history: Optional[RedisChatMessageHistory] = None):
        self.bot = bot
        self.history = history

    async def process_message(self, message: str, msg_id: str) -> AsyncGenerator[str, None]:
        """Process a message and yield content chunks as they become available"""
        if self.bot.type == Bot.Type.KNOWLEDGE:
            collections = await self.bot.collections()
            if len(collections) > 0:
                collection = collections[0]
            else:
                raise ValueError("No collection found for bot")
            pipeline = await create_knowledge_pipeline(bot=self.bot, collection=collection, history=self.history)
            async for msg in pipeline.run(message, message_id=msg_id):
                yield msg

        elif self.bot.type == Bot.Type.COMMON:
            pipeline = CommonPipeline(bot=self.bot, collection=None, history=self.history)
            async for msg in pipeline.run(message, message_id=msg_id):
                yield msg

        else:
            raise ValueError("Unsupported bot type")


class BaseConsumer(AsyncHttpConsumer):
    """Base class for SSE consumers"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.formatter = None  # To be set by subclasses

    async def send_event(self, data: dict, more_body=True):
        """Send a single SSE event"""
        event = f"data: {json.dumps(data)}\n\n"
        await self.send_body(event.encode("utf-8"), more_body=more_body)

    async def send_error(self, error: str):
        """Send an error response"""
        error_data = self.formatter.format_error(error)
        await self.send_event(error_data, more_body=False)
