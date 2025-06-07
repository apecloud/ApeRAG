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
import uuid
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.chat.sse.base import APIRequest
from aperag.chat.sse.openai_consumer import OpenAIFormatter
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser

logger = logging.getLogger(__name__)


class ChatCompletionService:
    """Chat completion service that handles business logic for OpenAI-compatible API"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def stream_openai_sse_response(self, generator: AsyncGenerator[str, None], formatter, msg_id: str):
        """Stream SSE response for OpenAI API format"""
        yield f"data: {json.dumps(formatter.format_stream_start(msg_id))}\n\n"
        async for chunk in generator:
            await asyncio.sleep(0.001)
            yield f"data: {json.dumps(formatter.format_stream_content(msg_id, chunk))}\n\n"
        yield f"data: {json.dumps(formatter.format_stream_end(msg_id))}\n\n"

    async def openai_chat_completions(self, user, body_data, query_params):
        """Handle OpenAI-compatible chat completions"""
        bot_id = query_params.get("bot_id") or query_params.get("app_id")
        if not bot_id:
            return None, OpenAIFormatter.format_error("bot_id is required")

        api_request = APIRequest(
            user=user,
            bot_id=bot_id,
            msg_id=str(uuid.uuid4()),
            stream=body_data.get("stream", False),
            messages=body_data.get("messages", []),
        )

        bot = await self.db_ops.query_bot(api_request.user, api_request.bot_id)
        if not bot:
            return None, OpenAIFormatter.format_error("Bot not found")

        formatter = OpenAIFormatter()

        # Get bot's flow configuration
        bot_config = json.loads(bot.config or "{}")
        flow_config = bot_config.get("flow")
        if not flow_config:
            return None, OpenAIFormatter.format_error("Bot flow config not found")

        flow = FlowParser.parse(flow_config)
        engine = FlowEngine()
        initial_data = {
            "query": api_request.messages[-1]["content"],
            "user": api_request.user,
            "message_id": api_request.msg_id,
        }

        try:
            _, system_outputs = await engine.execute_flow(flow, initial_data)
            logger.info("Flow executed successfully!")
        except Exception as e:
            logger.exception(e)
            return None, OpenAIFormatter.format_error(str(e))

        async_generator = None
        nodes = engine.find_end_nodes(flow)
        for node in nodes:
            async_generator = system_outputs[node].get("async_generator")
            if async_generator:
                break

        if not async_generator:
            return None, OpenAIFormatter.format_error("No output node found")

        return (api_request, formatter, async_generator), None


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
chat_completion_service = ChatCompletionService()
