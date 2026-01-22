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
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops

logger = logging.getLogger(__name__)


@dataclass
class APIRequest:
    """API request parameters for direct API calls"""

    user: str
    bot_id: str
    msg_id: str
    stream: bool
    messages: List[Dict[str, str]]


class OpenAIFormatter:
    """Format responses according to OpenAI API specification"""

    @staticmethod
    def format_stream_start(msg_id: str) -> Dict[str, Any]:
        """Format the start event for streaming"""
        return {
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "aperag",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

    @staticmethod
    def format_stream_content(msg_id: str, content: str) -> Dict[str, Any]:
        """Format a content chunk for streaming"""
        return {
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "aperag",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }

    @staticmethod
    def format_stream_end(msg_id: str) -> Dict[str, Any]:
        """Format the end event for streaming"""
        return {
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "aperag",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    @staticmethod
    def format_complete_response(msg_id: str, content: str) -> Dict[str, Any]:
        """Format a complete response for non-streaming mode"""
        return {
            "id": msg_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "aperag",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    @staticmethod
    def format_error(error: str) -> Dict[str, Any]:
        """Format an error response"""
        return {"error": {"message": error, "type": "server_error", "code": "internal_error"}}


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
        """Handle OpenAI-compatible chat completions - Not implemented"""
        return None, OpenAIFormatter.format_error(
            "The /v1/chat/completions endpoint is not implemented. Please use WebSocket API for agent-type bots."
        )


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
chat_completion_service = ChatCompletionService()
