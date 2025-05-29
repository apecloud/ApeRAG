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
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import parse_qsl

from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.sse.base import BaseConsumer, BaseFormatter, ChatRequest, MessageProcessor, logger
from aperag.chat.utils import get_async_redis_client, now_unix_milliseconds
from aperag.db.models import Bot, Chat


class FrontendFormatter(BaseFormatter):
    """Format responses according to Aperag custom format"""

    @staticmethod
    def format_stream_start(msg_id: str) -> Dict[str, Any]:
        """Format the start event for streaming"""
        return {
            "type": "start",
            "id": msg_id,
            "timestamp": now_unix_milliseconds(),
        }

    @staticmethod
    def format_stream_content(msg_id: str, content: str) -> Dict[str, Any]:
        """Format a content chunk for streaming"""
        return {
            "type": "message",
            "id": msg_id,
            "data": content,
            "timestamp": now_unix_milliseconds(),
        }

    @staticmethod
    def format_stream_end(
        msg_id: str,
        references: List[str] = None,
        memory_count: int = 0,
        urls: List[str] = None,
    ) -> Dict[str, Any]:
        """Format the end event for streaming"""
        if references is None:
            references = []
        if urls is None:
            urls = []

        return {
            "type": "stop",
            "id": msg_id,
            "data": references,
            "memoryCount": memory_count,
            "urls": urls,
            "timestamp": now_unix_milliseconds(),
        }

    @staticmethod
    def format_complete_response(msg_id: str, content: str) -> Dict[str, Any]:
        """Format a complete response for non-streaming mode"""
        return {
            "type": "message",
            "id": msg_id,
            "data": content,
            "timestamp": now_unix_milliseconds(),
        }

    @staticmethod
    def format_error(error: str) -> Dict[str, Any]:
        """Format an error response"""
        return {
            "type": "error",
            "id": str(uuid.uuid4()),
            "data": error,
            "timestamp": now_unix_milliseconds(),
        }


class FrontendConsumer(BaseConsumer):
    """Handle chat requests from frontend with chat history"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.formatter = FrontendFormatter()

    async def handle(self, body):
        try:
            await self._handle(body)
        except Exception as e:
            logger.exception(e)
            await self.send_error(str(e))

    async def _handle(self, body):
        # Parse request parameters
        request = await self._parse_request(body)

        # Get bot
        bot = await self._get_bot(request)
        if not bot:
            await self.send_error("Bot not found")
            return

        try:
            # Get or create chat
            chat = await self._get_or_create_chat(bot, request)

            # Initialize message processor
            history = RedisChatMessageHistory(session_id=str(chat.id), redis_client=get_async_redis_client())
            processor = MessageProcessor(bot, history)

            # Process message and send response based on stream mode
            if request.stream:
                await self._send_streaming_response(
                    request.msg_id, processor.process_message(request.message, request.msg_id)
                )
            else:
                await self._send_complete_response(
                    request.msg_id, processor.process_message(request.message, request.msg_id)
                )

        except ValueError as e:
            await self.send_error(str(e))
        except Exception as e:
            logger.exception(e)
            await self.send_error(str(e))

    async def _parse_request(self, body) -> ChatRequest:
        """Parse request parameters from body and query string"""
        query_params = dict(parse_qsl(self.scope["query_string"].decode("utf-8")))
        return ChatRequest(
            user=query_params.get("user", ""),
            bot_id=query_params.get("bot_id", ""),
            chat_id=query_params.get("chat_id", ""),
            msg_id=query_params.get("msg_id", "") or str(uuid.uuid4()),
            stream=query_params.get("stream", "false").lower() == "true",
            message=body.decode("utf-8"),
        )

    async def _get_bot(self, request: ChatRequest) -> Optional[Bot]:
        """Get bot by ID"""
        from aperag.db.ops import query_bot

        return await query_bot(request.user, request.bot_id)

    async def _get_or_create_chat(self, bot: Bot, request: ChatRequest) -> Chat:
        """Get or create chat session"""
        from aperag.db.ops import query_chat_by_peer

        chat = await query_chat_by_peer(bot.user, Chat.PeerType.FEISHU, request.chat_id)
        if chat is None:
            chat = Chat(user=bot.user, bot_id=bot.id, peer_type=Chat.PeerType.FEISHU, peer_id=request.chat_id)
            await chat.asave()
        return chat

    async def _send_streaming_response(self, msg_id: str, content_generator: AsyncGenerator[str, None]):
        """Send streaming response using SSE"""
        # Set SSE headers
        await self.send_headers(
            headers=[
                (b"Cache-Control", b"no-cache"),
                (b"Content-Type", b"text/event-stream"),
                (b"Transfer-Encoding", b"chunked"),
                (b"Connection", b"keep-alive"),
                (b"X-Accel-Buffering", b"no"),
            ]
        )

        # Send start event
        await self.send_event(self.formatter.format_stream_start(msg_id))

        # Send content chunks as they become available
        async for chunk in content_generator:
            await self.send_event(self.formatter.format_stream_content(msg_id, chunk))

        # Send end event
        await self.send_event(self.formatter.format_stream_end(msg_id), more_body=False)

    async def _send_complete_response(self, msg_id: str, content_generator: AsyncGenerator[str, None]):
        """Send complete response as JSON"""
        await self.send_headers(
            headers=[
                (b"Content-Type", b"application/json"),
            ]
        )

        # Collect all content
        full_content = ""
        async for chunk in content_generator:
            full_content += chunk

        response = self.formatter.format_complete_response(msg_id, full_content)
        await self.send_body(json.dumps(response).encode("utf-8"), more_body=False)
