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
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db import models as db_models
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import ChatNotFoundException, ResourceNotFoundException
from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser
from aperag.schema import view_models
from aperag.schema.view_models import Chat, ChatDetails
from aperag.utils.constant import DOC_QA_REFERENCES, DOCUMENT_URLS
from aperag.utils.history import (
    RedisChatMessageHistory,
    fail_response,
    get_async_redis_client,
    references_response,
    start_response,
    stop_response,
    success_response,
)

logger = logging.getLogger(__name__)


class FrontendFormatter:
    """Format responses according to Aperag custom format"""

    @staticmethod
    def format_stream_start(msg_id: str) -> Dict[str, Any]:
        """Format the start event for streaming"""
        return {
            "type": "start",
            "id": msg_id,
            "timestamp": int(time.time()),
        }

    @staticmethod
    def format_stream_content(msg_id: str, content: str) -> Dict[str, Any]:
        """Format a content chunk for streaming"""
        return {
            "type": "message",
            "id": msg_id,
            "data": content,
            "timestamp": int(time.time()),
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
            "timestamp": int(time.time()),
        }

    @staticmethod
    def format_complete_response(msg_id: str, content: str) -> Dict[str, Any]:
        """Format a complete response for non-streaming mode"""
        return {
            "type": "message",
            "id": msg_id,
            "data": content,
            "timestamp": int(time.time()),
        }

    @staticmethod
    def format_error(error: str) -> Dict[str, Any]:
        """Format an error response"""
        return {
            "type": "error",
            "id": str(uuid.uuid4()),
            "data": error,
            "timestamp": int(time.time()),
        }


class ChatService:
    """Chat service that handles business logic for chats"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    def build_chat_response(self, chat: db_models.Chat) -> view_models.Chat:
        """Build Chat response object for API return."""
        return Chat(
            id=chat.id,
            title=chat.title,
            bot_id=chat.bot_id,
            peer_type=chat.peer_type,
            peer_id=chat.peer_id,
            created=chat.gmt_created.isoformat(),
            updated=chat.gmt_updated.isoformat(),
        )

    async def create_chat(self, user: str, bot_id: str) -> view_models.Chat:
        # First check if bot exists
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            raise ResourceNotFoundException("Bot", bot_id)

        # Direct call to repository method, which handles its own transaction
        chat = await self.db_ops.create_chat(user=user, bot_id=bot_id)

        return self.build_chat_response(chat)

    async def list_chats(
        self,
        user: str,
        bot_id: str,
        page: int = 1,
        page_size: int = 50,
    ):
        """List chats with pagination, sorting and search capabilities."""

        # Define sort field mapping
        sort_mapping = {
            "created": db_models.Chat.gmt_created,
        }

        # Define search fields mapping
        search_fields = {"title": db_models.Chat.title}

        async def _execute_paginated_query(session):
            from sqlalchemy import and_, desc, select

            # Build base query
            query = select(db_models.Chat).where(
                and_(
                    db_models.Chat.user == user,
                    db_models.Chat.bot_id == bot_id,
                    db_models.Chat.status != db_models.ChatStatus.DELETED,
                )
            )

            # Build query parameters
            from aperag.utils.pagination import ListParams, PaginationHelper, PaginationParams, SortParams

            params = ListParams(
                pagination=PaginationParams(page=page, page_size=page_size),
                sort=SortParams(sort_by="created", sort_order="desc"),
            )

            # Use pagination helper
            items, total = await PaginationHelper.paginate_query(
                query=query,
                session=session,
                params=params,
                sort_mapping=sort_mapping,
                search_fields=search_fields,
                default_sort=desc(db_models.Chat.gmt_created),
            )

            # Build chat responses
            chat_responses = []
            for chat in items:
                chat_responses.append(self.build_chat_response(chat))

            return PaginationHelper.build_response(items=chat_responses, total=total, page=page, page_size=page_size)

        return await self.db_ops._execute_query(_execute_paginated_query)

    async def get_chat(self, user: str, bot_id: str, chat_id: str) -> view_models.ChatDetails:
        # Import here to avoid circular imports
        from aperag.utils.history import query_chat_messages

        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            raise ChatNotFoundException(chat_id)

        # Get chat history
        messages = await query_chat_messages(user, chat_id)

        # Build response object
        chat_obj = self.build_chat_response(chat)
        return ChatDetails(**chat_obj.model_dump(), history=messages)

    async def update_chat(
        self, user: str, bot_id: str, chat_id: str, chat_in: view_models.ChatUpdate
    ) -> view_models.Chat:
        # First check if chat exists
        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            raise ChatNotFoundException(chat_id)

        # Direct call to repository method, which handles its own transaction
        updated_chat = await self.db_ops.update_chat_by_id(user, bot_id, chat_id, chat_in.title)

        if not updated_chat:
            raise ChatNotFoundException(chat_id)

        return self.build_chat_response(updated_chat)

    async def delete_chat(self, user: str, bot_id: str, chat_id: str) -> Optional[view_models.Chat]:
        """Delete chat by ID (idempotent operation)

        Returns the deleted chat or None if already deleted/not found
        """
        # Check if chat exists - if not, silently succeed (idempotent)
        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            return None

        # Direct call to repository method, which handles its own transaction
        deleted_chat = await self.db_ops.delete_chat_by_id(user, bot_id, chat_id)

        if deleted_chat:
            # Clear chat history from Redis
            history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
            await history.clear()

            return self.build_chat_response(deleted_chat)

        return None

    def stream_frontend_sse_response(
        self, generator: AsyncGenerator[Any, Any], formatter: FrontendFormatter, msg_id: str
    ):
        """Yield SSE events for FastAPI StreamingResponse."""

        async def event_stream():
            yield f"data: {json.dumps(formatter.format_stream_start(msg_id))}\n\n"
            async for chunk in generator:
                yield f"data: {json.dumps(formatter.format_stream_content(msg_id, chunk))}\n\n"
            yield f"data: {json.dumps(formatter.format_stream_end(msg_id))}\n\n"

        return event_stream()

    async def frontend_chat_completions(
        self, user: str, message: str, stream: bool, bot_id: str, chat_id: str, msg_id: str
    ) -> Any:
        """Frontend chat completions with special error handling for UI responses"""

        # Validate bot_id - return formatted error for frontend
        if not bot_id:
            return FrontendFormatter.format_error("bot_id is required")

        bot = await self.db_ops.query_bot(user, bot_id)
        if not bot:
            return FrontendFormatter.format_error("Bot not found")

        # Get or create chat session
        chat = await self.db_ops.query_chat_by_peer(bot.user, db_models.ChatPeerType.FEISHU, chat_id)

        if chat is None:
            # Create chat with peer info atomically in single transaction
            chat = await self.db_ops.create_chat(
                user=bot.user,
                bot_id=bot.id,
                title="Feishu Chat",
                peer_type=db_models.ChatPeerType.FEISHU,
                peer_id=chat_id,
            )

        # Use flow engine instead of MessageProcessor/pipeline
        formatter = FrontendFormatter()

        # Get bot's flow configuration
        bot_config = json.loads(bot.config or "{}")
        flow_config = bot_config.get("flow")
        if not flow_config:
            return FrontendFormatter.format_error("Bot flow config not found")

        try:
            flow = FlowParser.parse(flow_config)
            engine = FlowEngine()

            # Prepare initial data for flow execution
            initial_data = {
                "query": message,
                "user": user,
                "message_id": msg_id or str(uuid.uuid4()),
            }

            # Execute flow
            _, system_outputs = await engine.execute_flow(flow, initial_data)
            logger.info("Flow executed successfully!")

            # Find the async generator from flow outputs
            async_generator = None
            nodes = engine.find_end_nodes(flow)
            for node in nodes:
                async_generator = system_outputs[node].get("async_generator")
                if async_generator:
                    break

            if not async_generator:
                return FrontendFormatter.format_error("No output node found")

            # Return streaming or non-streaming response
            if stream:
                return StreamingResponse(
                    self.stream_frontend_sse_response(
                        async_generator(),
                        formatter,
                        msg_id or str(uuid.uuid4()),
                    ),
                    media_type="text/event-stream",
                )
            else:
                # Collect all content for non-streaming response
                full_content = ""
                async for chunk in async_generator():
                    full_content += chunk
                return formatter.format_complete_response(msg_id or str(uuid.uuid4()), full_content)

        except Exception as e:
            logger.exception(e)
            return FrontendFormatter.format_error(str(e))

    async def feedback_message(
        self,
        user: str,
        chat_id: str,
        message_id: str,
        feedback_type: str = None,
        feedback_tag: str = None,
        feedback_message: str = None,
    ) -> dict:
        """Handle message feedback for chat messages"""
        # Get message from Redis history to validate it exists and get context
        history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
        ai_msg = None
        human_msg = None
        for message in await history.messages:
            if message.message_id != message_id:
                continue
            if message.role == "ai":
                ai_msg = message
            if message.role == "human":
                human_msg = message

        if not ai_msg:
            raise ResourceNotFoundException("AI Message", message_id)
        if not human_msg:
            raise ResourceNotFoundException("Human Message", message_id)

        # Handle feedback state change based on UX design principles
        if feedback_type is None:
            # User wants to remove feedback (cancel like/dislike)
            success_removed = await self.db_ops.remove_message_feedback(user, chat_id, message_id)
            result = {"action": "deleted", "success": success_removed}
        else:
            # User wants to set feedback state (like/dislike)
            feedback = await self.db_ops.set_message_feedback_state(
                user=user,
                chat_id=chat_id,
                message_id=message_id,
                feedback_type=feedback_type,
                feedback_tag=feedback_tag,
                feedback_message=feedback_message,
                question=human_msg.get_main_content(),
                original_answer=ai_msg.get_main_content(),
            )
            result = {"action": "upserted", "feedback": feedback}
        return result

    async def handle_websocket_chat(self, websocket: WebSocket, user: str, bot_id: str, chat_id: str):
        """Handle WebSocket chat connections and message streaming"""
        await websocket.accept()

        try:
            # Get bot configuration first to determine bot type
            bot = await self.db_ops.query_bot(user, bot_id)
            if not bot:
                await websocket.send_text(fail_response("error", "Bot not found"))
                return

            # Route to appropriate service based on bot type
            if bot.type == db_models.BotType.AGENT:
                # Use AgentChatService for agent-type bots
                from aperag.service.agent_chat_service import AgentChatService

                agent_service = AgentChatService()
                await agent_service.handle_websocket_agent_chat(websocket, user, bot_id, chat_id)
                return

            # Continue with existing flow for knowledge and common bots
            history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())

            while True:
                # Receive message from client
                text_data = await websocket.receive_text()
                data = json.loads(text_data)

                # Extract message content - support both "data" and "message" fields
                message_content = data.get("data") or data.get("message", "")
                if not message_content:
                    await websocket.send_text(fail_response("error", "Message content is required"))
                    continue

                # Generate message ID
                message_id = str(uuid.uuid4())

                try:
                    # Get or create chat session
                    try:
                        await self.db_ops.query_chat(user, bot_id, chat_id)
                    except Exception:
                        # If chat doesn't exist, create it with direct repository call
                        await self.db_ops.create_chat(user=user, bot_id=bot_id, title="WebSocket Chat")

                    # Get bot's flow configuration
                    bot_config = json.loads(bot.config or "{}")
                    flow_config = bot_config.get("flow")
                    if not flow_config:
                        await websocket.send_text(fail_response(message_id, "Bot flow config not found"))
                        continue

                    flow = FlowParser.parse(flow_config)
                    engine = FlowEngine()

                    # Prepare initial data for flow execution
                    initial_data = {
                        "query": message_content,
                        "user": user,
                        "message_id": message_id,
                        "history": history,
                    }

                    # Send start message
                    await websocket.send_text(start_response(message_id))

                    # Execute flow
                    _, system_outputs = await engine.execute_flow(flow, initial_data)
                    logger.info("Flow executed successfully for WebSocket!")

                    # Find the async generator from flow outputs
                    async_generator = None
                    nodes = engine.find_end_nodes(flow)
                    for node in nodes:
                        async_generator = system_outputs[node].get("async_generator")
                        if async_generator:
                            break

                    if not async_generator:
                        await websocket.send_text(fail_response(message_id, "No output node found"))
                        continue

                    # Stream response tokens
                    full_message = ""
                    references = []
                    urls = []

                    async for chunk in async_generator():
                        # Handle special tokens for references and URLs (similar to original implementation)
                        if chunk.startswith(DOC_QA_REFERENCES):
                            try:
                                references = json.loads(chunk[len(DOC_QA_REFERENCES) :])
                                continue
                            except Exception as e:
                                logger.exception(f"Error parsing doc qa references: {chunk}, {e}")

                        if chunk.startswith(DOCUMENT_URLS):
                            try:
                                urls = eval(chunk[len(DOCUMENT_URLS) :])  # Using eval as in original code
                                continue
                            except Exception as e:
                                logger.exception(f"Error parsing document urls: {chunk}, {e}")

                        # Send streaming response
                        await websocket.send_text(success_response(message_id, chunk))
                        full_message += chunk

                    # Send stop message with references and URLs
                    memory_count = 0  # You might want to implement memory counting if needed
                    await websocket.send_text(references_response(message_id, references, memory_count, urls))
                    await websocket.send_text(stop_response(message_id))

                except Exception as e:
                    logger.exception(f"Error processing WebSocket message: {e}")
                    await websocket.send_text(fail_response(message_id, str(e)))

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for bot {bot_id}, chat {chat_id}")
        except Exception as e:
            logger.exception(f"WebSocket error: {e}")
            try:
                await websocket.send_text(fail_response("error", str(e)))
            except Exception as e:
                logger.exception(f"Error sending fail response: {e}")


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
chat_service_global = ChatService()
