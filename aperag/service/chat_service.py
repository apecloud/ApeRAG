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
import uuid
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db import models as db_models
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser
from aperag.schema import view_models
from aperag.schema.view_models import Chat, ChatDetails, ChatList
from aperag.utils.constant import DOC_QA_REFERENCES, DOCUMENT_URLS
from aperag.utils.history import (
    RedisChatMessageHistory,
    fail_response,
    get_async_redis_client,
    start_response,
    stop_response,
    success_response,
)
from aperag.utils.utils import now_unix_milliseconds
from aperag.views.utils import fail, success

logger = logging.getLogger(__name__)


class FrontendFormatter:
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
            return fail(HTTPStatus.NOT_FOUND, "Bot not found")

        async def _create_operation(session):
            # Use DatabaseOps to create chat
            db_ops_session = AsyncDatabaseOps(session)
            chat_instance = await db_ops_session.create_chat(user=user, bot_id=bot_id, title="New Chat")
            return self.build_chat_response(chat_instance)

        try:
            result = await self.db_ops.execute_with_transaction(_create_operation)
            return success(result)
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to create chat: {str(e)}")

    async def list_chats(self, user: str, bot_id: str) -> view_models.ChatList:
        chats = await self.db_ops.query_chats(user, bot_id)
        response = []
        for chat in chats:
            response.append(self.build_chat_response(chat))
        return success(ChatList(items=response))

    async def get_chat(self, user: str, bot_id: str, chat_id: str) -> view_models.ChatDetails:
        # Import here to avoid circular imports
        from aperag.views.utils import query_chat_messages

        try:
            chat = await self.db_ops.query_chat(user, bot_id, chat_id)
            if chat is None:
                return fail(HTTPStatus.NOT_FOUND, "Chat not found")

            messages = await query_chat_messages(user, chat_id)

            chat_obj = self.build_chat_response(chat)
            return success(ChatDetails(**chat_obj.model_dump(), history=messages))

        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to get chat: {str(e)}")

    async def update_chat(
        self, user: str, bot_id: str, chat_id: str, chat_in: view_models.ChatUpdate
    ) -> view_models.Chat:
        # First check if chat exists
        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            return fail(HTTPStatus.NOT_FOUND, "Chat not found")

        async def _update_operation(session):
            # Use DatabaseOps to update chat
            db_ops_session = AsyncDatabaseOps(session)
            updated_chat = await db_ops_session.update_chat_by_id(
                user=user, bot_id=bot_id, chat_id=chat_id, title=chat_in.title
            )

            if not updated_chat:
                raise ValueError("Chat not found")

            return self.build_chat_response(updated_chat)

        try:
            result = await self.db_ops.execute_with_transaction(_update_operation)
            return success(result)
        except ValueError as e:
            return fail(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to update chat: {str(e)}")

    async def delete_chat(self, user: str, bot_id: str, chat_id: str) -> view_models.Chat:
        # First check if chat exists
        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            return fail(HTTPStatus.NOT_FOUND, "Chat not found")

        async def _delete_operation(session):
            # Use DatabaseOps to delete chat
            db_ops_session = AsyncDatabaseOps(session)
            deleted_chat = await db_ops_session.delete_chat_by_id(user, bot_id, chat_id)

            if not deleted_chat:
                raise ValueError("Chat not found")

            # Clear chat history from Redis
            history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
            await history.clear()

            return self.build_chat_response(deleted_chat)

        try:
            result = await self.db_ops.execute_with_transaction(_delete_operation)
            return success(result)
        except ValueError as e:
            return fail(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to delete chat: {str(e)}")

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
        try:
            # Validate bot_id
            if not bot_id:
                return success(FrontendFormatter.format_error("bot_id is required"))

            bot = await self.db_ops.query_bot(user, bot_id)
            if not bot:
                return success(FrontendFormatter.format_error("Bot not found"))

            # Get or create chat session
            chat = await self.db_ops.query_chat_by_peer(bot.user, db_models.ChatPeerType.FEISHU, chat_id)

            async def _ensure_chat_operation(session):
                if chat is None:
                    db_ops_session = AsyncDatabaseOps(session)
                    new_chat = await db_ops_session.create_chat(user=bot.user, bot_id=bot.id, title="Feishu Chat")
                    # Set peer info manually since DatabaseOps doesn't support this yet
                    new_chat.peer_type = db_models.ChatPeerType.FEISHU
                    new_chat.peer_id = chat_id
                    session.add(new_chat)
                    await session.flush()
                    await session.refresh(new_chat)
                    return new_chat
                return chat

            if chat is None:
                chat = await self.db_ops.execute_with_transaction(_ensure_chat_operation)

            # Use flow engine instead of MessageProcessor/pipeline
            formatter = FrontendFormatter()

            # Get bot's flow configuration
            bot_config = json.loads(bot.config or "{}")
            flow_config = bot_config.get("flow")
            if not flow_config:
                return success(FrontendFormatter.format_error("Bot flow config not found"))

            flow = FlowParser.parse(flow_config)
            engine = FlowEngine()

            # Prepare initial data for flow execution
            initial_data = {
                "query": message,
                "user": user,
                "message_id": msg_id or str(uuid.uuid4()),
            }

            # Execute flow
            try:
                _, system_outputs = await engine.execute_flow(flow, initial_data)
                logger.info("Flow executed successfully!")
            except Exception as e:
                logger.exception(e)
                return success(FrontendFormatter.format_error(str(e)))

            # Find the async generator from flow outputs
            async_generator = None
            nodes = engine.find_end_nodes(flow)
            for node in nodes:
                async_generator = system_outputs[node].get("async_generator")
                if async_generator:
                    break

            if not async_generator:
                return success(FrontendFormatter.format_error("No output node found"))

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
                return success(formatter.format_complete_response(msg_id or str(uuid.uuid4()), full_content))

        except Exception as e:
            logger.exception(e)
            return success(FrontendFormatter.format_error(str(e)))

    async def feedback_message(
        self,
        user: str,
        chat_id: str,
        message_id: str,
        feedback_type: str = None,
        feedback_tag: str = None,
        feedback_message: str = None,
    ) -> view_models.Chat:
        """Handle message feedback for chat messages"""
        try:
            # Get message from Redis history to validate it exists and get context
            history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
            msg = None
            for message in await history.messages:
                item = json.loads(message.content)
                if item["id"] != message_id:
                    continue
                if message.additional_kwargs.get("role", "") != "ai":
                    continue
                msg = item
                break

            if msg is None:
                return fail(HTTPStatus.NOT_FOUND, "Message not found")

            async def _feedback_operation(session):
                db_ops_session = AsyncDatabaseOps(session)

                # If feedback_type is None, delete the feedback record
                if feedback_type is None:
                    success_deleted = await db_ops_session.delete_message_feedback(user, chat_id, message_id)
                    return {"action": "deleted", "success": success_deleted}

                # Otherwise create or update the feedback record
                feedback = await db_ops_session.upsert_message_feedback(
                    user=user,
                    chat_id=chat_id,
                    message_id=message_id,
                    feedback_type=feedback_type,
                    feedback_tag=feedback_tag,
                    feedback_message=feedback_message,
                    question=msg.get("query"),
                    original_answer=msg.get("response", ""),
                    collection_id=msg.get("collection_id"),
                )

                return {"action": "upserted", "feedback": feedback}

            result = await self.db_ops.execute_with_transaction(_feedback_operation)
            return success(result)

        except Exception as e:
            logger.exception(f"Error in feedback_message: {e}")
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to process feedback: {str(e)}")

    async def handle_websocket_chat(self, websocket: WebSocket, user: str, bot_id: str, chat_id: str):
        """Handle WebSocket chat connections and message streaming"""
        await websocket.accept()

        history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())

        try:
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
                    # Get bot configuration
                    bot = await self.db_ops.query_bot(user, bot_id)
                    if not bot:
                        await websocket.send_text(fail_response(message_id, "Bot not found"))
                        continue

                    # Get or create chat session
                    try:
                        await self.db_ops.query_chat(user, bot_id, chat_id)
                    except Exception:
                        # If chat doesn't exist, create it
                        async def _create_chat_operation(session):
                            db_ops_session = AsyncDatabaseOps(session)
                            new_chat = await db_ops_session.create_chat(
                                user=user, bot_id=bot_id, title="WebSocket Chat"
                            )
                            return new_chat

                        await self.db_ops.execute_with_transaction(_create_chat_operation)

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
                    await websocket.send_text(stop_response(message_id, references, memory_count, urls))

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
