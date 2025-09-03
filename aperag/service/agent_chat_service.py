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
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import WebSocket
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.agent import (
    AgentHistoryManager,
    AgentMemoryManager,
    AgentMessageQueue,
    agent_session_manager,
    extract_tool_call_references,
    format_agent_setup_error,
    format_invalid_json_error,
    format_invalid_model_spec_error,
    format_mcp_connection_error,
    format_processing_error,
    format_query_required_error,
    format_stream_content,
    format_stream_end,
    format_stream_start,
)
from aperag.agent.agent_config import AgentConfig
from aperag.agent.agent_event_listener import agent_event_listener
from aperag.agent.exceptions import (
    AgentConfigurationError,
    JSONParsingError,
    MCPAppInitializationError,
    MCPConnectionError,
    handle_agent_error,
    safe_json_parse,
)
from aperag.agent.response_types import AgentErrorResponse, AgentToolCallResultResponse
from aperag.chat.history.message import StoredChatMessage, create_assistant_message
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.schema import view_models
from aperag.service.prompt_template_service import build_agent_query_prompt, get_agent_system_prompt
from aperag.trace import trace_async_function

logger = logging.getLogger(__name__)


def format_websocket_error(error: Exception, data: str) -> AgentErrorResponse:
    try:
        parsed = safe_json_parse(data, "language_detection")
        language = parsed.get("language", "en-US")
    except Exception:
        language = "en-US"

    if isinstance(error, JSONParsingError):
        return format_invalid_json_error(str(error), language)

    if isinstance(error, AgentConfigurationError):
        error_msg = str(error).lower()
        if "query" in error_msg:
            return format_query_required_error(language)
        if "completion" in error_msg or "modelspec" in error_msg:
            return format_invalid_model_spec_error(str(error), language)

    return format_processing_error(str(error), language)


class AgentChatService:
    """
    Chat service specifically for agent-type bots that uses MCPApp for intelligent conversation.

    This service uses AgentSessionManager for efficient session lifecycle management,
    including collection selection, model choice, and web search capabilities.

    Refactored to use message queue for clean separation of concerns.
    """

    def __init__(self, session: AsyncSession = None):
        if session is None:
            self.db_ops = async_db_ops
        else:
            self.db_ops = AsyncDatabaseOps(session)

        # Initialize memory and history managers
        self.memory_manager = AgentMemoryManager()
        self.history_manager = AgentHistoryManager()

    def _parse_websocket_message(
        self, raw_data: str
    ) -> Tuple[Optional[view_models.AgentMessage], Optional[AgentErrorResponse]]:
        """
        Parse WebSocket message using Go-style error handling.

        Args:
            raw_data: Raw JSON string from WebSocket

        Returns:
            Tuple of (agent_message, error_response):
            - If successful: (agent_message, None)
            - If failed: (None, error_response_dict)
        """
        try:
            # Step 1: Safe JSON parsing using agent module utilities
            message_data = safe_json_parse(raw_data, "websocket_message")

            # Step 2: Validate required query field early
            query = message_data.get("query", "").strip()
            if not query:
                from aperag.agent.exceptions import agent_config_invalid

                error = agent_config_invalid("query", "Query is required and cannot be empty")
                error_response = format_websocket_error(error, raw_data)
                return None, error_response

            # Step 3: Parse and validate AgentMessage using Pydantic
            agent_message = view_models.AgentMessage(**message_data)
            return agent_message, None

        except (JSONParsingError, AgentConfigurationError) as e:
            error_response = format_websocket_error(e, raw_data)
            return None, error_response
        except Exception as e:
            # Handle unexpected errors
            from aperag.agent.exceptions import agent_config_invalid

            config_error = agent_config_invalid("agent_message", f"Unexpected error: {str(e)}")
            error_response = format_websocket_error(config_error, raw_data)
            return None, error_response

    @handle_agent_error("websocket_agent_chat", reraise=False)
    async def handle_websocket_agent_chat(self, websocket: WebSocket, user: str, bot_id: str, chat_id: str):
        """Handle WebSocket connections for agent-type bot chats with message queue architecture"""
        while True:
            # Receive message from WebSocket
            data = await websocket.receive_text()

            # Parse WebSocket message using Go-style error handling
            agent_message, error_response = self._parse_websocket_message(data)
            if error_response:
                await websocket.send_text(json.dumps(error_response))
                continue

            # Process each message in a new trace context
            await self._handle_single_message(websocket, agent_message, user, chat_id)

    @trace_async_function("name=handle_single_websocket_message", new_trace=True)
    async def _handle_single_message(
        self, websocket: WebSocket, agent_message: view_models.AgentMessage, user: str, chat_id: str
    ):
        """Handle a single WebSocket message with its own trace"""
        trace_id = None
        try:
            message_id = str(uuid.uuid4())
            message_queue = AgentMessageQueue()
            trace_id = await self.register_message_queue(agent_message.language, chat_id, message_id, message_queue)

            # Get document metadata and associate documents with message if files are provided
            from aperag.service.chat_document_service import chat_document_service
            files = await chat_document_service.associate_documents_with_message(
                chat_id=chat_id, 
                message_id=message_id, 
                files=[file.id for file in agent_message.files], 
                user=user
            )

            # Message Producer: Start background task to process agent generation message
            process_task = asyncio.create_task(
                self.process_agent_message(agent_message, user, chat_id, message_id, message_queue)
            )
            # Message Consumer
            consumer_task = asyncio.create_task(
                self._consume_messages_from_queue(chat_id, message_id, trace_id, message_queue, websocket)
            )
            process_result, consumer_result = await asyncio.gather(process_task, consumer_task, return_exceptions=True)

            # Handle process_task exceptions with unified error formatting
            if isinstance(process_result, Exception):
                logger.error(f"Process task failed: {process_result}")
                error_response = self._format_exception_to_error_response(
                    process_result, agent_message.language or "en-US"
                )
                await websocket.send_text(json.dumps(error_response))
                return

            # Handle consumer_task exceptions
            if isinstance(consumer_result, Exception):
                logger.error(f"Consumer task failed: {consumer_result}")
                error_response = format_processing_error(str(consumer_result), agent_message.language or "en-US")
                await websocket.send_text(json.dumps(error_response))
                return

            # Handle history saving at WebSocket layer (better separation of concerns)
            # process_result now contains {query, content, references} on success
            query = process_result.get("query", "")
            ai_response = process_result.get("content", "")
            references = process_result.get("references", "")
            tool_use_list = consumer_result
            await self._save_conversation_history(
                chat_id, message_id, trace_id, query, ai_response, tool_use_list, references, files
            )

        except Exception as e:
            # This catches any other unexpected errors not handled above
            logger.error(f"Unexpected error processing agent websocket message: {e}")
            error_response = format_processing_error(str(e), agent_message.language or "en-US")
            await websocket.send_text(json.dumps(error_response))
        finally:
            if trace_id:
                await agent_event_listener.unregister_listener(str(trace_id))

    async def register_message_queue(self, language, chat_id, message_id, message_queue):
        # Get the trace_id from the current span
        from aperag.trace.mcp_integration import get_current_trace_info

        trace_id, _ = get_current_trace_info()
        if not trace_id:
            logger.error("Could not get trace_id from current span, event dispatching will fail.")
        else:
            # Register a listener for this request with the global proxy.
            await agent_event_listener.register_listener(
                trace_id=str(trace_id),
                chat_id=chat_id,
                message_id=message_id,
                queue=message_queue,
                language=language,
            )
        return trace_id

    async def _stream_message_content(
        self, message: Dict[str, Any], websocket: WebSocket, chunk_size: int = 5, delay: float = 0.01
    ) -> None:
        """
        Stream message content in small chunks to simulate typing effect.

        Args:
            message: The message dict with type="message"
            websocket: WebSocket connection to send chunks
            chunk_size: Number of characters per chunk
            delay: Delay in seconds between chunks
        """
        content = message.get("data", "")
        if not content:
            # If no content, send the original message
            await websocket.send_text(json.dumps(message))
            return

        # Split content into chunks
        chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]

        for i, chunk in enumerate(chunks):
            # Create a chunk message with same structure but partial content
            chunk_message = {
                "type": "message",
                "id": message.get("id"),
                "data": chunk,
                "timestamp": message.get("timestamp", int(time.time())),
            }

            await websocket.send_text(json.dumps(chunk_message))
            logger.debug(f"Sent message chunk {i + 1}/{len(chunks)}: {len(chunk)} chars")

            # Add delay between chunks (except for the last one)
            if i < len(chunks) - 1:
                await asyncio.sleep(delay)

    async def _consume_messages_from_queue(
        self, chat_id: str, message_id: str, trace_id: str, message_queue: AgentMessageQueue, websocket: WebSocket
    ) -> List[AgentToolCallResultResponse]:
        """
        Consume messages from queue, send to WebSocket, and collect AgentToolCallResultResponse messages.

        This method runs as a separate task to avoid race conditions.
        Returns a list of all AgentToolCallResultResponse messages.
        """
        try:
            # Properly initialize list to collect AgentToolCallResultResponse messages
            tool_call_results: List[Dict] = []

            while True:
                # Get message from queue (blocks until message is available)
                message = await message_queue.get()

                # None message signals end of stream
                if message is None:
                    logger.debug("Received end-of-stream signal from message queue")
                    break

                # Collect AgentToolCallResultResponse messages
                if isinstance(message, dict) and message.get("type") == "tool_call_result":
                    tool_call_results.append(message)

                # Special handling for type="message" - stream it in chunks
                if isinstance(message, dict) and message.get("type") == "message":
                    await self._stream_message_content(message, websocket)
                    logger.debug(f"Streamed message content: {message.get('type', 'unknown')}")
                else:
                    # Send other message types normally (start, stop, tool_call_result, etc.)
                    await websocket.send_text(json.dumps(message))
                    logger.debug(f"Sent message to WebSocket: {message.get('type', 'unknown')}")

            return tool_call_results

        except Exception as e:
            logger.error(f"Error in message consumer: {e}")
            raise

    async def _get_agent_session(self, agent_message: view_models.AgentMessage, user: str, chat_id: str):
        """Get or create chat session using AgentConfig."""
        # Query provider details and API key from database
        provider_info = await self.db_ops.query_llm_provider_by_name(agent_message.completion.model_service_provider)
        if not provider_info:
            error_msg = f"Provider '{agent_message.completion.model_service_provider}' not found in database"
            logger.error(error_msg)
            raise AgentConfigurationError(error_msg)

        api_key = await self.db_ops.query_provider_api_key(
            agent_message.completion.model_service_provider, user_id=user, need_public=True
        )
        if not api_key:
            error_msg = f"No API key available for provider '{agent_message.completion.model_service_provider}'"
            logger.error(error_msg)
            raise AgentConfigurationError(error_msg)

        aperag_api_keys = await self.db_ops.query_api_keys(user, is_system=True)
        for item in aperag_api_keys:
            aperag_api_key = item.key
        if not aperag_api_key:
            # Auto-create a new system aperag API key for the user if none exists
            logger.info(f"No aperag API key found for user {user}, creating a new system key")
            try:
                api_key_result = await self.db_ops.create_api_key(user=user, description="aperag", is_system=True)
                aperag_api_key = api_key_result.key
                logger.info(f"Successfully created new system aperag API key for user {user}")
            except Exception as e:
                error_msg = f"Failed to create aperag API key for user {user}: {str(e)}"
                logger.error(error_msg)
                raise AgentConfigurationError(error_msg)

        # Create AgentConfig with all needed parameters including chat_id
        config = AgentConfig(
            user_id=user,
            chat_id=chat_id,
            provider_name=agent_message.completion.model_service_provider,
            api_key=api_key,
            base_url=provider_info.base_url,
            default_model=agent_message.completion.model,
            language=agent_message.language if agent_message.language else "en-US",
            instruction=get_agent_system_prompt(language=agent_message.language),
            server_names=["aperag"],
            aperag_api_key=aperag_api_key,
            aperag_mcp_url=os.getenv("APERAG_MCP_URL", "http://localhost:8000/mcp/"),
            temperature=0.7,
            max_tokens=60000,
        )

        # Get or create chat session using config
        session = await agent_session_manager.get_or_create_session(config)

        return session

    async def process_agent_message(
        self,
        agent_message: view_models.AgentMessage,
        user: str,
        chat_id: str,
        message_id: str,
        message_queue: AgentMessageQueue,
    ) -> Dict[str, Any]:
        # Validate ModelSpec early
        if not agent_message.completion or not agent_message.completion.model:
            raise AgentConfigurationError(
                config_key="completion.model", reason="Model specification is required for AI response generation"
            )

        try:
            # Send start message
            await message_queue.put(format_stream_start(message_id))

            # Create memory from chat history
            history = await self.history_manager.get_chat_history(chat_id)
            memory = await self.memory_manager.create_memory_from_history(history, context_limit=4)

            # Get chat session
            session = await self._get_agent_session(agent_message, user, chat_id)
            llm = await session.get_llm(agent_message.completion.model)
            llm.history = memory

            comprehensive_prompt = build_agent_query_prompt(chat_id, agent_message=agent_message, user=user)
            request_params = RequestParams(
                maxTokens=8192,
                model=agent_message.completion.model,
                use_history=True,
                max_iterations=10,
                parallel_tool_calls=True,
                temperature=0.7,
                user=user,
            )
            response = await llm.generate_str(comprehensive_prompt, request_params)
            full_content = response if response else "No response generated"

            await asyncio.sleep(0.1)  # Allow time for the message to be processed in listener

            await message_queue.put(format_stream_content(message_id, full_content))

            tool_references = extract_tool_call_references(llm.history)
            urls = []

            await message_queue.put(format_stream_end(message_id, references=tool_references, urls=urls))

            return {
                "query": agent_message.query,
                "content": full_content,
                "references": tool_references,
            }

        finally:
            await message_queue.close()

    def _format_exception_to_error_response(self, exception: Exception, language: str) -> AgentErrorResponse:
        """
        Convert exception to properly formatted error response using unified error handling.

        Args:
            exception: The exception to format
            language: Language code for i18n error messages

        Returns:
            Formatted error response for WebSocket
        """
        # Use existing exception hierarchy and formatting utilities
        if isinstance(exception, AgentConfigurationError):
            # Check for specific configuration error types
            error_msg = str(exception).lower()
            if "model" in error_msg or "completion" in error_msg:
                return format_invalid_model_spec_error(str(exception), language)
            else:
                return format_agent_setup_error(str(exception), language)

        elif isinstance(exception, MCPConnectionError):
            return format_mcp_connection_error(language)

        elif isinstance(exception, MCPAppInitializationError):
            return format_agent_setup_error(str(exception), language)

        else:
            # Handle unexpected errors with generic processing error
            return format_processing_error(str(exception), language)

    async def chat_for_evaluation(
        self,
        query: str,
        user_id: str,
        model_name: str,
        model_service_provider: str,
        custom_llm_provider: Optional[Dict],
        collections: List[view_models.Collection],
        language: str = "en-US",
    ) -> StoredChatMessage | AgentErrorResponse:
        """
        Handle internal chat requests for evaluation tasks, bypassing WebSockets.
        Returns the AI response as a dictionary representation of StoredChatMessage.
        """
        # Construct AgentMessage
        agent_message = view_models.AgentMessage(
            query=query,
            completion=view_models.ModelSpec(
                model=model_name,
                model_service_provider=model_service_provider,
                custom_llm_provider=custom_llm_provider,
            ),
            collections=collections,
            language=language,
        )

        # Generate unique IDs for this interaction
        chat_id = f"eval-chat-{uuid.uuid4()}"
        message_id = str(uuid.uuid4())
        trace_id = None

        try:
            message_queue = AgentMessageQueue()
            trace_id = await self.register_message_queue(agent_message.language, chat_id, message_id, message_queue)

            # Simplified consumer that just collects results without a websocket
            async def consume_and_collect():
                tool_calls = []
                while True:
                    message = await message_queue.get()
                    if message is None:
                        break
                    if isinstance(message, dict) and message.get("type") == "tool_call_result":
                        tool_calls.append(message)
                return tool_calls

            process_task = asyncio.create_task(
                self.process_agent_message(agent_message, user_id, chat_id, message_id, message_queue)
            )
            consumer_task = asyncio.create_task(consume_and_collect())

            process_result, consumer_result = await asyncio.gather(process_task, consumer_task, return_exceptions=True)

            # Handle process_task exceptions with unified error formatting
            if isinstance(process_result, Exception):
                logger.error(f"Process task failed: {process_result}")
                error_response = self._format_exception_to_error_response(
                    process_result, agent_message.language or "en-US"
                )
                return error_response

            # Handle consumer_task exceptions
            if isinstance(consumer_result, Exception):
                logger.error(f"Consumer task failed: {consumer_result}")
                error_response = format_processing_error(str(consumer_result), agent_message.language or "en-US")
                return error_response

            query = process_result.get("query", "")
            ai_response = process_result.get("content", "")
            references = process_result.get("references", "")
            tool_use_list = consumer_result

            # AI message
            ai_message = create_assistant_message(
                content=ai_response,
                chat_id=chat_id,
                message_id=message_id,
                trace_id=trace_id,
                tool_use_list=tool_use_list,
                references=references,
                # urls=,
            )
            return ai_message

        except Exception as e:
            logger.error(f"Error during internal agent chat for evaluation: {e}")
            error_response = self._format_exception_to_error_response(e, agent_message.language or "en-US")
            return error_response
        finally:
            if trace_id:
                await agent_event_listener.unregister_listener(str(trace_id))

    async def _save_conversation_history(
        self,
        chat_id: str,
        message_id: str,
        trace_id: str,
        query: str,
        ai_response: str,
        files: List[Dict[str, Any]],
        tool_use_list: List[Dict],
        tool_references: List[Dict[str, Any]],
    ) -> None:
        """
        Save conversation history from successful agent processing.

        Args:
            chat_id: Chat session ID
            conversation_data: Dictionary containing query, content, and references
        """
        try:
            # Get history instance through history manager
            history = await self.history_manager.get_chat_history(chat_id)

            # Save conversation turn with data from successful processing
            history_saved = await self.history_manager.save_conversation_turn(
                message_id=message_id,
                trace_id=trace_id,
                history=history,
                user_query=query,
                ai_response=ai_response,
                files=files,
                tool_use_list=tool_use_list,
                tool_references=tool_references,
            )

            if not history_saved:
                logger.warning(f"Failed to save conversation history for chat: {chat_id}")

        except Exception as e:
            # Don't let history saving errors break the flow
            logger.error(f"Error saving conversation history for chat {chat_id}: {e}")
