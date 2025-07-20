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
from typing import Any, AsyncGenerator, Dict

from fastapi import WebSocket
from mcp_agent.logging.transport import AsyncEventBus
from mcp_agent.workflows.llm.augmented_llm import RequestParams, SimpleMemory
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.agent import (
    UniversalEventListener,
    agent_session_manager,
    extract_tool_call_references,
    format_agent_execution_error,
    format_agent_setup_error,
    format_invalid_json_error,
    format_invalid_model_spec_error,
    format_llm_generation_error,
    format_model_spec_required_error,
    format_processing_error,
    format_query_required_error,
    format_stream_content,
    format_stream_end,
    format_stream_start,
)
from aperag.agent.agent_config import AgentConfig
from aperag.agent.exceptions import (
    AgentConfigurationError,
    MCPAppInitializationError,
    MCPConnectionError,
    handle_agent_error,
    safe_json_parse,
)
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.flow.runners.llm import add_ai_message, add_human_message

# Import MCP server for direct collection search access
from aperag.schema import view_models
from aperag.service.prompt_template_service import build_agent_query_prompt, get_agent_system_prompt
from aperag.utils.history import RedisChatMessageHistory, get_async_redis_client

logger = logging.getLogger(__name__)


class AgentChatService:
    """
    Chat service specifically for agent-type bots that uses MCPApp for intelligent conversation.

    This service uses AgentSessionManager for efficient session lifecycle management,
    including collection selection, model choice, and web search capabilities.
    """

    def __init__(self, session: AsyncSession = None):
        if session is None:
            self.db_ops = async_db_ops
        else:
            self.db_ops = AsyncDatabaseOps(session)

    @handle_agent_error("websocket_agent_chat", reraise=False)
    async def handle_websocket_agent_chat(self, websocket: WebSocket, user: str, bot_id: str, chat_id: str):
        """Handle WebSocket connections for agent-type bot chats"""
        try:
            while True:
                # Receive message from WebSocket
                data = await websocket.receive_text()

                # Safe JSON parsing
                try:
                    message_data = safe_json_parse(data, "websocket_message")
                except Exception as e:
                    # Default to en-US for parsing errors since we don't have language info yet
                    error_response = format_invalid_json_error(str(e), "en-US")
                    await websocket.send_text(json.dumps(error_response))
                    continue

                # Generate message ID
                message_id = str(uuid.uuid4())
                query = message_data.get("query", "")
                language = message_data.get("language", "en-US")  # Get language preference

                if not query or not query.strip():
                    error_response = format_query_required_error(language)
                    await websocket.send_text(json.dumps(error_response))
                    continue

                try:
                    # Create fresh SimpleMemory for each conversation to prevent tool call format conflicts
                    memory = SimpleMemory()

                    # Create ModelSpec from completion data
                    completion_spec = None
                    if message_data.get("completion"):
                        try:
                            completion_spec = view_models.ModelSpec(**message_data["completion"])
                        except Exception as e:
                            error_response = format_invalid_model_spec_error(str(e), language)
                            await websocket.send_text(json.dumps(error_response))
                            continue

                    agent_message = view_models.AgentMessage(
                        query=query,
                        collections=message_data.get("collections"),
                        completion=completion_spec,
                        web_search_enabled=message_data.get("web_search_enabled", False),
                        language=language,  # Include language in agent message
                    )

                    # Process the agent message and stream responses
                    async for response_chunk in self.process_agent_message(
                        agent_message, user, chat_id, message_id, memory
                    ):
                        await websocket.send_text(json.dumps(response_chunk))

                except (AgentConfigurationError, MCPAppInitializationError, MCPConnectionError) as e:
                    logger.error(f"Agent configuration error in websocket: {e}")
                    error_response = format_agent_setup_error(str(e), language)
                    await websocket.send_text(json.dumps(error_response))
                except Exception as e:
                    logger.error(f"Unexpected error processing agent websocket message: {e}")
                    error_response = format_processing_error(str(e), language)
                    await websocket.send_text(json.dumps(error_response))

        except Exception as e:
            logger.error(f"WebSocket connection error in agent chat: {e}")

    async def _get_provider_session(self, agent_message: view_models.AgentMessage, user: str, chat_id: str):
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
            # aperag_api_key and aperag_url will be set by __post_init__
            # server_names will default to ["aperag"]
        )

        # Get or create chat session using config
        session = await agent_session_manager.get_or_create_session(config)

        return session

    @handle_agent_error("agent_message_processing", reraise=False)
    async def process_agent_message(
        self,
        agent_message: view_models.AgentMessage,
        user: str,
        chat_id: str,
        msg_id: str,
        memory,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process an agent message using session management for optimized resource reuse.

        This method uses AgentSessionManager to reuse existing MCPApp instances
        and provides efficient session lifecycle management.
        """
        # Get language preference from agent message
        language = agent_message.language if agent_message.language else "en-US"

        # Validate ModelSpec
        if not agent_message.completion or not agent_message.completion.model:
            yield format_model_spec_required_error(language)
            return

        # Yield start message
        yield format_stream_start(msg_id)

        # Get chat history for context
        history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())

        try:
            # Get chat session - super simple!
            session = await self._get_provider_session(agent_message, user, chat_id)

            # Get fresh LLM instance for this specific model and conversation
            llm = await session.get_llm(agent_message.completion.model)

            # Process message with session
            full_content = ""

            # Create universal event listener with msg_id
            event_listener = UniversalEventListener(msg_id)

            # Register the listener with AsyncEventBus
            event_bus = AsyncEventBus.get()
            event_bus.add_listener("universal_event_monitor", event_listener)

            try:
                request_params = RequestParams(
                    max_iterations=10,
                    parallel_tool_calls=True,
                    model=agent_message.completion.model,  # Use the specific model
                )

                # Set memory for this conversation (clean state)
                llm.history = memory

                # Build comprehensive prompt with context and pre-search results
                comprehensive_prompt = build_agent_query_prompt(agent_message=agent_message, user=user)

                # Start generate_str in background and monitor events
                generate_task = asyncio.create_task(llm.generate_str(comprehensive_prompt, request_params))

                # Monitor events while generate_str is running
                sent_message_count = 0
                while not generate_task.done():
                    # Check for new formatted messages from event listener
                    current_message_count = event_listener.get_message_count()

                    if current_message_count > sent_message_count:
                        # Get and yield new messages directly
                        new_messages = event_listener.get_new_messages(sent_message_count)
                        for message in new_messages:
                            yield message
                        sent_message_count = current_message_count

                    # Small delay to avoid busy waiting
                    await asyncio.sleep(0.05)

                # Get the final response
                response = await generate_task
                full_content = response if response else "No response generated"

                # Send any remaining messages that might have been missed
                final_message_count = event_listener.get_message_count()
                if final_message_count > sent_message_count:
                    remaining_messages = event_listener.get_new_messages(sent_message_count)
                    for message in remaining_messages:
                        yield message

                # Stream the response content using utils function
                yield format_stream_content(msg_id, full_content)
                memory = llm.history

            except Exception as e:
                logger.error(f"Error in LLM generation: {e}")
                yield format_llm_generation_error(str(e), language)
                return
            finally:
                # Clean up: remove the listener
                try:
                    event_bus.remove_listener("universal_event_monitor")
                except Exception as e:
                    logger.warning(f"Failed to remove event listener: {e}")

        except AgentConfigurationError as e:
            logger.error(f"Agent configuration error: {e}")
            yield format_agent_setup_error(str(e), language)
            return
        except Exception as e:
            logger.error(f"Error in agent session processing: {e}")
            yield format_agent_execution_error(str(e), language)
            return

        # Generate references - either from tool calls or direct search results
        # Extract tool call results from history and format as references
        tool_references = extract_tool_call_references(memory)

        # Store messages in history
        try:
            await add_human_message(history, agent_message.query, "")
            await add_ai_message(history, agent_message.query, "", full_content, tool_references, [])
        except Exception as e:
            logger.warning(f"Failed to store chat history: {e}")

        # Prepare references and URLs
        urls = []

        yield format_stream_end(msg_id, references=tool_references, urls=urls)
