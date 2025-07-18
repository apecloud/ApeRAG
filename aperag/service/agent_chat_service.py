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
import uuid
from typing import Any, AsyncGenerator, Dict

from fastapi import WebSocket
from mcp_agent.agents.agent import Agent
from mcp_agent.logging.transport import AsyncEventBus
from mcp_agent.workflows.llm.augmented_llm import RequestParams, SimpleMemory
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.agent import (
    MCPAppFactory,
    UniversalEventListener,
    extract_tool_call_references,
    format_error,
    format_stream_content,
    format_stream_end,
    format_stream_start,
)
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

# Only set default values if environment variables are not already set
if not os.getenv("APERAG_API_KEY"):
    os.environ["APERAG_API_KEY"] = "sk-test"
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-test"
if not os.getenv("APERAG_URL"):
    os.environ["APERAG_URL"] = "http://localhost:8000/mcp/"
if not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
if not os.getenv("DEFAULT_MODEL"):
    os.environ["DEFAULT_MODEL"] = "gpt-4o-mini"


class AgentChatService:
    """
    Chat service specifically for agent-type bots that uses MCPApp for intelligent conversation.

    This service dynamically constructs MCPApp instances based on agent message parameters,
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
                    error_response = format_error(f"Invalid JSON format: {str(e)}")
                    await websocket.send_text(json.dumps(error_response))
                    continue

                # Generate message ID
                message_id = str(uuid.uuid4())
                query = message_data.get("query", "")
                if not query or not query.strip():
                    error_response = format_error("Query is required and cannot be empty")
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
                            error_response = format_error(f"Invalid ModelSpec format: {str(e)}")
                            await websocket.send_text(json.dumps(error_response))
                            continue

                    agent_message = view_models.AgentMessage(
                        query=query,
                        collections=message_data.get("collections"),
                        completion=completion_spec,
                        web_search_enabled=message_data.get("web_search_enabled", False),
                    )

                    # Process the agent message and stream responses
                    async for response_chunk in self.process_agent_message(
                        agent_message, user, chat_id, message_id, memory
                    ):
                        await websocket.send_text(json.dumps(response_chunk))

                except (AgentConfigurationError, MCPAppInitializationError, MCPConnectionError) as e:
                    logger.error(f"Agent configuration error in websocket: {e}")
                    error_response = format_error(f"Agent setup failed: {str(e)}")
                    await websocket.send_text(json.dumps(error_response))
                except Exception as e:
                    logger.error(f"Unexpected error processing agent websocket message: {e}")
                    error_response = format_error(f"Processing error: {str(e)}")
                    await websocket.send_text(json.dumps(error_response))

        except Exception as e:
            logger.error(f"WebSocket connection error in agent chat: {e}")

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
        Process an agent message and yield streaming responses.

        This method creates a dynamic MCPApp instance based on the message parameters
        and uses it to generate intelligent responses.
        """
        # Validate collections if specified
        if agent_message.collections:
            for collection in agent_message.collections:
                collection_id = collection.id
                if not collection_id:
                    yield format_error("Collection object missing 'id' field")
                    return
                try:
                    db_collection = await self.db_ops.query_collection(user, collection_id)
                    if not db_collection:
                        yield format_error(f"Collection {collection_id} not found")
                        return
                except Exception as e:
                    yield format_error(f"Failed to validate collection {collection_id}: {str(e)}")
                    return

        # Create dynamic agent app from ModelSpec
        if not agent_message.completion:
            yield format_error("ModelSpec is required in completion field")
            return

        try:
            mcp_app = await MCPAppFactory.create_mcp_app_from_model_spec(
                model_spec=agent_message.completion, user_id=user
            )
        except (AgentConfigurationError, MCPAppInitializationError, MCPConnectionError) as e:
            logger.error(f"Failed to create MCP app: {e}")
            yield format_error(f"Agent setup failed: {str(e)}")
            return
        except Exception as e:
            logger.error(f"Unexpected error creating MCP app: {e}")
            yield format_error(f"Failed to initialize agent: {str(e)}")
            return

        # Yield start message
        yield format_stream_start(msg_id)

        # Get chat history for context
        history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())

        # Use agent app for intelligent conversation
        # This integrates with the MCP system for dynamic tool usage
        full_content = ""

        try:
            async with mcp_app.run() as running_app:
                # Create agent with instruction and server names
                agent = Agent(
                    name="aperag_assistant",
                    instruction=get_agent_system_prompt(),
                    server_names=["aperag"],
                )

                # Verify server connection
                if "aperag" not in running_app.server_registry.registry:
                    yield format_error("ApeRAG MCP Server connection failed")
                    return

                async with agent:
                    # Create universal event listener with msg_id
                    event_listener = UniversalEventListener(msg_id)

                    # Register the listener with AsyncEventBus
                    event_bus = AsyncEventBus.get()
                    event_bus.add_listener("universal_event_monitor", event_listener)

                    try:
                        # Attach LLM to agent
                        llm = await agent.attach_llm(OpenAIAugmentedLLM)

                        request_params = RequestParams(
                            max_iterations=10,
                            parallel_tool_calls=True,
                            model="google/gemini-2.5-flash",
                        )

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
                        yield format_error(f"Error in LLM generation: {str(e)}")
                        return
                    finally:
                        # Clean up: remove the listener
                        try:
                            event_bus.remove_listener("universal_event_monitor")
                        except Exception as e:
                            logger.warning(f"Failed to remove event listener: {e}")

        except Exception as e:
            logger.error(f"Error in MCP agent execution: {e}")
            yield format_error(f"Error in agent execution: {str(e)}")
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
