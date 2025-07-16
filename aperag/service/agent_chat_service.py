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
import traceback
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import WebSocket
from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.config import LoggerSettings, MCPServerSettings, MCPSettings, OpenAISettings, Settings
from mcp_agent.logging.transport import AsyncEventBus
from mcp_agent.workflows.llm.augmented_llm import RequestParams, SimpleMemory
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.agent import (
    UniversalEventListener,
    format_error,
    format_stream_content,
    format_stream_end,
    format_stream_start,
)
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.flow.runners.llm import add_ai_message, add_human_message

# Import MCP server for direct collection search access
from aperag.schema import view_models
from aperag.service.prompt_template_service import get_agent_system_prompt
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

    def _get_aperag_api_settings(self) -> Dict[str, str]:
        """Get ApeRAG API settings for MCP connection"""
        return {
            "aperag_api_key": os.getenv("APERAG_API_KEY", "sk-test"),
            "aperag_url": os.getenv("APERAG_URL", "http://localhost:8000/mcp/"),
        }

    def _get_openai_settings(self, model_name: Optional[str] = None) -> Dict[str, str]:
        """Get OpenAI settings for LLM calls"""
        return {
            "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            "openai_api_key": os.getenv("OPENAI_API_KEY", "sk-test"),
            "default_model": model_name or os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        }

    def _extract_tool_call_references(self, memory) -> List[Dict[str, Any]]:
        """
        Extract tool call results from MCP agent history and format as references.

        Args:
            memory: SimpleMemory instance containing agent history

        Returns:
            List of reference dictionaries in the format expected by llm.py
        """
        references = []

        try:
            # Get history from memory
            history_messages = memory.get() if hasattr(memory, "get") else []

            for message in history_messages:
                # Check if message has tool calls (message is a dict)
                if isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        # Debug: log the actual structure
                        logger.debug(f"Tool call structure: {tool_call}, type: {type(tool_call)}")

                        # Process tool call information
                        # Handle different tool call structures (dict vs object)
                        tool_name = "unknown_tool"
                        tool_args = "{}"
                        tool_call_id = ""

                        # Handle OpenAI ChatCompletionMessageToolCall objects
                        if hasattr(tool_call, "id"):
                            tool_call_id = tool_call.id
                            if hasattr(tool_call, "function"):
                                tool_name = (
                                    tool_call.function.name if hasattr(tool_call.function, "name") else "unknown_tool"
                                )
                                tool_args = (
                                    tool_call.function.arguments if hasattr(tool_call.function, "arguments") else "{}"
                                )
                        # Handle dictionary format
                        elif isinstance(tool_call, dict):
                            tool_call_id = tool_call.get("id", "")
                            if "function" in tool_call:
                                tool_name = tool_call["function"].get("name", "unknown_tool")
                                tool_args = tool_call["function"].get("arguments", "{}")
                            elif "name" in tool_call:
                                tool_name = tool_call.get("name", "unknown_tool")
                                tool_args = tool_call.get("arguments", "{}")
                            elif "type" in tool_call and tool_call["type"] == "function":
                                tool_name = tool_call.get("function", {}).get("name", "unknown_tool")
                                tool_args = tool_call.get("function", {}).get("arguments", "{}")

                        logger.debug(
                            f"Extracted tool_name: {tool_name}, tool_args: {tool_args}, tool_call_id: {tool_call_id}"
                        )

                        # Parse tool arguments
                        try:
                            args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        except json.JSONDecodeError:
                            args_dict = {"raw_args": tool_args}

                        # Find corresponding tool result message
                        tool_result = self._find_tool_result(history_messages, tool_call_id)

                        if tool_result:
                            # Format reference based on tool type
                            if tool_name == "aperag_search_collection":
                                ref = self._format_search_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            elif tool_name == "aperag_list_collections":
                                ref = self._format_list_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            elif tool_name == "aperag_web_search":
                                ref = self._format_web_search_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            elif tool_name == "aperag_web_read":
                                ref = self._format_web_read_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            else:
                                # Generic tool result reference
                                ref = self._format_generic_reference(tool_name, tool_result, args_dict)
                                if ref:
                                    references.append(ref)

        except Exception as e:
            logger.error(f"Error extracting tool call references: {e}")

        return references

    def _find_tool_result(self, messages, tool_call_id: str) -> Optional[str]:
        """Find the tool result message for a given tool call ID"""
        for message in messages:
            if (
                isinstance(message, dict)
                and message.get("role") == "tool"
                and message.get("tool_call_id") == tool_call_id
            ):
                content = message.get("content", "")
                logger.debug(f"Found tool result for {tool_call_id}: {type(content)} - {content}")

                # Handle both string and list content
                if isinstance(content, list):
                    return json.dumps(content)
                return content
        return None

    def _format_search_reference(self, tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format search_collection tool result as reference"""
        try:
            # Parse tool result - handle both string and already parsed data
            if isinstance(tool_result, str):
                try:
                    result_data = json.loads(tool_result)
                except json.JSONDecodeError:
                    result_data = {"raw_result": tool_result}
            else:
                result_data = tool_result

            logger.debug(f"Search reference result_data: {result_data}")

            # Handle array format where data is in first element's text field
            if isinstance(result_data, list) and len(result_data) > 0:
                first_item = result_data[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    try:
                        # Parse the text field as JSON
                        text_data = json.loads(first_item["text"])
                        result_data = text_data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                        return None

            # Extract search parameters
            collection_id = args.get("collection_id", "unknown")
            query = args.get("query", "")

            # Format search results
            if "items" in result_data:
                items = result_data["items"]
                if items:
                    # Combine all search results into a single reference
                    combined_text = ""
                    combined_metadata = {
                        "type": "search_collection",
                        "collection_id": collection_id,
                        "query": query,
                        "result_count": len(items),
                    }

                    for item in items:
                        content = item.get("content", "")
                        metadata = item.get("metadata", {})
                        combined_text += f"Document: {metadata.get('title', 'Untitled')}\n"
                        combined_text += f"Content: {content}\n\n"

                    return {
                        "text": combined_text.strip(),
                        "metadata": combined_metadata,
                        "score": 1.0,  # Default score for search results
                    }

            return None

        except Exception as e:
            logger.error(f"Error formatting search reference: {e}")
            return None

    def _format_list_reference(self, tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format list_collections tool result as reference"""
        try:
            # Parse tool result - handle both string and already parsed data
            if isinstance(tool_result, str):
                try:
                    result_data = json.loads(tool_result)
                except json.JSONDecodeError:
                    result_data = {"raw_result": tool_result}
            else:
                result_data = tool_result

            logger.debug(f"List reference result_data: {result_data}")

            # Handle array format where data is in first element's text field
            if isinstance(result_data, list) and len(result_data) > 0:
                first_item = result_data[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    try:
                        # Parse the text field as JSON
                        text_data = json.loads(first_item["text"])
                        result_data = text_data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                        return None

            # Look for items field (which contains collections)
            if "items" in result_data:
                collections = result_data["items"]
                text = "Available Collections:\n"
                for collection in collections:
                    title = collection.get("title", collection.get("name", "Unknown"))
                    description = collection.get("description", "No description")
                    collection_id = collection.get("id", "Unknown ID")
                    status = collection.get("status", "Unknown")

                    text += f"- {title} (ID: {collection_id})\n"
                    text += f"  Status: {status}\n"
                    if description:
                        text += f"  Description: {description}\n"
                    text += "\n"

                return {
                    "text": text.strip(),
                    "metadata": {"type": "list_collections", "collection_count": len(collections)},
                    "score": 1.0,
                }

            return None

        except Exception as e:
            logger.error(f"Error formatting list reference: {e}")
            return None

    def _format_web_search_reference(self, tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format web_search tool result as reference"""
        try:
            # Parse tool result - handle both string and already parsed data
            if isinstance(tool_result, str):
                try:
                    result_data = json.loads(tool_result)
                except json.JSONDecodeError:
                    result_data = {"raw_result": tool_result}
            else:
                result_data = tool_result

            logger.debug(f"Web search reference result_data: {result_data}")

            # Handle array format where data is in first element's text field
            if isinstance(result_data, list) and len(result_data) > 0:
                first_item = result_data[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    try:
                        # Parse the text field as JSON
                        text_data = json.loads(first_item["text"])
                        result_data = text_data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                        return None

            query = args.get("query", "")

            if "results" in result_data:
                results = result_data["results"]
                if results:
                    combined_text = f"Web Search Results for: {query}\n\n"

                    for result in results:
                        title = result.get("title", "No title")
                        url = result.get("url", "No URL")
                        snippet = result.get("snippet", "")

                        combined_text += f"Title: {title}\n"
                        combined_text += f"URL: {url}\n"
                        combined_text += f"Snippet: {snippet}\n\n"

                    return {
                        "text": combined_text.strip(),
                        "metadata": {"type": "web_search", "query": query, "result_count": len(results)},
                        "score": 1.0,
                    }

            return None

        except Exception as e:
            logger.error(f"Error formatting web search reference: {e}")
            return None

    def _format_web_read_reference(self, tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format web_read tool result as reference"""
        try:
            # Parse tool result - handle both string and already parsed data
            if isinstance(tool_result, str):
                try:
                    result_data = json.loads(tool_result)
                except json.JSONDecodeError:
                    result_data = {"raw_result": tool_result}
            else:
                result_data = tool_result

            logger.debug(f"Web read reference result_data: {result_data}")

            # Handle array format where data is in first element's text field
            if isinstance(result_data, list) and len(result_data) > 0:
                first_item = result_data[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    try:
                        # Parse the text field as JSON
                        text_data = json.loads(first_item["text"])
                        result_data = text_data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                        return None

            urls = args.get("url_list", [])

            if "results" in result_data:
                results = result_data["results"]
                if results:
                    combined_text = "Web Page Content:\n\n"

                    for result in results:
                        url = result.get("url", "No URL")
                        title = result.get("title", "No title")
                        content = result.get("content", "")

                        combined_text += f"URL: {url}\n"
                        combined_text += f"Title: {title}\n"
                        combined_text += f"Content: {content}\n\n"

                    return {
                        "text": combined_text.strip(),
                        "metadata": {"type": "web_read", "urls": urls, "result_count": len(results)},
                        "score": 1.0,
                    }

            return None

        except Exception as e:
            logger.error(f"Error formatting web read reference: {e}")
            return None

    def _format_generic_reference(
        self, tool_name: str, tool_result: str, args: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Format generic tool result as reference"""
        try:
            # Parse the tool result to handle array format
            parsed_result = tool_result
            if isinstance(tool_result, str):
                try:
                    parsed_result = json.loads(tool_result)
                except json.JSONDecodeError:
                    parsed_result = tool_result

            # Handle array format where data is in first element's text field
            if isinstance(parsed_result, list) and len(parsed_result) > 0:
                first_item = parsed_result[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    try:
                        # Parse the text field as JSON
                        text_data = json.loads(first_item["text"])
                        parsed_result = text_data
                    except json.JSONDecodeError:
                        # If parsing fails, use the original text
                        parsed_result = first_item["text"]

            # For generic tools, create a simple reference
            text = f"Tool: {tool_name}\n"
            if args:
                text += f"Arguments: {json.dumps(args, indent=2)}\n"

            # Handle both string and non-string results
            if isinstance(parsed_result, str):
                text += f"Result: {parsed_result}"
            else:
                text += f"Result: {json.dumps(parsed_result, indent=2)}"

            return {
                "text": text,
                "metadata": {"type": "tool_result", "tool_name": tool_name, "args": args},
                "score": 1.0,
            }

        except Exception as e:
            logger.error(f"Error formatting generic reference: {e}")
            return None

    def _build_llm_query_prompt(self, agent_message: view_models.AgentMessage, user: str) -> str:
        """
        Build a comprehensive prompt for LLM that includes context about user preferences,
        available collections, and web search status.
        """
        # Determine collection context
        if agent_message.collections:
            collection_context = ", ".join(
                [
                    " ".join(
                        [
                            f"collection_title={c.title}" if getattr(c, "title", None) else "",
                            f"collection_id={c.id}" if getattr(c, "id", None) else "",
                        ]
                    ).strip()
                    for c in agent_message.collections
                ]
            )
            collection_instruction = (
                "PRIORITY: Search these collections first, then decide if additional sources are needed"
            )
        else:
            collection_context = "None specified by user"
            collection_instruction = "discover and select relevant collections automatically"

        # Determine web search context
        web_status = "enabled" if agent_message.web_search_enabled else "disabled"
        if agent_message.web_search_enabled:
            web_instruction = "Use web search strategically for current information, verification, or gap-filling"
        else:
            web_instruction = "Rely entirely on knowledge collections; inform user if web search would be helpful"

        # Use template for cleaner formatting
        prompt_template = """**User Query**: {query}

**Session Context**:
- **User-Specified Collections**: {collection_context} ({collection_instruction})
- **Web Search**: {web_status} ({web_instruction})

**Research Instructions**:
1. **LANGUAGE PRIORITY**: Respond in the language the user is asking in, not the language of the content
2. If user specified collections (@mentions), search those first (REQUIRED)  
3. Use appropriate search keywords in multiple languages when beneficial
4. Assess result quality and decide if additional collections are needed
5. Use web search strategically if enabled and relevant
6. Provide comprehensive, well-structured response with clear source attribution
7. Distinguish between user-specified and additional sources in your response

Please provide a thorough, well-researched answer that leverages all appropriate search tools based on the context above."""

        return prompt_template.format(
            query=agent_message.query,
            collection_context=collection_context,
            collection_instruction=collection_instruction,
            web_status=web_status,
            web_instruction=web_instruction,
        )

    def _create_mcp_settings(
        self,
        model_name: Optional[str] = None,
    ) -> Optional[Settings]:
        """Create MCP settings dynamically based on agent message parameters"""
        if not MCPApp:
            logger.error("MCP components not available")
            return None

        aperag_settings = self._get_aperag_api_settings()
        openai_settings = self._get_openai_settings(model_name)

        try:
            return Settings(
                execution_engine="asyncio",
                logger=LoggerSettings(type="console", level="info"),
                mcp=MCPSettings(
                    servers={
                        "aperag": MCPServerSettings(
                            transport="streamable_http",
                            url=aperag_settings["aperag_url"],
                            headers={
                                "Authorization": f"Bearer {aperag_settings['aperag_api_key']}",
                                "Content-Type": "application/json",
                            },
                            http_timeout_seconds=30,
                            read_timeout_seconds=120,
                            description="ApeRAG knowledge base server",
                            env={"APERAG_API_KEY": aperag_settings["aperag_api_key"]},
                        )
                    }
                ),
                openai=OpenAISettings(
                    api_key=openai_settings["openai_api_key"],
                    base_url=openai_settings["openai_base_url"],
                    default_model=openai_settings["default_model"],
                    temperature=0.7,
                    max_tokens=2000,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to create MCP settings: {e}")
            return None

    def _create_mcp_app(
        self,
    ) -> Optional[MCPApp]:
        """Create MCPApp instance dynamically based on agent parameters"""
        settings = self._create_mcp_settings()
        if not settings:
            return None

        try:
            return MCPApp(name="aperag_agent", settings=settings)
        except Exception as e:
            logger.error(f"Failed to create MCPApp: {e}")
            return None

    async def handle_websocket_agent_chat(self, websocket: WebSocket, user: str, bot_id: str, chat_id: str):
        """Handle WebSocket connections for agent-type bot chats"""
        try:
            while True:
                # Receive message from WebSocket
                data = await websocket.receive_text()
                message_data = json.loads(data)

                # Generate message ID
                message_id = str(uuid.uuid4())
                query = message_data.get("query", "")
                if not query or not query.strip():
                    error_response = format_error("Invalid message format")
                    await websocket.send_text(json.dumps(error_response))
                    continue

                try:
                    # Create fresh SimpleMemory for each conversation to prevent tool call format conflicts
                    memory = SimpleMemory()

                    agent_message = view_models.AgentMessage(
                        query=query,
                        collections=message_data.get("collections"),
                        model_name=message_data.get("model_name"),
                        web_search_enabled=message_data.get("web_search_enabled", False),
                    )
                    # Process the agent message and stream responses
                    async for response_chunk in self.process_agent_message(
                        agent_message, user, chat_id, message_id, memory
                    ):
                        await websocket.send_text(json.dumps(response_chunk))

                except Exception as e:
                    logger.error(f"Error processing agent websocket message: {e}")
                    error_response = format_error(str(e))
                    await websocket.send_text(json.dumps(error_response))

        except Exception as e:
            logger.error(f"WebSocket error in agent chat: {e}")

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
        try:
            # Validate collections if specified
            if agent_message.collections:
                for collection in agent_message.collections:
                    collection_id = collection.id
                    if not collection_id:
                        yield self._format_error("Collection object missing 'id' field")
                        return
                    db_collection = await self.db_ops.query_collection(user, collection_id)
                    if not db_collection:
                        yield self._format_error(f"Collection {collection_id} not found")
                        return

            # Create dynamic agent app
            mcp_app = self._create_mcp_app()

            if not mcp_app:
                yield format_error("Failed to initialize agent")
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
                            comprehensive_prompt = self._build_llm_query_prompt(agent_message=agent_message, user=user)

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

                        finally:
                            # Clean up: remove the listener
                            event_bus.remove_listener("universal_event_monitor")

            except Exception as e:
                logger.error(f"Error in MCP agent execution: {e}")
                yield format_error(f"Error in agent execution: {str(e)}")
                return

            # Generate references - either from tool calls or direct search results
            # Extract tool call results from history and format as references
            tool_references = self._extract_tool_call_references(memory)

            # Store messages in history
            await add_human_message(history, agent_message.query, "")
            await add_ai_message(history, agent_message.query, "", full_content, tool_references, [])

            # Prepare references and URLs
            urls = []

            yield format_stream_end(msg_id, references=tool_references, urls=urls)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in agent message processing: {e}")
            yield format_error(f"Error processing agent message: {str(e)}")
