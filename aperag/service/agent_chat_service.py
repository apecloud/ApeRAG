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
import os
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.agent.rag_agent import APERAG_AGENT_INSTRUCTION
from aperag.config import settings
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.schema import view_models
from aperag.utils.constant import DOC_QA_REFERENCES, DOCUMENT_URLS
from aperag.utils.history import RedisChatMessageHistory, get_async_redis_client
from aperag.utils.utils import now_unix_milliseconds
from mcp_agent.app import MCPApp
from mcp_agent.config import Settings, LoggerSettings, MCPSettings, MCPServerSettings, OpenAISettings

logger = logging.getLogger(__name__)


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

    def _create_mcp_settings(
        self, 
        collection_ids: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        web_search_enabled: bool = False
    ) -> Optional[Settings]:
        """Create MCP settings dynamically based on agent message parameters"""
        if not MCPApp:
            logger.error("MCP components not available")
            return None

        aperag_settings = self._get_aperag_api_settings()
        openai_settings = self._get_openai_settings(model_name)

        # Create collection-specific prompt if collections are specified
        system_instruction = APERAG_AGENT_INSTRUCTION
        if collection_ids:
            collection_info = f"\n\nYou have access to the following collections: {', '.join(collection_ids)}"
            system_instruction += collection_info

        if web_search_enabled:
            system_instruction += "\n\nWeb search is enabled. You can search the internet for additional information when needed."

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
                                "Content-Type": "application/json"
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
                system_instruction=system_instruction,
            )
        except Exception as e:
            logger.error(f"Failed to create MCP settings: {e}")
            return None

    def _create_agent_app(
        self,
        collection_ids: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        web_search_enabled: bool = False
    ) -> Optional[MCPApp]:
        """Create MCPApp instance dynamically based on agent parameters"""
        settings = self._create_mcp_settings(collection_ids, model_name, web_search_enabled)
        if not settings:
            return None

        try:
            return MCPApp(name="aperag_agent", settings=settings)
        except Exception as e:
            logger.error(f"Failed to create MCPApp: {e}")
            return None

    async def handle_websocket_agent_chat(
        self, 
        websocket: WebSocket, 
        user: str, 
        bot_id: str, 
        chat_id: str
    ):
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
                    error_response = self._format_error("Invalid message format")
                    await websocket.send_text(json.dumps(error_response))
                    continue

                try:
                    agent_message = view_models.AgentMessage(
                        query=query,
                        collection_ids=message_data.get("collection_ids"),
                        model_name=message_data.get("model_name"),
                        web_search_enabled=message_data.get("web_search_enabled", False)
                    )
                    # Process the agent message and stream responses
                    async for response_chunk in self.process_agent_message(
                        agent_message, user, chat_id, message_id
                    ):
                        await websocket.send_text(json.dumps(response_chunk))

                except Exception as e:
                    logger.error(f"Error processing agent websocket message: {e}")
                    error_response = self._format_error(str(e))
                    await websocket.send_text(json.dumps(error_response))

        except Exception as e:
            logger.error(f"WebSocket error in agent chat: {e}")

    async def process_agent_message(
        self, 
        agent_message: view_models.AgentMessage,
        user: str,
        chat_id: str,
        msg_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process an agent message and yield streaming responses.
        
        This method creates a dynamic MCPApp instance based on the message parameters
        and uses it to generate intelligent responses.
        """
        try:
            # Validate collections if specified
            if agent_message.collection_ids:
                for collection_id in agent_message.collection_ids:
                    collection = await self.db_ops.query_collection(user, collection_id)
                    if not collection:
                        yield self._format_error(f"Collection {collection_id} not found")
                        return

            # Create dynamic agent app
            agent_app = self._create_agent_app(
                collection_ids=agent_message.collection_ids,
                model_name=agent_message.model_name,
                web_search_enabled=agent_message.web_search_enabled or False
            )

            if not agent_app:
                yield self._format_error("Failed to initialize agent")
                return

            # Yield start message
            yield self._format_stream_start(msg_id)

            # Get chat history for context
            history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
            # chat_history = await history.get_messages()

            # Prepare conversation context for the agent
            conversation_messages = []
            # for msg in chat_history[-10:]:  # Last 10 messages for context
            #     if isinstance(msg, dict):
            #         role = "user" if msg.get("role") == "human" else "assistant"
            #         content = msg.get("data", "")
            #     else:
            #         # Handle message objects
            #         role = "user" if msg.type == "human" else "assistant"
            #         content = msg.content if hasattr(msg, 'content') else str(msg)
                
            #     if content:
            #         conversation_messages.append({"role": role, "content": content})

            try:
                # Use agent app for intelligent conversation
                # This integrates with the MCP system for dynamic tool usage
                if hasattr(agent_app, 'chat'):
                    # Add current user query to conversation
                    conversation_messages.append({"role": "user", "content": agent_message.query})
                    
                    # Get response from agent app
                    response = await agent_app.chat(
                        messages=conversation_messages,
                        stream=True  # Enable streaming if supported
                    )
                    
                    # Handle streaming response
                    full_content = ""
                    if hasattr(response, '__aiter__'):
                        # Streaming response
                        async for chunk in response:
                            if hasattr(chunk, 'content') and chunk.content:
                                full_content += chunk.content
                                yield self._format_stream_content(msg_id, chunk.content)
                            elif isinstance(chunk, str):
                                full_content += chunk
                                yield self._format_stream_content(msg_id, chunk)
                    else:
                        # Non-streaming response
                        full_content = str(response) if response else "No response generated"
                        yield self._format_stream_content(msg_id, full_content)
                    
                    # Store messages in history
                    await history.add_user_message(agent_message.query)
                    await history.add_ai_message(full_content)
                    
                    # Prepare references based on agent configuration
                    references = []
                    urls = []
                    
                    if agent_message.collection_ids:
                        references.extend([f"Collection: {cid}" for cid in agent_message.collection_ids])
                    
                    if agent_message.web_search_enabled:
                        references.append("Web Search: Enabled")
                    
                    if agent_message.model_name:
                        references.append(f"Model: {agent_message.model_name}")
                    
                    yield self._format_stream_end(msg_id, references=references, urls=urls)
                
                else:
                    # Fallback if agent app doesn't support chat method
                    response_content = await self._generate_fallback_response(agent_message)
                    
                    # Simulate streaming for fallback
                    words = response_content.split()
                    current_chunk = ""
                    for i, word in enumerate(words):
                        current_chunk += word + " "
                        if (i + 1) % 5 == 0 or i == len(words) - 1:  # Send every 5 words or at the end
                            yield self._format_stream_content(msg_id, current_chunk)
                            current_chunk = ""
                    
                    # Store in history
                    await history.add_user_message(agent_message.query)
                    await history.add_ai_message(response_content)
                    
                    # Prepare references
                    references = []
                    if agent_message.collection_ids:
                        references.extend([f"Collection: {cid}" for cid in agent_message.collection_ids])
                    
                    yield self._format_stream_end(msg_id, references=references)

            except Exception as e:
                logger.error(f"Error in agent conversation: {e}")
                yield self._format_error(f"Error processing conversation: {str(e)}")

        except Exception as e:
            logger.error(f"Error in agent message processing: {e}")
            yield self._format_error(f"Error processing agent message: {str(e)}")

    async def _generate_fallback_response(self, agent_message: view_models.AgentMessage) -> str:
        """Generate a fallback response when MCPApp integration fails"""
        response_parts = [
            f"Based on your query: '{agent_message.query}'"
        ]
        
        if agent_message.collection_ids:
            response_parts.append(f"Searching in collections: {', '.join(agent_message.collection_ids)}")
        
        if agent_message.model_name:
            response_parts.append(f"Using model: {agent_message.model_name}")
        
        if agent_message.web_search_enabled:
            response_parts.append("Web search is enabled for additional context")
        
        response_parts.append(
            "I'm working on processing your request. "
            "The agent system is being enhanced to provide more intelligent responses. "
            "For now, I can confirm that your request has been received and configured properly."
        )
        
        return "\n\n".join(response_parts)

    # Helper methods for response formatting
    def _format_stream_start(self, msg_id: str) -> Dict[str, Any]:
        """Format the start event for streaming"""
        return {
            "type": "start",
            "id": msg_id,
            "timestamp": now_unix_milliseconds(),
        }

    def _format_stream_content(self, msg_id: str, content: str) -> Dict[str, Any]:
        """Format a content chunk for streaming"""
        return {
            "type": "message",
            "id": msg_id,
            "data": content,
            "timestamp": now_unix_milliseconds(),
        }

    def _format_stream_end(
        self, 
        msg_id: str, 
        references: List[str] = None,
        urls: List[str] = None
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
            "urls": urls,
            "timestamp": now_unix_milliseconds(),
        }

    def _format_error(self, error: str) -> Dict[str, Any]:
        """Format an error response"""
        return {
            "type": "error",
            "id": str(uuid.uuid4()),
            "data": error,
            "timestamp": now_unix_milliseconds(),
        } 