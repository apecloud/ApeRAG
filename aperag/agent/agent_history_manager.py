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

"""Agent chat history management for persistent storage - pure functions for testability."""

import logging
from typing import Dict, List, Optional

from langchain.schema import AIMessage, HumanMessage

from aperag.utils.history import RedisChatMessageHistory, get_async_redis_client

from .exceptions import handle_agent_error

logger = logging.getLogger(__name__)


class AgentHistoryManager:
    """
    Manages chat history persistence and retrieval using pure functions.

    This class provides pure functions that accept external dependencies,
    making it highly testable and free from hidden dependencies.

    Most methods require external RedisChatMessageHistory instances to be passed in,
    eliminating internal state and hidden dependencies.
    """

    @handle_agent_error("history_creation", reraise=True)
    async def get_chat_history(self, chat_id: str) -> RedisChatMessageHistory:
        """
        Get chat history instance for a given chat ID.

        This method encapsulates the creation of RedisChatMessageHistory instances,
        providing a central point for history management configuration.

        Args:
            chat_id: Chat session identifier

        Returns:
            RedisChatMessageHistory: Configured history instance
        """
        logger.debug(f"Creating chat history instance for chat_id: {chat_id}")

        # Create history instance with Redis client
        history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())

        logger.debug(f"Successfully created chat history instance for chat_id: {chat_id}")
        return history

    @handle_agent_error("conversation_save", reraise=False)
    async def save_conversation_turn(
        self,
        history: RedisChatMessageHistory,
        user_query: str,
        ai_response: str,
        tool_references: List,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Save a complete conversation turn to persistent storage.

        This is a pure function that accepts external history instance.
        Uses agent-specific saving format (plain text) instead of flow-based Message JSON.

        Args:
            history: External RedisChatMessageHistory instance
            user_query: User's query message
            ai_response: AI's response message
            tool_references: Tool call references from the conversation
            metadata: Optional metadata for the conversation turn (unused for now)

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            logger.debug(f"Saving conversation turn for history session: {history.session_id}")

            # Save human message (plain text for agent conversations)
            await history.add_message(HumanMessage(content=user_query))

            # Save AI message (plain text for agent conversations)
            await history.add_message(AIMessage(content=ai_response))

            logger.debug(f"Successfully saved conversation turn for session: {history.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save conversation turn for session {history.session_id}: {e}")
            return False

    @handle_agent_error("context_retrieval", reraise=False)
    async def get_recent_context(self, history: RedisChatMessageHistory, limit: int = 10) -> List[Dict]:
        """
        Get recent conversation context for prompt building.

        This is a pure function that accepts external history instance.

        Args:
            history: External RedisChatMessageHistory instance
            limit: Maximum number of recent messages to retrieve

        Returns:
            List[Dict]: Recent conversation context
        """
        try:
            logger.debug(f"Getting recent context for session: {history.session_id}, limit: {limit}")

            # Get recent messages
            messages = await history.messages

            # Convert to context format (limit to recent messages)
            recent_messages = messages[-limit:] if len(messages) > limit else messages

            context = []
            for message in recent_messages:
                context.append(
                    {"type": message.type, "content": message.content, "timestamp": getattr(message, "timestamp", None)}
                )

            logger.debug(f"Retrieved {len(context)} recent context items for session: {history.session_id}")
            return context

        except Exception as e:
            logger.warning(f"Failed to retrieve recent context for session {history.session_id}: {e}")
            return []

    @handle_agent_error("context_string_build", reraise=False)
    async def build_context_string(self, history: RedisChatMessageHistory, limit: int = 5) -> str:
        """
        Build a context string from recent conversation history.

        This is useful for including recent context in prompts.
        Pure function that accepts external history instance.

        Args:
            history: External RedisChatMessageHistory instance
            limit: Maximum number of recent messages to include

        Returns:
            str: Formatted context string
        """
        try:
            context_items = await self.get_recent_context(history, limit)

            if not context_items:
                return ""

            context_lines = []
            for item in context_items:
                role = "User" if item["type"] == "human" else "Assistant"
                content = item["content"][:200] + "..." if len(item["content"]) > 200 else item["content"]
                context_lines.append(f"{role}: {content}")

            context_string = "\n".join(context_lines)
            logger.debug(f"Built context string for session {history.session_id}: {len(context_string)} characters")

            return context_string

        except Exception as e:
            logger.warning(f"Failed to build context string for session {history.session_id}: {e}")
            return ""

    @handle_agent_error("history_cleanup", reraise=False)
    async def clear_chat_history(self, history: RedisChatMessageHistory) -> bool:
        """
        Clear all history for a specific chat session.

        Pure function that accepts external history instance.

        Args:
            history: External RedisChatMessageHistory instance

        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            logger.debug(f"Clearing chat history for session: {history.session_id}")

            await history.clear()

            logger.debug(f"Successfully cleared chat history for session: {history.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear chat history for session {history.session_id}: {e}")
            return False

    async def get_history_statistics(self, history: RedisChatMessageHistory) -> Dict:
        """
        Get statistics about chat history.

        Pure function that accepts external history instance.

        Args:
            history: External RedisChatMessageHistory instance

        Returns:
            Dict: History statistics
        """
        try:
            messages = await history.messages

            stats = {
                "total_messages": len(messages),
                "human_messages": len([m for m in messages if m.type == "human"]),
                "ai_messages": len([m for m in messages if m.type == "ai"]),
                "session_id": history.session_id,
            }

            logger.debug(f"History statistics for session {history.session_id}: {stats}")
            return stats

        except Exception as e:
            logger.warning(f"Failed to get history statistics for session {history.session_id}: {e}")
            return {"error": str(e)}
