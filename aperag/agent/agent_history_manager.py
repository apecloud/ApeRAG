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
from typing import Dict, List

from aperag.utils.history import RedisChatMessageHistory, get_async_redis_client

from ..chat.history import create_assistant_message, create_user_message
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
        message_id: str,
        trace_id: str,
        history: RedisChatMessageHistory,
        user_query: str,
        ai_response: str,
        tool_use_list: List[Dict],
        tool_references: List,
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

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            logger.debug(f"Saving conversation turn for history session: {history.session_id}")

            user_message = create_user_message(
                content=user_query,
                chat_id=history.session_id,
                message_id=message_id,
                trace_id=trace_id,
            )
            # Save human message (plain text for agent conversations)
            await history.add_stored_message(user_message)

            # Save AI message (plain text for agent conversations)
            ai_message = create_assistant_message(
                content=ai_response,
                chat_id=history.session_id,
                message_id=message_id,
                trace_id=trace_id,
                tool_use_list=tool_use_list,
                references=tool_references,
                # urls=,
            )
            await history.add_stored_message(ai_message)

            logger.debug(f"Successfully saved conversation turn for session: {history.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save conversation turn for session {history.session_id}: {e}")
            return False
