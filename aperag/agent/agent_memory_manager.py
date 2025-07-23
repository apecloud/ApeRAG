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

"""Agent memory management for conversation sessions - Pure function implementation."""

import logging

from mcp_agent.workflows.llm.augmented_llm import SimpleMemory

from aperag.utils.history import RedisChatMessageHistory

from .exceptions import handle_agent_error

logger = logging.getLogger(__name__)


class AgentMemoryManager:
    """
    Pure function-based memory manager for LLM conversations.

    Responsibilities:
    - Create memory from chat history (pure function)
    - Apply context window limitations and summarization
    - Return memory objects ready for LLM use
    - Build context summaries for special scenarios
    - No direct LLM object manipulation
    """

    @handle_agent_error("memory_creation_from_history", reraise=True)
    async def create_memory_from_history(
        self, history: RedisChatMessageHistory, context_limit: int = 4
    ) -> SimpleMemory:
        """
        Create LLM memory from chat history (pure function).

        This method:
        1. Retrieves recent messages from history
        2. Applies context window limit (default: 4 recent turns)
        3. Converts to SimpleMemory format with proper message types
        4. Returns memory ready for LLM use

        Args:
            history: Chat history instance
            context_limit: Number of recent conversation turns to include

        Returns:
            SimpleMemory: Memory populated with recent conversation context
        """
        from langchain_core.messages.utils import convert_to_openai_messages

        logger.debug(f"Creating memory from history with context_limit: {context_limit}")

        # Create fresh memory instance
        memory = SimpleMemory()

        try:
            # Get recent messages from history
            messages = await history.messages

            if not messages:
                logger.debug("No history found, returning empty memory")
                return memory

            # Apply context limit - take the most recent conversation turns
            # Each turn = user message + AI response, so we take last (context_limit * 2) messages
            recent_messages = messages[-(context_limit * 2) :] if len(messages) > context_limit * 2 else messages

            logger.debug(f"Retrieved {len(recent_messages)} recent messages from history")

            # Use LangChain's official utility to convert messages to OpenAI format
            openai_messages = convert_to_openai_messages(recent_messages)

            # Add converted messages to memory
            for openai_msg in openai_messages:
                memory.append(openai_msg)

            logger.debug(f"Successfully created memory with {len(memory.history)} message(s)")

            # Debug log to verify message formats
            for i, msg in enumerate(memory.history):
                msg_type = type(msg).__name__
                role = msg.get("role", "unknown") if isinstance(msg, dict) else "not_dict"
                logger.debug(f"Memory message [{i}]: {msg_type}, role: {role}")

            return memory

        except Exception as e:
            logger.warning(f"Failed to load history: {e}, returning empty memory")
            return memory

    @handle_agent_error("context_summary_build", reraise=False)
    async def build_context_summary(self, history: RedisChatMessageHistory, limit: int = 5) -> str:
        """
        Build a context summary string from recent conversation history.

        This is useful for including recent context in prompts for special scenarios.
        Pure function that accepts external history instance.

        Args:
            history: External RedisChatMessageHistory instance
            limit: Maximum number of recent messages to include

        Returns:
            str: Formatted context summary string
        """
        try:
            # Get recent messages from history
            messages = await history.messages

            if not messages:
                return ""

            # Convert to context format (limit to recent messages)
            recent_messages = messages[-limit:] if len(messages) > limit else messages

            context_lines = []
            for message in recent_messages:
                role = "User" if message.type == "human" else "Assistant"
                content = message.content[:200] + "..." if len(message.content) > 200 else message.content
                context_lines.append(f"{role}: {content}")

            context_summary = "\n".join(context_lines)
            logger.debug(f"Built context summary for session {history.session_id}: {len(context_summary)} characters")

            return context_summary

        except Exception as e:
            logger.warning(f"Failed to build context summary for session {history.session_id}: {e}")
            return ""
