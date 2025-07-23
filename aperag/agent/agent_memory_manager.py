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

    def extract_memory_from_llm(self, llm) -> SimpleMemory:
        """
        Extract updated memory from LLM after generation (pure function).

        Args:
            llm: LLM instance to extract from

        Returns:
            SimpleMemory: Updated conversation memory
        """
        logger.debug("Extracting memory from LLM")

        updated_memory = getattr(llm, "history", None)

        if updated_memory is None:
            logger.warning("LLM history is None, creating fresh memory")
            updated_memory = SimpleMemory()

        logger.debug("Successfully extracted updated memory")
        return updated_memory
