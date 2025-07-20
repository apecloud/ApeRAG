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
from typing import Any, Dict

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
        self, 
        history: RedisChatMessageHistory, 
        context_limit: int = 4
    ) -> SimpleMemory:
        """
        Create LLM memory from chat history (pure function).
        
        This method:
        1. Retrieves recent messages from history
        2. Applies context window limit (default: 4 recent turns)
        3. Converts to SimpleMemory format
        4. Returns memory ready for LLM use
        
        Args:
            history: Chat history instance
            context_limit: Number of recent conversation turns to include
            
        Returns:
            SimpleMemory: Memory populated with recent conversation context
        """
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
            recent_messages = messages[-(context_limit * 2):] if len(messages) > context_limit * 2 else messages
            
            logger.debug(f"Retrieved {len(recent_messages)} recent messages from history")
            
            # Convert history messages to SimpleMemory format
            for msg in recent_messages:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    # Use the correct API for SimpleMemory - append method
                    memory.append(msg)
                        
            logger.debug(f"Successfully created memory with {len(recent_messages)} message(s)")
            return memory
            
        except Exception as e:
            logger.warning(f"Failed to load history: {e}, returning empty memory")
            return memory

    def prepare_memory_for_llm(self, memory: SimpleMemory) -> SimpleMemory:
        """
        Prepare memory for LLM use (pure function).
        
        This method can apply additional processing like:
        - Context summarization
        - Token limit enforcement  
        - Message filtering
        
        Args:
            memory: Input memory instance
            
        Returns:
            SimpleMemory: Processed memory ready for LLM
        """
        logger.debug("Preparing memory for LLM use")
        
        # For now, return as-is
        # Future: Add summarization, token counting, etc.
        
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
        
        updated_memory = getattr(llm, 'history', None)
        
        if updated_memory is None:
            logger.warning("LLM history is None, creating fresh memory")
            updated_memory = SimpleMemory()
            
        logger.debug("Successfully extracted updated memory")
        return updated_memory

    async def get_memory_stats(self, memory: SimpleMemory) -> Dict[str, Any]:
        """
        Get memory statistics (pure function).
        
        Args:
            memory: Memory instance to analyze
            
        Returns:
            Dict containing memory statistics
        """
        try:
            # SimpleMemory stores messages in .history attribute
            messages = getattr(memory, 'history', [])
            
            stats = {
                'total_messages': len(messages),
                'human_messages': len([m for m in messages if getattr(m, 'type', None) == 'human']),
                'ai_messages': len([m for m in messages if getattr(m, 'type', None) == 'ai']),
                'has_context': len(messages) > 0
            }
            
            logger.debug(f"Memory stats: {stats}")
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")
            return {
                'total_messages': 0,
                'human_messages': 0, 
                'ai_messages': 0,
                'has_context': False
            }
