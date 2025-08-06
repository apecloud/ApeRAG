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

"""Simple agent session management - optimized for ease of maintenance and minimal bugs."""

import asyncio
import logging
import time
from typing import Dict, Optional

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from aperag.agent.agent_config import AgentConfig
from aperag.agent.exceptions import AgentConfigurationError
from aperag.agent.mcp_app_factory import MCPAppFactory

logger = logging.getLogger(__name__)


class ChatSession:
    """
    Chat session per user+chat+provider combination.

    Key insight: Each chat session maintains its own MCPApp, Agent, and LLM instances
    to preserve conversation state and memory. Same provider can serve multiple models,
    but each chat has its own isolated session.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.last_used = time.time()

        # MCP resources - created once per chat session
        self.mcp_app = None
        self.mcp_app_context_manager = None
        self.mcp_running_app = None
        self.agent = None
        self.llm = None  # Cache LLM instance for this chat

        # Simple state flags
        self._ready = False

    async def initialize(self):
        """Initialize with provider settings from config."""
        if self._ready:
            return

        try:
            logger.info(f"Initializing provider session {self.config.get_session_key()}")

            # Create MCP app for this provider using config
            self.mcp_app = MCPAppFactory.create_mcp_app_from_config(self.config)

            # Start MCP app
            self.mcp_app_context_manager = self.mcp_app.run()
            self.mcp_running_app = await self.mcp_app_context_manager.__aenter__()

            self.mcp_running_app.context.session_id = self.config.chat_id

            # Create reusable agent for this chat session
            self.agent = Agent(
                name=f"aperag_agent_{self.config.user_id}_{self.config.chat_id}_{self.config.provider_name}",
                instruction=self.config.instruction,
                server_names=self.config.server_names,
            )

            await self.agent.__aenter__()

            # Create and cache LLM instance for this chat session
            self.llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
            from mcp_agent.logging.logger import get_logger

            self.llm.logger = get_logger(self.llm.name, session_id=self.llm.context.session_id)
            self._ready = True

            logger.info(f"Chat session {self.config.get_session_key()} ready")

        except Exception as e:
            logger.error(f"Failed to initialize session {self.config.get_session_key()}: {e}")
            await self._cleanup()
            raise AgentConfigurationError(f"Session init failed: {e}")

    async def get_llm(self, model: str) -> OpenAIAugmentedLLM:
        """Get cached LLM instance for this chat session."""
        if not self._ready:
            raise AgentConfigurationError("Session not ready")

        # Return the cached LLM instance
        # This preserves conversation state and memory for the chat session
        return self.llm

    def touch(self):
        """Update last used time."""
        self.last_used = time.time()

    def is_expired(self, timeout: int = 1800) -> bool:  # 30 min default
        """Check if session expired."""
        return time.time() - self.last_used > timeout

    async def _cleanup(self):
        """Clean up all resources."""
        logger.info(f"Cleaning up chat session {self.config.get_session_key()}")

        # LLM cleanup is handled by agent cleanup
        self.llm = None

        if self.agent:
            try:
                await self.agent.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Agent cleanup error: {e}")
            self.agent = None

        if self.mcp_app_context_manager:
            try:
                await self.mcp_app_context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"MCP app cleanup error: {e}")
            self.mcp_running_app = None

        self.mcp_app = None
        self._ready = False


# Simple global state - no complex singleton patterns
_chat_sessions: Dict[str, ChatSession] = {}
_cleanup_task: Optional[asyncio.Task] = None


def generate_session_key(user_id: str, chat_id: str, provider_name: str) -> str:
    """Generate session key based on user, chat, and provider."""
    return f"{user_id}:{chat_id}:{provider_name}"


async def get_or_create_session(config: AgentConfig) -> ChatSession:
    """
    Get or create chat session using AgentConfig. Super simple - no complex locking.

    We accept some minor race conditions for simplicity. Worst case:
    we create an extra session that gets cleaned up later.
    """
    session_key = config.get_session_key()

    # Quick check if session exists and is ready
    session = _chat_sessions.get(session_key)
    if session and session._ready and not session.is_expired():
        session.touch()
        return session

    # Need new session - clean up old one if exists
    if session:
        try:
            await session._cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up old session: {e}")

    # Create fresh session with config
    session = ChatSession(config)
    await session.initialize()

    # Store in global dict
    _chat_sessions[session_key] = session
    logger.info(f"Created new chat session: {session_key}")

    return session


async def cleanup_expired_sessions():
    """Simple cleanup - remove expired chat sessions."""
    expired_keys = []

    for key, session in _chat_sessions.items():
        if session.is_expired():
            expired_keys.append(key)

    for key in expired_keys:
        session = _chat_sessions.pop(key, None)
        if session:
            try:
                await session._cleanup()
                logger.info(f"Cleaned up expired chat session: {key}")
            except Exception as e:
                logger.error(f"Error cleaning chat session {key}: {e}")


async def _cleanup_loop():
    """Background cleanup task."""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            await cleanup_expired_sessions()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")


async def start_cleanup():
    """Start background cleanup task."""
    global _cleanup_task
    if _cleanup_task is None:
        _cleanup_task = asyncio.create_task(_cleanup_loop())
        logger.info("Started session cleanup task")


async def shutdown_all():
    """Shutdown all chat sessions and cleanup task."""
    global _cleanup_task

    # Stop cleanup task
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        _cleanup_task = None

    # Clean up all chat sessions
    sessions = list(_chat_sessions.values())
    _chat_sessions.clear()

    for session in sessions:
        try:
            await session._cleanup()
        except Exception as e:
            logger.error(f"Error during shutdown cleanup: {e}")

    logger.info("All chat sessions cleaned up")


def get_stats() -> Dict:
    """Get simple stats."""
    return {
        "total_sessions": len(_chat_sessions),
        "active_sessions": sum(1 for s in _chat_sessions.values() if s._ready),
        "expired_sessions": sum(1 for s in _chat_sessions.values() if s.is_expired()),
    }
