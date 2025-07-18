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


class ProviderSession:
    """
    Ultra-simple session per user+provider combination.

    Key insight: Same provider (OpenAI, Anthropic) can serve multiple models.
    We create MCPApp per provider, specify model at runtime.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.last_used = time.time()

        # MCP resources - created once per provider
        self.mcp_app = None
        self.mcp_running_app = None
        self.agent = None

        # Simple state flags
        self._ready = False

    async def initialize(self):
        """Initialize with provider settings from config."""
        if self._ready:
            return

        try:
            logger.info(f"Initializing provider session {self.config.get_session_key()}")

            # Create MCP app for this provider using config
            self.mcp_app = MCPAppFactory.create_mcp_app(
                model=self.config.default_model,  # Default model, can override later
                llm_provider_name=self.config.provider_name,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                aperag_api_key=self.config.aperag_api_key,
                aperag_url=self.config.aperag_url,
            )

            # Start MCP app
            self.mcp_running_app = await self.mcp_app.run().__aenter__()

            # Create reusable agent
            self.agent = Agent(
                name=f"aperag_agent_{self.config.user_id}_{self.config.provider_name}",
                instruction=self.config.instruction,
                server_names=self.config.server_names,
            )

            await self.agent.__aenter__()
            self._ready = True

            logger.info(f"Provider session {self.config.get_session_key()} ready")

        except Exception as e:
            logger.error(f"Failed to initialize session {self.config.get_session_key()}: {e}")
            await self._cleanup()
            raise AgentConfigurationError(f"Session init failed: {e}")

    async def get_llm(self, model: str) -> OpenAIAugmentedLLM:
        """Get LLM for specific model. Creates new LLM each time for clean state."""
        if not self._ready:
            raise AgentConfigurationError("Session not ready")

        # Create fresh LLM with specified model
        # This ensures clean memory state for each conversation
        llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

        # Update model in the LLM config if needed
        # (The model will be specified in RequestParams anyway)

        return llm

    def touch(self):
        """Update last used time."""
        self.last_used = time.time()

    def is_expired(self, timeout: int = 1800) -> bool:  # 30 min default
        """Check if session expired."""
        return time.time() - self.last_used > timeout

    async def _cleanup(self):
        """Clean up all resources."""
        logger.info(f"Cleaning up session {self.config.get_session_key()}")

        if self.agent:
            try:
                await self.agent.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Agent cleanup error: {e}")
            self.agent = None

        if self.mcp_running_app:
            try:
                await self.mcp_running_app.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Agent app cleanup error: {e}")
            self.mcp_running_app = None

        self.mcp_app = None
        self._ready = False


# Simple global state - no complex singleton patterns
_provider_sessions: Dict[str, ProviderSession] = {}
_cleanup_task: Optional[asyncio.Task] = None


def generate_session_key(user_id: str, provider_name: str) -> str:
    """Generate session key based on user and provider only."""
    return f"{user_id}:{provider_name}"


async def get_or_create_session(config: AgentConfig) -> ProviderSession:
    """
    Get or create provider session using AgentConfig. Super simple - no complex locking.

    We accept some minor race conditions for simplicity. Worst case:
    we create an extra session that gets cleaned up later.
    """
    session_key = config.get_session_key()

    # Quick check if session exists and is ready
    session = _provider_sessions.get(session_key)
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
    session = ProviderSession(config)
    await session.initialize()

    # Store in global dict
    _provider_sessions[session_key] = session
    logger.info(f"Created new provider session: {session_key}")

    return session


async def cleanup_expired_sessions():
    """Simple cleanup - remove expired sessions."""
    expired_keys = []

    for key, session in _provider_sessions.items():
        if session.is_expired():
            expired_keys.append(key)

    for key in expired_keys:
        session = _provider_sessions.pop(key, None)
        if session:
            try:
                await session._cleanup()
                logger.info(f"Cleaned up expired session: {key}")
            except Exception as e:
                logger.error(f"Error cleaning session {key}: {e}")


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
    """Shutdown all sessions and cleanup task."""
    global _cleanup_task

    # Stop cleanup task
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        _cleanup_task = None

    # Clean up all sessions
    sessions = list(_provider_sessions.values())
    _provider_sessions.clear()

    for session in sessions:
        try:
            await session._cleanup()
        except Exception as e:
            logger.error(f"Error during shutdown cleanup: {e}")

    logger.info("All sessions cleaned up")


def get_stats() -> Dict:
    """Get simple stats."""
    return {
        "total_sessions": len(_provider_sessions),
        "active_sessions": sum(1 for s in _provider_sessions.values() if s._ready),
        "expired_sessions": sum(1 for s in _provider_sessions.values() if s.is_expired()),
    }
