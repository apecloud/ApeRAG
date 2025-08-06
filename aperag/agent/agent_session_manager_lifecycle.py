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

"""Super simple lifecycle management for agent sessions."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from aperag.agent import agent_session_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def agent_session_manager_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ultra-simple lifecycle management - just start/stop cleanup task."""

    # Startup: start background cleanup
    logger.info("Starting agent session cleanup")
    await agent_session_manager.start_cleanup()

    try:
        yield
    finally:
        # Shutdown: clean everything up
        logger.info("Shutting down agent sessions")
        await agent_session_manager.shutdown_all()
        logger.info("Agent sessions shutdown complete")
