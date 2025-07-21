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
import logging
from typing import Dict

from mcp_agent.logging.listeners import EventListener
from mcp_agent.logging.transport import AsyncEventBus, Event

from aperag.agent.event_listener import UniversalEventListener
from aperag.agent.message_queue import AgentMessageQueue

logger = logging.getLogger(__name__)


class GlobalProxyListener(EventListener):
    """
    A thread-safe, singleton proxy listener that is registered once and never removed.
    It solves the "dictionary changed size during iteration" race condition by
    managing its own internal, locked collection of temporary UniversalEventListeners.
    """

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalProxyListener, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    async def initialize(self):
        """Initializes the singleton instance and registers itself with the event bus."""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            self._request_listeners: Dict[str, UniversalEventListener] = {}
            self._bus = AsyncEventBus.get()
            self._bus.add_listener("global", self)  # Register self, permanently
            self._initialized = True
            logger.info("GlobalProxyListener initialized and registered permanently.")

    async def register_listener(self, message_id: str, queue: AgentMessageQueue):
        """
        Safely creates and registers a UniversalEventListener for a specific request.
        """
        listener = UniversalEventListener(message_id, queue)
        async with self._lock:
            self._request_listeners[message_id] = listener
            logger.debug(f"Registered temporary listener for message_id: {message_id}")

    async def unregister_listener(self, message_id: str):
        """Safely unregisters a temporary listener."""
        async with self._lock:
            if message_id in self._request_listeners:
                del self._request_listeners[message_id]
                logger.debug(f"Unregistered temporary listener for message_id: {message_id}")

    async def handle_event(self, event: Event):
        """
        Handles events from the main bus and safely forwards them to all
        currently registered temporary listeners.
        """
        if not self._request_listeners:
            return

        # Iterate over a copy of the values to avoid issues if the dict is
        # changed by another coroutine. This is the most robust approach.
        async with self._lock:
            listeners = list(self._request_listeners.values())

        # Await all listener handlers concurrently
        await asyncio.gather(*(listener.handle_event(event) for listener in listeners))


# Create a single instance for the application to use
global_proxy_listener = GlobalProxyListener()
