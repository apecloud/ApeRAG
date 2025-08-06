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

from aperag.agent import AgentMessageQueue
from aperag.agent.agent_event_processor import AgentEventProcessor

logger = logging.getLogger(__name__)


class AgentEventListener(EventListener):
    """
    A thread-safe, singleton proxy listener that is registered once and never removed.
    It solves the "dictionary changed size during iteration" race condition by
    managing its own internal, locked collection of temporary AgentEventProcessors,
    and uses the trace_id from the event to dispatch it to the correct listener.
    """

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentEventListener, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    async def initialize(self):
        """Initializes the singleton instance and registers itself with the event bus."""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            self._request_listeners: Dict[str, AgentEventProcessor] = {}
            self._bus = AsyncEventBus.get()
            self._bus.add_listener("global", self)  # Register self, permanently
            self._initialized = True
            logger.info("AgentEventListener initialized and registered permanently.")

    async def register_listener(
        self,
        trace_id: str,
        chat_id: str,
        message_id: str,
        queue: AgentMessageQueue,
        language,
    ):
        """
        Safely creates and registers a AgentEventProcessor for a specific request,
        keyed by its trace_id.
        """
        listener = AgentEventProcessor(
            message_queue=queue,
            trace_id=trace_id,
            chat_id=chat_id,
            message_id=message_id,
            language=language,
        )
        async with self._lock:
            self._request_listeners[trace_id] = listener
            logger.debug(f"Registered temporary listener for trace_id: {trace_id}")

    async def unregister_listener(self, trace_id: str):
        """Safely unregisters a temporary listener by its trace_id."""
        async with self._lock:
            if trace_id in self._request_listeners:
                del self._request_listeners[trace_id]
                logger.debug(f"Unregistered temporary listener for trace_id: {trace_id}")

    async def handle_event(self, event: Event):
        """
        Handles an event from the main bus and forwards it to the specific
        listener associated with the event's trace_id.
        """
        # Assuming the mcp-agent's OTel instrumentation adds trace_id to the event.
        # This is a critical assumption for this pattern to work.
        trace_id = event.trace_id
        if not trace_id:
            logger.warning("Received event without a trace_id. Cannot dispatch.")
            return

        # async with self._lock:
        #     # Find the specific listener for this trace_id
        #     listener = self._request_listeners.get(str(trace_id))

        listener = self._request_listeners.get(str(trace_id))

        if listener:
            # Dispatch the event only to the correct listener
            await listener.handle_event(event)
        else:
            logger.warning(f"Received event for trace_id {trace_id} but no listener was registered.")


# Create a single instance for the application to use
agent_event_listener = AgentEventListener()
