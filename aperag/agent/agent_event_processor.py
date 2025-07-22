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

"""Universal event listener for MCP agent events."""

import logging

from mcp_agent.logging.events import Event
from mcp_agent.logging.listeners import EventListener

from .agent_message_queue import AgentMessageQueue
from .exceptions import EventListenerError, handle_agent_error
from .tool_use_message_formatters import (
    detect_interface_type,
    format_tool_call_end,
    format_tool_use_response,
)

logger = logging.getLogger(__name__)


class AgentEventProcessor(EventListener):
    def __init__(
        self,
        message_queue: AgentMessageQueue,
        trace_id: str,
        chat_id: str,
        message_id: str,
        language: str = "en-US",
    ):
        self.message_queue = message_queue
        self.trace_id = trace_id
        self.chat_id = chat_id
        self.message_id = message_id
        self.language = language

    @handle_agent_error("event_handling", reraise=False)
    async def handle_event(self, event: Event):
        print(event)
        if not event or not event.message:
            return
        if self.trace_id != event.trace_id:
            logger.warning(
                f"Event trace_id {event.trace_id} does not match listener trace_id {self.trace_id}, ignoring event."
            )
            return

        if event.message == "send_request: response=":
            await self._handle_tool_response(event)
        else:
            await self._handle_generic_event(event)

    @handle_agent_error("tool_response_handling", reraise=False)
    async def _handle_tool_response(self, event: Event):
        if not event.data or not isinstance(event.data, dict):
            raise EventListenerError(
                "tool_response", "Invalid event data structure", event_data={"has_data": bool(event.data)}
            )

        data_field = event.data.get("data")
        if not data_field or not isinstance(data_field, dict):
            raise EventListenerError(
                "tool_response", "Missing or invalid data field", event_data={"data_type": type(data_field).__name__}
            )

        structured_content = data_field.get("structuredContent")
        is_error = data_field.get("isError", False)

        interface_type, result = detect_interface_type(structured_content)
        if interface_type == "unknown" or result is None:
            return

        display_text = format_tool_use_response(self.language, interface_type, structured_content, is_error)

        formatted_message = format_tool_call_end(self.message_id, display_text, interface_type, structured_content)
        await self.message_queue.put(formatted_message)

        logger.debug(f"Tool response captured for message {self.message_id}: {interface_type}")

    async def _handle_generic_event(self, event: Event):
        pass
