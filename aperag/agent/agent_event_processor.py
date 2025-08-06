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
from typing import Any, Dict, Optional

from mcp_agent.logging.events import Event
from mcp_agent.logging.listeners import EventListener

from .agent_message_queue import AgentMessageQueue
from .exceptions import EventListenerError, handle_agent_error
from .stream_formatters import format_tool_call_result
from .tool_use_message_formatters import (
    ToolResultFormatter,
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
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message_queue = message_queue
        self.trace_id = trace_id
        self.chat_id = chat_id
        self.message_id = message_id
        self.language = language
        self.formatter = ToolResultFormatter(language, context)

    @handle_agent_error("event_handling", reraise=False)
    async def handle_event(self, event: Event):
        if not event or not event.message:
            return
        if self.trace_id != event.trace_id:
            logger.warning(
                f"Event trace_id {event.trace_id} does not match listener trace_id {self.trace_id}, ignoring event."
            )
            return

        if event.message == "send_request: response=":
            await self._handle_tool_response(event)

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

        # Skip error calls as requested by user feedback
        if is_error:
            return

        interface_type, typed_result = self.formatter.detect_and_parse_result(structured_content)
        if interface_type == "unknown":
            return

        # Use simplified logic to determine if we should display this result
        if not self.formatter.should_display_result(interface_type, typed_result, structured_content):
            return

        display_text = self.formatter.format_tool_response(interface_type, typed_result, structured_content, is_error)

        formatted_message = format_tool_call_result(self.message_id, display_text + "\n\n", interface_type, None)
        await self.message_queue.put(formatted_message)

        logger.debug(
            f"Tool response captured for message {self.message_id}: {interface_type} (typed: {typed_result is not None})"
        )
