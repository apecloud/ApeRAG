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
from typing import Any

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

        interface_type, typed_result = detect_interface_type(structured_content)
        if interface_type == "unknown":
            return

        # Skip error calls as requested by user feedback
        if is_error:
            return

        # Skip empty or meaningless results
        if self._should_skip_empty_result(interface_type, typed_result, structured_content):
            return

        display_text = format_tool_use_response(self.language, interface_type, typed_result, is_error)

        formatted_message = format_tool_call_end(
            self.message_id, display_text, interface_type, typed_result or structured_content
        )
        await self.message_queue.put(formatted_message)

        logger.debug(
            f"Tool response captured for message {self.message_id}: {interface_type} (typed: {typed_result is not None})"
        )

    def _should_skip_empty_result(self, interface_type: str, typed_result: Any, structured_content: Any) -> bool:
        """Check if we should skip displaying empty or meaningless results"""
        try:
            if interface_type == "search_collection":
                # Skip if query is empty or just whitespace (meaningless search)
                if typed_result:
                    from aperag.schema.view_models import SearchResult

                    if isinstance(typed_result, SearchResult):
                        if not typed_result.query or not typed_result.query.strip():
                            return True
                        # Don't skip zero results if query is valid - show search action
                elif isinstance(structured_content, dict):
                    query = structured_content.get("query", "")
                    if not query or not query.strip():
                        return True
                    # Don't skip zero results if query is valid - show search action

            elif interface_type == "list_collections":
                # Skip if no collections found
                if typed_result:
                    from aperag.schema.view_models import CollectionList

                    if isinstance(typed_result, CollectionList):
                        if not typed_result.items or len(typed_result.items) == 0:
                            return True
                elif isinstance(structured_content, dict):
                    items = structured_content.get("items", [])
                    if not items or len(items) == 0:
                        return True

            elif interface_type == "web_search":
                # Skip if no web search results
                if typed_result:
                    from aperag.schema.view_models import WebSearchResponse

                    if isinstance(typed_result, WebSearchResponse):
                        if not typed_result.results or len(typed_result.results) == 0:
                            return True
                elif isinstance(structured_content, dict):
                    results = structured_content.get("results", [])
                    if not results or len(results) == 0:
                        return True

            elif interface_type == "web_read":
                # Skip if no pages successfully read
                if typed_result:
                    from aperag.schema.view_models import WebReadResponse

                    if isinstance(typed_result, WebReadResponse):
                        if typed_result.successful == 0:
                            return True
                elif isinstance(structured_content, dict):
                    successful = structured_content.get("successful", 0)
                    if successful == 0:
                        return True

            return False

        except Exception as e:
            logger.warning(f"Error checking if should skip result: {e}")
            # When in doubt, don't skip - better to show something than miss important info
            return False

    async def _handle_generic_event(self, event: Event):
        pass
