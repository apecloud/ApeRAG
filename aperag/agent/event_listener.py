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

from .exceptions import EventListenerError, handle_agent_error
from .message_queue import AgentMessageQueue
from .tool_formatters import (
    detect_interface_type,
    format_tool_call_end,
    format_tool_call_start,
    format_tool_request_display,
    format_tool_response_display,
)

logger = logging.getLogger(__name__)


class UniversalEventListener(EventListener):
    """通用事件监听器，支持多种事件类型的监听和处理"""

    def __init__(self, msg_id: str, message_queue: AgentMessageQueue):
        self.msg_id = msg_id
        self.message_queue = message_queue

    @handle_agent_error("event_handling", reraise=False)
    async def handle_event(self, event: Event):
        """处理各种类型的事件"""
        if not event.message:
            return

        # 根据消息类型分发到不同的处理函数
        if event.message == "send_request: request=":
            await self._handle_tool_request(event)
        elif event.message == "send_request: response=":
            await self._handle_tool_response(event)
        else:
            await self._handle_generic_event(event)

    @handle_agent_error("tool_request_handling", reraise=False)
    async def _handle_tool_request(self, event: Event):
        """处理工具调用请求事件"""
        if not event.data or not isinstance(event.data, dict):
            raise EventListenerError(
                "tool_request", "Invalid event data structure", event_data={"has_data": bool(event.data)}
            )

        data_field = event.data.get("data")
        if not data_field or not isinstance(data_field, dict):
            raise EventListenerError(
                "tool_request", "Missing or invalid data field", event_data={"data_type": type(data_field).__name__}
            )

        method = data_field.get("method", "")
        params = data_field.get("params", {})

        if method == "tools/call" and params:
            tool_name = params.get("name", "unknown")
            tool_args = params.get("arguments", {})

            # 使用工具函数格式化显示文本
            display_text = format_tool_request_display(tool_name, tool_args)

            # 使用工具函数创建格式化消息，发送到队列
            formatted_message = format_tool_call_start(self.msg_id, display_text, tool_name, tool_args)
            await self.message_queue.put(formatted_message)

            logger.debug(f"Tool request captured: {tool_name}")

    @handle_agent_error("tool_response_handling", reraise=False)
    async def _handle_tool_response(self, event: Event):
        """处理工具调用响应事件"""
        if not event.data or not isinstance(event.data, dict):
            raise EventListenerError(
                "tool_response", "Invalid event data structure", event_data={"has_data": bool(event.data)}
            )

        data_field = event.data.get("data")
        if not data_field or not isinstance(data_field, dict):
            raise EventListenerError(
                "tool_response", "Missing or invalid data field", event_data={"data_type": type(data_field).__name__}
            )

        # 解析响应内容
        structured_content = data_field.get("structuredContent")
        is_error = data_field.get("isError", False)

        # 使用工具函数检测接口类型
        interface_type = detect_interface_type(structured_content)

        # 使用工具函数格式化显示文本
        display_text = format_tool_response_display(interface_type, structured_content, is_error)

        # 使用工具函数创建格式化消息，发送到队列
        formatted_message = format_tool_call_end(self.msg_id, display_text, interface_type, structured_content)
        await self.message_queue.put(formatted_message)

        logger.debug(f"Tool response captured: {interface_type}")

    async def _handle_generic_event(self, event: Event):
        """处理其他通用事件"""
        # 可以根据需要扩展处理其他类型的事件
        pass

    # Deprecated methods - kept for backward compatibility during transition
    def get_new_messages(self, last_count: int = 0):
        """Deprecated: Use message queue instead"""
        logger.warning("get_new_messages is deprecated, use message queue instead")
        return []

    def get_message_count(self) -> int:
        """Deprecated: Use message queue instead"""
        logger.warning("get_message_count is deprecated, use message queue instead")
        return 0

    def clear_messages(self):
        """Deprecated: Use message queue instead"""
        logger.warning("clear_messages is deprecated, use message queue instead")
        pass
