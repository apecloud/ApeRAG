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
from typing import Any, Dict, List

from mcp_agent.logging.events import Event
from mcp_agent.logging.listeners import EventListener

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

    def __init__(self, msg_id: str):
        self.msg_id = msg_id
        self.formatted_messages = []  # 存储格式化好的消息，可直接yield

    async def handle_event(self, event: Event):
        """处理各种类型的事件"""
        try:
            if not event.message:
                return

            # 根据消息类型分发到不同的处理函数
            if event.message == "send_request: request=":
                await self._handle_tool_request(event)
            elif event.message == "send_request: response=":
                await self._handle_tool_response(event)
            else:
                await self._handle_generic_event(event)

        except Exception as e:
            logger.error(f"Error in universal event listener: {e}")

    async def _handle_tool_request(self, event: Event):
        """处理工具调用请求事件"""
        try:
            if not event.data or not isinstance(event.data, dict):
                return

            data_field = event.data.get("data")
            if not data_field or not isinstance(data_field, dict):
                return

            method = data_field.get("method", "")
            params = data_field.get("params", {})

            if method == "tools/call" and params:
                tool_name = params.get("name", "unknown")
                tool_args = params.get("arguments", {})

                # 使用工具函数格式化显示文本
                display_text = format_tool_request_display(tool_name, tool_args)

                # 使用工具函数创建格式化消息，直接可以yield
                formatted_message = format_tool_call_start(self.msg_id, display_text, tool_name, tool_args)
                self.formatted_messages.append(formatted_message)

                logger.debug(f"Tool request captured: {tool_name}")

        except Exception as e:
            logger.error(f"Error handling tool request: {e}")

    async def _handle_tool_response(self, event: Event):
        """处理工具调用响应事件"""
        try:
            if not event.data or not isinstance(event.data, dict):
                return

            data_field = event.data.get("data")
            if not data_field or not isinstance(data_field, dict):
                return

            # 解析响应内容
            structured_content = data_field.get("structuredContent")
            is_error = data_field.get("isError", False)

            # 使用工具函数检测接口类型
            interface_type = detect_interface_type(structured_content)

            # 使用工具函数格式化显示文本
            display_text = format_tool_response_display(interface_type, structured_content, is_error)

            # 使用工具函数创建格式化消息，直接可以yield
            formatted_message = format_tool_call_end(self.msg_id, display_text, interface_type, structured_content)
            self.formatted_messages.append(formatted_message)

            logger.debug(f"Tool response captured: {interface_type}")

        except Exception as e:
            logger.error(f"Error handling tool response: {e}")

    async def _handle_generic_event(self, event: Event):
        """处理其他通用事件"""
        # 可以根据需要扩展处理其他类型的事件
        pass

    def get_new_messages(self, last_count: int = 0) -> List[Dict[str, Any]]:
        """获取新的格式化消息"""
        return self.formatted_messages[last_count:]

    def get_message_count(self) -> int:
        """获取当前消息总数"""
        return len(self.formatted_messages)

    def clear_messages(self):
        """清空消息队列"""
        self.formatted_messages.clear()
