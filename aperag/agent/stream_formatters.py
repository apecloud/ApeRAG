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

"""Stream response formatters for agent chat."""

import uuid
from typing import Any, Dict, List

from aperag.utils.utils import now_unix_milliseconds

from .response_types import (
    AgentErrorResponse,
    AgentMessageResponse,
    AgentStartResponse,
    AgentStopResponse,
    AgentThinkingResponse,
)


def format_stream_start(msg_id: str) -> AgentStartResponse:
    """格式化流式开始事件"""
    return AgentStartResponse(
        type="start",
        id=msg_id,
        timestamp=now_unix_milliseconds(),
    )


def format_stream_content(msg_id: str, content: str) -> AgentMessageResponse:
    """格式化流式内容事件"""
    return AgentMessageResponse(
        type="message",
        id=msg_id,
        data=content,
        timestamp=now_unix_milliseconds(),
    )


def format_stream_end(
    msg_id: str, references: List[Dict[str, Any]] = None, urls: List[str] = None
) -> AgentStopResponse:
    """格式化流式结束事件"""
    if references is None:
        references = []
    if urls is None:
        urls = []

    return AgentStopResponse(
        type="stop",
        id=msg_id,
        data=references,
        urls=urls,
        timestamp=now_unix_milliseconds(),
    )


def format_error(error: str) -> AgentErrorResponse:
    """格式化错误响应"""
    return AgentErrorResponse(
        type="error",
        id=str(uuid.uuid4()),
        data=error,
        timestamp=now_unix_milliseconds(),
    )


def format_thinking(msg_id: str, content: str) -> AgentThinkingResponse:
    """格式化思考步骤事件"""
    return AgentThinkingResponse(
        type="thinking",
        id=msg_id,
        data=content,
        timestamp=now_unix_milliseconds(),
    )
