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

"""Stream message formatters for agent responses."""

from typing import Any, Dict, List

from aperag.utils.utils import now_unix_milliseconds

from .i18n import ERROR_MESSAGES
from .response_types import (
    AgentErrorResponse,
    AgentMessageResponse,
    AgentStartResponse,
    AgentStopResponse,
    AgentThinkingResponse,
    AgentToolCallResultResponse,
)


def format_i18n_error(error_key: str, language: str = "en-US", **kwargs) -> AgentErrorResponse:
    """Format internationalized error message."""
    messages = ERROR_MESSAGES.get(language, ERROR_MESSAGES["en-US"])

    if error_key in messages:
        error_message = messages[error_key]
        # If the message contains placeholders, format it with kwargs
        if kwargs:
            try:
                error_message = error_message.format(**kwargs)
            except KeyError:
                # If formatting fails, just use the base message
                pass
    else:
        # Fallback to unknown error if key doesn't exist
        error_message = messages.get("unknown_error", "An error occurred")

    return AgentErrorResponse(
        type="error",
        id="error",
        data=error_message,
        timestamp=now_unix_milliseconds(),
    )


# Backward compatibility functions
def format_stream_start(msg_id: str) -> AgentStartResponse:
    """Format stream start event - backward compatibility"""
    return AgentStartResponse(
        type="start",
        id=msg_id,
        timestamp=now_unix_milliseconds(),
    )


def format_stream_content(msg_id: str, content: str) -> AgentMessageResponse:
    """Format stream content event - backward compatibility"""
    return AgentMessageResponse(
        type="message",
        id=msg_id,
        data=content,
        timestamp=now_unix_milliseconds(),
    )


def format_stream_end(
    msg_id: str, references: List[Dict[str, Any]] = None, urls: List[str] = None
) -> AgentStopResponse:
    """Format stream end event - backward compatibility"""
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


def format_thinking(msg_id: str, content: str) -> AgentThinkingResponse:
    """Format thinking step event - backward compatibility"""
    return AgentThinkingResponse(
        type="thinking",
        id=msg_id,
        data=content,
        timestamp=now_unix_milliseconds(),
    )


def format_tool_call_result(msg_id: str, data: str, tool_name: str, result: Any) -> AgentMessageResponse:
    return AgentToolCallResultResponse(
        type="tool_call_result",
        id=msg_id,
        data=data,
        tool_name=tool_name,
        result=result,
        timestamp=now_unix_milliseconds(),
    )


# New unified formatter functions
def format_agent_start_message(trace_id: str, language: str = "en-US") -> Dict[str, Any]:
    """Format agent start message."""
    return {
        "type": "start",
        "id": trace_id,
        "timestamp": now_unix_milliseconds(),
    }


def format_agent_stop_message(trace_id: str, references: list = None, urls: list = None) -> Dict[str, Any]:
    """Format agent stop message with references."""
    return {
        "type": "stop",
        "id": trace_id,
        "data": references or [],
        "urls": urls or [],
        "timestamp": now_unix_milliseconds(),
    }


def format_agent_thinking_message(trace_id: str, thinking_content: str) -> Dict[str, Any]:
    """Format agent thinking message."""
    return {
        "type": "thinking",
        "id": trace_id,
        "data": thinking_content,
        "timestamp": now_unix_milliseconds(),
    }


def format_agent_message(trace_id: str, content: str) -> Dict[str, Any]:
    """Format regular agent message."""
    return {
        "type": "message",
        "id": trace_id,
        "data": content,
        "timestamp": now_unix_milliseconds(),
    }
