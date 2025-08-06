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

"""Unified response type definitions for agent chat."""

from typing import Any, Dict, List, Literal, TypedDict, Union


class BaseAgentResponse(TypedDict):
    """Base response structure for all agent messages."""

    id: str
    timestamp: int


class AgentStartResponse(BaseAgentResponse):
    """Stream start response."""

    type: Literal["start"]


class AgentMessageResponse(BaseAgentResponse):
    """Regular message content response."""

    type: Literal["message"]
    data: str


class AgentStopResponse(BaseAgentResponse):
    """Stream end response with references and URLs."""

    type: Literal["stop"]
    data: List[Dict[str, Any]]  # references
    urls: List[str]


class AgentErrorResponse(BaseAgentResponse):
    """Error response."""

    type: Literal["error"]
    data: str  # Error message


class AgentThinkingResponse(BaseAgentResponse):
    """Thinking step response."""

    type: Literal["thinking"]
    data: str


class AgentToolCallResultResponse(BaseAgentResponse):
    """Tool call end response."""

    type: Literal["tool_call_result"]
    data: str  # Display text
    tool_name: str
    result: Any


# Union type for all possible agent responses
AgentResponse = Union[
    AgentStartResponse,
    AgentMessageResponse,
    AgentStopResponse,
    AgentErrorResponse,
    AgentThinkingResponse,
    AgentToolCallResultResponse,
]


# Type aliases for backward compatibility
AgentChatResponse = AgentResponse
WebSocketResponse = AgentResponse
