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

"""Agent module for MCP-based intelligent conversation."""

from .event_listener import UniversalEventListener
from .stream_formatters import (
    format_error,
    format_stream_content,
    format_stream_end,
    format_stream_start,
    format_thinking,
)
from .tool_formatters import (
    detect_interface_type,
    format_tool_call_end,
    format_tool_call_start,
    format_tool_request_display,
    format_tool_response_display,
)
from .tool_reference_extractor import extract_tool_call_references

__all__ = [
    "UniversalEventListener",
    # Stream formatters
    "format_error",
    "format_stream_content",
    "format_stream_end",
    "format_stream_start",
    "format_thinking",
    # Tool formatters
    "detect_interface_type",
    "format_tool_call_end",
    "format_tool_call_start",
    "format_tool_request_display",
    "format_tool_response_display",
    # Tool reference extractor
    "extract_tool_call_references",
]
