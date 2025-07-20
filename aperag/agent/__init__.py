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

# Use simple session management instead of complex singleton patterns
from .agent_session_manager import get_or_create_session, get_stats
from .agent_session_manager_lifecycle import agent_session_manager_lifespan
from .event_listener import UniversalEventListener
from .exceptions import (
    AgentConfigError,
    AgentError,
    AgentTimeoutError,
    EventListenerError,
    JSONParsingError,
    MCPAppInitializationError,
    MCPConnectionError,
    StreamFormattingError,
    ToolExecutionError,
    ToolReferenceExtractionError,
    agent_config_invalid,
    agent_timeout,
    extract_tool_result_data,
    handle_agent_error,
    mcp_connection_failed,
    mcp_init_failed,
    safe_json_parse,
    tool_execution_failed,
    with_retry,
)
from .i18n_formatter import (
    I18nErrorFormatter,
    format_agent_execution_error,
    format_agent_initialization_error,
    format_agent_setup_error,
    format_bot_flow_config_not_found_error,
    format_bot_id_required_error,
    format_bot_not_found_error,
    format_invalid_json_error,
    format_invalid_model_spec_error,
    format_llm_generation_error,
    format_mcp_connection_error,
    format_model_spec_required_error,
    format_no_output_node_error,
    format_processing_error,
    format_query_required_error,
)
from .mcp_app_factory import MCPAppFactory
from .response_types import (
    AgentChatResponse,
    AgentErrorResponse,
    AgentMessageResponse,
    AgentResponse,
    AgentStartResponse,
    AgentStopResponse,
    AgentThinkingResponse,
    AgentToolCallEndResponse,
    AgentToolCallStartResponse,
    WebSocketResponse,
)
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
    # Event listener
    "UniversalEventListener",
    # Simple session management
    "get_or_create_session",
    "get_stats",
    # Agent session manager lifecycle management
    "agent_session_manager_lifespan",
    # MCP App Factory
    "MCPAppFactory",
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
    # Exception classes
    "AgentError",
    "MCPConnectionError",
    "MCPAppInitializationError",
    "ToolExecutionError",
    "EventListenerError",
    "StreamFormattingError",
    "AgentConfigError",
    "ToolReferenceExtractionError",
    "JSONParsingError",
    "AgentTimeoutError",
    # Exception utilities
    "safe_json_parse",
    "extract_tool_result_data",
    "with_retry",
    "handle_agent_error",
    # Convenience functions
    "mcp_connection_failed",
    "mcp_init_failed",
    "tool_execution_failed",
    "agent_config_invalid",
    "agent_timeout",
    # I18n error formatters
    "I18nErrorFormatter",
    "format_agent_execution_error",
    "format_agent_initialization_error",
    "format_agent_setup_error",
    "format_bot_flow_config_not_found_error",
    "format_bot_id_required_error",
    "format_bot_not_found_error",
    "format_invalid_json_error",
    "format_invalid_model_spec_error",
    "format_llm_generation_error",
    "format_mcp_connection_error",
    "format_model_spec_required_error",
    "format_no_output_node_error",
    "format_processing_error",
    "format_query_required_error",
    # Response types
    "AgentResponse",
    "AgentErrorResponse", 
    "AgentStartResponse",
    "AgentMessageResponse",
    "AgentStopResponse",
    "AgentThinkingResponse",
    "AgentToolCallStartResponse",
    "AgentToolCallEndResponse",
    "AgentChatResponse",
    "WebSocketResponse",
]
