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

"""Internationalized error message formatter for agent chat."""

from .response_types import AgentErrorResponse
from .stream_formatters import format_i18n_error


# Convenience functions for common error types
def format_invalid_json_error(error: str, language: str = "en-US") -> AgentErrorResponse:
    """Format invalid JSON error with i18n support."""
    return format_i18n_error("invalid_json_format", language, error=error)


def format_query_required_error(language: str = "en-US") -> AgentErrorResponse:
    """Format query required error with i18n support."""
    return format_i18n_error("query_required", language)


def format_invalid_model_spec_error(error: str, language: str = "en-US") -> AgentErrorResponse:
    """Format invalid model spec error with i18n support."""
    return format_i18n_error("invalid_model_spec", language, error=error)


def format_agent_setup_error(error: str, language: str = "en-US") -> AgentErrorResponse:
    """Format agent setup error with i18n support."""
    return format_i18n_error("agent_setup_failed", language, error=error)


def format_processing_error(error: str, language: str = "en-US") -> AgentErrorResponse:
    """Format processing error with i18n support."""
    return format_i18n_error("processing_error", language, error=error)


def format_model_spec_required_error(language: str = "en-US") -> AgentErrorResponse:
    """Format model spec required error with i18n support."""
    return format_i18n_error("model_spec_required", language)


def format_agent_initialization_error(error: str, language: str = "en-US") -> AgentErrorResponse:
    """Format agent initialization error with i18n support."""
    return format_i18n_error("agent_initialization_failed", language, error=error)


def format_mcp_connection_error(language: str = "en-US") -> AgentErrorResponse:
    """Format MCP server connection error with i18n support."""
    return format_i18n_error("mcp_server_connection_failed", language)


def format_llm_generation_error(error: str, language: str = "en-US") -> AgentErrorResponse:
    """Format LLM generation error with i18n support."""
    return format_i18n_error("llm_generation_error", language, error=error)


def format_agent_execution_error(error: str, language: str = "en-US") -> AgentErrorResponse:
    """Format agent execution error with i18n support."""
    return format_i18n_error("agent_execution_error", language, error=error)


def format_bot_id_required_error(language: str = "en-US") -> AgentErrorResponse:
    """Format bot ID required error with i18n support."""
    return format_i18n_error("bot_id_required", language)


def format_bot_not_found_error(language: str = "en-US") -> AgentErrorResponse:
    """Format bot not found error with i18n support."""
    return format_i18n_error("bot_not_found", language)


def format_bot_flow_config_not_found_error(language: str = "en-US") -> AgentErrorResponse:
    """Format bot flow config not found error with i18n support."""
    return format_i18n_error("bot_flow_config_not_found", language)


def format_no_output_node_error(language: str = "en-US") -> AgentErrorResponse:
    """Format no output node found error with i18n support."""
    return format_i18n_error("no_output_node_found", language)
