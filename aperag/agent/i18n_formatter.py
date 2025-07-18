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

import uuid
from typing import Any, Dict

from aperag.utils.utils import now_unix_milliseconds

# Error message translations
ERROR_MESSAGES = {
    "en-US": {
        "invalid_json_format": "Invalid message format. Please try again or refresh the page.",
        "query_required": "Please enter your question or message",
        "invalid_model_spec": "AI model configuration error. Please select a valid AI model.",
        "agent_setup_failed": "Unable to start the AI assistant. Please try again later.",
        "processing_error": "Unable to process your request. Please try again.",
        "model_spec_required": "Please select an AI model to continue",
        "agent_initialization_failed": "Unable to start the AI assistant. Please try again or contact support.",
        "mcp_server_connection_failed": "AI assistant is temporarily unavailable. Please try again later.",
        "llm_generation_error": "AI response generation failed. Please try again.",
        "agent_execution_error": "AI assistant encountered an error. Please try again.",
        "bot_id_required": "AI assistant not found. Please refresh and try again.",
        "bot_not_found": "The selected AI assistant is not available. Please choose another one.",
        "bot_flow_config_not_found": "AI assistant configuration is missing. Please contact support.",
        "no_output_node_found": "AI assistant configuration error. Please contact support.",
        "websocket_connection_error": "Connection lost. Please refresh the page and try again.",
        "chat_history_error": "Unable to save conversation history.",
        "event_listener_cleanup_error": "Connection cleanup error occurred.",
        "unknown_error": "Something went wrong. Please try again or contact support if the problem persists.",
    },
    "zh-CN": {
        "invalid_json_format": "消息格式错误，请重试或刷新页面",
        "query_required": "请输入您的问题或消息",
        "invalid_model_spec": "AI模型配置错误，请选择有效的AI模型",
        "agent_setup_failed": "无法启动AI助手，请稍后重试",
        "processing_error": "无法处理您的请求，请重试",
        "model_spec_required": "请选择AI模型以继续",
        "agent_initialization_failed": "无法启动AI助手，请重试或联系客服",
        "mcp_server_connection_failed": "AI助手暂时不可用，请稍后重试",
        "llm_generation_error": "AI回复生成失败，请重试",
        "agent_execution_error": "AI助手遇到错误，请重试",
        "bot_id_required": "AI助手未找到，请刷新页面重试",
        "bot_not_found": "所选的AI助手不可用，请选择其他助手",
        "bot_flow_config_not_found": "AI助手配置缺失，请联系客服",
        "no_output_node_found": "AI助手配置错误，请联系客服",
        "websocket_connection_error": "连接中断，请刷新页面重试",
        "chat_history_error": "无法保存对话历史",
        "event_listener_cleanup_error": "连接清理时发生错误",
        "unknown_error": "出现了问题，请重试或联系客服",
    },
}


class I18nErrorFormatter:
    """Internationalized error message formatter."""

    @staticmethod
    def get_error_message(error_key: str, language: str = "en-US", **kwargs) -> str:
        """
        Get localized error message.

        Args:
            error_key: The error message key
            language: Language code (en-US, zh-CN)
            **kwargs: Format parameters for the error message

        Returns:
            Localized error message
        """
        # Fallback to en-US if language not supported
        if language not in ERROR_MESSAGES:
            language = "en-US"

        # Get the message template
        messages = ERROR_MESSAGES[language]
        message_template = messages.get(error_key, messages.get("unknown_error", "Unknown error: {error}"))

        # Format the message with provided parameters
        try:
            return message_template.format(**kwargs)
        except KeyError:
            # If formatting fails, return the template with available info
            return message_template

    @staticmethod
    def format_error(error_key: str, language: str = "en-US", **kwargs) -> Dict[str, Any]:
        """
        Format an internationalized error response.

        Args:
            error_key: The error message key
            language: Language code (en-US, zh-CN)
            **kwargs: Format parameters for the error message

        Returns:
            Formatted error response
        """
        error_message = I18nErrorFormatter.get_error_message(error_key, language, **kwargs)

        return {
            "type": "error",
            "id": str(uuid.uuid4()),
            "data": error_message,
            "timestamp": now_unix_milliseconds(),
        }


# Convenience functions for common error types
def format_invalid_json_error(error: str, language: str = "en-US") -> Dict[str, Any]:
    """Format invalid JSON error with i18n support."""
    return I18nErrorFormatter.format_error("invalid_json_format", language, error=error)


def format_query_required_error(language: str = "en-US") -> Dict[str, Any]:
    """Format query required error with i18n support."""
    return I18nErrorFormatter.format_error("query_required", language)


def format_invalid_model_spec_error(error: str, language: str = "en-US") -> Dict[str, Any]:
    """Format invalid model spec error with i18n support."""
    return I18nErrorFormatter.format_error("invalid_model_spec", language, error=error)


def format_agent_setup_error(error: str, language: str = "en-US") -> Dict[str, Any]:
    """Format agent setup error with i18n support."""
    return I18nErrorFormatter.format_error("agent_setup_failed", language, error=error)


def format_processing_error(error: str, language: str = "en-US") -> Dict[str, Any]:
    """Format processing error with i18n support."""
    return I18nErrorFormatter.format_error("processing_error", language, error=error)


def format_model_spec_required_error(language: str = "en-US") -> Dict[str, Any]:
    """Format model spec required error with i18n support."""
    return I18nErrorFormatter.format_error("model_spec_required", language)


def format_agent_initialization_error(error: str, language: str = "en-US") -> Dict[str, Any]:
    """Format agent initialization error with i18n support."""
    return I18nErrorFormatter.format_error("agent_initialization_failed", language, error=error)


def format_mcp_connection_error(language: str = "en-US") -> Dict[str, Any]:
    """Format MCP server connection error with i18n support."""
    return I18nErrorFormatter.format_error("mcp_server_connection_failed", language)


def format_llm_generation_error(error: str, language: str = "en-US") -> Dict[str, Any]:
    """Format LLM generation error with i18n support."""
    return I18nErrorFormatter.format_error("llm_generation_error", language, error=error)


def format_agent_execution_error(error: str, language: str = "en-US") -> Dict[str, Any]:
    """Format agent execution error with i18n support."""
    return I18nErrorFormatter.format_error("agent_execution_error", language, error=error)


def format_bot_id_required_error(language: str = "en-US") -> Dict[str, Any]:
    """Format bot ID required error with i18n support."""
    return I18nErrorFormatter.format_error("bot_id_required", language)


def format_bot_not_found_error(language: str = "en-US") -> Dict[str, Any]:
    """Format bot not found error with i18n support."""
    return I18nErrorFormatter.format_error("bot_not_found", language)


def format_bot_flow_config_not_found_error(language: str = "en-US") -> Dict[str, Any]:
    """Format bot flow config not found error with i18n support."""
    return I18nErrorFormatter.format_error("bot_flow_config_not_found", language)


def format_no_output_node_error(language: str = "en-US") -> Dict[str, Any]:
    """Format no output node found error with i18n support."""
    return I18nErrorFormatter.format_error("no_output_node_found", language)
