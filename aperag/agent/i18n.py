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
