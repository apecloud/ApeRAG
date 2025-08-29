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

"""Internationalization messages for agent events and tool usage."""

# Tool use event message translations
TOOL_USE_EVENT_MESSAGES = {
    "en-US": {
        # Tool display names
        "tool_names": {
            "list_collections": "List Collections",
            "search_collection": "Search Collection",
            "search_chat_files": "Search Chat Files",
            "web_search": "Web Search",
            "web_read": "Read Web Pages",
            "unknown": "Tool Call",
        },
        # Tool request descriptions (what the tool is doing)
        "requests": {
            "list_collections": "Getting all available collections",
            "search_collection": 'Searching for "{query}" using {search_types}, returning top {topk} results',
            "search_chat_files": 'Searching chat files for "{query}" using {search_types}, returning top {topk} results',
            "web_search": 'Searching the web for "{query}", returning {max_results} results',
            "web_read": "Reading content from {count} web pages",
        },
        # Tool response summaries (what the tool accomplished)
        "responses": {
            "list_collections": {"success": "Found {count} collections", "error": "Failed to retrieve collections"},
            "search_collection": {
                "success": 'Found {count} results for "{query}"',
                "searching": 'Searched for "{query}"',  # Used when 0 results but valid query
                "error": "Search failed",
            },
            "search_chat_files": {
                "success": 'Found {count} results for "{query}" in chat files',
                "searching": 'Searched chat files for "{query}"',  # Used when 0 results but valid query
                "error": "Chat files search failed",
            },
            "web_search": {"success": "Found {count} web results", "error": "Web search failed"},
            "web_read": {"success": "Successfully read {count} pages", "error": "Failed to read web pages"},
            "unknown": {"success": "Operation completed", "error": "Operation failed"},
        },
        # Search type names for display
        "search_types": {
            "vector_search": "vector search",
            "graph_search": "graph search",
            "fulltext_search": "full-text search",
        },
        # Detailed information for tool results
        "details": {
            "collections_found": "Collections: {collection_names}",
            "search_results_detail": "{vector_count} vector, {graph_count} graph, {fulltext_count} full-text",
            "web_pages_read": "Pages: {page_titles}",
            "web_search_sources": "Sources: {domains}",
        },
    },
    "zh-CN": {
        # Tool display names
        "tool_names": {
            "list_collections": "获取集合列表",
            "search_collection": "搜索集合",
            "search_chat_files": "搜索聊天文件",
            "web_search": "网页搜索",
            "web_read": "读取网页",
            "unknown": "工具调用",
        },
        # Tool request descriptions (what the tool is doing)
        "requests": {
            "list_collections": "获取所有可用集合",
            "search_collection": "使用{search_types}搜索「{query}」，返回前 {topk} 条结果",
            "search_chat_files": "使用{search_types}搜索聊天文件中的「{query}」，返回前 {topk} 条结果",
            "web_search": "在网上搜索「{query}」，返回 {max_results} 条结果",
            "web_read": "读取 {count} 个网页的内容",
        },
        # Tool response summaries (what the tool accomplished)
        "responses": {
            "list_collections": {"success": "找到 {count} 个集合", "error": "获取集合失败"},
            "search_collection": {
                "success": "搜索「{query}」找到 {count} 条结果",
                "searching": "已搜索「{query}」",  # Used when 0 results but valid query
                "error": "搜索失败",
            },
            "search_chat_files": {
                "success": "在聊天文件中搜索「{query}」找到 {count} 条结果",
                "searching": "已在聊天文件中搜索「{query}」",  # Used when 0 results but valid query
                "error": "聊天文件搜索失败",
            },
            "web_search": {"success": "网页搜索找到 {count} 条结果", "error": "网页搜索失败"},
            "web_read": {"success": "成功读取 {count} 个网页", "error": "读取网页失败"},
            "unknown": {"success": "操作完成", "error": "操作失败"},
        },
        # Search type names for display
        "search_types": {"vector_search": "向量搜索", "graph_search": "图搜索", "fulltext_search": "全文搜索"},
        # Detailed information for tool results
        "details": {
            "collections_found": "集合：{collection_names}",
            "search_results_detail": "{vector_count} 个向量，{graph_count} 个图谱，{fulltext_count} 个全文",
            "web_pages_read": "页面：{page_titles}",
            "web_search_sources": "来源：{domains}",
        },
    },
}

# Error message translations for various agent errors
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
