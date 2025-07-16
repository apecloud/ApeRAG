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

"""Tool call formatters for agent events."""

import json
from typing import Any, Dict

from aperag.utils.utils import now_unix_milliseconds


def format_tool_call_content(msg_id: str, content: str) -> Dict[str, Any]:
    """格式化工具调用内容事件"""
    return {
        "type": "message",
        "id": msg_id,
        "data": f"<tool_call>{content}</tool_call>\n\n",
        "timestamp": now_unix_milliseconds(),
    }


def format_tool_call_start(msg_id: str, data: str, tool_name: str, arguments: dict) -> Dict[str, Any]:
    return {
        "type": "message",  # todo: change to tool_call_start
        "id": msg_id,
        "data": f"<tool_call_start>{data}</tool_call_start>\n\n",  # todo: remove format
        "tool_name": tool_name,
        "arguments": arguments,
        "timestamp": now_unix_milliseconds(),
    }


def format_tool_call_end(msg_id: str, data: str, tool_name: str, result: Any) -> Dict[str, Any]:
    return {
        "type": "message",  # todo: change to tool_call_end
        "id": msg_id,
        "data": f"<tool_call_end>{data}</tool_call_end>\n\n",  # todo: remove format
        "tool_name": tool_name,
        "result": result,
        "timestamp": now_unix_milliseconds(),
    }


def format_tool_arguments(tool_name: str, arguments: dict) -> str:
    """格式化工具参数显示"""
    if tool_name == "list_collections":
        return "获取所有集合列表"
    elif tool_name == "search_collection":
        query = arguments.get("query", "")
        use_vector = arguments.get("use_vector_index", True)
        use_graph = arguments.get("use_graph_index", True)
        use_fulltext = arguments.get("use_fulltext_index", False)
        topk = arguments.get("topk", 5)

        search_types = []
        if use_vector:
            search_types.append("向量搜索")
        if use_graph:
            search_types.append("图搜索")
        if use_fulltext:
            search_types.append("全文搜索")

        return f"在知识库中搜索「{query}」，使用 {'/'.join(search_types)}，返回 {topk} 条结果"
    elif tool_name == "web_search":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        return f"搜索「{query}」，返回 {max_results} 条结果"
    elif tool_name == "web_read":
        url_list = arguments.get("url_list", [])
        return f"读取 {len(url_list)} 个网页内容"
    else:
        return f"参数: {json.dumps(arguments, ensure_ascii=False)}"


def format_tool_response_summary(interface_type: str, content, is_error: bool) -> str:
    """格式化工具响应摘要"""
    if is_error:
        return "❌ 调用失败"

    if interface_type == "list_collections":
        if isinstance(content, dict) and "items" in content:
            count = len(content["items"])
            return f"找到 {count} 个集合"
    elif interface_type == "search_collection":
        if isinstance(content, dict) and "items" in content:
            count = len(content["items"])
            query = content.get("query", "")
            return f"搜索 '{query}' 找到 {count} 条结果"
    elif interface_type == "web_search":
        if isinstance(content, dict) and "results" in content:
            count = len(content["results"])
            return f"网页搜索找到 {count} 条结果"
    elif interface_type == "web_read":
        if isinstance(content, dict) and "results" in content:
            count = len(content["results"])
            return f"成功读取 {count} 个网页"

    return "✅ 调用成功"


def detect_interface_type(structured_content):
    """根据响应内容检测接口类型"""
    if not structured_content:
        return "unknown"

    if not isinstance(structured_content, dict):
        return "unknown"

    # 检测 search_collection 接口 - 优先检测，因为它有明确的query字段
    if "query" in structured_content and "items" in structured_content:
        return "search_collection"

    # 检测 list_collections 接口
    if "items" in structured_content:
        items = structured_content["items"]
        if isinstance(items, list) and len(items) > 0:
            first_item = items[0]
            if isinstance(first_item, dict) and "title" in first_item and "config" in first_item:
                return "list_collections"

    # 检测 web_search 和 web_read 接口
    if "results" in structured_content:
        results = structured_content["results"]
        if isinstance(results, list):
            # 即使results为空也认为是web_search/web_read
            if len(results) == 0:
                return "web_search"  # 默认为web_search

            first_result = results[0]
            if isinstance(first_result, dict):
                # 检测web_read: 有content字段
                if "content" in first_result:
                    return "web_read"
                # 检测web_search: 有url字段（snippet可选）
                elif "url" in first_result:
                    return "web_search"
                # 宽松检测：只要有results数组就认为是web_search
                else:
                    return "web_search"

    return "unknown"


def format_tool_request_display(tool_name: str, arguments: dict) -> str:
    """格式化工具请求的显示文本"""
    details = format_tool_arguments(tool_name, arguments)

    # 友好的工具名显示
    tool_names = {
        "list_collections": "获取集合列表",
        "search_collection": "搜索集合",
        "web_search": "网页搜索",
        "web_read": "读取网页",
    }

    display_name = tool_names.get(tool_name, tool_name)
    return f"🔧 {display_name}\n{details}"


def format_tool_response_display(interface_type: str, content, is_error: bool) -> str:
    """格式化工具响应的显示文本"""
    summary = format_tool_response_summary(interface_type, content, is_error)

    # 友好的接口类型显示
    interface_names = {
        "list_collections": "获取集合列表",
        "search_collection": "搜索集合",
        "web_search": "网页搜索",
        "web_read": "读取网页",
        "unknown": "工具调用",
    }

    display_name = interface_names.get(interface_type, interface_type)
    return f"✅ {display_name}\n{summary}"
