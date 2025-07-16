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

import json
import uuid
from typing import Any, Dict, List
from aperag.utils.utils import now_unix_milliseconds


def format_stream_start(msg_id: str) -> Dict[str, Any]:
    """格式化流式开始事件"""
    return {
        "type": "start",
        "id": msg_id,
        "timestamp": now_unix_milliseconds(),
    }


def format_stream_content(msg_id: str, content: str) -> Dict[str, Any]:
    """格式化流式内容事件"""
    return {
        "type": "message",
        "id": msg_id,
        "data": content,
        "timestamp": now_unix_milliseconds(),
    }


def format_tool_call_content(msg_id: str, content: str) -> Dict[str, Any]:
    """格式化工具调用内容事件"""
    return {
        "type": "message",
        "id": msg_id,
        "data": f"<tool_call>{content}</tool_call>\n\n",
        "timestamp": now_unix_milliseconds(),
    }


def format_stream_end(msg_id: str, references: List[str] = None, urls: List[str] = None) -> Dict[str, Any]:
    """格式化流式结束事件"""
    if references is None:
        references = []
    if urls is None:
        urls = []

    return {
        "type": "stop",
        "id": msg_id,
        "data": references,
        "urls": urls,
        "timestamp": now_unix_milliseconds(),
    }


def format_error(error: str) -> Dict[str, Any]:
    """格式化错误响应"""
    return {
        "type": "error",
        "id": str(uuid.uuid4()),
        "data": error,
        "timestamp": now_unix_milliseconds(),
    }


def format_tool_arguments(tool_name: str, arguments: dict) -> str:
    """格式化工具参数显示"""
    if tool_name == "list_collections":
        return "获取所有集合列表"
    elif tool_name == "search_collection":
        collection_id = arguments.get("collection_id", "unknown")
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
            
        return f"在集合 '{collection_id}' 中搜索 '{query}'，使用 {'/'.join(search_types)}，返回 {topk} 条结果"
    elif tool_name == "web_search":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        search_engine = arguments.get("search_engine", "duckduckgo")
        return f"使用 {search_engine} 搜索 '{query}'，返回 {max_results} 条结果"
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
        
    # 检测 list_collections 接口
    if isinstance(structured_content, dict) and "items" in structured_content:
        items = structured_content["items"]
        if isinstance(items, list) and len(items) > 0:
            first_item = items[0]
            if isinstance(first_item, dict) and "title" in first_item and "config" in first_item:
                return "list_collections"
    
    # 检测 search_collection 接口
    if isinstance(structured_content, dict) and "query" in structured_content and "items" in structured_content:
        return "search_collection"
    
    # 检测 web_search 接口
    if isinstance(structured_content, dict) and "results" in structured_content:
        results = structured_content["results"]
        if isinstance(results, list) and len(results) > 0:
            first_result = results[0]
            if isinstance(first_result, dict) and "url" in first_result and "snippet" in first_result:
                return "web_search"
            elif isinstance(first_result, dict) and "content" in first_result:
                return "web_read"
    
    return "unknown"


def format_tool_request_display(tool_name: str, arguments: dict) -> str:
    """格式化工具请求的显示文本"""
    details = format_tool_arguments(tool_name, arguments)
    return f"🔧 调用工具: {tool_name}\n{details}"


def format_tool_response_display(interface_type: str, content, is_error: bool) -> str:
    """格式化工具响应的显示文本"""
    summary = format_tool_response_summary(interface_type, content, is_error)
    return f"✅ 工具响应: {interface_type}\n{summary}" 