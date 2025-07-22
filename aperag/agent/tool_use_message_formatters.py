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
from typing import Any

from aperag.utils.utils import now_unix_milliseconds

from .i18n import TOOL_USE_EVENT_MESSAGES
from .response_types import AgentMessageResponse


def format_tool_call_content(msg_id: str, content: str) -> AgentMessageResponse:
    """格式化工具调用内容事件"""
    return AgentMessageResponse(
        type="message",
        id=msg_id,
        data=f"<tool_call>{content}</tool_call>\n\n",
        timestamp=now_unix_milliseconds(),
    )


def format_tool_call_start(msg_id: str, data: str, tool_name: str, arguments: dict) -> AgentMessageResponse:
    return AgentMessageResponse(
        type="message",  # todo: change to tool_call_start
        id=msg_id,
        data=f"<tool_call_start>{data}</tool_call_start>\n\n",  # todo: remove format
        timestamp=now_unix_milliseconds(),
    )


def format_tool_call_end(msg_id: str, data: str, tool_name: str, result: Any) -> AgentMessageResponse:
    return AgentMessageResponse(
        type="message",  # todo: change to tool_call_end
        id=msg_id,
        data=f"<tool_call_end>{data}</tool_call_end>\n\n",  # todo: remove format
        timestamp=now_unix_milliseconds(),
    )


def get_i18n_messages(language: str) -> dict:
    """Get i18n messages for the specified language, fallback to en-US"""
    return TOOL_USE_EVENT_MESSAGES.get(language, TOOL_USE_EVENT_MESSAGES["en-US"])


def format_tool_arguments(tool_name: str, arguments: dict, language: str = "en-US") -> str:
    """Format tool arguments display with i18n support"""
    messages = get_i18n_messages(language)

    if tool_name == "list_collections":
        return messages["requests"]["list_collections"]

    elif tool_name == "search_collection":
        query = arguments.get("query", "")
        use_vector = arguments.get("use_vector_index", True)
        use_graph = arguments.get("use_graph_index", True)
        use_fulltext = arguments.get("use_fulltext_index", False)
        topk = arguments.get("topk", 5)

        search_types = []
        if use_vector:
            search_types.append(messages["search_types"]["vector_search"])
        if use_graph:
            search_types.append(messages["search_types"]["graph_search"])
        if use_fulltext:
            search_types.append(messages["search_types"]["fulltext_search"])

        return messages["requests"]["search_collection"].format(
            query=query, search_types="/".join(search_types), topk=topk
        )

    elif tool_name == "web_search":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        return messages["requests"]["web_search"].format(query=query, max_results=max_results)

    elif tool_name == "web_read":
        url_list = arguments.get("url_list", [])
        return messages["requests"]["web_read"].format(count=len(url_list))

    else:
        return f"Arguments: {json.dumps(arguments, ensure_ascii=False)}"


def format_tool_response_summary_with_typed_content(
    interface_type: str, typed_result: Any, is_error: bool, language: str = "en-US"
) -> str:
    """Format tool response summary using strongly typed results"""
    messages = get_i18n_messages(language)

    if is_error:
        return messages["responses"][interface_type]["error"]

    if interface_type == "list_collections":
        from aperag.schema.view_models import CollectionList

        if isinstance(typed_result, CollectionList) and typed_result.items:
            count = len(typed_result.items)
            return messages["responses"]["list_collections"]["success"].format(count=count)

    elif interface_type == "search_collection":
        from aperag.schema.view_models import SearchResult

        if isinstance(typed_result, SearchResult):
            count = len(typed_result.items) if typed_result.items else 0
            query = typed_result.query or ""

            # If no results but valid query, show search action instead of "Found 0 results"
            if count == 0 and query.strip():
                return messages["responses"]["search_collection"]["searching"].format(query=query)
            else:
                return messages["responses"]["search_collection"]["success"].format(count=count, query=query)

    elif interface_type == "web_search":
        from aperag.schema.view_models import WebSearchResponse

        if isinstance(typed_result, WebSearchResponse):
            count = len(typed_result.results)
            return messages["responses"]["web_search"]["success"].format(count=count)

    elif interface_type == "web_read":
        from aperag.schema.view_models import WebReadResponse

        if isinstance(typed_result, WebReadResponse):
            count = typed_result.successful
            return messages["responses"]["web_read"]["success"].format(count=count)

    return messages["responses"]["unknown"]["success"]


def format_tool_response_summary(interface_type: str, content, is_error: bool, language: str = "en-US") -> str:
    """Legacy format tool response summary - kept for backward compatibility"""
    messages = get_i18n_messages(language)

    if is_error:
        return messages["responses"].get(interface_type, messages["responses"]["unknown"])["error"]

    if interface_type == "list_collections":
        if isinstance(content, dict) and "items" in content:
            count = len(content["items"])
            return messages["responses"]["list_collections"]["success"].format(count=count)
    elif interface_type == "search_collection":
        if isinstance(content, dict) and "items" in content:
            count = len(content["items"])
            query = content.get("query", "")
            return messages["responses"]["search_collection"]["success"].format(count=count, query=query)
    elif interface_type == "web_search":
        if isinstance(content, dict) and "results" in content:
            count = len(content["results"])
            return messages["responses"]["web_search"]["success"].format(count=count)
    elif interface_type == "web_read":
        if isinstance(content, dict) and "results" in content:
            count = len(content["results"])
            return messages["responses"]["web_read"]["success"].format(count=count)

    return messages["responses"]["unknown"]["success"]


def detect_interface_type(structured_content):
    """根据响应内容检测接口类型"""
    if not structured_content:
        return "unknown", None

    if not isinstance(structured_content, dict):
        return "unknown", None

    try:
        from aperag.schema.view_models import SearchResult

        result = SearchResult.model_validate(structured_content)
        if result and isinstance(result, SearchResult):
            return "search_collection", result
    except Exception:
        pass

    try:
        from aperag.schema.view_models import CollectionList

        result = CollectionList.model_validate(structured_content)
        if result and isinstance(result, CollectionList):
            return "list_collections", result
    except Exception:
        pass

    try:
        from aperag.schema.view_models import WebSearchResponse

        result = WebSearchResponse.model_validate(structured_content)
        if result and isinstance(result, WebSearchResponse):
            return "web_search", result
    except Exception:
        pass

    try:
        from aperag.schema.view_models import WebReadResponse

        result = WebReadResponse.model_validate(structured_content)
        if result and isinstance(result, WebReadResponse):
            return "web_read", result
    except Exception:
        pass

    return "unknown", None


def format_tool_request_display(tool_name: str, arguments: dict, language: str = "en-US") -> str:
    """Format tool request display text with i18n support"""
    messages = get_i18n_messages(language)
    details = format_tool_arguments(tool_name, arguments, language)

    display_name = messages["tool_names"].get(tool_name, tool_name)
    return f"{display_name}\n{details}"


def format_enhanced_tool_details(interface_type: str, typed_result: Any, language: str = "en-US") -> str:
    """Format enhanced tool details using strongly typed results"""
    messages = get_i18n_messages(language)

    try:
        if interface_type == "list_collections":
            from aperag.schema.view_models import CollectionList

            if isinstance(typed_result, CollectionList) and typed_result.items:
                collection_names = [item.title or item.id or "Unknown" for item in typed_result.items[:3]]
                if len(typed_result.items) > 3:
                    collection_names.append("...")
                return messages["details"]["collections_found"].format(collection_names=", ".join(collection_names))

        elif interface_type == "search_collection":
            from aperag.schema.view_models import SearchResult

            if isinstance(typed_result, SearchResult) and typed_result.items:
                # Count results by recall type
                vector_count = sum(1 for item in typed_result.items if item.recall_type == "vector_search")
                graph_count = sum(1 for item in typed_result.items if item.recall_type == "graph_search")
                fulltext_count = sum(1 for item in typed_result.items if item.recall_type == "fulltext_search")

                # Only show details if we have meaningful counts (not all zeros)
                if vector_count > 0 or graph_count > 0 or fulltext_count > 0:
                    return messages["details"]["search_results_detail"].format(
                        vector_count=vector_count, graph_count=graph_count, fulltext_count=fulltext_count
                    )

        elif interface_type == "web_search":
            from aperag.schema.view_models import WebSearchResponse

            if isinstance(typed_result, WebSearchResponse):
                domains = list(set([result.domain for result in typed_result.results[:5]]))
                return messages["details"]["web_search_sources"].format(domains=", ".join(domains))

        elif interface_type == "web_read":
            from aperag.schema.view_models import WebReadResponse

            if isinstance(typed_result, WebReadResponse):
                page_titles = [
                    result.title or result.url for result in typed_result.results if result.status == "success"
                ][:3]
                if len(page_titles) > 3:
                    page_titles.append("...")
                return messages["details"]["web_pages_read"].format(page_titles=", ".join(page_titles))

    except Exception:
        pass

    return ""


def format_tool_use_response(language: str, interface_type: str, typed_result: Any, is_error: bool) -> str:
    """Enhanced tool response formatter using strongly typed results and detailed information"""
    messages = get_i18n_messages(language)

    # Get main summary
    summary = format_tool_response_summary_with_typed_content(interface_type, typed_result, is_error, language)

    # Get display name
    display_name = messages["tool_names"].get(interface_type, interface_type)

    # Get enhanced details
    details = format_enhanced_tool_details(interface_type, typed_result, language)

    result = f"{display_name}\n{summary}"
    if details:
        result += f"\n{details}"
    return result
