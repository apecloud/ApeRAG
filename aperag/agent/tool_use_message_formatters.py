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
from typing import Any, Dict, Optional, Tuple

from aperag.utils.utils import now_unix_milliseconds

from .i18n import TOOL_USE_EVENT_MESSAGES
from .response_types import AgentMessageResponse


def format_tool_call_content(msg_id: str, content: str) -> AgentMessageResponse:
    """Format tool call content event"""
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


class ToolResultFormatter:
    """Unified tool result formatter with simplified logic"""

    def __init__(self, language: str = "en-US"):
        self.language = language
        self.messages = get_i18n_messages(language)

    def detect_and_parse_result(self, content: Any) -> Tuple[str, Optional[Any]]:
        """Detect interface type and parse typed result"""
        if not content or not isinstance(content, dict):
            return "unknown", None

        # Try to parse different result types
        parsers = [
            ("search_collection", self._parse_search_result),
            ("list_collections", self._parse_collection_list),
            ("web_search", self._parse_web_search),
            ("web_read", self._parse_web_read),
        ]

        for interface_type, parser in parsers:
            try:
                result = parser(content)
                if result:
                    return interface_type, result
            except Exception:
                continue

        return "unknown", None

    def _parse_search_result(self, content: dict):
        """Parse search result"""
        from aperag.schema.view_models import SearchResult

        return SearchResult.model_validate(content)

    def _parse_collection_list(self, content: dict):
        """Parse collection list"""
        from aperag.schema.view_models import CollectionList

        return CollectionList.model_validate(content)

    def _parse_web_search(self, content: dict):
        """Parse web search result"""
        from aperag.schema.view_models import WebSearchResponse

        return WebSearchResponse.model_validate(content)

    def _parse_web_read(self, content: dict):
        """Parse web read result"""
        from aperag.schema.view_models import WebReadResponse

        return WebReadResponse.model_validate(content)

    def should_display_result(self, interface_type: str, typed_result: Any, content: Any) -> bool:
        """Simplified logic to determine if result should be displayed"""
        # Always display search actions even with 0 results if query is valid
        if interface_type == "search_collection":
            if typed_result:
                from aperag.schema.view_models import SearchResult

                if isinstance(typed_result, SearchResult):
                    return bool(typed_result.query and typed_result.query.strip())
            elif isinstance(content, dict):
                query = content.get("query", "")
                return bool(query and query.strip())

        # For other types, display if we have meaningful results
        return self._has_meaningful_results(interface_type, typed_result, content)

    def _has_meaningful_results(self, interface_type: str, typed_result: Any, content: Any) -> bool:
        """Check if result has meaningful content to display"""
        if interface_type == "list_collections":
            if typed_result:
                from aperag.schema.view_models import CollectionList

                if isinstance(typed_result, CollectionList):
                    return bool(typed_result.items)
            elif isinstance(content, dict):
                return bool(content.get("items"))

        elif interface_type == "web_search":
            if typed_result:
                from aperag.schema.view_models import WebSearchResponse

                if isinstance(typed_result, WebSearchResponse):
                    return bool(typed_result.results)
            elif isinstance(content, dict):
                return bool(content.get("results"))

        elif interface_type == "web_read":
            if typed_result:
                from aperag.schema.view_models import WebReadResponse

                if isinstance(typed_result, WebReadResponse):
                    return typed_result.successful > 0
            elif isinstance(content, dict):
                return content.get("successful", 0) > 0

        return True

    def format_tool_request(self, tool_name: str, arguments: dict) -> str:
        """Format tool request display"""
        display_name = self.messages["tool_names"].get(tool_name, tool_name)
        details = self._format_request_details(tool_name, arguments)
        return f"{display_name}\n{details}"

    def _format_request_details(self, tool_name: str, arguments: dict) -> str:
        """Format tool request details"""
        if tool_name == "list_collections":
            return self.messages["requests"]["list_collections"]

        elif tool_name == "search_collection":
            query = arguments.get("query", "")
            use_vector = arguments.get("use_vector_index", True)
            use_graph = arguments.get("use_graph_index", True)
            use_fulltext = arguments.get("use_fulltext_index", False)
            topk = arguments.get("topk", 5)

            search_types = []
            if use_vector:
                search_types.append(self.messages["search_types"]["vector_search"])
            if use_graph:
                search_types.append(self.messages["search_types"]["graph_search"])
            if use_fulltext:
                search_types.append(self.messages["search_types"]["fulltext_search"])

            return self.messages["requests"]["search_collection"].format(
                query=query, search_types="/".join(search_types), topk=topk
            )

        elif tool_name == "web_search":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)
            return self.messages["requests"]["web_search"].format(query=query, max_results=max_results)

        elif tool_name == "web_read":
            url_list = arguments.get("url_list", [])
            return self.messages["requests"]["web_read"].format(count=len(url_list))

        else:
            return f"Arguments: {json.dumps(arguments, ensure_ascii=False)}"

    def format_tool_response(self, interface_type: str, typed_result: Any, content: Any, is_error: bool = False) -> str:
        """Format complete tool response with summary and details"""
        if is_error:
            return self._format_error_response(interface_type)

        display_name = self.messages["tool_names"].get(interface_type, interface_type)
        summary = self._format_response_summary(interface_type, typed_result, content)
        details = self._format_response_details(interface_type, typed_result, content)

        result = f"{display_name}\n{summary}"
        if details:
            result += f"\n{details}"
        return result

    def _format_error_response(self, interface_type: str) -> str:
        """Format error response"""
        display_name = self.messages["tool_names"].get(interface_type, interface_type)
        error_msg = self.messages["responses"].get(interface_type, self.messages["responses"]["unknown"])["error"]
        return f"{display_name}\n{error_msg}"

    def _format_response_summary(self, interface_type: str, typed_result: Any, content: Any) -> str:
        """Format response summary"""
        response_config = self.messages["responses"].get(interface_type, self.messages["responses"]["unknown"])

        if interface_type == "list_collections":
            count = self._get_collection_count(typed_result, content)
            return response_config["success"].format(count=count)

        elif interface_type == "search_collection":
            count, query = self._get_search_info(typed_result, content)
            # Show search action for valid queries even with 0 results
            if count == 0 and query.strip():
                return response_config["searching"].format(query=query)
            else:
                return response_config["success"].format(count=count, query=query)

        elif interface_type == "web_search":
            count = self._get_web_search_count(typed_result, content)
            return response_config["success"].format(count=count)

        elif interface_type == "web_read":
            count = self._get_web_read_count(typed_result, content)
            return response_config["success"].format(count=count)

        return response_config["success"]

    def _format_response_details(self, interface_type: str, typed_result: Any, content: Any) -> str:
        """Format detailed response information"""
        try:
            if interface_type == "list_collections" and self._get_collection_count(typed_result, content) > 0:
                collection_names = self._get_collection_names(typed_result, content)
                if collection_names:
                    return self.messages["details"]["collections_found"].format(
                        collection_names=", ".join(collection_names)
                    )

            elif interface_type == "search_collection":
                detail_info = self._get_search_detail_info(typed_result, content)
                if detail_info:
                    return self.messages["details"]["search_results_detail"].format(**detail_info)

            elif interface_type == "web_search":
                domains = self._get_web_search_domains(typed_result, content)
                if domains:
                    return self.messages["details"]["web_search_sources"].format(domains=", ".join(domains))

            elif interface_type == "web_read":
                page_titles = self._get_web_read_titles(typed_result, content)
                if page_titles:
                    return self.messages["details"]["web_pages_read"].format(page_titles=", ".join(page_titles))

        except Exception:
            pass

        return ""

    # Helper methods for extracting information
    def _get_collection_count(self, typed_result: Any, content: Any) -> int:
        """Get collection count"""
        if typed_result:
            from aperag.schema.view_models import CollectionList

            if isinstance(typed_result, CollectionList):
                return len(typed_result.items) if typed_result.items else 0
        elif isinstance(content, dict):
            items = content.get("items", [])
            return len(items) if items else 0
        return 0

    def _get_collection_names(self, typed_result: Any, content: Any) -> list:
        """Get collection names for display"""
        if typed_result:
            from aperag.schema.view_models import CollectionList

            if isinstance(typed_result, CollectionList) and typed_result.items:
                names = [item.title or item.id or "Unknown" for item in typed_result.items[:3]]
                if len(typed_result.items) > 3:
                    names.append("...")
                return names
        return []

    def _get_search_info(self, typed_result: Any, content: Any) -> Tuple[int, str]:
        """Get search count and query"""
        if typed_result:
            from aperag.schema.view_models import SearchResult

            if isinstance(typed_result, SearchResult):
                count = len(typed_result.items) if typed_result.items else 0
                query = typed_result.query or ""
                return count, query
        elif isinstance(content, dict):
            items = content.get("items", [])
            count = len(items) if items else 0
            query = content.get("query", "")
            return count, query
        return 0, ""

    def _get_search_detail_info(self, typed_result: Any, content: Any) -> Optional[Dict[str, int]]:
        """Get search result detail breakdown"""
        if typed_result:
            from aperag.schema.view_models import SearchResult

            if isinstance(typed_result, SearchResult) and typed_result.items:
                vector_count = sum(1 for item in typed_result.items if item.recall_type == "vector_search")
                graph_count = sum(1 for item in typed_result.items if item.recall_type == "graph_search")
                fulltext_count = sum(1 for item in typed_result.items if item.recall_type == "fulltext_search")

                if vector_count > 0 or graph_count > 0 or fulltext_count > 0:
                    return {"vector_count": vector_count, "graph_count": graph_count, "fulltext_count": fulltext_count}
        return None

    def _get_web_search_count(self, typed_result: Any, content: Any) -> int:
        """Get web search result count"""
        if typed_result:
            from aperag.schema.view_models import WebSearchResponse

            if isinstance(typed_result, WebSearchResponse):
                return len(typed_result.results)
        elif isinstance(content, dict):
            results = content.get("results", [])
            return len(results) if results else 0
        return 0

    def _get_web_search_domains(self, typed_result: Any, content: Any) -> list:
        """Get web search domains"""
        if typed_result:
            from aperag.schema.view_models import WebSearchResponse

            if isinstance(typed_result, WebSearchResponse):
                domains = list(set([result.domain for result in typed_result.results[:5]]))
                return domains
        return []

    def _get_web_read_count(self, typed_result: Any, content: Any) -> int:
        """Get successful web read count"""
        if typed_result:
            from aperag.schema.view_models import WebReadResponse

            if isinstance(typed_result, WebReadResponse):
                return typed_result.successful
        elif isinstance(content, dict):
            return content.get("successful", 0)
        return 0

    def _get_web_read_titles(self, typed_result: Any, content: Any) -> list:
        """Get web read page titles"""
        if typed_result:
            from aperag.schema.view_models import WebReadResponse

            if isinstance(typed_result, WebReadResponse):
                titles = [result.title or result.url for result in typed_result.results if result.status == "success"][
                    :3
                ]
                if len([r for r in typed_result.results if r.status == "success"]) > 3:
                    titles.append("...")
                return titles
        return []


# Legacy functions for backward compatibility
def detect_interface_type(structured_content):
    """Legacy function - detect interface type and return typed result"""
    formatter = ToolResultFormatter()
    interface_type, typed_result = formatter.detect_and_parse_result(structured_content)
    return interface_type, typed_result


def format_tool_request_display(tool_name: str, arguments: dict, language: str = "en-US") -> str:
    """Legacy function - format tool request display"""
    formatter = ToolResultFormatter(language)
    return formatter.format_tool_request(tool_name, arguments)


def format_tool_use_response(language: str, interface_type: str, typed_result: Any, is_error: bool) -> str:
    """Legacy function - format tool response"""
    formatter = ToolResultFormatter(language)
    return formatter.format_tool_response(interface_type, typed_result, None, is_error)
