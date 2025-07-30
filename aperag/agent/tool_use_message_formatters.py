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

from aperag.schema.view_models import CollectionList, SearchResult, WebReadResponse, WebSearchResponse

from .i18n import TOOL_USE_EVENT_MESSAGES


def get_i18n_messages(language: str) -> dict:
    """Get i18n messages for the specified language, fallback to en-US"""
    return TOOL_USE_EVENT_MESSAGES.get(language, TOOL_USE_EVENT_MESSAGES["en-US"])


class ToolResultFormatter:
    """Unified tool result formatter with simplified logic"""

    def __init__(self, language: str = "en-US", context: Optional[Dict[str, Any]] = None):
        self.language = language
        self.messages = get_i18n_messages(language)
        self.context = context or {}  # Store context like collection info

    def set_context(self, context: Dict[str, Any]):
        """Update context information"""
        self.context.update(context)

    def detect_and_parse_result(self, content: Any) -> Tuple[str, Optional[Any]]:
        """Detect interface type and parse typed result"""
        if not content or not isinstance(content, dict):
            return "unknown", None

        # Try to parse each type with clear criteria
        try:
            # SearchResult: has 'query' field
            if "query" in content:
                if "results" in content:
                    # WebSearchResponse: has both 'query' and 'results'
                    return "web_search", WebSearchResponse.model_validate(content)
                else:
                    # SearchResult: has 'query' but not 'results' (has 'items')
                    return "search_collection", SearchResult.model_validate(content)

            # CollectionList: has 'items' but no 'query'
            elif "items" in content:
                return "list_collections", CollectionList.model_validate(content)

            # WebReadResponse: has 'results' but no 'query', and has 'successful' field
            elif "results" in content and "successful" in content:
                return "web_read", WebReadResponse.model_validate(content)

        except Exception:
            pass

        return "unknown", None

    def should_display_result(self, interface_type: str, typed_result: Any, content: Any) -> bool:
        """Determine if result should be displayed"""
        if interface_type == "search_collection":
            result: SearchResult = typed_result
            return bool(result.query and result.query.strip())

        # For other types, always display if we have a valid result
        return typed_result is not None

    def format_tool_response(self, interface_type: str, typed_result: Any, content: Any, is_error: bool = False) -> str:
        """Format tool response using type-specific formatters"""
        if is_error:
            return self._format_error_response(interface_type)

        # Route to type-specific formatters
        if interface_type == "search_collection":
            return self._format_search_collection(typed_result)
        elif interface_type == "list_collections":
            return self._format_list_collections(typed_result)
        elif interface_type == "web_search":
            return self._format_web_search(typed_result)
        elif interface_type == "web_read":
            return self._format_web_read(typed_result)

        # Fallback for unknown types
        return self.messages["tool_names"].get(interface_type, interface_type)

    def _format_search_collection(self, result: SearchResult) -> str:
        """Format search collection result"""
        collection_name = self._extract_collection_name()
        query = result.query or ""

        # Count results by type
        vector_count = sum(1 for item in (result.items or []) if item.recall_type == "vector_search")
        graph_count = sum(1 for item in (result.items or []) if item.recall_type == "graph_search")
        fulltext_count = sum(1 for item in (result.items or []) if item.recall_type == "fulltext_search")
        total_count = len(result.items or [])

        # Determine search types used (infer from results)
        search_types = []
        if graph_count > 0:
            search_types.append(self.messages["search_types"]["graph_search"])
        if vector_count > 0:
            search_types.append(self.messages["search_types"]["vector_search"])
        if fulltext_count > 0:
            search_types.append(self.messages["search_types"]["fulltext_search"])

        # Default if no results
        if not search_types:
            if self.language == "zh-CN":
                search_types = ["向量搜索、图谱搜索"]
            else:
                search_types = ["vector and graph search"]

        search_methods = ", ".join(search_types)

        # Part 1: Search execution
        if self.language == "zh-CN":
            execution = f"**使用{search_methods}在{collection_name}中查找「{query}」**"
        else:
            execution = f'**Using {search_methods} to find "{query}" in {collection_name}**'

        # Part 2: Results (only if has results)
        if total_count == 0:
            if self.language == "zh-CN":
                results = "没有找到任何相关结果"
            else:
                results = "No relevant results found"
            return f"{execution}\n\n{results}"
        else:
            if self.language == "zh-CN":
                results = f"找到 {total_count} 条相关结果"
            else:
                results = f"Found {total_count} relevant results"

        # Add breakdown of result types (only non-zero)
        breakdown_parts = []
        if graph_count > 0:
            if self.language == "zh-CN":
                breakdown_parts.append(f"图谱搜索：{graph_count} 条")
            else:
                breakdown_parts.append(f"Graph: {graph_count}")
        if vector_count > 0:
            if self.language == "zh-CN":
                breakdown_parts.append(f"向量搜索：{vector_count} 条")
            else:
                breakdown_parts.append(f"Vector: {vector_count}")
        if fulltext_count > 0:
            if self.language == "zh-CN":
                breakdown_parts.append(f"全文搜索：{fulltext_count} 条")
            else:
                breakdown_parts.append(f"Full-text: {fulltext_count}")

        if breakdown_parts:
            results += "\n\n • " + "\n\n • ".join(breakdown_parts)

        return f"{execution}\n\n{results}"

    def _format_list_collections(self, result: CollectionList) -> str:
        """Format list collections result"""
        count = len(result.items or [])

        # Part 1: Action execution
        if self.language == "zh-CN":
            execution = "**正在搜索可用知识库**"
        else:
            execution = "**Searching available knowledge collections**"

        # Part 2: Results (only if has collections)
        if count == 0:
            if self.language == "zh-CN":
                results = "没有找到任何知识库"
            else:
                results = "No knowledge collections found"
        else:
            if self.language == "zh-CN":
                results = f"找到 {count} 个知识库"
            else:
                results = f"Found {count} knowledge collections"

        # Add collection names (first 5)
        if result.items:
            collection_names = [item.title or item.id or "Unknown" for item in result.items[:5]]
            if len(result.items) > 5:
                collection_names.append("...")

            if self.language == "zh-CN":
                results += "\n\n • " + "\n\n • ".join(collection_names)
            else:
                results += "\n\n • " + "\n\n • ".join(collection_names)

        return f"{execution}\n\n{results}"

    def _format_web_search(self, result: WebSearchResponse) -> str:
        """Format web search result"""
        query = result.query or ""
        count = len(result.results or [])

        # Part 1: Search execution
        if self.language == "zh-CN":
            execution = f"**在互联网上搜索「{query}」**"
        else:
            execution = f'**Searching the web for "{query}"**'

        # Part 2: Results (only if has results)
        if count == 0:
            if self.language == "zh-CN":
                results = "没有找到任何网页结果"
            else:
                results = "No web results found"
            return f"{execution}\n\n{results}"
        else:
            if self.language == "zh-CN":
                results = f"找到 {count} 个网页结果"
            else:
                results = f"Found {count} web results"

        # Add search results as markdown links (first 5)
        if result.results:
            links = []
            for item in result.results[:5]:
                # Create markdown link format [title](url)
                title = item.title or item.domain or "Untitled"
                url = item.url
                links.append(f"[{title}]({url})")

            if len(result.results) > 5:
                links.append("...")

            if self.language == "zh-CN":
                results += "\n\n • " + "\n\n • ".join(links)
            else:
                results += "\n\n • " + "\n\n • ".join(links)

        return f"{execution}\n\n{results}"

    def _format_web_read(self, result: WebReadResponse) -> str:
        """Format web read result"""
        total_count = result.total_urls or 0
        success_count = result.successful or 0

        # Part 1: Action execution
        if self.language == "zh-CN":
            execution = f"**读取 {total_count} 个网页的详细内容**"
        else:
            execution = f"**Reading detailed content from {total_count} web pages**"

        # Part 2: Results (only if has successful reads)
        if success_count == 0:
            if self.language == "zh-CN":
                results = "没有成功读取任何网页"
            else:
                results = "No web pages read successfully"
            return f"{execution}\n\n{results}"
        else:
            if self.language == "zh-CN":
                results = f"成功读取 {success_count} 个网页"
            else:
                results = f"Successfully read {success_count} web pages"

        # Add page links as markdown links (first 5 successful ones)
        if result.results:
            successful_results = [item for item in result.results if item.status == "success"]
            links = []
            for item in successful_results[:5]:
                # Create markdown link format [title](url)
                title = item.title or "Untitled"
                url = item.url
                links.append(f"[{title}]({url})")

            if len(successful_results) > 5:
                links.append("...")

            if links:
                if self.language == "zh-CN":
                    results += "\n\n • " + "\n\n • ".join(links)
                else:
                    results += "\n\n • " + "\n\n • ".join(links)

        return f"{execution}\n\n{results}"

    def _format_error_response(self, interface_type: str) -> str:
        """Format error response"""
        display_name = self.messages["tool_names"].get(interface_type, interface_type)
        error_msg = self.messages["responses"].get(interface_type, self.messages["responses"]["unknown"])["error"]
        return f"{display_name}\n\n{error_msg}"

    def _extract_collection_name(self, collection_id: Optional[str] = None) -> str:
        """Extract collection name from context or use default"""
        # Try to get from context first (passed by caller)
        if "current_collection" in self.context:
            collection = self.context["current_collection"]
            if isinstance(collection, dict):
                return collection.get("title") or collection.get("name") or collection.get("id", "当前知识库")
            elif hasattr(collection, "title") and collection.title:
                return collection.title
            elif hasattr(collection, "name") and collection.name:
                return collection.name
            elif hasattr(collection, "id") and collection.id:
                return collection.id

        # Try to find by collection_id in collections list
        if collection_id and "collections" in self.context:
            collections = self.context["collections"]
            if isinstance(collections, list):
                for collection in collections:
                    if isinstance(collection, dict):
                        if collection.get("id") == collection_id:
                            return collection.get("title") or collection.get("name") or collection_id
                    elif hasattr(collection, "id") and collection.id == collection_id:
                        return getattr(collection, "title", None) or getattr(collection, "name", None) or collection_id

        # Try to get collection_id from arguments (stored in context)
        if "tool_arguments" in self.context:
            args = self.context["tool_arguments"]
            if isinstance(args, dict) and "collection_id" in args:
                collection_id = args["collection_id"]
                # Try to find this collection in the context
                if "collections" in self.context:
                    collections = self.context["collections"]
                    if isinstance(collections, list):
                        for collection in collections:
                            if isinstance(collection, dict) and collection.get("id") == collection_id:
                                return collection.get("title") or collection.get("name") or collection_id
                            elif hasattr(collection, "id") and collection.id == collection_id:
                                return (
                                    getattr(collection, "title", None)
                                    or getattr(collection, "name", None)
                                    or collection_id
                                )
                # If not found in collections list, use the ID itself
                return collection_id

        # Fallback to default
        if self.language == "zh-CN":
            return "当前知识库"
        else:
            return "current knowledge base"


# Legacy functions for backward compatibility
def detect_interface_type(structured_content):
    """Legacy function - detect interface type and return typed result"""
    formatter = ToolResultFormatter()
    interface_type, typed_result = formatter.detect_and_parse_result(structured_content)
    return interface_type, typed_result


def format_tool_request_display(tool_name: str, arguments: dict, language: str = "en-US") -> str:
    """Legacy function - format tool request display"""
    messages = get_i18n_messages(language)

    display_name = messages["tool_names"].get(tool_name, tool_name)

    if tool_name == "list_collections":
        details = messages["requests"]["list_collections"]
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

        details = messages["requests"]["search_collection"].format(
            query=query, search_types="/".join(search_types), topk=topk
        )
    elif tool_name == "web_search":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        details = messages["requests"]["web_search"].format(query=query, max_results=max_results)
    elif tool_name == "web_read":
        url_list = arguments.get("url_list", [])
        details = messages["requests"]["web_read"].format(count=len(url_list))
    else:
        details = f"Arguments: {json.dumps(arguments, ensure_ascii=False)}"

    return f"{display_name}\n\n{details}"


def format_tool_use_response(language: str, interface_type: str, typed_result: Any, is_error: bool) -> str:
    """Legacy function - format tool response"""
    formatter = ToolResultFormatter(language)
    return formatter.format_tool_response(interface_type, typed_result, None, is_error)
