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

"""Tool call reference extractor for agent conversations."""

import json
import logging
from typing import Any, Dict, List, Optional

from .exceptions import (
    JSONParsingError,
    ToolReferenceExtractionError,
    handle_agent_error,
    safe_json_parse,
)

logger = logging.getLogger(__name__)


@handle_agent_error("tool_call_reference_extraction", default_return=[], reraise=False)
def extract_tool_call_references(memory) -> List[Dict[str, Any]]:
    """
    Extract tool call results from MCP agent history and format as references.

    Args:
        memory: SimpleMemory instance containing agent history

    Returns:
        List of reference dictionaries in the format expected by llm.py
    """
    references = []

    # Get history from memory
    history_messages = memory.get() if hasattr(memory, "get") else []
    if not history_messages:
        logger.debug("No history messages found in memory")
        return references

    for message in history_messages:
        # Check if message has tool calls (message is a dict)
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                try:
                    # Debug: log the actual structure
                    logger.debug(f"Tool call structure: {tool_call}, type: {type(tool_call)}")

                    # Process tool call information
                    # Handle different tool call structures (dict vs object)
                    tool_name = "unknown_tool"
                    tool_args = "{}"
                    tool_call_id = ""

                    # Handle OpenAI ChatCompletionMessageToolCall objects
                    if hasattr(tool_call, "id"):
                        tool_call_id = tool_call.id
                        if hasattr(tool_call, "function"):
                            tool_name = (
                                tool_call.function.name if hasattr(tool_call.function, "name") else "unknown_tool"
                            )
                            tool_args = (
                                tool_call.function.arguments if hasattr(tool_call.function, "arguments") else "{}"
                            )
                    # Handle dictionary format
                    elif isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id", "")
                        if "function" in tool_call:
                            tool_name = tool_call["function"].get("name", "unknown_tool")
                            tool_args = tool_call["function"].get("arguments", "{}")
                        elif "name" in tool_call:
                            tool_name = tool_call.get("name", "unknown_tool")
                            tool_args = tool_call.get("arguments", "{}")
                        elif "type" in tool_call and tool_call["type"] == "function":
                            tool_name = tool_call.get("function", {}).get("name", "unknown_tool")
                            tool_args = tool_call.get("function", {}).get("arguments", "{}")

                    logger.debug(
                        f"Extracted tool_name: {tool_name}, tool_args: {tool_args}, tool_call_id: {tool_call_id}"
                    )

                    # Parse tool arguments using safe parsing
                    try:
                        args_dict = (
                            safe_json_parse(tool_args, f"tool_args_{tool_name}")
                            if isinstance(tool_args, str)
                            else tool_args
                        )
                    except JSONParsingError:
                        logger.warning(f"Failed to parse tool arguments for {tool_name}, using raw args")
                        args_dict = {"raw_args": tool_args}

                    # Find corresponding tool result message
                    tool_result = _find_tool_result(history_messages, tool_call_id)

                    if tool_result:
                        # Format reference based on tool type
                        ref = None
                        try:
                            if tool_name == "aperag_search_collection":
                                ref = _format_search_reference(tool_result, args_dict)
                            elif tool_name == "aperag_search_chat_files":
                                ref = _format_search_chat_files_reference(tool_result, args_dict)
                            elif tool_name == "aperag_list_collections":
                                ref = _format_list_reference(tool_result, args_dict)
                            elif tool_name == "aperag_web_search":
                                ref = _format_web_search_reference(tool_result, args_dict)
                            elif tool_name == "aperag_web_read":
                                ref = _format_web_read_reference(tool_result, args_dict)
                            else:
                                # Generic tool result reference
                                ref = _format_generic_reference(tool_name, tool_result, args_dict)

                            if ref:
                                references.append(ref)

                        except (JSONParsingError, ToolReferenceExtractionError) as e:
                            logger.warning(f"Failed to format reference for tool {tool_name}: {e}")
                            continue

                except Exception as e:
                    logger.warning(f"Error processing individual tool call: {e}")
                    continue

    return references


def _find_tool_result(messages, tool_call_id: str) -> Optional[str]:
    """Find the tool result message for a given tool call ID"""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool" and message.get("tool_call_id") == tool_call_id:
            content = message.get("content", "")
            logger.debug(f"Found tool result for {tool_call_id}: {type(content)} - {content}")

            # Handle both string and list content
            if isinstance(content, list):
                return json.dumps(content)
            return content
    return None


def _format_search_reference(tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format search_collection tool result as reference"""
    try:
        # Parse tool result - handle both string and already parsed data
        if isinstance(tool_result, str):
            try:
                result_data = json.loads(tool_result)
            except json.JSONDecodeError:
                result_data = {"raw_result": tool_result}
        else:
            result_data = tool_result

        logger.debug(f"Search reference result_data: {result_data}")

        # Handle array format where data is in first element's text field
        if isinstance(result_data, list) and len(result_data) > 0:
            first_item = result_data[0]
            if isinstance(first_item, dict) and "text" in first_item:
                try:
                    # Parse the text field as JSON
                    text_data = json.loads(first_item["text"])
                    result_data = text_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                    return None

        # Extract search parameters
        collection_id = args.get("collection_id", "unknown")
        query = args.get("query", "")

        # Format search results
        if "items" in result_data:
            items = result_data["items"]
            if items:
                # Combine all search results into a single reference
                combined_text = ""
                combined_metadata = {
                    "type": "search_collection",
                    "collection_id": collection_id,
                    "query": query,
                    "result_count": len(items),
                }

                for item in items:
                    content = item.get("content", "")
                    metadata = item.get("metadata", {})
                    combined_text += f"Document: {metadata.get('source', 'Untitled')}\n\n"
                    combined_text += f"Content: {content}\n\n"

                    if metadata.get("asset_id") and metadata.get("document_id") and metadata.get("collection_id"):
                        asset_url = f"asset://{metadata.get('asset_id')}?document_id={metadata.get('document_id')}&collection_id={metadata.get('collection_id')}"
                        if metadata.get("mimetype"):
                            asset_url = asset_url + "&mime_type=" + metadata.get("mimetype")
                        combined_text += f"![]({asset_url})\n\n"

                return {
                    "text": combined_text.strip(),
                    "metadata": combined_metadata,
                    "score": 1.0,  # Default score for search results
                }

        return None

    except Exception as e:
        logger.error(f"Error formatting search reference: {e}")
        return None


def _format_search_chat_files_reference(tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format search_chat_files tool result as reference"""
    try:
        # Parse tool result - handle both string and already parsed data
        if isinstance(tool_result, str):
            try:
                result_data = json.loads(tool_result)
            except json.JSONDecodeError:
                result_data = {"raw_result": tool_result}
        else:
            result_data = tool_result

        logger.debug(f"Search chat files reference result_data: {result_data}")

        # Handle array format where data is in first element's text field
        if isinstance(result_data, list) and len(result_data) > 0:
            first_item = result_data[0]
            if isinstance(first_item, dict) and "text" in first_item:
                try:
                    # Parse the text field as JSON
                    text_data = json.loads(first_item["text"])
                    result_data = text_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                    return None

        # Extract search parameters
        chat_id = args.get("chat_id", "unknown")
        query = args.get("query", "")

        # Format search results
        if "items" in result_data:
            items = result_data["items"]
            if items:
                # Combine all search results into a single reference
                combined_text = ""
                combined_metadata = {
                    "type": "search_chat_files",
                    "chat_id": chat_id,
                    "query": query,
                    "result_count": len(items),
                }

                for item in items:
                    content = item.get("content", "")
                    metadata = item.get("metadata", {})
                    combined_text += f"Document: {metadata.get('source', 'Untitled')}\n\n"
                    combined_text += f"Content: {content}\n\n"

                    if metadata.get("asset_id") and metadata.get("document_id") and metadata.get("collection_id"):
                        asset_url = f"asset://{metadata.get('asset_id')}?document_id={metadata.get('document_id')}&collection_id={metadata.get('collection_id')}"
                        if metadata.get("mimetype"):
                            asset_url = asset_url + "&mime_type=" + metadata.get("mimetype")
                        combined_text += f"![]({asset_url})\n\n"

                return {
                    "text": combined_text.strip(),
                    "metadata": combined_metadata,
                    "score": 1.0,  # Default score for search results
                }

        return None

    except Exception as e:
        logger.error(f"Error formatting search chat files reference: {e}")
        return None


def _format_list_reference(tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format list_collections tool result as reference"""
    try:
        # Parse tool result - handle both string and already parsed data
        if isinstance(tool_result, str):
            try:
                result_data = json.loads(tool_result)
            except json.JSONDecodeError:
                result_data = {"raw_result": tool_result}
        else:
            result_data = tool_result

        logger.debug(f"List reference result_data: {result_data}")

        # Handle array format where data is in first element's text field
        if isinstance(result_data, list) and len(result_data) > 0:
            first_item = result_data[0]
            if isinstance(first_item, dict) and "text" in first_item:
                try:
                    # Parse the text field as JSON
                    text_data = json.loads(first_item["text"])
                    result_data = text_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                    return None

        # Look for items field (which contains collections)
        if "items" in result_data:
            collections = result_data["items"]
            text = "Available Collections:\n"
            for collection in collections:
                title = collection.get("title", collection.get("name", "Unknown"))
                description = collection.get("description", "No description")
                collection_id = collection.get("id", "Unknown ID")
                status = collection.get("status", "Unknown")

                text += f"- {title} (ID: {collection_id})\n"
                text += f"  Status: {status}\n"
                if description:
                    text += f"  Description: {description}\n"
                text += "\n"

            return {
                "text": text.strip(),
                "metadata": {"type": "list_collections", "collection_count": len(collections)},
                "score": 1.0,
            }

        return None

    except Exception as e:
        logger.error(f"Error formatting list reference: {e}")
        return None


def _format_web_search_reference(tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format web_search tool result as reference"""
    try:
        # Parse tool result - handle both string and already parsed data
        if isinstance(tool_result, str):
            try:
                result_data = json.loads(tool_result)
            except json.JSONDecodeError:
                result_data = {"raw_result": tool_result}
        else:
            result_data = tool_result

        logger.debug(f"Web search reference result_data: {result_data}")

        # Handle array format where data is in first element's text field
        if isinstance(result_data, list) and len(result_data) > 0:
            first_item = result_data[0]
            if isinstance(first_item, dict) and "text" in first_item:
                try:
                    # Parse the text field as JSON
                    text_data = json.loads(first_item["text"])
                    result_data = text_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                    return None

        query = args.get("query", "")

        if "results" in result_data:
            results = result_data["results"]
            if results:
                combined_text = f"Web Search Results for: {query}\n\n"

                for result in results:
                    title = result.get("title", "No title")
                    url = result.get("url", "No URL")
                    snippet = result.get("snippet", "")

                    combined_text += f"Title: {title}\n"
                    combined_text += f"URL: {url}\n"
                    combined_text += f"Snippet: {snippet}\n\n"

                return {
                    "text": combined_text.strip(),
                    "metadata": {"type": "web_search", "query": query, "result_count": len(results)},
                    "score": 1.0,
                }

        return None

    except Exception as e:
        logger.error(f"Error formatting web search reference: {e}")
        return None


def _format_web_read_reference(tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format web_read tool result as reference"""
    try:
        # Parse tool result - handle both string and already parsed data
        if isinstance(tool_result, str):
            try:
                result_data = json.loads(tool_result)
            except json.JSONDecodeError:
                result_data = {"raw_result": tool_result}
        else:
            result_data = tool_result

        logger.debug(f"Web read reference result_data: {result_data}")

        # Handle array format where data is in first element's text field
        if isinstance(result_data, list) and len(result_data) > 0:
            first_item = result_data[0]
            if isinstance(first_item, dict) and "text" in first_item:
                try:
                    # Parse the text field as JSON
                    text_data = json.loads(first_item["text"])
                    result_data = text_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                    return None

        urls = args.get("url_list", [])

        if "results" in result_data:
            results = result_data["results"]
            if results:
                combined_text = "Web Page Content:\n\n"

                for result in results:
                    url = result.get("url", "No URL")
                    title = result.get("title", "No title")
                    content = result.get("content", "")

                    combined_text += f"URL: {url}\n"
                    combined_text += f"Title: {title}\n"
                    combined_text += f"Content: {content}\n\n"

                return {
                    "text": combined_text.strip(),
                    "metadata": {"type": "web_read", "urls": urls, "result_count": len(results)},
                    "score": 1.0,
                }

        return None

    except Exception as e:
        logger.error(f"Error formatting web read reference: {e}")
        return None


def _format_generic_reference(tool_name: str, tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format generic tool result as reference"""
    try:
        # Parse the tool result to handle array format
        parsed_result = tool_result
        if isinstance(tool_result, str):
            try:
                parsed_result = json.loads(tool_result)
            except json.JSONDecodeError:
                parsed_result = tool_result

        # Handle array format where data is in first element's text field
        if isinstance(parsed_result, list) and len(parsed_result) > 0:
            first_item = parsed_result[0]
            if isinstance(first_item, dict) and "text" in first_item:
                try:
                    # Parse the text field as JSON
                    text_data = json.loads(first_item["text"])
                    parsed_result = text_data
                except json.JSONDecodeError:
                    # If parsing fails, use the original text
                    parsed_result = first_item["text"]

        # For generic tools, create a simple reference
        text = f"Tool: {tool_name}\n"
        if args:
            text += f"Arguments: {json.dumps(args, indent=2)}\n"

        # Handle both string and non-string results
        if isinstance(parsed_result, str):
            text += f"Result: {parsed_result}"
        else:
            text += f"Result: {json.dumps(parsed_result, indent=2)}"

        return {
            "text": text,
            "metadata": {"type": "tool_result", "tool_name": tool_name, "args": args},
            "score": 1.0,
        }

    except Exception as e:
        logger.error(f"Error formatting generic reference: {e}")
        return None
