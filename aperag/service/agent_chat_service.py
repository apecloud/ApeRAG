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
import logging
import os
import traceback
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import WebSocket
from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.config import LoggerSettings, MCPServerSettings, MCPSettings, OpenAISettings, Settings
from mcp_agent.workflows.llm.augmented_llm import RequestParams, SimpleMemory
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.flow.runners.llm import add_ai_message, add_human_message

# Import MCP server for direct collection search access
from aperag.schema import view_models
from aperag.utils.history import RedisChatMessageHistory, get_async_redis_client
from aperag.utils.utils import now_unix_milliseconds

logger = logging.getLogger(__name__)

# Only set default values if environment variables are not already set
if not os.getenv("APERAG_API_KEY"):
    os.environ["APERAG_API_KEY"] = "sk-test"
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-test"
if not os.getenv("APERAG_URL"):
    os.environ["APERAG_URL"] = "http://localhost:8000/mcp/"
if not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
if not os.getenv("DEFAULT_MODEL"):
    os.environ["DEFAULT_MODEL"] = "gpt-4o-mini"

# Optimized ApeRAG Agent Instruction with collection and web search support
APERAG_AGENT_INSTRUCTION = """
# ApeRAG Knowledge Assistant

You are an advanced AI knowledge assistant powered by ApeRAG's comprehensive search and information retrieval capabilities. Your primary mission is to help users find, understand, and utilize information from their knowledge collections and the web with exceptional accuracy and thoroughness.

You operate as an intelligent research partner who can access multiple knowledge sources and provide well-researched, comprehensive answers. Each time you receive a query, you should autonomously search, analyze, and synthesize information until the user's question is completely resolved.

## Core Identity & Mission

You are pair-working with a USER to solve their information needs. Each query should be treated as a research task that requires:
1. **Complete autonomous resolution** - Keep working until the question is fully answered
2. **Multi-source integration** - Leverage both knowledge collections and web resources
3. **Comprehensive exploration** - Don't stop at the first result; explore multiple angles
4. **Quality synthesis** - Provide well-structured, accurate, and actionable information

Your main goal is to follow the USER's instructions and resolve their information needs to the best of your ability before yielding back to the user.

## Available Research Tools

You have access to 4 powerful tools for information retrieval:

### 1. Collection Management
- **`list_collections()`**: Discover all available knowledge collections with complete metadata
- **`search_collection(collection_id, query, ...)`**: Search within specific collections using hybrid search methods

### 2. Web Intelligence  
- **`web_search(query, ...)`**: Search the web using multiple engines (DuckDuckGo, Google, Bing) with domain targeting
- **`web_read(url_list, ...)`**: Extract and read content from web pages for detailed analysis

## Priority-Based Search Strategy

### 🎯 When User Specifies Collections (via "@" selection):
**CRITICAL**: When the user has selected specific collections using "@" mentions, you MUST:

1. **First Priority**: Search the user-specified collections immediately and thoroughly
2. **Quality Assessment**: Evaluate if the specified collections provide sufficient information
3. **Strategic Expansion**: Only if needed, autonomously search additional relevant collections
4. **Clear Attribution**: Always indicate which results come from user-specified vs. additional collections

**Example Workflow**:
```
User: "@documentation How do I deploy applications?"
→ 1. Search "documentation" collection first (REQUIRED)
→ 2. Assess result quality and coverage
→ 3. If needed, search additional collections like "tutorials" or "examples"
→ 4. Clearly distinguish sources in response
```

### 🔍 When No Collections Specified:
1. **Discovery**: Use `list_collections()` to explore available knowledge sources
2. **Strategic Selection**: Choose most relevant collections based on query analysis
3. **Multi-Collection Search**: Search multiple relevant collections for comprehensive coverage
4. **Autonomous Decision-Making**: You decide which collections to search and in what order

## Tool Usage Protocol

### Strategic Tool Deployment
1. **ALWAYS use tools autonomously** - Never ask permission; execute searches based on what you determine is needed
2. **Respect user preferences** - Honor "@" collection selections and web search settings
3. **Parallel execution** - Use multiple tools simultaneously when gathering information from different sources
4. **Comprehensive coverage** - Don't stop at one search; explore multiple collections, search terms, and sources
5. **Quality over quantity** - Prioritize relevant, high-quality information over volume

### Search Strategy Framework

#### Step 1: Query Analysis & Source Planning
1. **Check user specifications**: Identify any "@" mentioned collections and web search preferences
2. **Understand intent**: Analyze what type of information the user needs
3. **Plan search hierarchy**: Prioritize user-specified sources, then determine additional sources
4. **Design queries**: Create multiple search variations to ensure comprehensive coverage

#### Step 2: Autonomous Information Gathering
1. **Priority execution**: Search user-specified collections first (if any)
2. **Strategic collection selection**: Choose additional relevant collections based on query context
3. **Multi-method search**: Use hybrid search (vector + fulltext + graph) for comprehensive results
4. **Web augmentation**: Use web search for current information, verification, or gap-filling (if enabled)
5. **Content extraction**: Read full web pages when initial snippets are insufficient

#### Step 3: Synthesis & Response
1. **Information integration**: Combine findings from all sources with clear source hierarchy
2. **Quality assurance**: Verify accuracy and completeness
3. **Clear attribution**: Cite all sources with transparency, distinguishing user-specified vs. additional sources
4. **Actionable delivery**: Provide practical, well-structured responses

## Advanced Search Techniques

### Collection Search Optimization
- **Hybrid approach**: Enable all search types (vector, fulltext, graph) by default for maximum coverage
- **Query variations**: Try different phrasings and keywords if initial results are insufficient
- **Cross-collection search**: Search multiple relevant collections for comprehensive coverage
- **Iterative refinement**: Adjust search parameters based on result quality

### Web Search Intelligence
- **Conditional usage**: Only use web search when it's enabled in the session
- **Multi-engine strategy**: Use different search engines for varied perspectives
- **Domain targeting**: Use `source` parameter for site-specific searches when relevant
- **LLM.txt discovery**: Leverage `search_llms_txt` for AI-optimized content discovery
- **Content depth**: Always read full pages (`web_read`) when web search provides promising URLs

### Parallel Information Gathering
Execute multiple searches simultaneously:
- Search multiple collections in parallel
- Combine collection and web searches (when enabled)
- Read multiple web pages concurrently
- Cross-reference findings across sources

## Response Excellence Standards

### Structure Your Responses:
```
## Direct Answer
[Clear, actionable answer to the user's question]

## Comprehensive Analysis
[Detailed explanation with context, analysis, and insights]

## Supporting Evidence

📚 **User-Specified Collections** (if any):
- @[Collection Name]: [Key findings and insights]

📚 **Additional Collections Searched**:
- [Collection Name]: [Key findings and relevance]

🌐 **Web Sources** (if web search enabled):
- [Title] ([Domain]) - [Key Points]
- [URL for reference]

## Additional Context
[Related information, limitations, or follow-up suggestions]
```

### Quality Assurance:
- **Accuracy**: Only provide verified information from reliable sources
- **Completeness**: Address all aspects of the user's question thoroughly
- **Clarity**: Use clear, well-organized language with logical flow
- **Transparency**: Always cite sources and indicate confidence levels
- **Actionability**: Provide practical next steps or applications when relevant
- **Source Hierarchy**: Clearly distinguish between user-specified and additional sources

## Error Handling & Adaptation

### When User-Specified Collections Are Empty:
- Search the specified collections first (as required)
- Clearly report if specified collections have no relevant results
- Automatically search additional relevant collections
- Inform user about the expanded search strategy

### When Information is Limited:
- Try alternative search terms and approaches
- Search additional collections that might be relevant
- Use web search to supplement knowledge base gaps (if enabled)
- Clearly communicate what information is available vs. unavailable

### When Web Search is Disabled:
- Rely entirely on knowledge collections
- Be more thorough in collection searches
- Clearly indicate when web search might have provided additional current information
- Focus on comprehensive collection coverage

## Special Instructions

### User Preference Compliance:
- **@ Collection Priority**: Always search user-specified collections first, regardless of your assessment
- **Web Search Respect**: Only use web search when it's explicitly enabled
- **Transparent Expansion**: Clearly explain when and why you search additional sources beyond user specifications

### Thoroughness Requirements:
- **Explore comprehensively**: Don't stop at first results; dig deeper
- **Multiple perspectives**: Search different collections and web sources (when available)
- **Verify findings**: Cross-reference important information across sources
- **Complete coverage**: Ensure all aspects of complex queries are addressed

### Efficiency Optimization:
- **Parallel execution**: Run multiple searches simultaneously
- **Smart collection selection**: Choose most relevant collections first (after user-specified ones)
- **Strategic web usage**: Use web search for current info, verification, and gap-filling (when enabled)
- **Focused reading**: Extract specific content from web pages rather than full scanning

### Communication Excellence:
- **Source transparency**: Always clearly indicate where information comes from
- **Hierarchy clarity**: Distinguish between user-specified and additional sources
- **Confidence indicators**: Specify certainty levels for different claims
- **Balanced perspectives**: Present multiple viewpoints when they exist
- **Practical focus**: Emphasize actionable insights and applications

## Your Mission

Be the user's most capable research partner. Help them discover accurate, comprehensive, and actionable information by:

1. **Respecting user preferences**: Honor "@" collection selections and web search settings
2. **Autonomous exploration**: Search multiple sources without waiting for permission
3. **Comprehensive coverage**: Use all available tools to ensure complete information gathering
4. **Quality synthesis**: Combine findings into clear, well-structured responses
5. **Continuous improvement**: Adapt search strategies based on result quality
6. **Transparent attribution**: Always cite sources and acknowledge limitations

You have powerful tools at your disposal - use them strategically and thoroughly to provide exceptional research assistance. Your goal is to be so comprehensive and helpful that users can rely on you for their most important information needs while respecting their preferences and guidance.

Ready to assist with your research and knowledge discovery needs!
"""


class AgentChatService:
    """
    Chat service specifically for agent-type bots that uses MCPApp for intelligent conversation.

    This service dynamically constructs MCPApp instances based on agent message parameters,
    including collection selection, model choice, and web search capabilities.
    """

    def __init__(self, session: AsyncSession = None):
        if session is None:
            self.db_ops = async_db_ops
        else:
            self.db_ops = AsyncDatabaseOps(session)

    def _get_aperag_api_settings(self) -> Dict[str, str]:
        """Get ApeRAG API settings for MCP connection"""
        return {
            "aperag_api_key": os.getenv("APERAG_API_KEY", "sk-test"),
            "aperag_url": os.getenv("APERAG_URL", "http://localhost:8000/mcp/"),
        }

    def _get_openai_settings(self, model_name: Optional[str] = None) -> Dict[str, str]:
        """Get OpenAI settings for LLM calls"""
        return {
            "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            "openai_api_key": os.getenv("OPENAI_API_KEY", "sk-test"),
            "default_model": model_name or os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        }

    def _extract_tool_call_references(self, memory) -> List[Dict[str, Any]]:
        """
        Extract tool call results from MCP agent history and format as references.

        Args:
            memory: SimpleMemory instance containing agent history

        Returns:
            List of reference dictionaries in the format expected by llm.py
        """
        references = []

        try:
            # Get history from memory
            history_messages = memory.get() if hasattr(memory, "get") else []

            for message in history_messages:
                # Check if message has tool calls (message is a dict)
                if isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
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

                        # Parse tool arguments
                        try:
                            args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        except json.JSONDecodeError:
                            args_dict = {"raw_args": tool_args}

                        # Find corresponding tool result message
                        tool_result = self._find_tool_result(history_messages, tool_call_id)

                        if tool_result:
                            # Format reference based on tool type
                            if tool_name == "aperag_search_collection":
                                ref = self._format_search_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            elif tool_name == "aperag_list_collections":
                                ref = self._format_list_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            elif tool_name == "aperag_web_search":
                                ref = self._format_web_search_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            elif tool_name == "aperag_web_read":
                                ref = self._format_web_read_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            else:
                                # Generic tool result reference
                                ref = self._format_generic_reference(tool_name, tool_result, args_dict)
                                if ref:
                                    references.append(ref)

        except Exception as e:
            logger.error(f"Error extracting tool call references: {e}")

        return references

    def _find_tool_result(self, messages, tool_call_id: str) -> Optional[str]:
        """Find the tool result message for a given tool call ID"""
        for message in messages:
            if (
                isinstance(message, dict)
                and message.get("role") == "tool"
                and message.get("tool_call_id") == tool_call_id
            ):
                content = message.get("content", "")
                logger.debug(f"Found tool result for {tool_call_id}: {type(content)} - {content}")

                # Handle both string and list content
                if isinstance(content, list):
                    return json.dumps(content)
                return content
        return None

    def _format_search_reference(self, tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
                        combined_text += f"Document: {metadata.get('title', 'Untitled')}\n"
                        combined_text += f"Content: {content}\n\n"

                    return {
                        "text": combined_text.strip(),
                        "metadata": combined_metadata,
                        "score": 1.0,  # Default score for search results
                    }

            return None

        except Exception as e:
            logger.error(f"Error formatting search reference: {e}")
            return None

    def _format_list_reference(self, tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    def _format_web_search_reference(self, tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    def _format_web_read_reference(self, tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    def _format_generic_reference(
        self, tool_name: str, tool_result: str, args: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
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

    def _build_llm_query_prompt(self, agent_message: view_models.AgentMessage, user: str) -> str:
        """
        Build a comprehensive prompt for LLM that includes context about user preferences,
        available collections, and web search status using a template approach.
        """
        # Determine collection context
        if agent_message.collection_ids:
            collection_context = ", ".join([f"'{cid}'" for cid in agent_message.collection_ids])
            collection_instruction = (
                "PRIORITY: Search these collections first, then decide if additional sources are needed"
            )
        else:
            collection_context = "None specified by user"
            collection_instruction = "discover and select relevant collections automatically"

        # Determine web search context
        web_status = "enabled" if agent_message.web_search_enabled else "disabled"
        if agent_message.web_search_enabled:
            web_instruction = "Use web search strategically for current information, verification, or gap-filling"
        else:
            web_instruction = "Rely entirely on knowledge collections; inform user if web search would be helpful"

        # Use template for cleaner formatting
        prompt_template = """**User Query**: {query}

**Session Context**:
- **User-Specified Collections**: {collection_context} ({collection_instruction})
- **Web Search**: {web_status} ({web_instruction})

**Research Instructions**:
1. If user specified collections (@mentions), search those first (REQUIRED)
2. Assess result quality and decide if additional collections are needed
3. Use web search strategically if enabled and relevant
4. Provide comprehensive, well-structured response with clear source attribution
5. Distinguish between user-specified and additional sources in your response

Please provide a thorough, well-researched answer that leverages all appropriate search tools based on the context above."""

        return prompt_template.format(
            query=agent_message.query,
            collection_context=collection_context,
            collection_instruction=collection_instruction,
            web_status=web_status,
            web_instruction=web_instruction,
        )

    def _create_mcp_settings(
        self,
        model_name: Optional[str] = None,
    ) -> Optional[Settings]:
        """Create MCP settings dynamically based on agent message parameters"""
        if not MCPApp:
            logger.error("MCP components not available")
            return None

        aperag_settings = self._get_aperag_api_settings()
        openai_settings = self._get_openai_settings(model_name)

        try:
            return Settings(
                execution_engine="asyncio",
                logger=LoggerSettings(type="console", level="info"),
                mcp=MCPSettings(
                    servers={
                        "aperag": MCPServerSettings(
                            transport="streamable_http",
                            url=aperag_settings["aperag_url"],
                            headers={
                                "Authorization": f"Bearer {aperag_settings['aperag_api_key']}",
                                "Content-Type": "application/json",
                            },
                            http_timeout_seconds=30,
                            read_timeout_seconds=120,
                            description="ApeRAG knowledge base server",
                            env={"APERAG_API_KEY": aperag_settings["aperag_api_key"]},
                        )
                    }
                ),
                openai=OpenAISettings(
                    api_key=openai_settings["openai_api_key"],
                    base_url=openai_settings["openai_base_url"],
                    default_model=openai_settings["default_model"],
                    temperature=0.7,
                    max_tokens=2000,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to create MCP settings: {e}")
            return None

    def _create_mcp_app(
        self,
    ) -> Optional[MCPApp]:
        """Create MCPApp instance dynamically based on agent parameters"""
        settings = self._create_mcp_settings()
        if not settings:
            return None

        try:
            return MCPApp(name="aperag_agent", settings=settings)
        except Exception as e:
            logger.error(f"Failed to create MCPApp: {e}")
            return None

    async def handle_websocket_agent_chat(self, websocket: WebSocket, user: str, bot_id: str, chat_id: str):
        """Handle WebSocket connections for agent-type bot chats"""
        try:
            memory = SimpleMemory()
            while True:
                # Receive message from WebSocket
                data = await websocket.receive_text()
                message_data = json.loads(data)

                # Generate message ID
                message_id = str(uuid.uuid4())
                query = message_data.get("query", "")
                if not query or not query.strip():
                    error_response = self._format_error("Invalid message format")
                    await websocket.send_text(json.dumps(error_response))
                    continue

                try:
                    agent_message = view_models.AgentMessage(
                        query=query,
                        collection_ids=message_data.get("collection_ids"),
                        model_name=message_data.get("model_name"),
                        web_search_enabled=message_data.get("web_search_enabled", False),
                    )
                    # Process the agent message and stream responses
                    async for response_chunk in self.process_agent_message(
                        agent_message, user, chat_id, message_id, memory
                    ):
                        await websocket.send_text(json.dumps(response_chunk))

                except Exception as e:
                    logger.error(f"Error processing agent websocket message: {e}")
                    error_response = self._format_error(str(e))
                    await websocket.send_text(json.dumps(error_response))

        except Exception as e:
            logger.error(f"WebSocket error in agent chat: {e}")

    async def process_agent_message(
        self,
        agent_message: view_models.AgentMessage,
        user: str,
        chat_id: str,
        msg_id: str,
        memory,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process an agent message and yield streaming responses.

        This method creates a dynamic MCPApp instance based on the message parameters
        and uses it to generate intelligent responses.
        """
        try:
            # Validate collections if specified
            if agent_message.collection_ids:
                for collection_id in agent_message.collection_ids:
                    collection = await self.db_ops.query_collection(user, collection_id)
                    if not collection:
                        yield self._format_error(f"Collection {collection_id} not found")
                        return

            # Create dynamic agent app
            mcp_app = self._create_mcp_app()

            if not mcp_app:
                yield self._format_error("Failed to initialize agent")
                return

            # Yield start message
            yield self._format_stream_start(msg_id)

            # Get chat history for context
            history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())

            # Use agent app for intelligent conversation
            # This integrates with the MCP system for dynamic tool usage
            full_content = ""

            try:
                async with mcp_app.run() as running_app:
                    # Create agent with instruction and server names
                    agent = Agent(
                        name="aperag_assistant",
                        instruction=APERAG_AGENT_INSTRUCTION,
                        server_names=["aperag"],
                    )

                    # Verify server connection
                    if "aperag" not in running_app.server_registry.registry:
                        yield self._format_error("ApeRAG MCP Server connection failed")
                        return

                    async with agent:
                        # Attach LLM to agent
                        llm = await agent.attach_llm(OpenAIAugmentedLLM)

                        request_params = RequestParams(
                            max_iterations=10,
                            parallel_tool_calls=True,
                            model=agent_message.model_name,
                        )

                        llm.history = memory

                        # Build comprehensive prompt with context and pre-search results
                        comprehensive_prompt = self._build_llm_query_prompt(agent_message=agent_message, user=user)

                        # Generate response using LLM with comprehensive prompt
                        response = await llm.generate_str(comprehensive_prompt, request_params)
                        full_content = response if response else "No response generated"

                        # Stream the response content
                        yield self._format_stream_content(msg_id, full_content)
                        memory = llm.history

            except Exception as e:
                logger.error(f"Error in MCP agent execution: {e}")
                yield self._format_error(f"Error in agent execution: {str(e)}")
                return

            # Generate references - either from tool calls or direct search results
            # Extract tool call results from history and format as references
            tool_references = self._extract_tool_call_references(memory)

            # Store messages in history
            await add_human_message(history, agent_message.query, "")
            await add_ai_message(history, agent_message.query, "", full_content, tool_references, [])

            # Prepare references and URLs
            urls = []

            yield self._format_stream_end(msg_id, references=tool_references, urls=urls)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in agent message processing: {e}")
            yield self._format_error(f"Error processing agent message: {str(e)}")

    # Helper methods for response formatting
    def _format_stream_start(self, msg_id: str) -> Dict[str, Any]:
        """Format the start event for streaming"""
        return {
            "type": "start",
            "id": msg_id,
            "timestamp": now_unix_milliseconds(),
        }

    def _format_stream_content(self, msg_id: str, content: str) -> Dict[str, Any]:
        """Format a content chunk for streaming"""
        return {
            "type": "message",
            "id": msg_id,
            "data": content,
            "timestamp": now_unix_milliseconds(),
        }

    def _format_tool_call_content(self, msg_id: str, content: str) -> Dict[str, Any]:
        """Format a content chunk for streaming"""
        return {
            "type": "message",
            "id": msg_id,
            "data": f"<tool_call>{content}</tool_call>\n",
            "timestamp": now_unix_milliseconds(),
        }

    def _format_stream_end(self, msg_id: str, references: List[str] = None, urls: List[str] = None) -> Dict[str, Any]:
        """Format the end event for streaming"""
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

    def _format_error(self, error: str) -> Dict[str, Any]:
        """Format an error response"""
        return {
            "type": "error",
            "id": str(uuid.uuid4()),
            "data": error,
            "timestamp": now_unix_milliseconds(),
        }
