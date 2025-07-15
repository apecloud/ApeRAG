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
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.config import settings
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.flow.runners.llm import add_ai_message, add_human_message
from aperag.schema import view_models
from aperag.utils.constant import DOC_QA_REFERENCES, DOCUMENT_URLS
from aperag.utils.history import RedisChatMessageHistory, get_async_redis_client
from aperag.utils.utils import now_unix_milliseconds
from mcp_agent.app import MCPApp
from mcp_agent.config import Settings, LoggerSettings, MCPSettings, MCPServerSettings, OpenAISettings
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm import SimpleMemory


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
# ApeRAG Intelligent Assistant

You are an advanced AI assistant powered by ApeRAG's comprehensive search capabilities. Your primary mission is to provide accurate, well-researched answers by leveraging multiple search methods and knowledge sources.

## Core Capabilities

### 1. Knowledge Collection Management
- **Collection Discovery**: Automatically discover and list available knowledge collections
- **Smart Collection Selection**: Choose the most relevant collections based on user queries
- **Multi-Collection Search**: Search across multiple collections when needed for comprehensive coverage

### 2. Advanced Search Methods
- **Hybrid Search (Recommended)**: Combine vector, full-text, and graph search for optimal results
- **Vector Search**: Semantic similarity and conceptual understanding
- **Full-Text Search**: Exact keyword matching and specific terminology
- **Graph Search**: Relationship discovery and connected concepts

### 3. Web Search Integration
- **Real-time Information**: Access current information from the web when enabled
- **Fact Verification**: Cross-reference knowledge base information with web sources
- **Comprehensive Coverage**: Combine knowledge base and web search for complete answers

## Workflow Protocol

### Step 1: Query Analysis & Source Planning
1. **Analyze User Intent**: Understand what the user is asking and what type of information they need
2. **Collection Strategy**: 
   - If specific collections are provided, use only those collections
   - If no collections specified, list available collections and select the most relevant ones
   - Consider whether multiple collections might be needed for comprehensive coverage
3. **Search Strategy**: Determine if web search is needed based on:
   - Query type (current events, latest information, verification)
   - Web search availability (enabled/disabled)
   - Knowledge base coverage gaps

### Step 2: Information Retrieval
1. **Collection Search**: Execute searches on selected collections using appropriate methods
2. **Web Search** (if enabled and relevant): Search the web for additional or current information
3. **Result Integration**: Combine and synthesize information from all sources

### Step 3: Response Generation
1. **Synthesis**: Create a comprehensive answer combining all relevant information
2. **Source Attribution**: Clearly cite all sources (collections and web)
3. **Quality Assurance**: Ensure accuracy and completeness

## Collection Selection Logic

### When Collections Are Specified:
```
User specifies collections → Use only those collections → Search within specified scope
```

### When No Collections Are Specified:
```
List available collections → Analyze query relevance → Select best matching collections → Execute search
```

### Collection Selection Criteria:
- **Topic Relevance**: Match collection topics with query subject
- **Content Type**: Consider if query needs specific document types
- **Scope Coverage**: Ensure selected collections can provide comprehensive answers
- **User Context**: Consider user's domain or role if apparent

## Web Search Integration

### When to Use Web Search:
- **Current Events**: Recent news, updates, or time-sensitive information
- **Latest Developments**: Technology updates, policy changes, market information
- **Verification**: Cross-check knowledge base information with current sources
- **Gap Filling**: When knowledge base doesn't have sufficient information

### Web Search Guidelines:
- Only use when web search is enabled in the request
- Clearly distinguish between knowledge base and web sources
- Prioritize authoritative and recent web sources
- Use web search to supplement, not replace, knowledge base information

## Response Format

### Structure Your Responses:
```
**Direct Answer**: [Clear, concise answer to the user's question]

**Detailed Explanation**: [Comprehensive explanation with context and analysis]

**Sources Used**:
📚 **Knowledge Collections**:
- Collection: [Name] | Document: [Title] | Relevance: [Score/Context]

🌐 **Web Sources** (if used):
- Source: [Website/URL] | Title: [Page Title] | Date: [If available]

**Additional Information**: [Related topics, suggestions, or limitations]
```

### Quality Standards:
- **Accuracy**: Only provide verified information from reliable sources
- **Completeness**: Address all aspects of the user's question
- **Clarity**: Use clear, well-structured language
- **Transparency**: Clearly indicate source types and confidence levels
- **Helpfulness**: Provide actionable information when possible

## Search Optimization

### Collection Search Best Practices:
1. **Start with Hybrid Search**: Usually provides the best balance of precision and recall
2. **Use Vector Search**: For conceptual or semantic queries
3. **Use Full-Text Search**: For specific terms, names, or exact phrases
4. **Use Graph Search**: For relationship or connection queries

### Query Refinement:
- If initial search yields insufficient results, try alternative search methods
- Break complex queries into sub-queries when appropriate
- Use synonyms and related terms for broader coverage

## Error Handling & Limitations

### When Information Is Limited:
- Clearly state what information is available vs. unavailable
- Suggest alternative search approaches or collections
- Recommend web search if it might help and is available

### When Collections Are Empty or Inaccessible:
- Inform user about collection status
- Suggest alternative collections if available
- Fall back to web search if enabled

## Special Instructions

### Collection Management:
- Always list available collections when no specific collections are provided
- Help users understand what each collection contains
- Suggest the most relevant collections for their queries

### Web Search Usage:
- Use web search strategically, not as a default
- Clearly label web-sourced information
- Prefer recent and authoritative web sources
- Combine web and knowledge base information thoughtfully

### Multi-Source Integration:
- Synthesize information from multiple sources coherently
- Highlight agreements and discrepancies between sources
- Provide balanced perspectives when sources differ

## Your Mission
Be the user's intelligent research partner. Help them find accurate, comprehensive, and actionable information by leveraging both their knowledge collections and web resources effectively. Always prioritize accuracy, provide clear source attribution, and deliver well-structured, helpful responses.

Ready to assist with your research and information needs!
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
            history_messages = memory.get() if hasattr(memory, 'get') else []
            
            for message in history_messages:
                # Check if message has tool calls (message is a dict)
                if isinstance(message, dict) and message.get('role') == 'assistant' and message.get('tool_calls'):
                    for tool_call in message['tool_calls']:
                        # Debug: log the actual structure
                        logger.debug(f"Tool call structure: {tool_call}, type: {type(tool_call)}")
                        
                        # Process tool call information
                        # Handle different tool call structures (dict vs object)
                        tool_name = 'unknown_tool'
                        tool_args = '{}'
                        tool_call_id = ''
                        
                        # Handle OpenAI ChatCompletionMessageToolCall objects
                        if hasattr(tool_call, 'id'):
                            tool_call_id = tool_call.id
                            if hasattr(tool_call, 'function'):
                                tool_name = tool_call.function.name if hasattr(tool_call.function, 'name') else 'unknown_tool'
                                tool_args = tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else '{}'
                        # Handle dictionary format
                        elif isinstance(tool_call, dict):
                            tool_call_id = tool_call.get('id', '')
                            if 'function' in tool_call:
                                tool_name = tool_call['function'].get('name', 'unknown_tool')
                                tool_args = tool_call['function'].get('arguments', '{}')
                            elif 'name' in tool_call:
                                tool_name = tool_call.get('name', 'unknown_tool')
                                tool_args = tool_call.get('arguments', '{}')
                            elif 'type' in tool_call and tool_call['type'] == 'function':
                                tool_name = tool_call.get('function', {}).get('name', 'unknown_tool')
                                tool_args = tool_call.get('function', {}).get('arguments', '{}')
                        
                        logger.debug(f"Extracted tool_name: {tool_name}, tool_args: {tool_args}, tool_call_id: {tool_call_id}")
                        
                        # Parse tool arguments
                        try:
                            args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        except json.JSONDecodeError:
                            args_dict = {"raw_args": tool_args}
                        
                        # Find corresponding tool result message
                        tool_result = self._find_tool_result(history_messages, tool_call_id)
                        
                        if tool_result:
                            # Format reference based on tool type
                            if tool_name == 'aperag_search_collection':
                                ref = self._format_search_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            elif tool_name == 'aperag_list_collections':
                                ref = self._format_list_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            elif tool_name == 'aperag_web_search':
                                ref = self._format_web_search_reference(tool_result, args_dict)
                                if ref:
                                    references.append(ref)
                            elif tool_name == 'aperag_web_read':
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
            if (isinstance(message, dict) and message.get('role') == 'tool' and 
                message.get('tool_call_id') == tool_call_id):
                content = message.get('content', '')
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
                if isinstance(first_item, dict) and 'text' in first_item:
                    try:
                        # Parse the text field as JSON
                        text_data = json.loads(first_item['text'])
                        result_data = text_data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                        return None
            
            # Extract search parameters
            collection_id = args.get('collection_id', 'unknown')
            query = args.get('query', '')
            
            # Format search results
            if 'items' in result_data:
                items = result_data['items']
                if items:
                    # Combine all search results into a single reference
                    combined_text = ""
                    combined_metadata = {
                        "type": "search_collection",
                        "collection_id": collection_id,
                        "query": query,
                        "result_count": len(items)
                    }
                    
                    for item in items:
                        content = item.get('content', '')
                        metadata = item.get('metadata', {})
                        combined_text += f"Document: {metadata.get('title', 'Untitled')}\n"
                        combined_text += f"Content: {content}\n\n"
                    
                    return {
                        "text": combined_text.strip(),
                        "metadata": combined_metadata,
                        "score": 1.0  # Default score for search results
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
                if isinstance(first_item, dict) and 'text' in first_item:
                    try:
                        # Parse the text field as JSON
                        text_data = json.loads(first_item['text'])
                        result_data = text_data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                        return None
            
            # Look for items field (which contains collections)
            if 'items' in result_data:
                collections = result_data['items']
                text = "Available Collections:\n"
                for collection in collections:
                    title = collection.get('title', collection.get('name', 'Unknown'))
                    description = collection.get('description', 'No description')
                    collection_id = collection.get('id', 'Unknown ID')
                    status = collection.get('status', 'Unknown')
                    
                    text += f"- {title} (ID: {collection_id})\n"
                    text += f"  Status: {status}\n"
                    if description:
                        text += f"  Description: {description}\n"
                    text += "\n"
                
                return {
                    "text": text.strip(),
                    "metadata": {
                        "type": "list_collections",
                        "collection_count": len(collections)
                    },
                    "score": 1.0
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
                if isinstance(first_item, dict) and 'text' in first_item:
                    try:
                        # Parse the text field as JSON
                        text_data = json.loads(first_item['text'])
                        result_data = text_data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                        return None
            
            query = args.get('query', '')
            
            if 'results' in result_data:
                results = result_data['results']
                if results:
                    combined_text = f"Web Search Results for: {query}\n\n"
                    
                    for result in results:
                        title = result.get('title', 'No title')
                        url = result.get('url', 'No URL')
                        snippet = result.get('snippet', '')
                        
                        combined_text += f"Title: {title}\n"
                        combined_text += f"URL: {url}\n"
                        combined_text += f"Snippet: {snippet}\n\n"
                    
                    return {
                        "text": combined_text.strip(),
                        "metadata": {
                            "type": "web_search",
                            "query": query,
                            "result_count": len(results)
                        },
                        "score": 1.0
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
                if isinstance(first_item, dict) and 'text' in first_item:
                    try:
                        # Parse the text field as JSON
                        text_data = json.loads(first_item['text'])
                        result_data = text_data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse text field as JSON: {first_item['text']}")
                        return None
            
            urls = args.get('url_list', [])
            
            if 'results' in result_data:
                results = result_data['results']
                if results:
                    combined_text = "Web Page Content:\n\n"
                    
                    for result in results:
                        url = result.get('url', 'No URL')
                        title = result.get('title', 'No title')
                        content = result.get('content', '')
                        
                        combined_text += f"URL: {url}\n"
                        combined_text += f"Title: {title}\n"
                        combined_text += f"Content: {content}\n\n"
                    
                    return {
                        "text": combined_text.strip(),
                        "metadata": {
                            "type": "web_read",
                            "urls": urls,
                            "result_count": len(results)
                        },
                        "score": 1.0
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error formatting web read reference: {e}")
            return None

    def _format_generic_reference(self, tool_name: str, tool_result: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
                if isinstance(first_item, dict) and 'text' in first_item:
                    try:
                        # Parse the text field as JSON
                        text_data = json.loads(first_item['text'])
                        parsed_result = text_data
                    except json.JSONDecodeError:
                        # If parsing fails, use the original text
                        parsed_result = first_item['text']
            
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
                "metadata": {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "args": args
                },
                "score": 1.0
            }
            
        except Exception as e:
            logger.error(f"Error formatting generic reference: {e}")
            return None

    def _create_dynamic_instruction(
        self,
        collection_ids: Optional[List[str]] = None,
        web_search_enabled: bool = False
    ) -> str:
        """Create dynamic instruction based on agent parameters"""
        instruction = APERAG_AGENT_INSTRUCTION
        
        # Add collection-specific context
        if collection_ids:
            collection_context = f"""

## Current Session Context

### Specified Collections:
You have been configured to search within these specific collections:
{chr(10).join(f"- {collection_id}" for collection_id in collection_ids)}

**Important**: Only use these specified collections. Do not list or search other collections.
"""
            instruction += collection_context
        else:
            collection_context = """

## Current Session Context

### Collection Discovery Required:
No specific collections have been specified. You must:
1. List available collections using the `list_collections()` tool
2. Analyze the user's query to determine the most relevant collections
3. Select and search the appropriate collections for comprehensive answers
"""
            instruction += collection_context

        # Add web search context
        if web_search_enabled:
            web_context = """

### Web Search Enabled:
Web search is available for this session. Use it strategically to:
- Access current information and recent developments
- Verify or supplement knowledge base information
- Fill gaps when knowledge base coverage is insufficient
- Provide real-time data when relevant to the query

Remember to clearly distinguish between knowledge base and web sources in your responses.
"""
            instruction += web_context
        else:
            web_context = """

### Web Search Disabled:
Web search is not available for this session. Rely entirely on the knowledge collections available through the ApeRAG system.
"""
            instruction += web_context

        return instruction

    def _create_mcp_settings(
        self, 
        collection_ids: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        web_search_enabled: bool = False
    ) -> Optional[Settings]:
        """Create MCP settings dynamically based on agent message parameters"""
        if not MCPApp:
            logger.error("MCP components not available")
            return None

        aperag_settings = self._get_aperag_api_settings()
        openai_settings = self._get_openai_settings(model_name)

        # Create dynamic instruction
        system_instruction = self._create_dynamic_instruction(collection_ids, web_search_enabled)

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
                                "Content-Type": "application/json"
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
                system_instruction=system_instruction,
            )
        except Exception as e:
            logger.error(f"Failed to create MCP settings: {e}")
            return None

    def _create_agent_app(
        self,
        collection_ids: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        web_search_enabled: bool = False
    ) -> Optional[MCPApp]:
        """Create MCPApp instance dynamically based on agent parameters"""
        settings = self._create_mcp_settings(collection_ids, model_name, web_search_enabled)
        if not settings:
            return None

        try:
            return MCPApp(name="aperag_agent", settings=settings)
        except Exception as e:
            logger.error(f"Failed to create MCPApp: {e}")
            return None

    async def handle_websocket_agent_chat(
        self, 
        websocket: WebSocket, 
        user: str, 
        bot_id: str, 
        chat_id: str
    ):
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
                        web_search_enabled=message_data.get("web_search_enabled", False)
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
            agent_app = self._create_agent_app(
                collection_ids=agent_message.collection_ids,
                model_name=agent_message.model_name,
                web_search_enabled=agent_message.web_search_enabled or False
            )

            if not agent_app:
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
                async with agent_app.run() as running_app:
                    # Create agent with instruction and server names
                    agent = Agent(
                        name="aperag_assistant", 
                        instruction=self._create_dynamic_instruction(
                            agent_message.collection_ids,
                            agent_message.web_search_enabled or False
                        ), 
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
                        )
                        
                        llm.history = memory
                        # Generate response using LLM
                        response = await llm.generate_str(agent_message.query, request_params)
                        full_content = response if response else "No response generated"
                        
                        # Stream the response content
                        yield self._format_stream_content(msg_id, full_content)
                        memory = llm.history
                        
            except Exception as e:
                logger.error(f"Error in MCP agent execution: {e}")
                yield self._format_error(f"Error in agent execution: {str(e)}")
                return
            
            # Extract tool call results from history and format as references
            tool_references = self._extract_tool_call_references(memory)
            
            # Store messages in history
            await add_human_message(history, agent_message.query, "")
            await add_ai_message(history, agent_message.query, "", full_content, tool_references, [])
            
            # Prepare references and URLs
            urls = []
            
            yield self._format_stream_end(msg_id, references=tool_references, urls=urls)
                
        except Exception as e:
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

    def _format_stream_end(
        self, 
        msg_id: str, 
        references: List[str] = None,
        urls: List[str] = None
    ) -> Dict[str, Any]:
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