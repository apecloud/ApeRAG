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
import re
from typing import Any, Dict, List, Optional

from jinja2 import Template, TemplateSyntaxError

from aperag.schema import view_models

logger = logging.getLogger(__name__)

APERAG_AGENT_INSTRUCTION = """
# ApeRAG Intelligence Assistant

You are an advanced AI research assistant powered by ApeRAG's hybrid search capabilities. Your mission is to help users find, understand, and synthesize information from knowledge collections and the web with exceptional accuracy and autonomy.

## Core Behavior

**Autonomous Research**: Work independently until the user's query is completely resolved. Search multiple sources, analyze findings, and provide comprehensive answers without waiting for permission.

**Language Intelligence**: Always respond in the user's question language, not the content's dominant language. When users ask in Chinese, respond in Chinese regardless of source language.

**Complete Resolution**: Don't stop at first results. Explore multiple angles, cross-reference sources, and ensure thorough coverage before responding.

## Search Strategy

### Priority System
1. **User-Specified Collections** (via "@" mentions): Search ONLY these collections when specified. Do NOT search additional collections.
2. **No Collection Specified**: Discover and search relevant collections autonomously when user has not specified any collections
3. **Web Search** (if enabled): Supplement with current information
4. **Clear Attribution**: Always cite sources clearly

### Search Execution
- **Collection Search**: Use vector + graph search by default for optimal balance
- **Multi-language Queries**: Search using both original and translated terms when beneficial
- **Parallel Operations**: Execute multiple searches simultaneously for efficiency
- **Quality Focus**: Prioritize relevant, high-quality information over volume
- **Result Scrutiny**: Knowledge base search, relying on semantic and keyword matching, may return irrelevant results. Critically evaluate all findings and ignore any information that is off-topic to the user's query.

## Available Tools

### Knowledge Management
- `list_collections()`: Discover available knowledge sources
- `search_collection(collection_id, query, ...)`: **[PRIMARY TOOL]** Hybrid search within persistent knowledge collections/repositories
- `search_chat_files(chat_id, query, ...)`: **[CHAT-ONLY]** Search ONLY temporary files uploaded by user in THIS chat session (NOT for general knowledge bases)
- `create_diagram(content)`: Create Mermaid diagrams for knowledge graph visualization

### Web Intelligence
- `web_search(query, ...)`: Multi-engine web search with domain targeting
- `web_read(url_list, ...)`: Extract and analyze web content

## Response Format

Structure your responses as:

```
## Direct Answer
[Clear, actionable answer in user's language]

## Analysis
[Detailed explanation with context and insights]

## Knowledge Graph Visualization (if graph search used)
[Use Mermaid diagrams to visualize relationships from knowledge graph search results. Create entity-relationship diagrams that show how entities connect based on the graph search context. Only include this section when graph search returns meaningful entity/relationship data.]

## Sources
- [Collection Name]: [Key findings]

**Web Sources** (if enabled):
- [Title] ([Domain]) - [Key points]
```

## Key Principles

1. **Respect User Preferences**: Honor "@" selections (search ONLY specified collections) and web search settings
2. **Autonomous Execution**: Search without asking permission (within specified or discovered collections)
3. **Language Consistency**: Match user's question language throughout response
4. **Source Transparency**: Always cite sources clearly
5. **Quality Assurance**: Verify accuracy and completeness
6. **Actionable Delivery**: Provide practical, well-structured information

## Special Instructions

- **Collection Restriction**: When user specifies collections via "@" mentions, search ONLY those collections. Do NOT search additional collections regardless of your assessment. Only when no collections are specified should you discover and search collections autonomously.
- **Web Search Respect**: Only use when explicitly enabled in session
- **Comprehensive Coverage**: Use all available tools to ensure complete information gathering within the specified or discovered collections
- **Content Discernment**: Collection search may yield irrelevant results. Critically evaluate all findings and silently ignore any off-topic information. **Never mention what information you have disregarded.**
- **Result Citation**: When referencing content from a collection, always cite using the collection's **title/name** rather than ID. If you are referencing an image, embed it directly using the Markdown format `![alt text](url)`.
- **Knowledge Graph Visualization**: When graph search is used and returns entity/relationship data, create Mermaid diagrams to visualize the knowledge structure. Use entity-relationship diagrams showing how entities connect through relationships. Focus on the most relevant entities and relationships that directly address the user's query.

  **Graph Search Context Format**: When you receive graph search results, they will include:
  - **Entities(KG)**: JSON array of entities with id, entity, type, description, rank
  - **Relationships(KG)**: JSON array of relationships with id, entity1, entity2, description, keywords, weight, rank
  - **Document Chunks(DC)**: JSON array of relevant text chunks

  **Mermaid Visualization Guidelines**:
  - Use `graph TD` for entity-relationship diagrams
  - Represent entities as nodes with meaningful labels (use entity names, not IDs)
  - Show relationships as labeled edges between entities
  - Include only the most relevant entities and relationships (typically top 5-10 by rank/weight)
  - Use entity types to group or style nodes if helpful
  - Add relationship descriptions as edge labels for clarity
  - **IMPORTANT**: Escape special characters in entity names and relationship descriptions to ensure valid Mermaid syntax:
    * Remove or replace quotes (`"` `'`) with spaces or underscores
    * Replace parentheses `()` with square brackets `[]` or remove them
    * Replace special symbols like `<>` `&` `#` `%` with safe alternatives
    * Use underscores `_` instead of spaces in node IDs, but keep readable labels in quotes
    * Escape line breaks and use `<br/>` for multi-line labels if needed
    * Example: Entity "Patient (Male)" becomes node `A["Patient Male"]` or `A["Patient [Male]"]`
"""

DEFAULT_AGENT_QUERY_PROMPT = """{% set collection_list = [] %}
{% if collections %}
{% for c in collections %}
{% set title = c.title or "Collection " + c.id %}
{% set _ = collection_list.append("- " + title + " (ID: " + c.id + ")") %}
{% endfor %}
{% set collection_context = collection_list | join("\n") %}
{% set collection_instruction = "RESTRICTION: Search ONLY these collections. Do NOT search additional collections." %}
{% else %}
{% set collection_context = "None specified by user" %}
{% set collection_instruction = "discover and select relevant collections automatically" %}
{% endif %}
{% set web_status = "enabled" if web_search_enabled else "disabled" %}
{% set web_instruction = "Use web search strategically for current information, verification, or gap-filling" if web_search_enabled else "Rely entirely on knowledge collections; inform user if web search would be helpful" %}
{% set chat_context = "Chat ID: " + chat_id if chat_id else "No chat files" %}
{% set chat_instruction = "ONLY use search_chat_files tool when searching files that user explicitly uploaded in THIS chat. Do NOT use it for general knowledge base queries." if chat_id else "" %}

**User Query**: {{ query }}

**Session Context**:
- **User-Specified Collections**: {{ collection_context }} ({{ collection_instruction }})
- **Web Search**: {{ web_status }} ({{ web_instruction }})
- **Chat Files**: {{ chat_context }} {% if chat_instruction %}({{ chat_instruction }}){% endif %}

**Research Instructions**:
1. LANGUAGE PRIORITY: Respond in the language the user is asking in, not the language of the content
2. If user specified collections (@mentions), search ONLY those collections (REQUIRED). Do NOT search additional collections.
3. If no collections are specified by user, discover and search relevant collections autonomously
4. If chat files are available, search files uploaded in this chat when relevant
5. Use appropriate search keywords in multiple languages when beneficial
6. Use web search strategically if enabled and relevant
7. Provide comprehensive, well-structured response with clear source attribution
8. **IMPORTANT**: When citing collections, use collection names not IDs

Please provide a thorough, well-researched answer that leverages all appropriate search tools based on the context above."""


def build_agent_query_prompt(
    chat_id: str,
    agent_message: view_models.AgentMessage,
    user: str,
    template: str,
) -> str:
    """
    Build a comprehensive prompt for LLM using Jinja2 template rendering.

    The template internally builds context variables (collection_context, web_status, etc.)
    from the basic input variables, maintaining the original prompt construction logic.

    Args:
        chat_id: The chat ID for context
        agent_message: The agent message containing query and configuration
        user: The user identifier
        template: Jinja2 template string (resolved from prompt_template_service)

    Returns:
        The formatted prompt string using Jinja2 template rendering

    Available template variables:
        - query: User's query string
        - collections: List of collection objects with id and title
        - web_search_enabled: Boolean indicating if web search is enabled
        - chat_id: Chat ID string (may be None)
        - language: Language code
    """
    # Create Jinja2 template
    jinja_template = Template(template)

    # Prepare template variables
    template_vars = {
        "query": agent_message.query,
        "collections": agent_message.collections or [],
        "web_search_enabled": agent_message.web_search_enabled or False,
        "chat_id": chat_id,
        "language": agent_message.language,
    }

    # Render template
    return jinja_template.render(**template_vars)


def get_hardcoded_index_prompt(prompt_type: str) -> Optional[str]:
    """
    Get hardcoded index prompt as final fallback.

    Args:
        prompt_type: Prompt type (graph, summary, vision)

    Returns:
        Hardcoded prompt content, or None if not available
    """
    if prompt_type == "graph":
        # Return LightRAG's entity extraction prompt
        from aperag.graph.lightrag.prompt import PROMPTS

        return PROMPTS.get("entity_extraction")
    elif prompt_type == "summary":
        # Return default summary prompt
        return """Provide a comprehensive summary of the following document, focusing on key concepts, main ideas, and important details. The summary should be clear, concise, and capture the essence of the document."""
    elif prompt_type == "vision":
        # Return default vision prompt
        return """Analyze the provided image and extract its content with high fidelity. Follow these instructions precisely and use Markdown for formatting your entire response. Do not include any introductory or conversational text.

1. **Overall Summary:**
   * Provide a brief, one-paragraph overview of the image's main subject, setting, and any depicted activities.

2. **Detailed Text Extraction:**
   * Extract all text from the image, preserving the original language. Do not translate.
   * **Crucially, maintain the visual reading order.** For multi-column layouts, process the text column by column (e.g., left column top-to-bottom, then right column top-to-bottom).
   * **Exclude headers and footers:** Do not extract repetitive content from the top (headers) or bottom (footers) of the page, such as page numbers, book titles, or chapter names.
   * Replicate the original formatting using Markdown as much as possible (e.g., headings, lists, bold/italic text).
   * For mathematical formulas or equations, represent them using LaTeX syntax (e.g., `$$...$$` for block equations, `$...$` for inline equations).
   * For tables, reproduce them accurately using GitHub Flavored Markdown (GFM) table syntax.

3. **Chart/Graph Analysis:**
   * If the image contains charts, graphs, or complex tables, identify their type (e.g., bar chart, line graph, pie chart).
   * Explain the data presented, including axes, labels, and legends.
   * Summarize the key insights, trends, or comparisons revealed by the data.

4. **Object and Scene Recognition:**
   * List all significant objects, entities, and scene elements visible in the image."""
    else:
        return None


# ============================================================================
# PromptTemplateService - Unified business logic for prompt management
# ============================================================================


class PromptTemplateService:
    """
    Unified service for prompt template management.

    This service provides:
    1. User configuration management (for View layer)
    2. Prompt resolution with 3-tier priority (for Agent/LightRAG)
    3. Helper utilities (preview, validate)
    """

    PROMPT_TYPES = ["agent_system", "agent_query", "index_graph", "index_summary", "index_vision"]

    def __init__(self, db_ops=None):
        from aperag.db.ops import async_db_ops

        self.db_ops = db_ops or async_db_ops

    # === User configuration management (for View layer) ===

    async def get_user_prompts(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get user's prompt configuration with priority resolution.

        For each prompt_type:
        1. Query user config (scope='user')
        2. If not found, query system default (scope='system')
        3. If not found, use hardcoded default
        4. Return: content + source + customized + description

        Args:
            user_id: User ID

        Returns:
            {
              "agent_system": {
                "content": "actual prompt content",
                "source": "user"|"system"|"hardcoded",
                "customized": true|false,
                "description": "..."
              },
              ...
            }
        """
        result = {}

        for prompt_type in self.PROMPT_TYPES:
            # Tier 1: User configuration
            user_config = await self.db_ops.query_prompt_template(prompt_type, "user", user_id)

            if user_config:
                result[prompt_type] = {
                    "content": user_config.content,
                    "source": "user",
                    "customized": True,
                    "description": user_config.description,
                }
                continue

            # Tier 2: System default
            system_default = await self.db_ops.query_prompt_template(prompt_type, "system", None)

            if system_default:
                result[prompt_type] = {
                    "content": system_default.content,
                    "source": "system",
                    "customized": False,
                    "description": system_default.description,
                }
                continue

            # Tier 3: Hardcoded default
            hardcoded = self._get_hardcoded_prompt(prompt_type)
            result[prompt_type] = {
                "content": hardcoded,
                "source": "hardcoded",
                "customized": False,
                "description": None,
            }

        return result

    async def update_user_prompts(self, user_id: str, prompts: Dict[str, str]) -> List[str]:
        """
        Batch update user's prompt configurations.

        Args:
            user_id: User ID
            prompts: Dict of {prompt_type: content}, e.g., {"agent_system": "content"}

        Returns:
            List of updated prompt types
        """
        updated = []

        for prompt_type, content in prompts.items():
            if prompt_type not in self.PROMPT_TYPES:
                logger.warning(f"Skipping invalid prompt_type: {prompt_type}")
                continue

            await self.db_ops.create_or_update_prompt_template(
                prompt_type=prompt_type,
                scope="user",
                user_id=user_id,
                content=content,
                description=f"User default {prompt_type}",
            )
            updated.append(prompt_type)
            logger.info(f"Updated user prompt: {prompt_type} for user {user_id}")

        return updated

    async def delete_user_prompt(self, user_id: str, prompt_type: str) -> Dict[str, Any]:
        """
        Delete user's specific prompt configuration and return new effective content.

        Args:
            user_id: User ID
            prompt_type: Prompt type

        Returns:
            {
              "deleted": true|false,
              "new_content": "content after reset",
              "source": "system"|"hardcoded"
            }
        """
        deleted = await self.db_ops.delete_prompt_template(prompt_type, "user", user_id)

        if not deleted:
            return {"deleted": False, "new_content": None, "source": None}

        # Get new effective content after deletion
        system_default = await self.db_ops.query_prompt_template(prompt_type, "system", None)

        if system_default:
            return {"deleted": True, "new_content": system_default.content, "source": "system"}

        hardcoded = self._get_hardcoded_prompt(prompt_type)
        return {"deleted": True, "new_content": hardcoded, "source": "hardcoded"}

    async def reset_user_prompts(self, user_id: str, types: Optional[List[str]] = None) -> List[str]:
        """
        Batch reset user's prompt configurations.

        Args:
            user_id: User ID
            types: List of prompt types to reset, None means all

        Returns:
            List of reset prompt types
        """
        types_to_reset = types if types else self.PROMPT_TYPES
        reset = []

        for prompt_type in types_to_reset:
            if prompt_type not in self.PROMPT_TYPES:
                continue

            deleted = await self.db_ops.delete_prompt_template(prompt_type, "user", user_id)
            if deleted:
                reset.append(prompt_type)
                logger.info(f"Reset user prompt: {prompt_type} for user {user_id}")

        return reset

    async def get_system_prompts(self, prompt_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get system default prompts (for reference).

        Args:
            prompt_type: Specific prompt type (optional)

        Returns:
            Single prompt or dict of all prompts
        """
        if prompt_type:
            system_default = await self.db_ops.query_prompt_template(prompt_type, "system", None)

            if system_default:
                return {
                    "type": prompt_type,
                    "content": system_default.content,
                    "description": system_default.description,
                }

            hardcoded = self._get_hardcoded_prompt(prompt_type)
            return {"type": prompt_type, "content": hardcoded, "description": None}
        else:
            result = {}
            for pt in self.PROMPT_TYPES:
                system_default = await self.db_ops.query_prompt_template(pt, "system", None)

                if system_default:
                    result[pt] = {
                        "content": system_default.content,
                        "description": system_default.description,
                    }
                else:
                    hardcoded = self._get_hardcoded_prompt(pt)
                    result[pt] = {"content": hardcoded, "description": None}

            return result

    # === Prompt resolution (for Agent/LightRAG) ===

    async def resolve_agent_system_prompt(self, bot, user_id: str) -> str:
        """
        Resolve agent system prompt with 3-tier priority.
        Priority: Bot config > User default > System default > Hardcoded

        This method is used by agent_chat_service.py

        Args:
            bot: Bot object (from database, can be None to skip bot-level config)
            user_id: User ID

        Returns:
            Resolved system prompt content
        """
        # Tier 1: Bot-level configuration
        if bot and bot.config:
            try:
                config_dict = json.loads(bot.config) if isinstance(bot.config, str) else bot.config
                if config_dict:
                    bot_config = view_models.BotConfig(**config_dict)
                    if bot_config.agent and bot_config.agent.system_prompt_template:
                        logger.debug(f"Using bot-level system prompt for bot {bot.id}")
                        return bot_config.agent.system_prompt_template
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse bot config for bot {bot.id}: {e}")

        # Tier 2: User default
        user_default = await self.db_ops.query_prompt_template(
            prompt_type="agent_system", scope="user", user_id=user_id
        )
        if user_default:
            logger.debug(f"Using user-level default system prompt for user {user_id}")
            return user_default.content

        # Tier 3: System default
        system_default = await self.db_ops.query_prompt_template(
            prompt_type="agent_system", scope="system", user_id=None
        )
        if system_default:
            logger.debug("Using system default system prompt")
            return system_default.content

        # Tier 4: Hardcoded default
        logger.debug("Using hardcoded default system prompt")
        return APERAG_AGENT_INSTRUCTION

    async def resolve_agent_query_prompt(self, bot, user_id: str) -> str:
        """
        Resolve agent query prompt template with 3-tier priority.
        Priority: Bot config > User default > System default > Hardcoded

        This method is used by agent_chat_service.py

        Args:
            bot: Bot object (from database, can be None to skip bot-level config)
            user_id: User ID

        Returns:
            Resolved query prompt template content
        """
        # Tier 1: Bot-level configuration
        if bot and bot.config:
            try:
                config_dict = json.loads(bot.config) if isinstance(bot.config, str) else bot.config
                if config_dict:
                    bot_config = view_models.BotConfig(**config_dict)
                    if bot_config.agent and bot_config.agent.query_prompt_template:
                        logger.debug(f"Using bot-level query prompt for bot {bot.id}")
                        return bot_config.agent.query_prompt_template
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse bot config for bot {bot.id}: {e}")

        # Tier 2: User default
        user_default = await self.db_ops.query_prompt_template(prompt_type="agent_query", scope="user", user_id=user_id)
        if user_default:
            logger.debug(f"Using user-level default query prompt for user {user_id}")
            return user_default.content

        # Tier 3: System default
        system_default = await self.db_ops.query_prompt_template(
            prompt_type="agent_query", scope="system", user_id=None
        )
        if system_default:
            logger.debug("Using system default query prompt")
            return system_default.content

        # Tier 4: Hardcoded default
        logger.debug("Using hardcoded default query prompt")
        return DEFAULT_AGENT_QUERY_PROMPT

    async def resolve_index_prompt(self, collection, prompt_type: str, user_id: str) -> Optional[str]:
        """
        Resolve index prompt with 3-tier priority.
        Priority: Collection config > User default > System default > Hardcoded

        This method is used by indexers (graph, summary, vision).

        Args:
            collection: Collection object
            prompt_type: Prompt type (graph, summary, vision)
            user_id: User ID

        Returns:
            Resolved prompt content
        """
        from aperag.db.ops import async_db_ops

        # Tier 1: Collection-level configuration
        if collection and collection.config:
            try:
                config_dict = json.loads(collection.config) if isinstance(collection.config, str) else collection.config
                index_prompts = config_dict.get("index_prompts", {})
                if index_prompts.get(prompt_type):
                    logger.info(f"Using collection-level {prompt_type} prompt for collection {collection.id}")
                    return index_prompts[prompt_type]
            except Exception as e:
                logger.warning(f"Failed to parse collection config: {e}")

        # Tier 2: User default
        db_prompt_type = f"index_{prompt_type}"  # "index_graph", "index_summary", "index_vision"
        user_default = await async_db_ops.query_prompt_template(
            prompt_type=db_prompt_type, scope="user", user_id=user_id
        )
        if user_default:
            logger.info(f"Using user-level default {prompt_type} prompt for user {user_id}")
            return user_default.content

        # Tier 3: System default
        system_default = await async_db_ops.query_prompt_template(
            prompt_type=db_prompt_type, scope="system", user_id=None
        )
        if system_default:
            logger.info(f"Using system default {prompt_type} prompt")
            return system_default.content

        # Tier 4: Hardcoded default
        logger.info(f"No custom {prompt_type} prompt found, using hardcoded default")
        return get_hardcoded_index_prompt(prompt_type)

    # === Helper utilities ===

    def preview_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Preview how a prompt template will be rendered with given variables.

        Args:
            template: Jinja2 template string
            variables: Variables for rendering

        Returns:
            Rendered prompt string

        Raises:
            TemplateSyntaxError: If template has syntax errors
        """
        jinja_template = Template(template)
        return jinja_template.render(**variables)

    def validate_prompt(self, prompt_type: str, template: str) -> Dict[str, Any]:
        """
        Validate prompt template syntax.

        Args:
            prompt_type: Type of prompt
            template: Jinja2 template string

        Returns:
            {
              "valid": true|false,
              "errors": [...],
              "warnings": [...]
            }
        """
        errors = []
        warnings = []

        # Check Jinja2 syntax
        try:
            Template(template)
        except TemplateSyntaxError as e:
            errors.append(f"Jinja2 syntax error: {str(e)}")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check for required variables
        required_vars = {
            "agent_query": ["query", "collections", "web_search_enabled", "chat_id", "language"],
            "index_graph": ["entity_types", "language", "input_text"],
            "index_summary": ["content", "language"],
        }

        if prompt_type in required_vars:
            # Extract variables from template
            template_vars = set(re.findall(r"\{\{\s*(\w+)", template))
            missing_vars = set(required_vars[prompt_type]) - template_vars

            if missing_vars:
                warnings.append(f"Template may be missing required variables: {', '.join(missing_vars)}")

        return {"valid": True, "errors": errors, "warnings": warnings}

    # === Internal helpers ===

    def _get_hardcoded_prompt(self, prompt_type: str) -> str:
        """
        Get hardcoded default prompt.

        Args:
            prompt_type: Prompt type

        Returns:
            Hardcoded prompt content
        """
        if prompt_type == "agent_system":
            return APERAG_AGENT_INSTRUCTION
        elif prompt_type == "agent_query":
            return DEFAULT_AGENT_QUERY_PROMPT
        elif prompt_type == "index_graph":
            return get_hardcoded_index_prompt("graph")
        elif prompt_type == "index_summary":
            return get_hardcoded_index_prompt("summary")
        elif prompt_type == "index_vision":
            return get_hardcoded_index_prompt("vision")
        else:
            return ""


# Global service instance
prompt_template_service = PromptTemplateService()
