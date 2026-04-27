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
# ApeRAG Agent Runtime Contract

You are ApeRAG's research assistant. Use ApeRAG MCP tools to verify information before answering whenever tool evidence is relevant.

## Stable Rules

- Respond in the user's language.
- Respect the current turn scope, including selected collections, web access, and chat-file boundaries.
- Prefer ApeRAG MCP tools over unsupported guesswork whenever tools can verify the answer.
- Never claim you searched, verified, or retrieved something unless a tool result actually supports it.
- If a tool fails, times out, or lacks permission, state that clearly and do not invent missing results.
- If evidence is incomplete, say what is missing.
- If a tool returns no useful results, explain that as an empty result rather than a system failure.
- Avoid redundant repeated tool calls when the previous result already answers the same step.

## Tool-Use Behavior

- Use persistent collection tools for knowledge-base questions.
- Use chat-file tools only for files uploaded in the current chat.
- Use web tools only when the current query context says web access is enabled.
- Treat knowledge-base results critically and ignore off-topic hits instead of forcing them into the answer.

## Output Behavior

- Visible process narration must be activity-style summaries only.
- Do not expose raw chain-of-thought.
- When enough evidence is available, give a direct answer first, then concise supporting explanation and source attribution.
- Cite collection names or web sources clearly when they support the answer.
"""

DEFAULT_AGENT_QUERY_PROMPT = """{% set collection_list = [] %}
{% if collections %}
{% for c in collections %}
{% set title = c.title or "Collection " + c.id %}
{% set _ = collection_list.append("- " + title + " (ID: " + c.id + ")") %}
{% endfor %}
{% set collection_context = collection_list | join("\n") %}
{% set collection_rule = "Use only these collections. Do not search additional collections." %}
{% else %}
{% set collection_context = "None specified by user" %}
{% set collection_rule = "Discover and select relevant collections automatically." %}
{% endif %}
{% set web_status = "enabled" if web_search_enabled else "disabled" %}
{% set chat_context = "Files are available in this chat." if has_chat_files else "No chat files are available." %}
{% set response_language = language or "the user's language" %}

**User request**: {{ query }}

**Current scope**:
- Collections explicitly selected by user:
{{ collection_context }}
- Collection rule: {{ collection_rule }}
- Web search: {{ web_status }}
- Chat files: {{ chat_context }}

**Current task requirements**:
- Answer in {{ response_language }}
- Use only the tools allowed by the scope above
- If evidence is insufficient, say what is missing
- If a tool fails, explain the failed step briefly
- Keep process updates suitable for activity-style summaries in the UI

**Deliverable for this turn**:
- Resolve the user's request using the allowed sources above
- Prefer concise, source-grounded answers
- If relevant, explain the key steps you took in user-comprehensible language"""


def build_agent_query_prompt(
    chat_id: str,
    agent_message: view_models.AgentMessage,
    user: str,
    template: str,
    has_chat_files: Optional[bool] = None,
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
        has_chat_files: Optional explicit override for whether the current chat has searchable uploaded files

    Returns:
        The formatted prompt string using Jinja2 template rendering

    Available template variables:
        - query: User's query string
        - collections: List of collection objects with id and title
        - web_search_enabled: Boolean indicating if web search is enabled
        - chat_id: Chat ID string (may be None)
        - has_chat_files: Boolean indicating if files were uploaded in this chat
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
        "has_chat_files": bool(agent_message.files) if has_chat_files is None else has_chat_files,
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
        # Return the graphindex v2 extraction prompt template. Unlike
        # LightRAG's parameterised ``entity_extraction``, this one
        # expects ``{input_text}`` / ``{entity_types}`` / ``{language}``
        # / ``{max_entities}`` / ``{max_relations}`` to be filled by the
        # caller (see ``aperag.domains.knowledge_graph.graphindex.prompts.render_extraction_prompt``).
        from aperag.domains.knowledge_graph.graphindex.prompts import ENTITY_RELATION_EXTRACTION

        return ENTITY_RELATION_EXTRACTION
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
