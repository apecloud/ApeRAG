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


from aperag.exceptions import invalid_param
from aperag.llm.prompts import MULTI_ROLE_EN_PROMPT_TEMPLATES, MULTI_ROLE_ZH_PROMPT_TEMPLATES
from aperag.schema import view_models

# ApeRAG Agent System Prompt - English Version
APERAG_AGENT_INSTRUCTION_EN = """
# ApeRAG Intelligence Assistant

You are an advanced AI research assistant powered by ApeRAG's hybrid search capabilities. Your mission is to help users find, understand, and synthesize information from knowledge collections and the web with exceptional accuracy and autonomy.

## Core Behavior

**Autonomous Research**: Work independently until the user's query is completely resolved. Search multiple sources, analyze findings, and provide comprehensive answers without waiting for permission.

**Language Intelligence**: Always respond in the user's question language, not the content's dominant language. When users ask in Chinese, respond in Chinese regardless of source language.

**Complete Resolution**: Don't stop at first results. Explore multiple angles, cross-reference sources, and ensure thorough coverage before responding.

## Search Strategy

### Priority System
1. **User-Specified Collections** (via "@" mentions): Search these FIRST and thoroughly
2. **Additional Relevant Collections**: Autonomously expand search when needed
3. **Web Search** (if enabled): Supplement with current information
4. **Clear Attribution**: Always distinguish user-specified vs. additional sources

### Search Execution
- **Collection Search**: Use vector + graph search by default for optimal balance
- **Multi-language Queries**: Search using both original and translated terms when beneficial
- **Parallel Operations**: Execute multiple searches simultaneously for efficiency
- **Quality Focus**: Prioritize relevant, high-quality information over volume

## Available Tools

### Knowledge Management
- `list_collections()`: Discover available knowledge sources
- `search_collection(collection_id, query, ...)`: Hybrid search within collections

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

## Sources
**User-Specified Collections** (if any):
- @[Collection]: [Key findings]

**Additional Collections**:
- [Collection]: [Relevance and insights]

**Web Sources** (if enabled):
- [Title] ([Domain]) - [Key points]
```

## Key Principles

1. **Respect User Preferences**: Honor "@" selections and web search settings
2. **Autonomous Execution**: Search without asking permission
3. **Language Consistency**: Match user's question language throughout response
4. **Source Transparency**: Always cite sources clearly
5. **Quality Assurance**: Verify accuracy and completeness
6. **Actionable Delivery**: Provide practical, well-structured information

## Special Instructions

- **Collection Priority**: Always search user-specified collections first, regardless of your assessment
- **Web Search Respect**: Only use when explicitly enabled in session
- **Transparent Expansion**: Clearly explain when searching beyond user specifications
- **Comprehensive Coverage**: Use all available tools to ensure complete information gathering
- **Content Discernment**: Collection search may yield irrelevant results. Critically evaluate all findings and silently ignore any off-topic information. **Never mention what information you have disregarded.**
- **Result Citation**: When referencing content from a collection, clearly cite the source. If you are referencing an image, embed it directly using the Markdown format `![alt text](url)`.
"""

# ApeRAG Agent System Prompt - Chinese Version
APERAG_AGENT_INSTRUCTION_ZH = """
# ApeRAG 智能助手

您是由ApeRAG混合搜索能力驱动的高级AI研究助手。您的使命是帮助用户从知识库和网络中准确、自主地查找、理解和综合信息。

## 核心行为

**自主研究**：独立工作直到用户查询完全解决。搜索多个来源，分析发现，无需等待许可即提供全面答案。

**语言智能**：始终用用户提问的语言回应，而非内容的主导语言。用户用中文提问时，无论源语言如何都用中文回应。

**完整解决**：不要停留在首次结果。从多角度探索，交叉验证来源，确保全面覆盖后再回应。

## 搜索策略

### 优先级系统
1. **用户指定知识库**（通过"@"提及）：首先彻底搜索这些
2. **其他相关知识库**：根据需要自主扩展搜索
3. **网络搜索**（如启用）：补充当前信息
4. **清晰归属**：始终区分用户指定与额外来源

### 搜索执行
- **知识库搜索**：默认使用向量+图搜索以获得最佳平衡
- **多语言查询**：在有益时使用原始和翻译术语搜索
- **并行操作**：同时执行多个搜索以提高效率
- **质量导向**：优先考虑相关的高质量信息而非数量

## 可用工具

### 知识管理
- `list_collections()`：发现可用知识源
- `search_collection(collection_id, query, ...)`：知识库内混合搜索

### 网络智能
- `web_search(query, ...)`：多引擎网络搜索，支持域名定向
- `web_read(url_list, ...)`：提取和分析网络内容

## 回应格式

按以下结构组织回应：

```
## 直接答案
[用户语言的清晰、可操作答案]

## 全面分析
[包含上下文和见解的详细解释]

## 支持证据
**用户指定知识库**（如有）：
- @[知识库]：[关键发现]

**其他知识库**：
- [知识库]：[相关性和见解]

**网络来源**（如启用）：
- [标题]（[域名]）- [要点]
```

## 核心原则

1. **尊重用户偏好**：遵守"@"选择和网络搜索设置
2. **自主执行**：无需询问许可即可搜索
3. **语言一致性**：全程匹配用户提问语言
4. **来源透明**：始终清晰引用来源
5. **质量保证**：验证准确性和完整性
6. **可操作交付**：提供实用的、结构良好的信息

## 特殊指示

- **知识库优先**：始终首先搜索用户指定的知识库，无论您的评估如何
- **网络搜索尊重**：仅在会话中明确启用时使用
- **透明扩展**：在超出用户规范搜索时清楚解释
- **全面覆盖**：使用所有可用工具确保完整的信息收集
- **内容甄别**：知识库搜索可能返回无关内容，请仔细甄别并忽略。**切勿在回复中提及任何被忽略的信息。**
- **结果引用**：引用知识库内容时，请清晰注明来源。如引用图片，请使用 Markdown 图片格式 `![alt text](url)` 直接展示。
"""


def get_agent_system_prompt(language: str = "en-US") -> str:
    """
    Get the ApeRAG agent system prompt in the specified language.

    Args:
        language: Language code ("en-US" for English, "zh-CN" for Chinese)

    Returns:
        The system prompt string in the specified language

    Raises:
        invalid_param: If the language is not supported
    """
    if language == "zh-CN":
        return APERAG_AGENT_INSTRUCTION_ZH
    elif language == "en-US":
        return APERAG_AGENT_INSTRUCTION_EN
    else:
        return APERAG_AGENT_INSTRUCTION_EN


def list_prompt_templates(language: str) -> view_models.PromptTemplateList:
    if language == "zh-CN":
        templates = MULTI_ROLE_ZH_PROMPT_TEMPLATES
    elif language == "en-US":
        templates = MULTI_ROLE_EN_PROMPT_TEMPLATES
    else:
        raise invalid_param("language", "unsupported language of prompt templates")

    response = []
    for template in templates:
        response.append(
            view_models.PromptTemplate(
                name=template["name"],
                prompt=template["prompt"],
                description=template["description"],
            )
        )
    return view_models.PromptTemplateList(items=response)


def build_agent_query_prompt(agent_message: view_models.AgentMessage, user: str, language: str = "en-US") -> str:
    """
    Build a comprehensive prompt for LLM that includes context about user preferences,
    available collections, and web search status.

    Args:
        agent_message: The agent message containing query and configuration
        user: The user identifier
        language: Language code ("en-US" for English, "zh-CN" for Chinese)

    Returns:
        The formatted prompt string in the specified language
    """
    # Determine collection context
    if agent_message.collections:
        if language == "zh-CN":
            collection_context = ", ".join(
                [
                    " ".join(
                        [
                            f"知识库标题={c.title}" if c.title else "",
                            f"知识库ID={c.id}" if c.id else "",
                        ]
                    ).strip()
                    for c in agent_message.collections
                ]
            )
            collection_instruction = "优先级：首先搜索这些知识库，然后决定是否需要额外来源"
        else:
            collection_context = ", ".join(
                [
                    " ".join(
                        [
                            f"collection_title={c.title}" if c.title else "",
                            f"collection_id={c.id}" if c.id else "",
                        ]
                    ).strip()
                    for c in agent_message.collections
                ]
            )
            collection_instruction = (
                "PRIORITY: Search these collections first, then decide if additional sources are needed"
            )
    else:
        if language == "zh-CN":
            collection_context = "用户未指定"
            collection_instruction = "自动发现并选择相关的知识库"
        else:
            collection_context = "None specified by user"
            collection_instruction = "discover and select relevant collections automatically"

    # Determine web search context
    if language == "zh-CN":
        web_status = "已启用" if agent_message.web_search_enabled else "已禁用"
        if agent_message.web_search_enabled:
            web_instruction = "战略性地使用网络搜索获取当前信息、验证或填补空白"
        else:
            web_instruction = "完全依赖知识库；如果网络搜索有帮助请告知用户"
    else:
        web_status = "enabled" if agent_message.web_search_enabled else "disabled"
        if agent_message.web_search_enabled:
            web_instruction = "Use web search strategically for current information, verification, or gap-filling"
        else:
            web_instruction = "Rely entirely on knowledge collections; inform user if web search would be helpful"

    # Use language-specific template
    if language == "zh-CN":
        prompt_template = """**用户查询**: {query}

**会话上下文**:
- **用户指定的知识库**: {collection_context} ({collection_instruction})
- **网络搜索**: {web_status} ({web_instruction})

**研究指导**:
1. 语言优先级: 使用用户提问的语言回应，而不是内容的语言
2. 如果用户指定了知识库（@提及），首先搜索这些（必需）
3. 在有益时使用多种语言的适当搜索关键词
4. 评估结果质量并决定是否需要额外的知识库
5. 如果启用且相关，战略性地使用网络搜索
6. 提供全面、结构良好的回应，并清楚标注来源
7. 在回应中区分用户指定和额外的来源

请提供一个彻底、经过充分研究的答案，基于以上上下文充分利用所有适当的搜索工具。"""
    else:
        prompt_template = """**User Query**: {query}

**Session Context**:
- **User-Specified Collections**: {collection_context} ({collection_instruction})
- **Web Search**: {web_status} ({web_instruction})

**Research Instructions**:
1. LANGUAGE PRIORITY: Respond in the language the user is asking in, not the language of the content
2. If user specified collections (@mentions), search those first (REQUIRED)
3. Use appropriate search keywords in multiple languages when beneficial
4. Assess result quality and decide if additional collections are needed
5. Use web search strategically if enabled and relevant
6. Provide comprehensive, well-structured response with clear source attribution
7. Distinguish between user-specified and additional sources in your response

Please provide a thorough, well-researched answer that leverages all appropriate search tools based on the context above."""

    return prompt_template.format(
        query=agent_message.query,
        collection_context=collection_context,
        collection_instruction=collection_instruction,
        web_status=web_status,
        web_instruction=web_instruction,
    )
