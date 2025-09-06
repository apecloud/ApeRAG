{% set collection_list = [] %} {% if collections %} {% for c in collections %} {% set title = c.title or "知识库" + c.id %} {% set _ = collection_list.append("- " + title + " (ID: " + c.id + ")") %} {% endfor %} {% set collection_context = collection_list | join("\n") %} {% set collection_instruction = "优先级：首先搜索这些用户指定的知识库" %} {% else %} {% set collection_context = "用户未指定" %} {% set collection_instruction = "自动发现并选择相关的知识库" %} {% endif %} {% set web_status = "已启用" if web_search_enabled else "已禁用" %} {% set web_instruction = "战略性地使用网络搜索获取当前信息、验证或填补空白" if web_search_enabled else "完全依赖知识库；如果网络搜索有帮助请告知用户" %} {% set chat_context = "聊天ID: " + chat_id if chat_id else "无" %} {% set chat_instruction = "可使用 search_chat_files 工具搜索此聊天中上传的文件" if chat_id else "" %}

**用户查询**: {{ query }}

**会话上下文**:

- **用户指定的知识库**: {{ collection_context }} ({{ collection_instruction }})
- **网络搜索**: {{ web_status }} ({{ web_instruction }})
- **聊天文件**: {{ chat_context }} {% if chat_instruction %}({{ chat_instruction }}){% endif %}

**研究指导**:

1. 语言优先级: 使用用户提问的语言回应，而不是内容的语言
2. 如果用户指定了知识库（@提及），首先搜索这些（必需）
3. 如果有聊天文件，可以搜索聊天中上传的文件
4. 在有益时使用多种语言的适当搜索关键词
5. 评估结果质量并决定是否需要额外的知识库
6. 如果启用且相关，战略性地使用网络搜索
7. 提供全面、结构良好的回应，并清楚标注来源
8. 在回应中区分用户指定和额外的来源
9. **重要**：引用知识库时，使用知识库名称而非ID

请提供一个彻底、经过充分研究的答案，基于以上上下文充分利用所有适当的搜索工具。
