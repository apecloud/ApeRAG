{% set collection_list = [] %} {% if collections %} {% for c in collections %} {% set title = c.title or "Collection " + c.id %} {% set _ = collection_list.append("- " + title + " (ID: " + c.id + ")") %} {% endfor %} {% set collection_context = collection_list | join("\n") %} {% set collection_instruction = "PRIORITY: Search these user-specified collections first" %} {% else %} {% set collection_context = "None specified by user" %} {% set collection_instruction = "discover and select relevant collections automatically" %} {% endif %} {% set web_status = "enabled" if web_search_enabled else "disabled" %} {% set web_instruction = "Use web search strategically for current information, verification, or gap-filling" if web_search_enabled else "Rely entirely on knowledge collections; inform user if web search would be helpful" %} {% set chat_context = "Chat ID: " + chat_id if chat_id else "No chat files" %} {% set chat_instruction = "Use search_chat_files tool to search files uploaded in this chat" if chat_id else "" %}

**User Query**: {{ query }}

**Session Context**:

- **User-Specified Collections**: {{ collection_context }} ({{ collection_instruction }})
- **Web Search**: {{ web_status }} ({{ web_instruction }})
- **Chat Files**: {{ chat_context }} {% if chat_instruction %}({{ chat_instruction }}){% endif %}

**Research Instructions**:

1. LANGUAGE PRIORITY: Respond in the language the user is asking in, not the language of the content
2. If user specified collections (@mentions), search those first (REQUIRED)
3. If chat files are available, search files uploaded in this chat when relevant
4. Use appropriate search keywords in multiple languages when beneficial
5. Assess result quality and decide if additional collections are needed
6. Use web search strategically if enabled and relevant
7. Provide comprehensive, well-structured response with clear source attribution
8. Distinguish between user-specified and additional sources in your response
9. **IMPORTANT**: When citing collections, use collection names not IDs

Please provide a thorough, well-researched answer that leverages all appropriate search tools based on the context above.
