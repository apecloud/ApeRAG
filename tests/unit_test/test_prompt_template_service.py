from aperag.domains.agent_runtime.schemas import AgentMessage
from aperag.domains.conversation.schemas import File
from aperag.domains.knowledge_base.schemas import Collection
from aperag.service import prompt_template_service as pts


def test_agent_system_prompt_keeps_activity_summary_boundary():
    prompt = pts.APERAG_AGENT_INSTRUCTION

    assert "activity-style summaries only" in prompt
    assert "Do not expose raw chain-of-thought." in prompt
    assert "**User request**" not in prompt


def test_agent_query_prompt_stays_dynamic_task_layer():
    prompt = pts.DEFAULT_AGENT_QUERY_PROMPT

    assert "**User request**: {{ query }}" in prompt
    assert "**Current scope**:" in prompt
    assert "**Deliverable for this turn**:" in prompt
    assert "Do not expose raw chain-of-thought." not in prompt


def test_build_agent_query_prompt_renders_dynamic_scope_only():
    agent_message = AgentMessage(
        query="What knowledge bases can I use?",
        collections=[Collection(id="col-1", title="Team Docs")],
        web_search_enabled=True,
        language="en-US",
    )

    prompt = pts.build_agent_query_prompt(
        chat_id="chat-123",
        agent_message=agent_message,
        user="user-1",
        template=pts.DEFAULT_AGENT_QUERY_PROMPT,
    )

    assert "**User request**: What knowledge bases can I use?" in prompt
    assert "Team Docs (ID: col-1)" in prompt
    assert "Web search: enabled" in prompt
    assert "Chat files: No chat files are available." in prompt
    assert "Use only the tools allowed by the scope above" in prompt


def test_build_agent_query_prompt_marks_chat_files_available_only_when_uploaded():
    agent_message = AgentMessage(
        query="Search the uploaded file",
        collections=[],
        web_search_enabled=False,
        language="en-US",
        files=[File(id="file-1", name="notes.pdf")],
    )

    prompt = pts.build_agent_query_prompt(
        chat_id="chat-123",
        agent_message=agent_message,
        user="user-1",
        template=pts.DEFAULT_AGENT_QUERY_PROMPT,
    )

    assert "Chat files: Files are available in this chat." in prompt


def test_build_agent_query_prompt_respects_chat_level_file_override():
    agent_message = AgentMessage(
        query="Continue using the files from earlier in this chat",
        collections=[],
        web_search_enabled=False,
        language="en-US",
        files=[],
    )

    prompt = pts.build_agent_query_prompt(
        chat_id="chat-123",
        agent_message=agent_message,
        user="user-1",
        template=pts.DEFAULT_AGENT_QUERY_PROMPT,
        has_chat_files=True,
    )

    assert "Chat files: Files are available in this chat." in prompt


def test_build_agent_query_prompt_handles_default_scope_and_language_fallback():
    """When ``language`` is None the prompt must fall back to the
    documented default — Simplified Chinese — instead of leaving the
    LLM to guess. Pins earayu2's "默认中文" requirement (msg=4d8373a4)."""
    agent_message = AgentMessage(
        query="Summarize the available sources",
        collections=[],
        web_search_enabled=False,
        language=None,
    )

    prompt = pts.build_agent_query_prompt(
        chat_id=None,
        agent_message=agent_message,
        user="user-1",
        template=pts.DEFAULT_AGENT_QUERY_PROMPT,
    )

    assert "None specified by user" in prompt
    assert "Collection rule: Discover and select relevant collections automatically." in prompt
    assert "Web search: disabled" in prompt
    assert "Chat files: No chat files are available." in prompt
    assert "请使用简体中文回答。" in prompt
    assert "do not switch languages mid-response" in prompt


def test_build_agent_query_prompt_emits_native_language_directive_for_known_locales():
    """Each supported BCP-47 code injects a *native-language* directive
    so the model can't fall back to its training-default language. The
    explicit native-script string is the lever — generic "respond in
    the user's language" copy does not survive provider hot-paths."""
    expectations = {
        "en-US": "Please answer in English.",
        "zh-CN": "请使用简体中文回答。",
        "zh-TW": "請使用繁體中文回答。",
        "ja-JP": "日本語で回答してください。",
        "ko-KR": "한국어로 답변해 주세요.",
        "fr-FR": "Veuillez répondre en français.",
        "de-DE": "Bitte antworten Sie auf Deutsch.",
        "es-ES": "Por favor, responda en español.",
        "it-IT": "Si prega di rispondere in italiano.",
        "pt-BR": "Por favor, responda em português.",
        "ru-RU": "Пожалуйста, ответьте на русском языке.",
    }
    for code, directive in expectations.items():
        agent_message = AgentMessage(
            query="dummy",
            collections=[],
            web_search_enabled=False,
            language=code,
        )
        prompt = pts.build_agent_query_prompt(
            chat_id=None,
            agent_message=agent_message,
            user="user-1",
            template=pts.DEFAULT_AGENT_QUERY_PROMPT,
        )
        assert directive in prompt, f"language={code} prompt missing expected directive {directive!r}"
