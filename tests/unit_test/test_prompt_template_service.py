from aperag.schema import view_models
from aperag.service import prompt_template_service as pts


def test_agent_system_prompt_keeps_activity_summary_boundary():
    prompt = pts.APERAG_AGENT_INSTRUCTION

    assert "activity-style summary" in prompt
    assert "Do not expose raw chain-of-thought." in prompt
    assert "**User request**" not in prompt


def test_agent_query_prompt_stays_dynamic_task_layer():
    prompt = pts.DEFAULT_AGENT_QUERY_PROMPT

    assert "**User request**: {{ query }}" in prompt
    assert "**Current scope**:" in prompt
    assert "**Deliverable for this turn**:" in prompt
    assert "Do not expose raw chain-of-thought." not in prompt


def test_build_agent_query_prompt_renders_dynamic_scope_only():
    agent_message = view_models.AgentMessage(
        query="What knowledge bases can I use?",
        collections=[view_models.Collection(id="col-1", title="Team Docs")],
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
    assert "Chat files: Files are available in this chat." in prompt
    assert "Use only the tools allowed by the scope above" in prompt
