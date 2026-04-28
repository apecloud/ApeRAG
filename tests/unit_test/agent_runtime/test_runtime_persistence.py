from aperag.domains.agent_runtime.runtime import (
    _compose_assistant_parts,
    _PersistedToolCall,
)
from aperag.domains.agent_runtime.schemas import ReferenceBundleItem
from aperag.domains.agent_runtime.uimessage import DataCitationPart, SourceUrlPart, TextPart, ToolPart


def test_compose_assistant_parts_persists_tool_lifecycle_for_reload():
    """Reload snapshots must retain tool calls, not just answer text.

    Live SSE renders tool lifecycle from timeline events. After a
    browser refresh the FE reads ``AgentTurnSnapshot.parts`` from the
    durable ``agent_message`` row, so the runtime must persist a
    collapsed ``ToolPart`` for every tool call.
    """

    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="Answer body",
        references=[
            ReferenceBundleItem(
                source_type="web",
                source_id="src-1",
                title="Doc A",
                snippet="Reference text",
                uri="https://example.com/doc-a",
            )
        ],
        tool_calls=[
            _PersistedToolCall(
                tool_call_id="call-1",
                tool_name="web.search",
                state="output-available",
            )
        ],
    )

    assert [part.type for part in parts] == [
        "tool-web_search",
        "text",
        "source-url",
        "data-citation",
    ]
    assert isinstance(parts[0], ToolPart)
    assert parts[0].tool_call_id == "call-1"
    assert parts[0].state == "output-available"
    assert parts[0].metadata == {"mcpToolName": "web.search"}
    assert parts[0].input is None, "raw tool args must not be persisted"
    assert isinstance(parts[1], TextPart)
    assert isinstance(parts[2], SourceUrlPart)
    assert isinstance(parts[3], DataCitationPart)


def test_compose_assistant_parts_persists_tool_error_state():
    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="",
        references=[],
        tool_calls=[
            _PersistedToolCall(
                tool_call_id="call-err",
                tool_name="knowledge_base.read_document",
                state="output-error",
                error_text="read timed out",
            )
        ],
    )

    assert len(parts) == 1
    assert isinstance(parts[0], ToolPart)
    assert parts[0].type == "tool-knowledge_base_read_document"
    assert parts[0].state == "output-error"
    assert parts[0].error_text == "read timed out"
