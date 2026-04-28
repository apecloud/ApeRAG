from aperag.domains.agent_runtime.runtime import (
    _TOOL_SUMMARY_MAX_LEN,
    _compose_assistant_parts,
    _extract_tool_summary,
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


def test_compose_assistant_parts_persists_tool_summary():
    """Reload paths must surface the user-visible activity subtitle
    (e.g. "搜索:张飞牛肉") via ``ToolPart.summary``. Without it the FE
    can only show generic "已完成网页搜索" copy after refresh because
    ``input`` is intentionally not persisted (D9 §A7)."""
    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="",
        references=[],
        tool_calls=[
            _PersistedToolCall(
                tool_call_id="call-1",
                tool_name="web.search",
                state="output-available",
                summary="搜索:张飞牛肉",
            )
        ],
    )
    assert len(parts) == 1
    assert isinstance(parts[0], ToolPart)
    assert parts[0].summary == "搜索:张飞牛肉"
    assert parts[0].input is None, "raw tool args must not be persisted (D9 §A7)"


def test_extract_tool_summary_search_query():
    assert _extract_tool_summary({"query": "张飞牛肉"}) == "搜索:张飞牛肉"
    assert _extract_tool_summary({"q": "beekeeping"}) == "搜索:beekeeping"
    assert _extract_tool_summary({"keyword": "wave 9"}) == "搜索:wave 9"
    # Whitespace-only / missing → None (FE falls back to generic copy)
    assert _extract_tool_summary({"query": "   "}) is None
    assert _extract_tool_summary({"unrelated": "x"}) is None


def test_extract_tool_summary_url_fields_compact_to_host_path():
    assert _extract_tool_summary({"url": "https://example.com/foo/bar"}) == "阅读:example.com/foo/bar"
    assert _extract_tool_summary({"uri": "https://example.com/"}) == "阅读:example.com"
    # Schemeless string passes through unchanged (still useful for FE).
    assert _extract_tool_summary({"link": "example.com/page"}) == "阅读:example.com/page"


def test_extract_tool_summary_truncates_long_values_with_ellipsis():
    long_query = "x" * (_TOOL_SUMMARY_MAX_LEN + 50)
    out = _extract_tool_summary({"query": long_query})
    assert out is not None
    assert len(out) == _TOOL_SUMMARY_MAX_LEN
    assert out.endswith("…")


def test_extract_tool_summary_query_takes_precedence_over_url():
    """A tool that takes both ``query`` and ``url`` (e.g. a hybrid
    search-with-fallback) prefers the human-readable query string."""
    out = _extract_tool_summary({"query": "张飞牛肉", "url": "https://example.com"})
    assert out == "搜索:张飞牛肉"


def test_extract_tool_summary_non_dict_returns_none():
    assert _extract_tool_summary(None) is None
    assert _extract_tool_summary("just a string") is None
    assert _extract_tool_summary(["a", "b"]) is None
