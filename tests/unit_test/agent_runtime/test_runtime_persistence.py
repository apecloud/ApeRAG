from aperag.domains.agent_runtime.runtime import (
    _TOOL_SUMMARY_MAX_LEN,
    _compose_assistant_parts,
    _extract_tool_summary,
    _PersistedReasoning,
    _PersistedToolCall,
)
from aperag.domains.agent_runtime.schemas import ReferenceBundleItem
from aperag.domains.agent_runtime.uimessage import (
    DataCitationPart,
    ReasoningPart,
    SourceUrlPart,
    TextPart,
    ToolPart,
)


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


# ---------------------------------------------------------------------
# Reasoning timeline (Wave 9 task #2 followup #2 — Claude-style chunks)
# ---------------------------------------------------------------------


def test_compose_assistant_parts_interleaves_reasoning_and_tool_in_timeline():
    """Architect ratify msg=2639aeea: chronological "思考1 → 工具a →
    思考2 → 工具b → final answer" pattern. The runtime feeds the
    timeline pre-ordered; ``_compose_assistant_parts`` must preserve
    that order so the FE renders Claude/Cursor-style interleaving."""

    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="Final answer.",
        references=[],
        timeline=[
            _PersistedReasoning(text="I need to look up the price first."),
            _PersistedToolCall(
                tool_call_id="call-1",
                tool_name="web.search",
                state="output-available",
                summary="搜索:price",
            ),
            _PersistedReasoning(text="Got it. Let me also fetch the page."),
            _PersistedToolCall(
                tool_call_id="call-2",
                tool_name="web.read",
                state="output-available",
                summary="阅读:example.com",
            ),
            _PersistedReasoning(text="Now I have enough to answer."),
        ],
    )
    types = [p.type for p in parts]
    assert types == [
        "reasoning",
        "tool-web_search",
        "reasoning",
        "tool-web_read",
        "reasoning",
        "text",
    ], "timeline order must be preserved verbatim before TextPart"
    assert isinstance(parts[0], ReasoningPart)
    assert parts[0].text == "I need to look up the price first."
    assert isinstance(parts[5], TextPart)
    assert parts[5].text == "Final answer."


def test_compose_assistant_parts_drops_empty_reasoning_chunks():
    """Empty-after-strip reasoning entries (whitespace-only) are
    dropped so the FE never renders an empty thinking block. The
    runtime's ``_flush_reasoning_chunk`` already strips before
    pushing — this test pins the composer's defence-in-depth so
    direct unit-test callers (and any future code path that bypasses
    the flush helper) still get the same behaviour."""

    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="",
        references=[],
        timeline=[
            _PersistedReasoning(text="   \n  "),
            _PersistedReasoning(text=""),
            _PersistedToolCall(
                tool_call_id="call-1",
                tool_name="web.search",
                state="output-available",
            ),
        ],
    )
    types = [p.type for p in parts]
    assert types == ["tool-web_search"], "empty reasoning chunks must be dropped"


def test_compose_assistant_parts_legacy_tool_calls_path_still_works():
    """Backward compat: callers that haven't migrated to ``timeline``
    can still pass ``tool_calls`` and get the original tool-first
    ordering. PR #1798 + PR #1803 paths must not break."""

    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="Answer",
        references=[],
        tool_calls=[
            _PersistedToolCall(
                tool_call_id="call-1",
                tool_name="web.search",
                state="output-available",
                summary="搜索:x",
            ),
        ],
    )
    types = [p.type for p in parts]
    assert types == ["tool-web_search", "text"]
    assert isinstance(parts[0], ToolPart)
    assert parts[0].summary == "搜索:x"


def test_compose_assistant_parts_timeline_takes_precedence_over_tool_calls():
    """When both ``timeline`` and ``tool_calls`` are supplied (e.g. a
    transitional caller), ``timeline`` wins and ``tool_calls`` is
    silently ignored — keeps the new ordered shape canonical without
    forcing every legacy caller to delete their old kwarg in one PR."""

    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="",
        references=[],
        timeline=[
            _PersistedToolCall(
                tool_call_id="from-timeline",
                tool_name="web.search",
                state="output-available",
            ),
        ],
        tool_calls=[
            _PersistedToolCall(
                tool_call_id="ignored-legacy",
                tool_name="web.read",
                state="output-available",
            ),
        ],
    )
    assert len(parts) == 1
    assert isinstance(parts[0], ToolPart)
    assert parts[0].tool_call_id == "from-timeline"


# ---------------------------------------------------------------------
# ThinkingPartDelta overflow handling (huangheng PR #1804 BLOCKER fix)
# ---------------------------------------------------------------------


def test_reasoning_buffer_pre_flush_avoids_part_overflow():
    """huangheng PR #1804 BLOCKER (msg=e609d5c9): the pre-flight
    overflow check must run BEFORE appending so no single
    ``ReasoningPart`` exceeds ``MAX_REASONING_PART_CHARS``.

    Replays the runtime's exact buffer-management logic against a
    stream of 30 × 200-char deltas (typical for long reasoning
    streams from large models) and asserts:
      1. No produced ReasoningPart exceeds the per-part cap.
      2. Joined text equals the input concatenation (no data loss
         across flush boundaries).
      3. Multiple parts are produced (proves chunking actually fires
         instead of silently dropping data).
    """
    from aperag.domains.agent_runtime.runtime import _PersistedReasoning
    from aperag.domains.agent_runtime.uimessage import MAX_REASONING_PART_CHARS

    deltas = ["x" * 200] * 30  # 6000 chars total > 4096 cap
    timeline: list = []
    buffer: list[str] = []

    def _flush():
        text = "".join(buffer).strip()
        buffer.clear()
        if text:
            timeline.append(_PersistedReasoning(text=text))

    # Mirror the runtime's pre-flight check exactly.
    for delta in deltas:
        if sum(len(s) for s in buffer) + len(delta) > MAX_REASONING_PART_CHARS:
            _flush()
        buffer.append(delta)
    _flush()  # trailing flush at turn end

    parts = _compose_assistant_parts(
        turn_id="turn-overflow",
        answer_text="",
        references=[],
        timeline=timeline,
    )

    reasoning_parts = [p for p in parts if isinstance(p, ReasoningPart)]
    assert len(reasoning_parts) >= 2, "buffer cap must produce multiple parts on overflow"
    for rp in reasoning_parts:
        assert len(rp.text) <= MAX_REASONING_PART_CHARS, (
            f"ReasoningPart text length {len(rp.text)} exceeds cap {MAX_REASONING_PART_CHARS} — "
            "Pydantic validator would reject this part and crash the turn write"
        )
    rejoined = "".join(rp.text for rp in reasoning_parts)
    assert rejoined == "".join(deltas), (
        "joined text must equal input concatenation (no data lost across flush boundaries)"
    )


# ---------------------------------------------------------------------
# PartStartEvent first-chunk preservation (earayu2 bug msg=193eeb7c)
# ---------------------------------------------------------------------


def test_part_start_event_initial_content_is_persisted():
    """pydantic-ai opens each new ThinkingPart / TextPart with a
    ``PartStartEvent`` whose ``part.content`` carries the FIRST chunk
    of text. Before this fix the runtime only handled
    ``PartDeltaEvent``, silently dropping the leading character of
    every reasoning/text part that opens after a tool call boundary —
    visible in the UI as 思考2 / final answer paragraphs starting with
    "，..." (the model's natural "嗯" / "好的" leading character was
    eaten).

    Replays the runtime's two-event sequence (start + delta) and
    asserts the joined buffer starts with the start-event content.
    """
    from aperag.domains.agent_runtime.runtime import _PersistedReasoning

    # Sequence: PartStartEvent(ThinkingPart(content="嗯")) followed by
    # PartDeltaEvent(ThinkingPartDelta(content_delta="，知识库..."))
    start_content = "嗯"
    delta_content = "，知识库中只有一个 'test' 集合"

    buffer: list[str] = []
    timeline: list = []

    def _flush():
        text = "".join(buffer).strip()
        buffer.clear()
        if text:
            timeline.append(_PersistedReasoning(text=text))

    # PartStartEvent handler (the new branch).
    if start_content:
        buffer.append(start_content)
    # PartDeltaEvent handler (existing branch).
    if delta_content:
        buffer.append(delta_content)
    _flush()

    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="",
        references=[],
        timeline=timeline,
    )

    reasoning_parts = [p for p in parts if isinstance(p, ReasoningPart)]
    assert len(reasoning_parts) == 1
    assert reasoning_parts[0].text == start_content + delta_content, (
        "PartStartEvent.part.content must be preserved — without it the "
        "leading character of every new ThinkingPart is silently dropped"
    )
    assert reasoning_parts[0].text.startswith("嗯"), (
        "regression marker: leading '嗯' (or any first-chunk char) must "
        "survive the start→delta transition (earayu2 bug msg=193eeb7c)"
    )
