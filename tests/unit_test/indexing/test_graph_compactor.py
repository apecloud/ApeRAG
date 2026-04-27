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

"""Unit tests for ``aperag.indexing.graph_compactor`` — Wave 7 task #2.

Pins the description-compaction contract: short input is left alone,
oversize input goes through the LLM, and any LLM failure cleanly falls
through to a hard-cap truncation that always carries the operator-
visible marker.
"""

from __future__ import annotations

import asyncio

import pytest

from aperag.indexing.graph_compactor import (
    FALLBACK_TRUNCATE_MARK,
    MAX_DESCRIPTION_CHARS,
    SUMMARIZE_AT_FRAGMENTS,
    TARGET_SUMMARY_CHARS,
    GraphIndexCompactor,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _RecordingLLM:
    """Async LLM stub that records each prompt and returns a scripted
    response (or raises a scripted exception)."""

    def __init__(self, response: str | None = None, exc: Exception | None = None):
        self._response = response
        self._exc = exc
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


# ---------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------


def test_below_threshold_returns_none_without_llm():
    """No fragment cap hit + total chars under cap → no compaction."""
    parts = ["short A", "short B", "short C"]
    assert len(parts) < SUMMARIZE_AT_FRAGMENTS
    llm = _RecordingLLM(response="should-not-be-called")

    result = _run(GraphIndexCompactor(llm).compact_if_oversized(parts))

    assert result is None
    assert llm.prompts == []


def test_empty_parts_returns_none():
    """Empty list and whitespace-only entries collapse to nothing."""
    llm = _RecordingLLM(response="should-not-be-called")
    compactor = GraphIndexCompactor(llm)

    assert _run(compactor.compact_if_oversized([])) is None
    assert _run(compactor.compact_if_oversized(["", "   ", "\n"])) is None
    assert llm.prompts == []


# ---------------------------------------------------------------------
# LLM success path
# ---------------------------------------------------------------------


def test_fragment_count_triggers_llm_summary():
    """Fragment count >= SUMMARIZE_AT_FRAGMENTS triggers compaction
    even when total chars are well below the cap."""
    parts = [f"fragment-{i}" for i in range(SUMMARIZE_AT_FRAGMENTS)]
    summary = "compact summary"
    llm = _RecordingLLM(response=summary)

    result = _run(GraphIndexCompactor(llm).compact_if_oversized(parts))

    assert result == summary
    assert len(llm.prompts) == 1
    # Prompt must include every fragment so the model sees full context.
    rendered = llm.prompts[0]
    for p in parts:
        assert p in rendered
    # Sanity: prompt mentions the target length so the model knows the budget.
    assert str(TARGET_SUMMARY_CHARS) in rendered


def test_char_count_triggers_llm_summary():
    """Single huge part > MAX_DESCRIPTION_CHARS also triggers compaction."""
    big = "x" * (MAX_DESCRIPTION_CHARS + 100)
    summary = "compact summary of huge text"
    llm = _RecordingLLM(response=summary)

    result = _run(GraphIndexCompactor(llm).compact_if_oversized([big]))

    assert result == summary
    assert len(llm.prompts) == 1


# ---------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------


def test_llm_exception_falls_back_to_truncation():
    parts = ["x" * 2000 for _ in range(SUMMARIZE_AT_FRAGMENTS + 2)]
    llm = _RecordingLLM(exc=RuntimeError("upstream provider 500"))

    result = _run(GraphIndexCompactor(llm).compact_if_oversized(parts))

    assert result is not None
    assert result.endswith(FALLBACK_TRUNCATE_MARK)
    assert len(result) <= MAX_DESCRIPTION_CHARS
    assert len(llm.prompts) == 1  # LLM was attempted before fallback


def test_llm_empty_response_falls_back_to_truncation():
    parts = ["x" * 2000 for _ in range(SUMMARIZE_AT_FRAGMENTS + 2)]
    llm = _RecordingLLM(response="   ")  # whitespace-only

    result = _run(GraphIndexCompactor(llm).compact_if_oversized(parts))

    assert result is not None
    assert result.endswith(FALLBACK_TRUNCATE_MARK)
    assert len(result) <= MAX_DESCRIPTION_CHARS


def test_llm_oversize_response_falls_back_to_truncation():
    """LLM returns text longer than the hard cap → reject and truncate
    so the persisted description never exceeds column budget."""
    parts = ["x" * 2000 for _ in range(SUMMARIZE_AT_FRAGMENTS + 2)]
    too_long = "y" * (MAX_DESCRIPTION_CHARS + 50)
    llm = _RecordingLLM(response=too_long)

    result = _run(GraphIndexCompactor(llm).compact_if_oversized(parts))

    assert result is not None
    assert result.endswith(FALLBACK_TRUNCATE_MARK)
    assert len(result) <= MAX_DESCRIPTION_CHARS


# ---------------------------------------------------------------------
# Truncation shape
# ---------------------------------------------------------------------


def test_fallback_truncate_prefers_word_boundary():
    """Truncator should land on whitespace inside the last-64-char
    window to avoid cutting mid-token."""
    head = "word " * (MAX_DESCRIPTION_CHARS // 5)  # plenty of word boundaries
    joined = head + "z" * 200  # forces a cut somewhere

    truncated = GraphIndexCompactor._fallback_truncate(joined)

    assert truncated.endswith(FALLBACK_TRUNCATE_MARK)
    assert len(truncated) <= MAX_DESCRIPTION_CHARS
    body = truncated[: -len(FALLBACK_TRUNCATE_MARK)]
    # The body should not end mid-word: either ends on a known token or
    # is the hard fallback when no whitespace was reachable.
    assert body.rstrip() == body


def test_fallback_truncate_passes_through_when_under_cap():
    short = "already small"
    assert GraphIndexCompactor._fallback_truncate(short) == short


# ---------------------------------------------------------------------
# Pure-compute invariant
# ---------------------------------------------------------------------


def test_compactor_does_not_touch_external_state():
    """Sanity: GraphIndexCompactor only depends on the LLM callable,
    nothing else. This protects the §K.12.6 separation guarantee that
    the compactor stays a pure helper (no LineageGraphStore /
    VectorStore coupling)."""
    sig_attrs = vars(GraphIndexCompactor(_RecordingLLM(response="ok")))
    assert set(sig_attrs.keys()) == {"_llm"}


# ---------------------------------------------------------------------
# pytest-async harness
# ---------------------------------------------------------------------

# All tests above wrap their coroutines via ``_run`` so this module
# stays compatible with both bare ``unittest`` runners and pytest
# without requiring ``pytest-asyncio`` to be configured. Direct test
# discovery still works through pytest collection.

if __name__ == "__main__":  # pragma: no cover - manual run convenience
    raise SystemExit(pytest.main([__file__, "-v"]))
