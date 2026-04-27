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
oversize input goes through the LLM, every LLM failure falls through to
a hard-cap truncation that always carries the operator-visible marker,
and the prompt the model receives carries the five legacy features
(``subject_kind`` framing, ``subject_label`` anchor, ``language``
control, relation S-P-O guidance, contradiction-keep-both).
"""

from __future__ import annotations

import pytest

from aperag.indexing.graph_compactor import (
    DESCRIPTION_SUMMARIZATION,
    FALLBACK_TRUNCATE_MARK,
    MAX_DESCRIPTION_CHARS,
    SUMMARIZE_AT_FRAGMENTS,
    TARGET_SUMMARY_CHARS,
    GraphIndexCompactor,
    render_summarization_prompt,
)


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


def _entity_kwargs(name: str = "OpenAI") -> dict:
    return {"subject_kind": "entity", "subject_label": name}


# ---------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_below_threshold_returns_none_without_llm():
    """No fragment cap hit + total chars under cap → no compaction."""
    parts = ["short A", "short B", "short C"]
    assert len(parts) < SUMMARIZE_AT_FRAGMENTS
    llm = _RecordingLLM(response="should-not-be-called")

    result = await GraphIndexCompactor(llm).compact_if_oversized(parts, **_entity_kwargs())

    assert result is None
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_empty_parts_returns_none():
    """Empty list and whitespace-only entries collapse to nothing."""
    llm = _RecordingLLM(response="should-not-be-called")
    compactor = GraphIndexCompactor(llm)

    assert await compactor.compact_if_oversized([], **_entity_kwargs()) is None
    assert await compactor.compact_if_oversized(["", "   ", "\n"], **_entity_kwargs()) is None
    assert llm.prompts == []


# ---------------------------------------------------------------------
# LLM success path + prompt feature coverage
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fragment_count_triggers_llm_summary_with_entity_framing():
    """Fragment count >= SUMMARIZE_AT_FRAGMENTS triggers compaction.

    Also pins that the rendered prompt carries the five legacy
    features required by ``feedback_announce_equals_landed.md`` /
    earayu2 directive 'do not lose good legacy algorithms'.
    """
    parts = [f"fragment-{i}" for i in range(SUMMARIZE_AT_FRAGMENTS)]
    summary = "compact summary"
    llm = _RecordingLLM(response=summary)

    result = await GraphIndexCompactor(llm).compact_if_oversized(
        parts, subject_kind="entity", subject_label="OpenAI Inc."
    )

    assert result == summary
    assert len(llm.prompts) == 1
    rendered = llm.prompts[0]

    # Feature 1: subject_kind framing ("entity" appears in the prompt body).
    assert "entity" in rendered
    # Feature 2: subject_label anchoring ("OpenAI Inc." reaches the model).
    assert "OpenAI Inc." in rendered
    # Feature 3: default language is rendered.
    assert "Chinese" in rendered
    # Feature 5: contradiction-keep-both rule is included.
    assert "Keep conflicting facts" in rendered
    # All fragments make it into the prompt body.
    for p in parts:
        assert p in rendered
    # Sanity: prompt mentions the target length so the model knows the budget.
    assert str(TARGET_SUMMARY_CHARS) in rendered


@pytest.mark.asyncio
async def test_relation_framing_uses_relation_kind():
    """Relations carry the S-P-O framing rule (legacy feature 4)."""
    parts = ["x" * 2000 for _ in range(SUMMARIZE_AT_FRAGMENTS + 1)]
    llm = _RecordingLLM(response="merged relation summary")

    result = await GraphIndexCompactor(llm).compact_if_oversized(
        parts,
        subject_kind="relation",
        subject_label="Apple → Tim Cook",
    )

    assert result == "merged relation summary"
    rendered = llm.prompts[0]
    assert "relation" in rendered
    assert "Apple → Tim Cook" in rendered
    # The S-P-O framing rule must reach the model so relation summaries
    # don't degenerate into entity-style listings.
    assert "subject" in rendered.lower() and "predicate" in rendered.lower()


@pytest.mark.asyncio
async def test_language_kwarg_overrides_default():
    """``language`` reaches the model so non-Chinese collections work."""
    parts = ["x" * 2000 for _ in range(SUMMARIZE_AT_FRAGMENTS + 1)]
    llm = _RecordingLLM(response="english summary")

    result = await GraphIndexCompactor(llm).compact_if_oversized(
        parts, subject_kind="entity", subject_label="OpenAI", language="English"
    )

    assert result == "english summary"
    rendered = llm.prompts[0]
    assert "English" in rendered
    # Default language should NOT leak in when an override is supplied.
    assert "Chinese" not in rendered


@pytest.mark.asyncio
async def test_char_count_triggers_llm_summary():
    """Single huge part > MAX_DESCRIPTION_CHARS also triggers compaction."""
    big = "x" * (MAX_DESCRIPTION_CHARS + 100)
    summary = "compact summary of huge text"
    llm = _RecordingLLM(response=summary)

    result = await GraphIndexCompactor(llm).compact_if_oversized([big], **_entity_kwargs())

    assert result == summary
    assert len(llm.prompts) == 1


# ---------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_exception_falls_back_to_truncation():
    parts = ["x" * 2000 for _ in range(SUMMARIZE_AT_FRAGMENTS + 2)]
    llm = _RecordingLLM(exc=RuntimeError("upstream provider 500"))

    result = await GraphIndexCompactor(llm).compact_if_oversized(parts, **_entity_kwargs())

    assert result is not None
    assert result.endswith(FALLBACK_TRUNCATE_MARK)
    assert len(result) <= MAX_DESCRIPTION_CHARS
    assert len(llm.prompts) == 1  # LLM was attempted before fallback


@pytest.mark.asyncio
async def test_llm_empty_response_falls_back_to_truncation():
    parts = ["x" * 2000 for _ in range(SUMMARIZE_AT_FRAGMENTS + 2)]
    llm = _RecordingLLM(response="   ")  # whitespace-only

    result = await GraphIndexCompactor(llm).compact_if_oversized(parts, **_entity_kwargs())

    assert result is not None
    assert result.endswith(FALLBACK_TRUNCATE_MARK)
    assert len(result) <= MAX_DESCRIPTION_CHARS


@pytest.mark.asyncio
async def test_llm_oversize_response_falls_back_to_truncation():
    """LLM returns text longer than the hard cap → reject and truncate
    so the persisted description never exceeds column budget."""
    parts = ["x" * 2000 for _ in range(SUMMARIZE_AT_FRAGMENTS + 2)]
    too_long = "y" * (MAX_DESCRIPTION_CHARS + 50)
    llm = _RecordingLLM(response=too_long)

    result = await GraphIndexCompactor(llm).compact_if_oversized(parts, **_entity_kwargs())

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
    # The body should not end mid-word.
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
# Prompt renderer surface (used by callers that want to log the prompt
# without going through the LLM call)
# ---------------------------------------------------------------------


def test_render_summarization_prompt_strips_blank_fragments():
    rendered = render_summarization_prompt(
        subject_kind="entity",
        subject_label="OpenAI",
        fragments=["alpha", "  ", "beta", "", "gamma"],
        language="English",
        target_chars=TARGET_SUMMARY_CHARS,
    )
    assert "alpha" in rendered and "beta" in rendered and "gamma" in rendered
    # Blank entries shouldn't add empty separators that confuse the model.
    assert "---\n\n---" not in rendered
    # Fragment count reflects only non-blank fragments.
    assert "3 short descriptions" in rendered


def test_template_carries_legacy_five_features():
    """Compile-time check on the template body, independent of caller
    invocations. If anyone simplifies the template inline, this test
    breaks before the per-feature behavioural tests do."""
    body = DESCRIPTION_SUMMARIZATION
    # Feature 1: subject_kind placeholder.
    assert "{subject_kind}" in body
    # Feature 2: subject_label placeholder.
    assert "{subject_label}" in body
    # Feature 3: language placeholder (rendered twice — body + final cue).
    assert body.count("{language}") >= 2
    # Feature 4: relation S-P-O framing rule.
    assert "subject" in body.lower() and "predicate" in body.lower()
    # Feature 5: contradiction-keep-both rule.
    assert "Keep conflicting facts" in body


if __name__ == "__main__":  # pragma: no cover - manual run convenience
    raise SystemExit(pytest.main([__file__, "-v"]))
