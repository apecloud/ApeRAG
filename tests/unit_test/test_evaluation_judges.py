# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the LLM-as-judge MVP path
(``aperag/domains/evaluation/judges.py``).

Architect spec ``#evaluation msg=2424afe2`` + @earayu2 ratify
``msg=e6a534a8``: the judge takes ``{question, expected_answer,
actual_answer}``, fires one LLM call, parses ``{score: 0-10, reason}``,
and returns a normalised ``[0, 1]`` score plus a Phase-5-forward-compat
``breakdown`` dict with ``correctness`` populated and the other three
dimensions ``None``.
"""

from __future__ import annotations

import json

import pytest

from aperag.domains.evaluation.db.models import EvaluationJudgeMode
from aperag.domains.evaluation.judges import (
    ExactMatchJudge,
    JudgeInput,
    LlmAsJudge,
    NoopJudge,
    _parse_llm_judge_response,
    build_judge,
)
from aperag.domains.evaluation.schemas import JudgeConfig

# ---------------------------------------------------------------------
# _parse_llm_judge_response — shape coercion
# ---------------------------------------------------------------------


def test_parse_llm_judge_response_accepts_plain_json():
    score, reason = _parse_llm_judge_response('{"score": 8, "reason": "good"}')
    assert score == 8
    assert reason == "good"


def test_parse_llm_judge_response_accepts_fenced_json():
    score, reason = _parse_llm_judge_response('```json\n{"score": 6, "reason": "ok"}\n```')
    assert score == 6
    assert reason == "ok"


def test_parse_llm_judge_response_clips_score_to_range():
    score, _ = _parse_llm_judge_response('{"score": 99, "reason": "x"}')
    assert score == 10
    score, _ = _parse_llm_judge_response('{"score": -3, "reason": "x"}')
    assert score == 0


def test_parse_llm_judge_response_rounds_floats():
    score, _ = _parse_llm_judge_response('{"score": 7.5, "reason": "x"}')
    assert score == 8


def test_parse_llm_judge_response_rejects_invalid_json():
    score, reason = _parse_llm_judge_response("model rambled, not JSON")
    assert score is None
    assert reason is None


def test_parse_llm_judge_response_rejects_missing_score():
    score, _ = _parse_llm_judge_response('{"reason": "no score"}')
    assert score is None


# ---------------------------------------------------------------------
# LlmAsJudge — orchestration
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_as_judge_returns_normalised_score_and_breakdown():
    async def fake_llm(_prompt: str) -> str:
        return json.dumps({"score": 9, "reason": "answer matches expected"})

    judge = LlmAsJudge(llm=fake_llm)
    verdict = await judge.judge(
        JudgeInput(
            case_key="case_1",
            question="What is 2+2?",
            expected_answer="4",
            reference_context=None,
            actual_answer="The answer is 4.",
        )
    )
    assert verdict.score == pytest.approx(0.9)
    assert verdict.passed is True
    assert verdict.reasoning == "answer matches expected"
    assert verdict.breakdown is not None
    assert verdict.breakdown["correctness"] == 9
    # Phase 5 forward-compat: other three dimensions stay null until
    # the multi-dim prompt ships.
    for key in ("completeness", "faithfulness", "relevance"):
        assert verdict.breakdown[key] is None


@pytest.mark.asyncio
async def test_llm_as_judge_low_score_fails():
    async def fake_llm(_prompt: str) -> str:
        return json.dumps({"score": 3, "reason": "answer is wrong"})

    judge = LlmAsJudge(llm=fake_llm)
    verdict = await judge.judge(
        JudgeInput(
            case_key="case_1",
            question="Q?",
            expected_answer="right",
            reference_context=None,
            actual_answer="wrong",
        )
    )
    assert verdict.score == pytest.approx(0.3)
    assert verdict.passed is False


@pytest.mark.asyncio
async def test_llm_as_judge_skips_when_expected_answer_missing():
    """No ground truth → cannot compute Correctness; surface a soft
    ``score=0`` rather than crashing or pretending to pass."""

    async def fake_llm(_prompt: str) -> str:
        raise AssertionError("LLM must not be called when expected_answer is empty")

    judge = LlmAsJudge(llm=fake_llm)
    verdict = await judge.judge(
        JudgeInput(
            case_key="case_1",
            question="Q?",
            expected_answer=None,
            reference_context=None,
            actual_answer="anything",
        )
    )
    assert verdict.score == 0.0
    assert verdict.passed is False
    assert "expected_answer" in (verdict.reasoning or "")


@pytest.mark.asyncio
async def test_llm_as_judge_marks_failed_on_invalid_json():
    async def fake_llm(_prompt: str) -> str:
        return "model produced no JSON"

    judge = LlmAsJudge(llm=fake_llm)
    verdict = await judge.judge(
        JudgeInput(
            case_key="case_1",
            question="Q?",
            expected_answer="A",
            reference_context=None,
            actual_answer="A",
        )
    )
    assert verdict.score == 0.0
    assert verdict.passed is False
    assert verdict.raw is not None


@pytest.mark.asyncio
async def test_llm_as_judge_handles_llm_exception():
    async def fake_llm(_prompt: str) -> str:
        raise RuntimeError("model offline")

    judge = LlmAsJudge(llm=fake_llm)
    verdict = await judge.judge(
        JudgeInput(
            case_key="case_1",
            question="Q?",
            expected_answer="A",
            reference_context=None,
            actual_answer="A",
        )
    )
    assert verdict.score == 0.0
    assert verdict.passed is False
    assert "failed" in (verdict.reasoning or "").lower()


# ---------------------------------------------------------------------
# build_judge — factory
# ---------------------------------------------------------------------


def test_build_judge_default_returns_noop():
    assert isinstance(build_judge(None), NoopJudge)


def test_build_judge_exact_match_mode():
    config = JudgeConfig(mode=EvaluationJudgeMode.EXACT_MATCH)
    assert isinstance(build_judge(config), ExactMatchJudge)


def test_build_judge_llm_as_judge_with_callable():
    async def fake_llm(_prompt: str) -> str:
        return '{"score": 7, "reason": "x"}'

    config = JudgeConfig(mode=EvaluationJudgeMode.LLM_AS_JUDGE)
    judge = build_judge(config, llm=fake_llm)
    assert isinstance(judge, LlmAsJudge)


def test_build_judge_llm_as_judge_falls_back_to_exact_match_without_llm():
    """``LLM_AS_JUDGE`` without an ``llm`` callable must degrade to
    ExactMatch so the run still finishes deterministically."""
    config = JudgeConfig(mode=EvaluationJudgeMode.LLM_AS_JUDGE)
    judge = build_judge(config, llm=None)
    assert isinstance(judge, ExactMatchJudge)
