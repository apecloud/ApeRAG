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

"""Judge interface for evaluation-v2.

The MVP single-dimension ``LlmAsJudge`` (architect spec
``#evaluation msg=2424afe2`` + @earayu2 ratify ``msg=e6a534a8``) ships
the real LLM-as-judge path: the worker hands the LLM the question +
ground-truth + actual answer and parses ``{score, reason}`` out of the
JSON. We persist ``score / 10`` into the existing ``judge_score``
``[0, 1]`` float and the LLM's ``reason`` into ``judge_reason``.

Phase 5 expansion (Completeness / Faithfulness / Relevance dimensions
written into ``EvaluationRunItem.judge_breakdown``) is forward-compat
in the schema this round but does not run yet.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from aperag.domains.evaluation.db.models import EvaluationJudgeMode
from aperag.domains.evaluation.schemas import JudgeConfig

logger = logging.getLogger(__name__)


@dataclass
class JudgeInput:
    case_key: str
    question: str
    expected_answer: Optional[str]
    reference_context: Optional[str]
    actual_answer: str


@dataclass
class JudgeOutput:
    score: float
    passed: bool
    reasoning: Optional[str] = None
    raw: Optional[dict[str, Any]] = None
    breakdown: Optional[dict[str, Any]] = None


class Judge(ABC):
    """Produces a per-case score + pass/fail verdict."""

    mode: EvaluationJudgeMode

    @abstractmethod
    async def judge(self, payload: JudgeInput) -> JudgeOutput: ...


class NoopJudge(Judge):
    mode = EvaluationJudgeMode.NONE

    async def judge(self, payload: JudgeInput) -> JudgeOutput:
        return JudgeOutput(score=0.0, passed=True, reasoning="judge disabled")


class ExactMatchJudge(Judge):
    mode = EvaluationJudgeMode.EXACT_MATCH

    async def judge(self, payload: JudgeInput) -> JudgeOutput:
        expected = (payload.expected_answer or "").strip()
        actual = (payload.actual_answer or "").strip()
        passed = bool(expected) and expected == actual
        return JudgeOutput(
            score=1.0 if passed else 0.0,
            passed=passed,
            reasoning="exact match" if passed else "answer does not match expected",
        )


# ---------------------------------------------------------------------
# LLM-as-judge MVP — single Correctness dimension
# ---------------------------------------------------------------------


_LLM_JUDGE_PROMPT_ZH: str = """\
评判 RAG 系统回答与期望答案的语义一致性。

问题: {question}
期望答案: {expected_answer}
实际答案: {actual_answer}

评分标准 (0-10 整数, 10 = 完全一致):
- 8-10: 核心一致, 关键事实正确
- 5-7: 部分对, 有偏差或缺关键点
- 0-4: 错误或答非所问

只输出 JSON, 不要任何解释或代码块包裹:
{{"score": <int 0-10>, "reason": "<简短中文原因>"}}"""


_LLM_JUDGE_PROMPT_EN: str = """\
Score the semantic correctness of a RAG system answer against the
ground truth.

Question: {question}
Expected answer: {expected_answer}
Actual answer: {actual_answer}

Scoring rubric (integer 0-10, 10 = perfect):
- 8-10: core meaning matches, key facts correct
- 5-7: partially correct, deviation or missing key points
- 0-4: incorrect or off-topic

Output JSON ONLY (no Markdown fences, no commentary):
{{"score": <int 0-10>, "reason": "<short English rationale>"}}"""


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(raw: str) -> str:
    match = _FENCE_RE.match(raw)
    return match.group(1).strip() if match else raw.strip()


def _parse_llm_judge_response(raw: str) -> tuple[Optional[int], Optional[str]]:
    """Coerce the LLM's response into ``(score, reason)`` or ``(None, None)``
    on any parse failure. The caller falls through to a deterministic
    rejection so a flaky judge model never crashes the run."""
    payload = _strip_code_fence(raw)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return (None, None)
    if not isinstance(parsed, dict):
        return (None, None)
    score = parsed.get("score")
    reason = parsed.get("reason") or parsed.get("Reason")
    if not isinstance(score, (int, float)):
        return (None, None)
    score_int = int(round(score))
    if score_int < 0:
        score_int = 0
    if score_int > 10:
        score_int = 10
    if not isinstance(reason, str):
        reason = ""
    return (score_int, reason.strip())


def _detect_language(text: str) -> str:
    if not text:
        return "en"
    cjk = len(re.findall(r"[一-鿿]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    return "zh" if cjk > latin else "en"


_PASS_THRESHOLD: float = 0.6  # ``score / 10 >= 0.6`` ⇒ ``passed = True``


class LlmAsJudge(Judge):
    """LLM-as-judge MVP — single Correctness dimension.

    Constructed by the worker after resolving the run's ``judge_model``
    (or the collection's default completion model) into an async
    ``(prompt) -> str`` callable. The prompt language follows the
    expected_answer / actual_answer scripts so a Chinese collection
    gets a Chinese rationale.
    """

    mode = EvaluationJudgeMode.LLM_AS_JUDGE

    def __init__(self, llm: Callable[[str], Awaitable[str]]):
        self._llm = llm

    async def judge(self, payload: JudgeInput) -> JudgeOutput:
        expected = (payload.expected_answer or "").strip()
        actual = (payload.actual_answer or "").strip()
        if not expected:
            # Without ground truth we cannot compute Correctness — fall
            # back to a soft pass so the run does not surface as FAILED
            # for a config-shape problem. Operators see this as
            # ``judge_score = 0`` + a clear reason.
            return JudgeOutput(
                score=0.0,
                passed=False,
                reasoning="LLM-as-judge skipped: dataset item has no expected_answer",
            )
        if not actual:
            return JudgeOutput(
                score=0.0,
                passed=False,
                reasoning="LLM-as-judge: actual answer is empty",
            )
        language = _detect_language(expected + actual)
        template = _LLM_JUDGE_PROMPT_ZH if language == "zh" else _LLM_JUDGE_PROMPT_EN
        prompt = template.format(
            question=(payload.question or "").strip(),
            expected_answer=expected,
            actual_answer=actual,
        )
        try:
            raw = await self._llm(prompt)
        except Exception:  # noqa: BLE001 — judge failure must not crash the worker
            logger.exception(
                "LlmAsJudge: LLM call failed for case_key=%s; surfacing as 0",
                payload.case_key,
            )
            return JudgeOutput(
                score=0.0,
                passed=False,
                reasoning="LLM-as-judge call failed; see worker logs",
            )
        score_int, reason = _parse_llm_judge_response(raw)
        if score_int is None:
            logger.warning(
                "LlmAsJudge: chunk_id=%s could not parse JSON: %r",
                payload.case_key,
                raw[:500],
            )
            return JudgeOutput(
                score=0.0,
                passed=False,
                reasoning="LLM-as-judge response was not valid JSON",
                raw={"text": raw[:500]},
            )
        score_norm = float(score_int) / 10.0
        # Phase 5 forward-compat: the breakdown column already exists on
        # the run-item row; populating ``correctness`` here means a
        # future Phase-5 PR only has to add Completeness / Faithfulness
        # / Relevance and aggregate, not invent the schema.
        breakdown = {
            "correctness": score_int,
            "completeness": None,
            "faithfulness": None,
            "relevance": None,
        }
        return JudgeOutput(
            score=score_norm,
            passed=score_norm >= _PASS_THRESHOLD,
            reasoning=reason or "",
            breakdown=breakdown,
        )


def build_judge(
    config: Optional[JudgeConfig],
    *,
    llm: Optional[Callable[[str], Awaitable[str]]] = None,
) -> Judge:
    """Factory used by the worker pipeline.

    ``llm`` is the per-run judge callable resolved upstream from
    ``run.judge_model`` (or the collection default). When the caller
    selects ``LLM_AS_JUDGE`` but ``llm`` is ``None`` we degrade to the
    deterministic ``ExactMatchJudge`` so a half-configured run still
    finishes — matching the explicit-fail-not-silent pattern but at the
    judge layer rather than the dispatch layer.
    """
    if not config or config.mode == EvaluationJudgeMode.NONE:
        return NoopJudge()
    if config.mode == EvaluationJudgeMode.EXACT_MATCH:
        return ExactMatchJudge()
    if config.mode == EvaluationJudgeMode.LLM_AS_JUDGE:
        if llm is None:
            logger.warning(
                "build_judge: LLM_AS_JUDGE requested but no llm callable wired; falling back to ExactMatchJudge",
            )
            return ExactMatchJudge()
        return LlmAsJudge(llm=llm)
    return NoopJudge()
