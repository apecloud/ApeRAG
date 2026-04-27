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

Phase 1 lands only the interface and the no-op + exact-match judges, which
are enough to let the orchestration service and worker pipeline be wired and
tested end-to-end. The LLM-as-judge implementation ships in Phase 3.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from aperag.domains.evaluation.db.models import EvaluationJudgeMode
from aperag.domains.evaluation.schemas import JudgeConfig


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


def build_judge(config: Optional[JudgeConfig]) -> Judge:
    """Factory used by the worker pipeline.

    The LLM-as-judge branch is a Phase 3 follow-up; we fall back to exact
    match so the worker has a deterministic baseline until then.
    """
    if not config or config.mode == EvaluationJudgeMode.NONE:
        return NoopJudge()
    if config.mode == EvaluationJudgeMode.EXACT_MATCH:
        return ExactMatchJudge()
    if config.mode == EvaluationJudgeMode.LLM_AS_JUDGE:
        return ExactMatchJudge()
    return NoopJudge()
