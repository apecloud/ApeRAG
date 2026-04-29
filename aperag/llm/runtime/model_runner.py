from __future__ import annotations

from typing import Protocol

from aperag.llm.runtime.types import ResolvedModelInvocation


class ModelRunner(Protocol):
    async def chat(self, invocation: ResolvedModelInvocation, messages: list[dict], **kwargs):
        raise NotImplementedError()

    async def embed(self, invocation: ResolvedModelInvocation, inputs: list[str]):
        raise NotImplementedError()


class ModelRunnerRegistry:
    def __init__(self):
        self._runners: dict[str, ModelRunner] = {}

    def register(self, runner_type: str, runner: ModelRunner) -> None:
        self._runners[runner_type] = runner

    def get(self, runner_type: str) -> ModelRunner:
        if runner_type not in self._runners:
            raise ValueError(f"Unknown model runner: {runner_type}")
        return self._runners[runner_type]


runner_registry = ModelRunnerRegistry()
