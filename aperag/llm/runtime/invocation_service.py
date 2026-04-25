from __future__ import annotations

from aperag.db.ops import async_db_ops
from aperag.llm.runtime.resolver import resolve_model_invocation_from_records
from aperag.llm.runtime.runners import runner_registry
from aperag.llm.runtime.types import ModelUnavailableError


class ModelInvocationService:
    async def resolve(self, model_id: str, user_id: str):
        row = await async_db_ops.query_model_runtime(model_id, user_id)
        if not row:
            raise ModelUnavailableError(f"Model '{model_id}' is not available")
        model, account = row
        return resolve_model_invocation_from_records(model=model, account=account)

    async def chat(self, model_id: str, user_id: str, messages: list[dict], **kwargs):
        invocation = await self.resolve(model_id, user_id)
        runner = runner_registry.get(invocation.runner_type)
        return await runner.chat(invocation, messages, **kwargs)

    async def embed(self, model_id: str, user_id: str, inputs: list[str]):
        invocation = await self.resolve(model_id, user_id)
        runner = runner_registry.get(invocation.runner_type)
        return await runner.embed(invocation, inputs)

    async def rerank(self, model_id: str, user_id: str, query: str, documents: list[str], **kwargs):
        invocation = await self.resolve(model_id, user_id)
        runner = runner_registry.get(invocation.runner_type)
        return await runner.rerank(invocation, query, documents, **kwargs)


model_invocation_service = ModelInvocationService()
