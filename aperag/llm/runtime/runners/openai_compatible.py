from __future__ import annotations

from openai import AsyncOpenAI

from aperag.llm.runtime.types import ResolvedModelInvocation


class OpenAICompatibleRunner:
    async def chat(self, invocation: ResolvedModelInvocation, messages: list[dict], **kwargs):
        client = AsyncOpenAI(api_key=invocation.api_key, base_url=invocation.base_url)
        return await client.chat.completions.create(
            model=invocation.provider_model_id,
            messages=messages,
            **kwargs,
        )

    async def embed(self, invocation: ResolvedModelInvocation, inputs: list[str]):
        client = AsyncOpenAI(api_key=invocation.api_key, base_url=invocation.base_url)
        return await client.embeddings.create(model=invocation.provider_model_id, input=inputs)
