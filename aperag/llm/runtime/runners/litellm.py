from __future__ import annotations

import litellm

from aperag.llm.runtime.types import ResolvedModelInvocation


class LiteLLMRunner:
    async def chat(self, invocation: ResolvedModelInvocation, messages: list[dict], **kwargs):
        return await litellm.acompletion(
            custom_llm_provider=invocation.runner_config.get("provider"),
            model=invocation.provider_model_id,
            base_url=invocation.base_url,
            api_key=invocation.api_key,
            messages=messages,
            **kwargs,
        )

    async def embed(self, invocation: ResolvedModelInvocation, inputs: list[str]):
        return litellm.embedding(
            custom_llm_provider=invocation.runner_config.get("provider"),
            model=invocation.provider_model_id,
            api_base=invocation.base_url,
            api_key=invocation.api_key,
            input=inputs,
            encoding_format="float",
        )
