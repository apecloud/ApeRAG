from __future__ import annotations

import httpx

from aperag.llm.runtime.types import ResolvedModelInvocation


class DashScopeRerankRunner:
    DEFAULT_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"

    async def rerank(self, invocation: ResolvedModelInvocation, query: str, documents: list[str], **kwargs):
        base_url = invocation.base_url.rstrip("/")
        if base_url.endswith("/compatible-mode/v1"):
            base_url = base_url[: -len("/compatible-mode/v1")]
        path = invocation.runner_config.get("path") or self.DEFAULT_PATH
        url = base_url if base_url.endswith(path) else f"{base_url}{path}"
        payload = {
            "model": invocation.provider_model_id,
            "input": {"query": query, "documents": documents},
            "parameters": {"return_documents": False, "top_n": kwargs.get("top_n") or len(documents)},
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {invocation.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        results = data.get("output", {}).get("results", [])
        return [
            {"index": item.get("index", idx), "relevance_score": item.get("relevance_score", 0.0)}
            for idx, item in enumerate(results)
        ]

    async def chat(self, invocation: ResolvedModelInvocation, messages: list[dict], **kwargs):
        raise NotImplementedError("DashScope rerank runner only supports rerank")

    async def embed(self, invocation: ResolvedModelInvocation, inputs: list[str]):
        raise NotImplementedError("DashScope rerank runner only supports rerank")
