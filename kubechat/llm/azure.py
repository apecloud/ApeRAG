import json
import os

import httpx
import openai
from openai import AsyncAzureOpenAI, AzureOpenAI

from kubechat.llm.base import LLMConfigError, Predictor


class AzureOpenAIPredictor(Predictor):
    """
    https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints#keyword-argument-for-model
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deployment_id = kwargs.get("deployment_id", "gpt-4")
        self.endpoint = kwargs.get("endpoint", "https://api.openai.com/v1")
        self.api_version = kwargs.get("api_version", "2023-05-15")

        self.token = kwargs.get("token", os.environ.get("OPENAI_API_KEY", ""))
        if not self.token:
            raise LLMConfigError("Please specify the API token")
        self.use_default_token = not kwargs.get("token", "")
        """
        # https://github.com/openai/openai-python/issues/279
        Example:
        openai.proxy = {
            "http":"http://127.0.0.1:7890",
            "https":"http://127.0.0.1:7890"
        }
        """
        proxy = json.loads(os.environ.get("OPENAI_API_PROXY", "{}"))
        if proxy:
            openai.proxy = proxy
        self.api_type = "azure"

    @staticmethod
    def provide_default_token():
        return bool(os.environ.get("OPENAI_API_KEY", ""))

    async def _agenerate_stream(self, history, prompt, memory=False):
        timeout = httpx.Timeout(None, connect=3)
        client = AsyncAzureOpenAI(
            timeout=timeout,
            api_key=self.token,
            api_version=self.api_version,
            max_retries=0,
            azure_endpoint=self.endpoint,
        )
        response = await client.chat.completions.create(
            model=self.deployment_id,
            stream=True,
            temperature=self.temperature,
            messages=history + [{"role": "user", "content": prompt}] if memory else [{"role": "user", "content": prompt}],
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason == "stop":
                return
            content = choice.delta.content
            if not content:
                continue
            yield choice.delta.content

    def _generate_stream(self, history, prompt, memory=False):
        timeout = httpx.Timeout(None, connect=3)
        client = AzureOpenAI(
            timeout=timeout,
            api_key=self.token,
            api_version=self.api_version,
            max_retries=0,
            azure_endpoint=self.endpoint,
        )
        response = client.chat.completions.create(
            model=self.deployment_id,
            stream=True,
            temperature=self.temperature,
            messages=history + [{"role": "user", "content": prompt}] if memory else [{"role": "user", "content": prompt}],
        )
        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason == "stop":
                return
            content = choice.delta.content
            if not content:
                continue
            yield choice.delta.content

    async def agenerate_stream(self, history, prompt, memory=False):
        async for tokens in self._agenerate_stream(history, prompt, memory):
            yield tokens

    def generate_stream(self, history, prompt, memory=False):
        for tokens in self._generate_stream(history, prompt, memory):
            yield tokens
