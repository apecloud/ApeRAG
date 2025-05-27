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


import httpx
import litellm

from aperag.llm.base import LLMConfigError, Predictor


class AzureOpenAIPredictor(Predictor):
    """
    https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints#keyword-argument-for-model
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deployment_id = kwargs.get("deployment_id", "")
        self.base_url = kwargs.get("endpoint", "")
        self.api_version = kwargs.get("api_version", "")
        self.api_key = kwargs.get("token", "")

        self.base_url = self.base_url.strip()
        self.api_key = self.api_key.strip()

        if not self.deployment_id:
            raise LLMConfigError("Please specify the deployment ID")

        if not self.base_url:
            raise LLMConfigError("Please specify the API endpoint")

        if not self.api_version:
            raise LLMConfigError("Please specify the API version")

        if not self.api_key:
            raise LLMConfigError("Please specify the API token")

    @staticmethod
    def provide_default_token():
        return False

    async def _agenerate_stream(self, history, prompt, memory=False):
        response = litellm.completion(
            model=self.deployment_id,
            stream=True,
            temperature=self.temperature,
            messages=history + [{"role": "user", "content": prompt}]
            if memory
            else [{"role": "user", "content": prompt}],
            timeout=httpx.Timeout(None, connect=3),
            api_key=self.api_key,
            api_version=self.api_version,
            max_retries=0,
            base_url=self.base_url,
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
        response = litellm.completion(
            model=self.deployment_id,
            stream=True,
            temperature=self.temperature,
            messages=history + [{"role": "user", "content": prompt}]
            if memory
            else [{"role": "user", "content": prompt}],
            timeout=httpx.Timeout(None, connect=3),
            api_key=self.api_key,
            api_version=self.api_version,
            max_retries=0,
            base_url=self.base_url,
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
