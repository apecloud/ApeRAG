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


class OpenAIPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model", "gpt-3.5-turbo")
        self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        self.api_key = kwargs.get("api_key", "")

        if not self.api_key:
            raise LLMConfigError("Please specify the API_KEY")

        self.api_key = self.api_key.strip()
        self.base_url = self.base_url.strip()

    def validate_params(self):
        if not self.kwargs.get("model"):
            raise LLMConfigError("Please specify the MODEL")
        if not self.kwargs.get("api_key"):
            raise LLMConfigError("Please specify the API_KEY")

    @staticmethod
    def provide_default_token():
        return False

    async def _agenerate_stream(self, history, prompt, memory=False):
        response = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            messages=history + [{"role": "user", "content": prompt}]
            if memory
            else [{"role": "user", "content": prompt}],
            stream=True,
            timeout=httpx.Timeout(None, connect=3),
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason == "stop":
                return
            yield choice.delta.content

    def _generate_stream(self, history, prompt, memory=False):
        response = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            messages=history + [{"role": "user", "content": prompt}]
            if memory
            else [{"role": "user", "content": prompt}],
            stream=True,
            timeout=httpx.Timeout(None, connect=3),
        )
        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason == "stop":
                return
            yield choice.delta.content

    async def agenerate_stream(self, history, prompt, memory=False):
        async for tokens in self._agenerate_stream(history, prompt, memory):
            yield tokens

    async def agenerate_by_tools(self, prompt, tools):
        response = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto",
        )
        tool_calls = response.choices[0].message.tool_calls
        return tool_calls, response.choices[0].message.content

    def generate_stream(self, history, prompt, memory=False):
        for tokens in self._generate_stream(history, prompt, memory):
            yield tokens
