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

import litellm

from aperag.llm.base import LLMConfigError, Predictor


class CompletionService(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def validate_params(self):
        if not self.kwargs.get("model"):
            raise LLMConfigError("Please specify the MODEL")

    @staticmethod
    def provide_default_token():
        return False

    async def _agenerate_stream(self, history, prompt, memory=False):
        try:
            response = await litellm.acompletion(
                **self.kwargs,
                messages=history + [{"role": "user", "content": prompt}]
                if memory
                else [{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason == "stop":
                    return
                yield choice.delta.content
        except Exception as e:
            raise e

    def _generate_stream(self, history, prompt, memory=False):
        try:
            response = litellm.completion(
                **self.kwargs,
                messages=history + [{"role": "user", "content": prompt}]
                if memory
                else [{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason == "stop":
                    return
                yield choice.delta.content
        except Exception as e:
            raise e

    async def agenerate_stream(self, history, prompt, memory=False):
        async for tokens in self._agenerate_stream(history, prompt, memory):
            yield tokens

    async def agenerate_by_tools(self, prompt, tools):
        try:
            response = await litellm.acompletion(
                **self.kwargs,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice="auto",
            )
            tool_calls = response.choices[0].message.tool_calls
            return tool_calls, response.choices[0].message.content
        except Exception as e:
            raise e

    def generate_stream(self, history, prompt, memory=False):
        for tokens in self._generate_stream(history, prompt, memory):
            yield tokens
