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

import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMConfigError(Exception):
    """
    LLMConfigError is raised when the LLM config is invalid.
    """


class LLMAPIError(Exception):
    """
    LLMAPIError is raised when error occurs when calling LLM API.
    """


class Predictor(ABC):
    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.1)
        self.trial = False

    @abstractmethod
    async def agenerate_stream(self, history, prompt, memory):
        pass

    @abstractmethod
    def generate_stream(self, history, prompt, memory):
        pass

    @staticmethod
    def match_predictor(model_service_provider, model_name, base_url, api_key, kwargs):
        kwargs["model"] = model_name
        kwargs["api_key"] = api_key
        kwargs["base_url"] = base_url

        if model_service_provider == "alibabacloud":
            from aperag.llm.qianwen import QianWenPredictor

            return QianWenPredictor

        from aperag.llm.completion_service import CompletionService

        return CompletionService

    @staticmethod
    def get_completion_service(model_service_provider, model_name, base_url, api_key, **kwargs):
        predictor = Predictor.match_predictor(model_service_provider, model_name, base_url, api_key, kwargs)
        return predictor(**kwargs)

    def get_latest_history(self, messages, limit_length, limit_count, use_ai_memory) -> str:
        latest_history = []
        length = 0
        count = 0

        for message in reversed(messages):
            if message.additional_kwargs["role"] == "human":
                history = {"role": "user", "content": json.loads(message.content)["query"] + "\n"}
            else:
                if not use_ai_memory:
                    continue
                history = {"role": "assistant", "content": json.loads(message.content)["response"] + "\n"}
            count += 1

            if count >= limit_count or length + len(history["content"]) > limit_length:
                break

            latest_history.append(history)
            length += len(history["content"])

        if len(latest_history) > 0 and latest_history[-1]["role"] == "assistant":
            latest_history = latest_history[:-1]

        return latest_history[::-1]
