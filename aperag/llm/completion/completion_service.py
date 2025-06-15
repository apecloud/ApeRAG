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
import logging
from typing import Optional

import litellm

from aperag.llm.llm_error_types import (
    CompletionError,
    InvalidPromptError,
    ResponseParsingError,
    ToolCallError,
    wrap_litellm_error,
)

logger = logging.getLogger(__name__)


class CompletionService:
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _agenerate_stream(self, history, prompt, memory=False):
        try:
            # Validate inputs
            if not prompt or not prompt.strip():
                raise InvalidPromptError("Prompt cannot be empty", prompt[:100] if prompt else "")

            response = await litellm.acompletion(
                custom_llm_provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
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
                if choice.delta and choice.delta.content:
                    yield choice.delta.content
        except Exception as e:
            logger.error(f"Completion streaming failed: {str(e)}")
            # Convert litellm exceptions to our custom types
            raise wrap_litellm_error(e, "completion", self.provider, self.model) from e

    def _generate_stream(self, history, prompt, memory=False):
        try:
            # Validate inputs
            if not prompt or not prompt.strip():
                raise InvalidPromptError("Prompt cannot be empty", prompt[:100] if prompt else "")

            response = litellm.completion(
                custom_llm_provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
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
                if choice.delta and choice.delta.content:
                    yield choice.delta.content
        except Exception as e:
            logger.error(f"Completion streaming failed: {str(e)}")
            # Convert litellm exceptions to our custom types
            raise wrap_litellm_error(e, "completion", self.provider, self.model) from e

    async def agenerate_stream(self, history, prompt, memory=False):
        try:
            async for tokens in self._agenerate_stream(history, prompt, memory):
                yield tokens
        except CompletionError:
            # Re-raise our custom completion errors
            raise
        except Exception as e:
            logger.error(f"Async completion generation failed: {str(e)}")
            raise wrap_litellm_error(e, "completion", self.provider, self.model) from e

    async def agenerate_by_tools(self, prompt, tools):
        try:
            # Validate inputs
            if not prompt or not prompt.strip():
                raise InvalidPromptError("Prompt cannot be empty", prompt[:100] if prompt else "")

            if not tools or not isinstance(tools, list):
                raise ToolCallError(reason="Tools parameter must be a non-empty list")

            response = await litellm.acompletion(
                custom_llm_provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice="auto",
            )

            if not response.choices or not response.choices[0].message:
                raise ResponseParsingError("No valid response received from completion API")

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            content = getattr(message, "content", None)

            return tool_calls, content
        except (InvalidPromptError, ToolCallError, ResponseParsingError):
            # Re-raise our custom errors
            raise
        except Exception as e:
            logger.error(f"Tool-based completion failed: {str(e)}")
            raise wrap_litellm_error(e, "completion", self.provider, self.model) from e

    def generate_stream(self, history, prompt, memory=False):
        try:
            for tokens in self._generate_stream(history, prompt, memory):
                yield tokens
        except CompletionError:
            # Re-raise our custom completion errors
            raise
        except Exception as e:
            logger.error(f"Sync completion generation failed: {str(e)}")
            raise wrap_litellm_error(e, "completion", self.provider, self.model) from e
