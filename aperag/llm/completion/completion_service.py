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
from typing import Any, AsyncGenerator, Dict, List, Optional

import litellm

from aperag.llm.llm_error_types import (
    CompletionError,
    InvalidPromptError,
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
        vision: bool = False,
        caching: bool = True,
    ):
        super().__init__()
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.vision = vision
        self.caching = caching

    def is_vision_model(self) -> bool:
        return self.vision

    def _validate_inputs(self, prompt: Optional[str], images: Optional[List[str]] = None) -> None:
        """Validate input parameters."""
        if not self.vision:
            images = []
        if not images and (not prompt or not prompt.strip()):
            raise InvalidPromptError(
                "Prompt cannot be empty when no images are provided", prompt[:100] if prompt else ""
            )

    def _build_messages(
        self, history: List[Dict], prompt: Optional[str], images: Optional[List[str]] = None, memory: bool = False
    ) -> List[Dict]:
        """Build the messages array for the API call."""
        if self.vision and images:
            content: List[Dict[str, Any]] = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            for image_data in images:
                content.append({"type": "image_url", "image_url": image_data})
            user_message = {"role": "user", "content": content}
        else:
            user_message = {"role": "user", "content": prompt}

        return history + [user_message] if memory else [user_message]

    def _extract_content_from_response(self, response: Any) -> str:
        """Extract content from non-streaming response."""
        if not response or not response.choices:
            raise CompletionError("Empty response from completion API")

        choice = response.choices[0]
        if not choice.message:
            raise CompletionError("No message in completion response")

        if hasattr(choice.message, "content") and choice.message.content and choice.message.content.strip():
            return choice.message.content
        elif hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            return choice.message.reasoning_content
        else:
            raise CompletionError("No content in completion response")

    async def _acompletion_non_stream(
        self, history: List[Dict], prompt: str, images: Optional[List[str]] = None, memory: bool = False
    ) -> str:
        """Core async completion method for non-streaming responses."""
        try:
            self._validate_inputs(prompt, images)
            messages = self._build_messages(history, prompt, images, memory)

            response = await litellm.acompletion(
                custom_llm_provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
                stream=False,
                caching=self.caching,
            )

            return self._extract_content_from_response(response)

        except CompletionError:
            # Re-raise our custom completion errors
            raise
        except Exception as e:
            logger.error(f"Async completion generation failed: {str(e)}")
            raise wrap_litellm_error(e, "completion", self.provider, self.model) from e

    async def _acompletion_stream_raw(
        self, history: List[Dict], prompt: str, images: Optional[List[str]] = None, memory: bool = False
    ) -> AsyncGenerator[str, None]:
        """Core async completion method for streaming responses."""
        try:
            self._validate_inputs(prompt, images)
            messages = self._build_messages(history, prompt, images, memory)

            response = await litellm.acompletion(
                custom_llm_provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
                stream=True,
                caching=self.caching,
            )

            # Process the raw stream and yield clean text chunks
            async for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason == "stop":
                    return
                content_to_yield = None
                if choice.delta and choice.delta.content:
                    content_to_yield = choice.delta.content
                elif hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                    content_to_yield = choice.delta.reasoning_content
                if content_to_yield:
                    yield content_to_yield

        except CompletionError:
            # Re-raise our custom completion errors
            raise
        except Exception as e:
            logger.error(f"Async streaming generation failed: {str(e)}")
            raise wrap_litellm_error(e, "completion", self.provider, self.model) from e

    def _completion_core(
        self, history: List[Dict], prompt: str, images: Optional[List[str]] = None, memory: bool = False
    ) -> str:
        """Core sync completion method (non-streaming only)."""
        try:
            self._validate_inputs(prompt, images)
            messages = self._build_messages(history, prompt, images, memory)

            response = litellm.completion(
                custom_llm_provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
                stream=False,
                caching=self.caching,
            )

            return self._extract_content_from_response(response)

        except CompletionError:
            # Re-raise our custom completion errors
            raise
        except Exception as e:
            logger.error(f"Sync completion generation failed: {str(e)}")
            raise wrap_litellm_error(e, "completion", self.provider, self.model) from e

    async def agenerate_stream(
        self, history: List[Dict], prompt: str, images: Optional[List[str]] = None, memory: bool = False
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response (async)."""
        async for chunk in self._acompletion_stream_raw(history, prompt, images, memory):
            yield chunk

    async def agenerate(
        self, history: List[Dict], prompt: str, images: Optional[List[str]] = None, memory: bool = False
    ) -> str:
        """Generate complete response (async, non-streaming)."""
        return await self._acompletion_non_stream(history, prompt, images, memory)

    def generate(
        self, history: List[Dict], prompt: str, images: Optional[List[str]] = None, memory: bool = False
    ) -> str:
        """Generate complete response (sync, non-streaming)."""
        return self._completion_core(history, prompt, images, memory)
