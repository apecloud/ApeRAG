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
import uuid
from typing import Dict, List, Optional, Tuple

from langchain.schema import AIMessage, HumanMessage
from litellm import BaseModel
from pydantic import Field

from aperag.db.models import APIType
from aperag.db.ops import async_db_ops
from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from aperag.llm.completion.completion_service import CompletionService
from aperag.llm.llm_error_types import InvalidConfigurationError
from aperag.query.query import DocumentWithScore
from aperag.utils.constant import DOC_QA_REFERENCES
from aperag.utils.history import BaseChatMessageHistory
from aperag.utils.utils import now_unix_milliseconds

# Character to token estimation ratio for Chinese/mixed content
# Conservative estimate: 1.5 characters = 1 token
TOKEN_TO_CHAR_RATIO = 1.5

# Reserve tokens for output generation (default 1000 tokens)
DEFAULT_OUTPUT_TOKENS = 1000

# Fallback max context length if model context_window is not available
FALLBACK_MAX_CONTEXT_LENGTH = 50000


class Message(BaseModel):
    id: str
    query: Optional[str] = None
    timestamp: Optional[int] = None
    response: Optional[str] = None
    urls: Optional[List[str]] = None
    references: Optional[List[Dict]] = None


def new_ai_message(message, message_id, response, references, urls):
    return Message(
        id=message_id,
        query=message,
        response=response,
        timestamp=now_unix_milliseconds(),
        references=references,
        urls=urls,
    )


def new_human_message(message, message_id):
    return Message(
        id=message_id,
        query=message,
        timestamp=now_unix_milliseconds(),
    )


async def add_human_message(history: BaseChatMessageHistory, message, message_id):
    if not message_id:
        message_id = str(uuid.uuid4())

    human_msg = new_human_message(message, message_id)
    human_msg = human_msg.model_dump_json(exclude_none=True)
    await history.add_message(HumanMessage(content=human_msg, additional_kwargs={"role": "human"}))


async def add_ai_message(history: BaseChatMessageHistory, message, message_id, response, references, urls):
    ai_msg = new_ai_message(message, message_id, response, references, urls)
    ai_msg = ai_msg.model_dump_json(exclude_none=True)
    await history.add_message(AIMessage(content=ai_msg, additional_kwargs={"role": "ai"}))


class LLMInput(BaseModel):
    model_service_provider: str = Field(..., description="Model service provider")
    model_name: str = Field(..., description="Model name")
    custom_llm_provider: str = Field(..., description="Custom LLM provider")
    prompt_template: str = Field(..., description="Prompt template")
    temperature: float = Field(..., description="Sampling temperature")
    docs: Optional[List[DocumentWithScore]] = Field(None, description="Documents")


class LLMOutput(BaseModel):
    text: str


async def calculate_model_token_limits(
    model_service_provider: str,
    model_name: str,
) -> Tuple[int, int]:
    """
    Calculate input and output token limits based on three constraints:

    1. Reserve at least 4096 tokens for output (or model's max_output_tokens if smaller)
    2. Input (query+context) should not exceed model's input limit (min of context_window, max_input_tokens)
    3. Total (input + output) should not exceed context_window

    Args:
        model_service_provider: Model service provider name
        model_name: Model name

    Returns:
        Tuple of (max_input_tokens, final_output_tokens)
    """
    # Get model configuration to determine token limits
    try:
        model_config = await async_db_ops.query_llm_provider_model(
            provider_name=model_service_provider, api=APIType.COMPLETION.value, model=model_name
        )
        if model_config:
            context_window = model_config.context_window
            max_input_tokens = model_config.max_input_tokens
            max_output_tokens = model_config.max_output_tokens
        else:
            context_window = None
            max_input_tokens = None
            max_output_tokens = None
    except Exception:
        context_window = None
        max_input_tokens = None
        max_output_tokens = None

    # Constraint 1: Determine output token reservation
    reserved_output_tokens = min(max_output_tokens or DEFAULT_OUTPUT_TOKENS, DEFAULT_OUTPUT_TOKENS)

    # Constraint 2: Determine maximum input tokens allowed
    input_limits = []
    if max_input_tokens:
        input_limits.append(max_input_tokens)
    if context_window:
        input_limits.append(context_window)

    if input_limits:
        max_allowed_input = min(input_limits)
    else:
        # Fallback if no limits available
        max_allowed_input = FALLBACK_MAX_CONTEXT_LENGTH // int(TOKEN_TO_CHAR_RATIO)

    # Constraint 3: Ensure total doesn't exceed context_window
    if context_window:
        # Make sure input + output <= context_window
        max_allowed_input = min(max_allowed_input, context_window - reserved_output_tokens)

    # Ensure we have at least some minimal input space
    if max_allowed_input <= 0:
        max_allowed_input = 100  # Minimal fallback

    return max_allowed_input, reserved_output_tokens


# Database operations interface
class LLMRepository:
    """Repository interface for LLM database operations"""

    pass


# Business logic service
class LLMService:
    """Service class containing LLM business logic"""

    def __init__(self, repository: LLMRepository):
        self.repository = repository

    async def generate_response(
        self,
        user,
        query: str,
        message_id: str,
        history: BaseChatMessageHistory,
        model_service_provider: str,
        model_name: str,
        custom_llm_provider: str,
        prompt_template: str,
        temperature: float,
        docs: Optional[List[DocumentWithScore]] = None,
    ) -> Tuple[str, Dict]:
        """Generate LLM response with given parameters"""
        api_key = await async_db_ops.query_provider_api_key(model_service_provider, user)
        if not api_key:
            raise InvalidConfigurationError(
                "api_key", None, f"API KEY not found for LLM Provider: {model_service_provider}"
            )

        try:
            llm_provider = await async_db_ops.query_llm_provider_by_name(model_service_provider)
            base_url = llm_provider.base_url
        except Exception:
            raise Exception(f"LLMProvider {model_service_provider} not found")

        # Calculate input and output limits based on model configuration
        max_input_tokens, max_output_tokens = await calculate_model_token_limits(
            model_service_provider=model_service_provider,
            model_name=model_name,
        )

        # Build context and references from documents
        max_input_chars = max_input_tokens * TOKEN_TO_CHAR_RATIO
        context = ""
        references = []
        if docs:
            for doc in docs:
                # Estimate final prompt length: template + query + current context + new doc
                estimated_prompt_length = len(prompt_template) + len(query) + len(context) + len(doc.text)
                if estimated_prompt_length > max_input_chars:
                    break
                context += doc.text
                references.append({"text": doc.text, "metadata": doc.metadata, "score": doc.score})

        prompt = prompt_template.format(query=query, context=context)
        if len(prompt) > max_input_chars:
            raise Exception(
                f"Prompt requires {len(prompt)} characters, which exceeds the calculated "
                f"input limit of {max_input_chars} characters"
            )

        cs = CompletionService(custom_llm_provider, model_name, base_url, api_key, temperature, max_output_tokens)

        async def async_generator():
            response = ""
            async for chunk in cs.agenerate_stream([], prompt, False):
                if not chunk:
                    continue
                yield chunk
                response += chunk

            if references:
                yield DOC_QA_REFERENCES + json.dumps(references)

            if history:
                await add_human_message(history, query, message_id)
                await add_ai_message(history, query, message_id, response, references, [])

        return "", {"async_generator": async_generator}


@register_node_runner(
    "llm",
    input_model=LLMInput,
    output_model=LLMOutput,
)
class LLMNodeRunner(BaseNodeRunner):
    def __init__(self):
        self.repository = LLMRepository()
        self.service = LLMService(self.repository)

    async def run(self, ui: LLMInput, si: SystemInput) -> Tuple[LLMOutput, dict]:
        """
        Run LLM node. ui: user input; si: system input (SystemInput).
        Returns (output, system_output)
        """
        text, system_output = await self.service.generate_response(
            user=si.user,
            query=si.query,
            message_id=si.message_id,
            history=si.history,
            model_service_provider=ui.model_service_provider,
            model_name=ui.model_name,
            custom_llm_provider=ui.custom_llm_provider,
            prompt_template=ui.prompt_template,
            temperature=ui.temperature,
            docs=ui.docs,
        )

        return LLMOutput(text=text), system_output
