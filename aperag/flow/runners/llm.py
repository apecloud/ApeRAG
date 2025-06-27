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
# Conservative estimate: 2 characters = 1 token
CHAR_TO_TOKEN_RATIO = 2.0

# Reserve tokens for output generation (default 1000 tokens)
DEFAULT_OUTPUT_TOKENS = 1000

# Fallback max context length if model context_window is not available
FALLBACK_MAX_CONTEXT_LENGTH = 50000

# Minimum required output tokens
MIN_OUTPUT_TOKENS = 100


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
    human_msg = human_msg.json(exclude_none=True)
    await history.add_message(HumanMessage(content=human_msg, additional_kwargs={"role": "human"}))


async def add_ai_message(history: BaseChatMessageHistory, message, message_id, response, references, urls):
    ai_msg = new_ai_message(message, message_id, response, references, urls)
    ai_msg = ai_msg.json(exclude_none=True)
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


def estimate_token_count(text: str) -> int:
    """
    Estimate token count from character count for Chinese/mixed content.
    Using conservative ratio: 2 characters = 1 token
    """
    return int(len(text) / CHAR_TO_TOKEN_RATIO)


def calculate_input_limits(
    context_window: Optional[int],
    max_input_tokens: Optional[int], 
    max_output_tokens: Optional[int],
    output_tokens: int = DEFAULT_OUTPUT_TOKENS
) -> Tuple[int, int]:
    """
    Calculate input context length and output token limits based on model configuration.
    
    Special handling for two scenarios:
    1. If context_window == max_input_tokens: max_input_tokens represents total context (input+output)
    2. If context_window > max_input_tokens: max_input_tokens is purely input limit
    
    Args:
        context_window: Total context window size (tokens)
        max_input_tokens: Maximum input tokens allowed
        max_output_tokens: Maximum output tokens allowed
        output_tokens: Desired output tokens (fallback)
    
    Returns:
        Tuple of (max_input_character_length, output_max_tokens)
    """
    # Determine output token limit
    if max_output_tokens is not None:
        output_max_tokens = max_output_tokens
    else:
        output_max_tokens = output_tokens
    
    # Determine input token limit with special handling
    if max_input_tokens is not None and context_window is not None:
        if context_window == max_input_tokens:
            # Case 1: max_input_tokens represents total context (input + output)
            max_input_token_limit = max_input_tokens - output_max_tokens
            if max_input_token_limit <= 0:
                # If total context is too small, use minimal allocation
                max_input_token_limit = max(max_input_tokens // 2, 100)
                # Adjust output tokens accordingly
                output_max_tokens = max(max_input_tokens - max_input_token_limit, MIN_OUTPUT_TOKENS)
        else:
            # Case 2: max_input_tokens is purely input limit, context_window is total
            max_input_token_limit = max_input_tokens
            # Ensure total doesn't exceed context_window
            if max_input_token_limit + output_max_tokens > context_window:
                # Prioritize input limit, adjust output if needed
                output_max_tokens = max(context_window - max_input_token_limit, MIN_OUTPUT_TOKENS)
    elif max_input_tokens is not None:
        # Only max_input_tokens available, assume it's pure input limit
        max_input_token_limit = max_input_tokens
    elif context_window is not None:
        # Only context_window available, calculate from total minus output
        max_input_token_limit = context_window - output_max_tokens
        if max_input_token_limit <= 0:
            # If context window is too small, use minimal allocation
            max_input_token_limit = max(context_window // 2, 100)
            # Adjust output tokens accordingly
            output_max_tokens = max(context_window - max_input_token_limit, MIN_OUTPUT_TOKENS)
    else:
        # Fallback to default values
        max_input_token_limit = FALLBACK_MAX_CONTEXT_LENGTH // int(CHAR_TO_TOKEN_RATIO)
    
    # Convert input token limit to character count
    max_input_character_length = int(max_input_token_limit * CHAR_TO_TOKEN_RATIO)
    
    return max_input_character_length, output_max_tokens


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

        # Get model configuration to determine token limits
        try:
            model_config = await async_db_ops.query_llm_provider_model(
                provider_name=model_service_provider,
                api=APIType.COMPLETION.value,
                model=model_name
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

        # Calculate input and output limits based on model configuration
        max_context_length, output_max_tokens = calculate_input_limits(
            context_window=context_window,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens
        )

        # Build context and references from documents
        context = ""
        references = []
        if docs:
            for doc in docs:
                if len(context) + len(doc.text) > max_context_length:
                    break
                context += doc.text
                references.append({"text": doc.text, "metadata": doc.metadata, "score": doc.score})

        prompt = prompt_template.format(query=query, context=context)
        
        # Validate prompt size against input limits
        prompt_tokens = estimate_token_count(prompt)
        
        # Validate based on the relationship between context_window and max_input_tokens
        if max_input_tokens and context_window:
            if context_window == max_input_tokens:
                # Case 1: max_input_tokens represents total context (input + output)
                if prompt_tokens + output_max_tokens > max_input_tokens:
                    raise Exception(
                        f"Prompt ({prompt_tokens} tokens) plus output ({output_max_tokens} tokens) exceeds the model's total context limit of {max_input_tokens} tokens"
                    )
            else:
                # Case 2: max_input_tokens is purely input limit
                if prompt_tokens > max_input_tokens:
                    raise Exception(
                        f"Prompt requires approximately {prompt_tokens} tokens, which exceeds the model's max_input_tokens limit of {max_input_tokens}"
                    )
                # Also check total context window
                if prompt_tokens + output_max_tokens > context_window:
                    raise Exception(
                        f"Prompt ({prompt_tokens} tokens) plus output ({output_max_tokens} tokens) exceeds the model's context_window of {context_window} tokens"
                    )
        elif max_input_tokens:
            # Only max_input_tokens available, treat as pure input limit
            if prompt_tokens > max_input_tokens:
                raise Exception(
                    f"Prompt requires approximately {prompt_tokens} tokens, which exceeds the model's max_input_tokens limit of {max_input_tokens}"
                )
        elif context_window:
            # Only context_window available
            if prompt_tokens + output_max_tokens > context_window:
                raise Exception(
                    f"Prompt ({prompt_tokens} tokens) plus output ({output_max_tokens} tokens) exceeds the model's context_window of {context_window} tokens"
                )

        cs = CompletionService(custom_llm_provider, model_name, base_url, api_key, temperature, output_max_tokens)

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
