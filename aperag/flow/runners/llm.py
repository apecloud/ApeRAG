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

# Fallback max context length if model max_tokens is not available
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


def calculate_max_context_length(model_max_tokens: Optional[int], output_tokens: int = DEFAULT_OUTPUT_TOKENS) -> int:
    """
    Calculate maximum context length based on model's max_tokens limit.
    Reserve tokens for output generation.
    """
    if not model_max_tokens:
        return FALLBACK_MAX_CONTEXT_LENGTH
    
    # Reserve tokens for output, convert to character count
    max_context_tokens = model_max_tokens - output_tokens
    if max_context_tokens <= 0:
        # If model max_tokens is too small, use a minimal context
        max_context_tokens = max(model_max_tokens // 2, 100)
    
    # Convert tokens to character count
    return int(max_context_tokens * CHAR_TO_TOKEN_RATIO)


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

        # Get model configuration to determine max_tokens
        try:
            model_config = await async_db_ops.query_llm_provider_model(
                provider_name=model_service_provider,
                api=APIType.COMPLETION.value,
                model=model_name
            )
            model_max_tokens = model_config.max_tokens if model_config else None
        except Exception:
            model_max_tokens = None

        # Calculate dynamic context length based on model's max_tokens
        max_context_length = calculate_max_context_length(model_max_tokens)

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
        
        # Estimate prompt tokens and calculate output tokens
        prompt_tokens = estimate_token_count(prompt)
        if model_max_tokens:
            output_max_tokens = model_max_tokens - prompt_tokens
            if output_max_tokens < 100:  # Ensure minimum output tokens
                raise Exception(
                    f"Model max_tokens {model_max_tokens} is too small to hold the prompt which requires approximately {prompt_tokens} tokens"
                )
        else:
            # Use default output tokens if model max_tokens is unknown
            output_max_tokens = DEFAULT_OUTPUT_TOKENS

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
