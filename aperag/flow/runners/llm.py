import json
import uuid
from typing import Dict, List, Optional, Tuple

from langchain.schema import AIMessage, HumanMessage
from litellm import BaseModel
from pydantic import Field

from aperag.chat.history.base import BaseChatMessageHistory
from aperag.db.ops import query_msp_dict
from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from aperag.llm.base import Predictor
from aperag.pipeline.base_pipeline import DOC_QA_REFERENCES
from aperag.query.query import DocumentWithScore
from aperag.utils.utils import now_unix_milliseconds

MAX_CONTEXT_LENGTH = 100000


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
    max_tokens: int = Field(..., description="Max tokens for generation")
    docs: Optional[List[DocumentWithScore]] = Field(None, description="Documents")


class LLMOutput(BaseModel):
    text: str


@register_node_runner(
    "llm",
    input_model=LLMInput,
    output_model=LLMOutput,
)
class LLMNodeRunner(BaseNodeRunner):
    async def run(self, ui: LLMInput, si: SystemInput) -> Tuple[LLMOutput, dict]:
        """
        Run LLM node. ui: user input; si: system input (SystemInput).
        Returns (output, system_output)
        """
        user = si.user
        query: str = si.query
        message_id: str = si.message_id
        history: BaseChatMessageHistory = si.history

        temperature: float = ui.temperature
        max_tokens: int = ui.max_tokens
        model_service_provider = ui.model_service_provider
        model_name = ui.model_name
        custom_llm_provider = ui.custom_llm_provider
        prompt_template = ui.prompt_template
        docs: List[DocumentWithScore] = ui.docs

        msp_dict = await query_msp_dict(user)
        if model_service_provider in msp_dict:
            msp = msp_dict[model_service_provider]
            base_url = msp.base_url
            api_key = msp.api_key
        else:
            raise Exception("Model service provider not found")
        context = ""
        references = []
        if docs:
            for doc in docs:
                if len(context) + len(doc.text) > MAX_CONTEXT_LENGTH:
                    break
                context += doc.text
                references.append({"text": doc.text, "metadata": doc.metadata, "score": doc.score})
        prompt = prompt_template.format(query=query, context=context)
        output_max_tokens = max_tokens - len(prompt)
        if output_max_tokens < 0:
            raise Exception(
                "max_tokens %d is too small to hold the prompt which size is %d" % (max_tokens, len(prompt))
            )
        llm_kwargs = {
            "custom_llm_provider": custom_llm_provider,
            "temperature": temperature,
            "max_tokens": output_max_tokens,
        }
        predictor = Predictor.get_completion_service(
            model_service_provider, model_name, base_url, api_key, **llm_kwargs
        )

        async def async_generator():
            response = ""
            async for chunk in predictor.agenerate_stream([], prompt, False):
                if not chunk:
                    continue
                yield chunk
                response += chunk
            if references:
                yield DOC_QA_REFERENCES + json.dumps(references)
            if history:
                await add_human_message(history, query, message_id)
                await add_ai_message(history, query, message_id, response, references, [])

        return LLMOutput(text=""), {"async_generator": async_generator}
