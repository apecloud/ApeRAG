import json
from typing import Any, Dict, List, Optional
import uuid

from litellm import BaseModel

from langchain.schema import AIMessage, HumanMessage
from aperag.chat.history.base import BaseChatMessageHistory
from aperag.db.models import Bot
from aperag.db.ops import query_msp_dict
from aperag.flow.base.models import BaseNodeRunner, register_node_runner, NodeInstance
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
    await history.add_message(
        HumanMessage(
            content=human_msg,
            additional_kwargs={"role": "human"}
        )
    )

async def add_ai_message(history: BaseChatMessageHistory, message, message_id, response, references, urls):
    ai_msg = new_ai_message(message, message_id, response, references, urls)
    ai_msg = ai_msg.json(exclude_none=True)
    await history.add_message(
        AIMessage(
            content=ai_msg,
            additional_kwargs={"role": "ai"}
        )
    )

@register_node_runner("llm")
class LLMNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        user = inputs.get("user")
        message_id: str = inputs["message_id"]
        query: str = inputs["query"]
        temperature: float = inputs.get("temperature", 0.2)
        max_tokens: int = inputs.get("max_tokens", 1000)
        model_service_provider = inputs.get("model_service_provider")
        model_name = inputs.get("model_name")
        custom_llm_provider = inputs.get("custom_llm_provider")
        prompt_template = inputs.get("prompt_template", "{context}\n{query}")
        docs: List[DocumentWithScore] = inputs.get("docs", [])

        history: BaseChatMessageHistory = inputs.get("history")
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
                references.append({
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "score": doc.score
                })
        prompt = prompt_template.format(query=query, context=context)
        llm_kwargs = {
            "custom_llm_provider": custom_llm_provider,
            "temperature": temperature,
            "max_tokens": max_tokens - len(prompt),
        }
        predictor = Predictor.get_completion_service(model_service_provider, model_name, base_url, api_key, **llm_kwargs)
        async def async_generator():
            response = ""
            async for chunk in predictor.agenerate_stream([], prompt, False):
                yield chunk
                if chunk:
                    response += chunk
            if references:
                yield DOC_QA_REFERENCES + json.dumps(references)
            if history:
                await add_human_message(history, query, message_id)
                await add_ai_message(history, query, message_id, response, references, [])
        return {"async_generator": async_generator} 
