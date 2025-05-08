import json
from typing import Any, Dict, List, Optional
import uuid

from litellm import BaseModel

from langchain.schema import AIMessage, HumanMessage
from aperag.chat.history.base import BaseChatMessageHistory
from aperag.db.ops import query_msp_dict
from aperag.flow.base.models import BaseNodeRunner, register_node_runner, NodeInstance
from aperag.llm.base import Predictor
from aperag.pipeline.base_pipeline import DOC_QA_REFERENCES
from aperag.utils.utils import now_unix_milliseconds

class Message(BaseModel):
    id: str
    query: Optional[str] = None
    timestamp: Optional[int] = None
    response: Optional[str] = None
    urls: Optional[List[str]] = None
    references: Optional[List[Dict]] = None
    collection_id: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_size: Optional[int] = None
    embedding_score_threshold: Optional[float] = None
    embedding_topk: Optional[int] = None
    llm_model: Optional[str] = None
    llm_prompt_template: Optional[str] = None
    llm_context_window: Optional[int] = None

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
        bot = inputs["bot"]
        docs = inputs["docs"]
        history: BaseChatMessageHistory = inputs.get("history")
        bot_config = json.loads(bot.config)
        model_service_provider = bot_config.get("model_service_provider")
        model_name = bot_config.get("model_name")
        llm_config = bot_config.get("llm", {})
        msp_dict = await query_msp_dict(bot.user)
        if model_service_provider in msp_dict:
            msp = msp_dict[model_service_provider]
            base_url = msp.base_url
            api_key = msp.api_key
            predictor = Predictor.get_completion_service(model_service_provider, model_name, base_url, api_key, **llm_config)
        else:
            raise Exception("Model service provider not found")
        prompt_template = llm_config.get("prompt_template", "{context}\n{query}")
        context = ""
        references = []
        if docs:
            max_length = 100000
            for doc in docs:
                if len(context) + len(doc.text) > max_length:
                    break
                context += doc.text
                references.append({
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "score": doc.score
                })
        prompt = prompt_template.format(query=inputs["query"], context=context)
        async def async_generator():
            response = ""
            async for chunk in predictor.agenerate_stream([], prompt, False):
                yield chunk
                response += chunk
            if references:
                yield DOC_QA_REFERENCES + json.dumps(references)
            if history:
                await add_human_message(history, inputs["query"], inputs["message_id"])
                await add_ai_message(history, inputs["query"], inputs["message_id"], response, references, [])
        return {"async_generator": async_generator} 
