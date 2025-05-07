import json
from typing import Any, Dict

from aperag.db.ops import query_msp_dict
from aperag.flow.base.models import BaseNodeRunner, register_node_runner, NodeInstance
from aperag.llm.base import Predictor
from aperag.query.query import get_packed_answer

@register_node_runner("llm")
class LLMNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        bot = inputs["bot"]
        docs = inputs["docs"]
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
        if docs:
            context = get_packed_answer(docs, 1000)
        else:
            context = ""
        prompt = prompt_template.format(query=inputs["query"], context=context)
        async def async_generator():
            async for chunk in predictor.agenerate_stream([], prompt, False):
                yield chunk
        return {"async_generator": async_generator} 
