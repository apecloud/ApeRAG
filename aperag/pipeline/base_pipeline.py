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
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from langchain.schema import AIMessage, HumanMessage
from pydantic import BaseModel

from aperag.chat.history.base import BaseChatMessageHistory
from aperag.db.ops import query_msp_dict
from aperag.llm.base import Predictor
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


DOC_QA_REFERENCES = "|DOC_QA_REFERENCES|"
DOCUMENT_URLS = "|DOCUMENT_URLS|"


class Pipeline(ABC):
    def __init__(
        self,
        bot,
        collection,
        history: BaseChatMessageHistory,
    ):
        self.bot = bot
        self.collection = collection
        self.history = history
        bot_config = json.loads(self.bot.config)
        self.llm_config = bot_config.get("llm", {})
        self.model = bot_config.get("model")
        self.model_service_provider = bot_config.get("model_service_provider")
        self.model_name = bot_config.get("model_name")
        self.memory = bot_config.get("memory", False)
        self.memory_count = 0
        self.memory_limit_length = bot_config.get("memory_length", 0)
        self.memory_limit_count = bot_config.get("memory_count", 10)
        self.use_ai_memory = bot_config.get("use_ai_memory", True)
        self.topk = self.llm_config.get("similarity_topk", 3)
        self.enable_keyword_recall = self.llm_config.get("enable_keyword_recall", False)
        self.score_threshold = self.llm_config.get("similarity_score_threshold", 0.5)
        self.context_window = self.llm_config.get("context_window", 4096)
        self.bot_context = ""

        self.prompt_template = self.llm_config.get("prompt_template", None)

    async def ainit(self):
        msp_dict = await query_msp_dict(self.bot.user)
        if self.model_service_provider in msp_dict:
            msp = msp_dict[self.model_service_provider]
            api_key = msp.api_key

            # Get base_url from LLMProvider
            try:
                from aperag.db.models import LLMProvider

                llm_provider = await LLMProvider.objects.aget(name=self.model_service_provider)
                base_url = llm_provider.base_url
            except LLMProvider.DoesNotExist:
                raise Exception(f"LLMProvider '{self.model_service_provider}' not found")

            self.predictor = Predictor.get_completion_service(
                self.model_service_provider, self.model_name, base_url, api_key, **self.llm_config
            )
        else:
            raise Exception("Model service provider not found")

    @staticmethod
    async def new_human_message(message, message_id):
        return Message(
            id=message_id,
            query=message,
            timestamp=now_unix_milliseconds(),
        )

    async def new_ai_message(self, message, message_id, response, references, urls):
        pass

    async def add_human_message(self, message, message_id):
        if not message_id:
            message_id = str(uuid.uuid4())

        human_msg = await self.new_human_message(message, message_id)
        human_msg = human_msg.json(exclude_none=True)
        await self.history.add_message(HumanMessage(content=human_msg, additional_kwargs={"role": "human"}))

    async def add_ai_message(self, message, message_id, response, references, urls):
        ai_msg = await self.new_ai_message(message, message_id, response, references, urls)
        ai_msg = ai_msg.json(exclude_none=True)
        await self.history.add_message(AIMessage(content=ai_msg, additional_kwargs={"role": "ai"}))

    @abstractmethod
    async def run(self, query, gen_references=False, message_id=""):
        pass

    async def update_bot_context(self, bot_context):
        self.bot_context = bot_context
