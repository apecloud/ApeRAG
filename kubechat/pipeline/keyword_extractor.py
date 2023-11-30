import logging
import os.path
import re
from pathlib import Path
from typing import Dict, Any

from elasticsearch import Elasticsearch, AsyncElasticsearch
from langchain import PromptTemplate

from kubechat.llm.base import Predictor, PredictorType
from kubechat.llm.prompts import KEYWORD_PROMPT_TEMPLATE


logger = logging.getLogger(__name__)


class KeywordExtractor(object):
    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx

    async def extract(self, text):
        raise NotImplementedError


class IKExtractor(KeywordExtractor):
    """
    Extract keywords from text using IK
    """

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        self.client = AsyncElasticsearch(ctx.get("es_host", "http://127.0.0.1:9200"))
        self.index_name = ctx["index_name"]
        # TODO move stop words to global
        stop_words_path = ctx.get("stop_words_path", Path(__file__).parent / "stopwords.txt")
        if os.path.exists(stop_words_path):
            with open(stop_words_path) as f:
                self.stop_words = set(f.read().splitlines())
        else:
            self.stop_words = set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()

    async def extract(self, text):
        resp = await self.client.indices.exists(index=self.index_name)
        if not resp.body:
            logger.warning("index %s not exists", self.index_name)
            return []

        resp = await self.client.indices.analyze(index=self.index_name, body={"text": text}, analyzer="ik_smart")
        tokens = {}
        for item in resp.body["tokens"]:
            token = item["token"]
            if token in self.stop_words:
                continue
            tokens[token] = True
        return tokens.keys()


class LLMExtractor(KeywordExtractor):
    """
    Extract keywords from text using LLM
    """

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        self.predictor = Predictor.from_model(ctx["model"], PredictorType.CUSTOM_LLM)

    async def extract(self, text):
        keyword_template = PromptTemplate(template=KEYWORD_PROMPT_TEMPLATE, input_variables=["query"])
        keyword_prompt = keyword_template.format(query=text)
        keyword_response = ""
        for tokens in self.predictor.generate_stream(keyword_prompt):
            keyword_response += tokens
        keywords = []
        # the output format of LLM maybe unstable, so we use a list of delimiters to split the response
        delimiters = "\n, ：，、。"
        parts = re.split(f"[{re.escape(delimiters)}]", keyword_response)
        for item in parts:
            if item.lower() not in text.lower():
                logger.info("ignore keyword %s not found in message", item)
                continue
            item = item.strip()
            if not item:
                continue
            keywords.append(item)
        return keywords
