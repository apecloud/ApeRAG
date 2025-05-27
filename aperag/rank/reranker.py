#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List

import litellm

from aperag.query.query import DocumentWithScore
from config.settings import (
    RERANK_BACKEND,
    RERANK_SERVICE_MODEL,
    RERANK_SERVICE_TOKEN_API_KEY,
    RERANK_SERVICE_URL,
)


class RankerService:
    def __init__(self):
        self.dialect = f"{RERANK_BACKEND}"
        self.model = f"{RERANK_SERVICE_MODEL}"
        self.api_base = RERANK_SERVICE_URL
        self.api_key = RERANK_SERVICE_TOKEN_API_KEY

    async def rank(self, query: str, results: List[DocumentWithScore]):
        documents = [d.text for d in results]
        resp = await litellm.arerank(
            custom_llm_provider=self.dialect,
            model=self.model,
            query=query,
            documents=documents,
            api_key=self.api_key,
            api_base=self.api_base,
            return_documents=False,
        )
        indices = [item["index"] for item in resp["results"]]
        return [results[i] for i in indices]


async def rerank(message, results):
    svc = RankerService()
    return await svc.rank(message, results)
