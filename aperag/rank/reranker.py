#!/usr/bin/env python3
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
