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

from aperag.config import settings
from aperag.query.query import DocumentWithScore


class RankerService:
    def __init__(self):
        self.dialect = f"{settings.rerank_backend}"
        self.model = f"{settings.rerank_service_model}"
        self.api_base = settings.rerank_service_url
        self.api_key = settings.rerank_service_token_api_key

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
