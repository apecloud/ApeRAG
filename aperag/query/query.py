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

from typing import List, Optional

from pydantic import BaseModel


class DocumentWithScore(BaseModel):
    text: Optional[str] = None
    score: float
    metadata: Optional[dict] = None


class Query(BaseModel):
    query: str
    top_k: Optional[int] = 3


class QueryWithEmbedding(Query):
    embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[DocumentWithScore]


def get_packed_answer(results, limit_length: Optional[int] = 0) -> str:
    text_chunks = []
    for r in results:
        prefix = ""
        if r.metadata.get("url"):
            prefix = "The following information is from: " + r.metadata.get("url") + "\n"
        text_chunks.append(prefix + r.text)
    answer_text = "\n\n".join(text_chunks)
    if limit_length != 0:
        return answer_text[:limit_length]
    else:
        return answer_text
