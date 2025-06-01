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

import logging
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from aperag.db.ops import query_collection
from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from aperag.query.query import DocumentWithScore
from aperag.utils.utils import generate_vector_db_collection_name
from config import settings

logger = logging.getLogger(__name__)


class KeywordSearchInput(BaseModel):
    query: str = Field(..., description="User's question or query")
    top_k: int = Field(5, description="Number of top results to return")
    collection_ids: Optional[List[str]] = Field(default_factory=list, description="Collection IDs")


class KeywordSearchOutput(BaseModel):
    docs: List[DocumentWithScore]


@register_node_runner(
    "keyword_search",
    input_model=KeywordSearchInput,
    output_model=KeywordSearchOutput,
)
class KeywordSearchNodeRunner(BaseNodeRunner):
    async def run(self, ui: KeywordSearchInput, si: SystemInput) -> Tuple[KeywordSearchOutput, dict]:
        """
        Run keyword search node. ui: user input; si: system input (SystemInput).
        Returns (output, system_output)
        """
        query = si.query
        topk = ui.top_k
        collection_ids = ui.collection_ids or []
        collection = None
        if collection_ids:
            collection = await query_collection(si.user, collection_ids[0])
        if not collection:
            return KeywordSearchOutput(docs=[]), {}

        from aperag.context.full_text import search_document
        from aperag.pipeline.keyword_extractor import IKExtractor

        index = generate_vector_db_collection_name(collection.id)
        async with IKExtractor({"index_name": index, "es_host": settings.ES_HOST}) as extractor:
            keywords = await extractor.extract(query)

        # find the related documents using keywords
        docs = await search_document(index, keywords, topk * 3)
        for doc in docs:
            doc.metadata["recall_type"] = "keyword_search"
        return KeywordSearchOutput(docs=docs), {}
