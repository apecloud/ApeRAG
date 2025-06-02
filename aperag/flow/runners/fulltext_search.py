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

from aperag.config import settings
from aperag.db.ops import query_collection
from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from aperag.query.query import DocumentWithScore
from aperag.utils.utils import generate_vector_db_collection_name

logger = logging.getLogger(__name__)


class FulltextSearchInput(BaseModel):
    query: str = Field(..., description="User's question or query")
    top_k: int = Field(5, description="Number of top results to return")
    collection_ids: Optional[List[str]] = Field(default_factory=list, description="Collection IDs")


class FulltextSearchOutput(BaseModel):
    docs: List[DocumentWithScore]


@register_node_runner(
    "fulltext_search",
    input_model=FulltextSearchInput,
    output_model=FulltextSearchOutput,
)
class FulltextSearchNodeRunner(BaseNodeRunner):
    async def run(self, ui: FulltextSearchInput, si: SystemInput) -> Tuple[FulltextSearchOutput, dict]:
        """
        Run fulltext search node. ui: user input; si: system input (SystemInput).
        Returns (output, system_output)
        """
        query = si.query
        topk = ui.top_k
        collection_ids = ui.collection_ids or []
        collection = None
        if collection_ids:
            collection = await query_collection(si.user, collection_ids[0])
        if not collection:
            return FulltextSearchOutput(docs=[]), {}

        from aperag.context.full_text import search_document
        from aperag.pipeline.keyword_extractor import IKExtractor

        index = generate_vector_db_collection_name(collection.id)
        async with IKExtractor({"index_name": index, "es_host": settings.es_host}) as extractor:
            keywords = await extractor.extract(query)

        # find the related documents using keywords
        docs = await search_document(index, keywords, topk * 3)
        for doc in docs:
            doc.metadata["recall_type"] = "fulltext_search"
        return FulltextSearchOutput(docs=docs), {}
