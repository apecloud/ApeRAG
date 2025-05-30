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
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)


# User input model for graph search node
class GraphSearchInput(BaseModel):
    top_k: int = Field(5, description="Number of top results to return")
    collection_ids: Optional[list[str]] = Field(default_factory=list, description="Collection IDs")


# User output model for graph search node
class GraphSearchOutput(BaseModel):
    docs: List[DocumentWithScore]


@register_node_runner(
    "graph_search",
    input_model=GraphSearchInput,
    output_model=GraphSearchOutput,
)
class GraphSearchNodeRunner(BaseNodeRunner):
    async def run(self, ui: GraphSearchInput, si: SystemInput) -> Tuple[GraphSearchOutput, dict]:
        """
        Run graph search node. up: user configurable params; sp: system injected params (SystemInput).
        Returns (uo, so)
        """
        query: str = si.query
        topk: int = ui.top_k
        collection_ids = ui.collection_ids or []
        collection = None
        if collection_ids:
            collection = await query_collection(si.user, collection_ids[0])
        if not collection:
            return GraphSearchOutput(docs=[]), {}

        config = parseCollectionConfig(collection.config)
        if not config.enable_knowledge_graph:
            logger.warning(f"Collection {collection.id} does not have knowledge graph enabled")
            return GraphSearchOutput(docs=[]), {}

        # Import LightRAG and run as in _run_light_rag
        from lightrag import QueryParam

        from aperag.graph import lightrag_holder
        from aperag.graph.lightrag_holder import LightRagHolder

        rag: LightRagHolder = await lightrag_holder.get_lightrag_holder(collection)
        param: QueryParam = QueryParam(
            mode="hybrid",
            only_need_context=True,
            top_k=topk,
        )
        context = await rag.aquery(query=query, param=param)
        return GraphSearchOutput(docs=[DocumentWithScore(text=context, metadata={"recall_type": "graph_search"})]), {}
