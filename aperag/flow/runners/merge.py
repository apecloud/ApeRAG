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

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from aperag.flow.base.exceptions import ValidationError
from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from aperag.query.query import DocumentWithScore


class MergeInput(BaseModel):
    merge_strategy: str = Field("union", description="How to merge results")
    deduplicate: bool = Field(True, description="Whether to deduplicate merged results")
    vector_search_docs: Optional[List[DocumentWithScore]] = Field(
        default_factory=list, description="Vector search docs"
    )
    keyword_search_docs: Optional[List[DocumentWithScore]] = Field(
        default_factory=list, description="Keyword search docs"
    )
    graph_search_docs: Optional[List[DocumentWithScore]] = Field(default_factory=list, description="Graph search docs")


class MergeOutput(BaseModel):
    docs: List[DocumentWithScore]


@register_node_runner(
    "merge",
    input_model=MergeInput,
    output_model=MergeOutput,
)
class MergeNodeRunner(BaseNodeRunner):
    async def run(self, ui: MergeInput, si: SystemInput) -> Tuple[MergeOutput, dict]:
        """
        Run merge node. ui: user input; si: system input (SystemInput).
        Returns (output, system_output)
        """
        docs_a: List[DocumentWithScore] = ui.vector_search_docs or []
        docs_b: List[DocumentWithScore] = ui.keyword_search_docs or []
        docs_c: List[DocumentWithScore] = ui.graph_search_docs or []
        merge_strategy: str = ui.merge_strategy
        deduplicate: bool = ui.deduplicate

        if merge_strategy not in ["union"]:
            raise ValidationError(f"Unknown merge strategy: {merge_strategy}")

        all_docs = docs_a + docs_b + docs_c
        if deduplicate:
            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc.text not in seen:
                    seen.add(doc.text)
                    unique_docs.append(doc)
            return MergeOutput(docs=unique_docs), {}
        return MergeOutput(docs=all_docs), {}
