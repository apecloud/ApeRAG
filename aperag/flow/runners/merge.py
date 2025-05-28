from typing import List, Tuple

from pydantic import BaseModel, Field

from aperag.flow.base.exceptions import ValidationError
from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from aperag.query.query import DocumentWithScore


class MergeInput(BaseModel):
    merge_strategy: str = Field("union", description="How to merge results")
    deduplicate: bool = Field(True, description="Whether to deduplicate merged results")
    vector_search_docs: List[DocumentWithScore]
    keyword_search_docs: List[DocumentWithScore]


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
        docs_a: List[DocumentWithScore] = ui.vector_search_docs
        docs_b: List[DocumentWithScore] = ui.keyword_search_docs
        merge_strategy: str = ui.merge_strategy
        deduplicate: bool = ui.deduplicate

        if merge_strategy == "union":
            all_docs = docs_a + docs_b
            if deduplicate:
                seen = set()
                unique_docs = []
                for doc in all_docs:
                    if doc.text not in seen:
                        seen.add(doc.text)
                        unique_docs.append(doc)
                return MergeOutput(docs=unique_docs), {}
            return MergeOutput(docs=all_docs), {}
        else:
            raise ValidationError(f"Unknown merge strategy: {merge_strategy}")
