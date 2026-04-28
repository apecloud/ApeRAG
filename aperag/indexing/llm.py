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

"""LLM helpers for the new indexing pipeline — Wave 5 P1 chunk 1.

Relocates two helpers out of the legacy ``graphindex/`` package so the
new indexing pipeline (``aperag/indexing/``) does not depend on legacy
package paths. Per architect msg=87e2b187 chunk 4d Option C ruling +
Wave 5 task #26 spec — the legacy ``aperag/domains/knowledge_graph/
graphindex/`` package is being phased out as a coordinated batch in
Wave 5; this module is the new home for the two LLM-callable + prompt-
template helpers that the new ``aperag.indexing.graph_extractor``
already consumes.

Public surface:

* :func:`build_collection_llm_callable` — read the collection's
  completion config and return an async ``(prompt) -> str`` closure
  bound to a single :class:`CompletionService` instance. The
  closure is reusable across many extractor calls for the same
  collection so we pay the LiteLLM import + HTTP session setup only
  once per collection per process.
* :func:`render_extraction_prompt` — fill in the canonical entity /
  relation extraction prompt template for one chunk batch. Kept as a
  separate function so tests can snapshot the rendered string and
  catch accidental prompt regressions.

The legacy modules (``graphindex.integration.build_collection_llm_callable``
and ``graphindex.prompts.render_extraction_prompt``) re-export from this
module during the Wave 5 deprecation window so external callers (legacy
graphindex retrieval / curation paths) keep working until the cross-
cutting refactor lands the migration. Once those callers move to the
§G.5 read primitives, the legacy modules can be deleted with no
behavioural change.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from aperag.db.ops import db_ops
from aperag.domains.knowledge_graph.ports import CollectionRow
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)


LLMCall = Callable[[str], Awaitable[str]]
"""Async callable returning the model's response to a single prompt.
The new indexing pipeline's graph extractor consumes this exact shape
(``aperag.indexing.graph_extractor`` line ~165)."""


def build_collection_llm_callable(collection: CollectionRow) -> LLMCall:
    """Construct the per-collection async LLM callable.

    Reads the collection's completion config (provider, model, base_url,
    api key from the user's provider record) and returns an async
    ``(prompt) -> str`` closure. The closure is bound to a single
    :class:`CompletionService` instance so concurrent extractor calls
    against the same collection share the underlying client without
    paying the LiteLLM import + HTTP session setup per chunk.

    Raises ``RuntimeError`` if the collection has no completion model
    configured or the configured model cannot be resolved against the
    runtime registry. The :class:`aperag.indexing.graph_extractor`
    builder catches this and wraps it in
    :class:`aperag.indexing.worker_factory.WorkerFactoryError` so the
    orchestrator finalises the row FAILED with operator-facing
    diagnostics (Wave 3 lesson #10 explicit-fail-not-silent pattern).
    """
    config = parseCollectionConfig(collection.config)
    if not config.completion or not config.completion.model_id:
        raise RuntimeError(f"indexing.llm: completion model not configured (collection {collection.id})")
    row = db_ops.query_model_runtime(config.completion.model_id, collection.user)
    if not row:
        raise RuntimeError(f"indexing.llm: model not found {config.completion.model_id!r} (collection {collection.id})")
    model, account = row

    # Local import: CompletionService pulls in litellm which is heavy
    # at import time and we don't want to pay it just for
    # ``import aperag.indexing.llm``.
    from aperag.llm.completion.completion_service import CompletionService
    from aperag.llm.runtime.resolver import resolve_model_invocation_from_records

    invocation = resolve_model_invocation_from_records(model=model, account=account)
    provider = invocation.runner_config.get("provider")
    if not provider:
        provider = "openai" if invocation.runner_type == "openai_compatible" else invocation.provider_type

    svc = CompletionService(
        provider=provider,
        model=invocation.provider_model_id,
        base_url=invocation.base_url,
        api_key=invocation.api_key,
        temperature=0.0,  # deterministic output for extraction
        max_tokens=None,
        caching=False,
    )

    async def _llm(prompt: str) -> str:
        # No history, no images, no memory. Single-turn JSON request.
        return await svc.agenerate(history=[], prompt=prompt, images=[], memory=False)

    return _llm


# ---------------------------------------------------------------------
# Prompt templates — relocated from ``graphindex.prompts``.
# ---------------------------------------------------------------------


ENTITY_RELATION_EXTRACTION: str = """\
You are an information-extraction assistant building a knowledge graph.

Read the TEXT below and return a single JSON object with two arrays:
``entities`` and ``relations``. Do not output anything outside the JSON
object.

**Rules**

1. Output language: {language}. Names keep their original script and
   case (e.g. English names capitalized, Chinese names unchanged).
2. Prefer existing entity types from this collection list:
   {entity_types}
3. If no existing entity type fits, create a concise new entity type
   string in {language}. Do not skip a valid entity only because its
   type is not already listed.
4. Every entity needs a short, self-contained description in
   {language}. Do not add information that isn't in the text.
5. Every relation must reference entities by the exact ``name`` you put
   in the ``entities`` list. No self-loops (source != target).
6. ``weight`` is an integer 1-10 expressing how strongly the text
   supports the relation; default to 5 when unsure.
7. Cap: at most {max_entities} entities and {max_relations} relations.
   If the text contains more, prefer the most specific and the
   most-mentioned.
8. If the text has no extractable entities, return
   ``{{"entities": [], "relations": []}}``.

**JSON schema**

```
{{
  "entities": [
    {{"name": "<string>", "entity_type": "<existing or new type string>",
      "description": "<string>"}}
  ],
  "relations": [
    {{"source": "<entity name>", "target": "<entity name>",
      "relation_type": "<short relation type>",
      "description": "<string>", "weight": <int 1-10>}}
  ]
}}
```

**Example** (English):

Text:
```
Alice, a researcher at Acme Labs, collaborated with Bob on the project.
```

Output:
```
{{
  "entities": [
    {{"name": "Alice", "entity_type": "Person",
      "description": "A researcher at Acme Labs."}},
    {{"name": "Bob", "entity_type": "Person",
      "description": "A collaborator on the project."}},
    {{"name": "Acme Labs", "entity_type": "Organization",
      "description": "Research organization where Alice works."}}
  ],
  "relations": [
    {{"source": "Alice", "target": "Bob",
      "relation_type": "collaborated_with",
      "description": "Alice and Bob collaborated on a project.",
      "weight": 7}},
    {{"source": "Alice", "target": "Acme Labs",
      "relation_type": "works_for",
      "description": "Alice is a researcher employed by Acme Labs.",
      "weight": 8}}
  ]
}}
```

---

TEXT:
```
{input_text}
```

Output (JSON only):"""


def render_extraction_prompt(
    *,
    input_text: str,
    entity_types: list[str] | tuple[str, ...],
    language: str,
    max_entities: int,
    max_relations: int,
) -> str:
    """Fill in the extraction prompt template for one chunk.

    Kept as a simple function so tests can snapshot the exact rendered
    string and catch accidental prompt regressions. The template is
    identical to the legacy ``graphindex.prompts.ENTITY_RELATION_EXTRACTION``
    text (relocated here verbatim during Wave 5 P1).
    """
    from aperag.indexing.entity_types import format_entity_types_for_prompt

    return ENTITY_RELATION_EXTRACTION.format(
        input_text=input_text,
        entity_types=format_entity_types_for_prompt(entity_types),
        language=language,
        max_entities=max_entities,
        max_relations=max_relations,
    )


__all__ = [
    "ENTITY_RELATION_EXTRACTION",
    "LLMCall",
    "build_collection_llm_callable",
    "render_extraction_prompt",
]
