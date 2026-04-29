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
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence

from aperag.db.ops import db_ops
from aperag.domains.knowledge_graph.ports import CollectionRow
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)


LLMCall = Callable[[str], Awaitable[str]]
"""Async callable returning the model's response to a single prompt.
The new indexing pipeline's graph extractor consumes this exact shape
(``aperag.indexing.graph_extractor`` line ~165)."""


def build_collection_llm_callable(
    collection: CollectionRow,
    *,
    response_format: Optional[Dict[str, Any]] = None,
) -> LLMCall:
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

    ``response_format`` (issue #1861, task #14): 透传给 ``CompletionService``
    的 LiteLLM provider-side 强约束. **默认 ``None``** 保留共享 builder 的
    向后兼容行为 — collection_regen / evaluation worker / dataset_generator /
    summary worker / graph curation 等非 graph extractor 调用方共用此
    builder, 不能默认注入 JSON 模式. 只有 graph extractor 调用入口
    (``aperag.indexing.graph_extractor.build_collection_graph_extractor``)
    显式传 ``{"type": "json_object"}`` 让 graph extraction 走 provider-side
    JSON 强约束.
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

    # issue #1861 (task #14): ``response_format`` 由 caller 显式传入决定是否
    # 启用 provider-side JSON 强约束. **默认 None 不开启** — 共享 builder 给
    # collection_regen / evaluation / summary worker / graph curation 等非
    # graph extractor 模块复用 (huangzhangshu grep 实证 msg=1c06455d), 在
    # builder 默认开启 JSON 模式会越界影响这些模块. 只有 graph extractor
    # 入口显式传 ``response_format={"type":"json_object"}``.
    svc = CompletionService(
        provider=provider,
        model=invocation.provider_model_id,
        base_url=invocation.base_url,
        api_key=invocation.api_key,
        temperature=0.0,  # deterministic output for extraction
        max_tokens=None,
        caching=False,
        response_format=response_format,
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

Read the TEXT below — it is composed of one or more chunks separated by
``[[chunk_id=<id> index=<n>]]`` boundary markers — and return a single
JSON object with two arrays: ``entities`` and ``relations``. Do not
output anything outside the JSON object.

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
7. Cap: at most {max_entities} entities and {max_relations} relations
   for the whole window. If the text contains more, prefer the most
   specific and the most-mentioned.
8. If the text has no extractable entities, return
   ``{{"entities": [], "relations": []}}``.
9. **Provenance — REQUIRED**: every entity and every relation MUST carry
   ``source_chunk_ids``: a non-empty list of the boundary-marker
   ``chunk_id`` values whose text supports it. Use only ids from this
   allowlist: {allowed_chunk_ids}. The list of allowed ids matches the
   ``chunk_id`` values in the boundary markers above; do not invent
   new ids.
10. **Cross-chunk relations**: when an entity defined in one chunk
    appears in another, prefer to extract a relation linking them. Set
    ``source_chunk_ids`` to every chunk_id whose text contributes
    evidence for the relation (typically the chunks that mention each
    side).
11. **Deduplication & normalisation**: when the same entity appears in
    multiple chunks (same name and type), emit it once with
    ``source_chunk_ids`` covering every supporting chunk. Pick a single
    canonical name across chunks — do not output two entities for an
    obvious surface variant of the same thing.
12. **No fabrication / fail-safe**: if you cannot ground a candidate
    entity or relation in the actual text, drop it. Low-confidence
    candidates may also be skipped. Returning fewer-but-correct beats
    over-eager extraction.

**JSON schema**

```
{{
  "entities": [
    {{"name": "<string>", "entity_type": "<existing or new type string>",
      "description": "<string>",
      "source_chunk_ids": ["<chunk_id>", "..."]}}
  ],
  "relations": [
    {{"source": "<entity name>", "target": "<entity name>",
      "relation_type": "<short relation type>",
      "description": "<string>", "weight": <int 1-10>,
      "source_chunk_ids": ["<chunk_id>", "..."]}}
  ]
}}
```

**Example** (single chunk):

Text:
```
[[chunk_id=c1 index=0]]
Alice, a researcher at Acme Labs, collaborated with Bob on the project.
```

Output:
```
{{
  "entities": [
    {{"name": "Alice", "entity_type": "Person",
      "description": "A researcher at Acme Labs.",
      "source_chunk_ids": ["c1"]}},
    {{"name": "Bob", "entity_type": "Person",
      "description": "A collaborator on the project.",
      "source_chunk_ids": ["c1"]}},
    {{"name": "Acme Labs", "entity_type": "Organization",
      "description": "Research organization where Alice works.",
      "source_chunk_ids": ["c1"]}}
  ],
  "relations": [
    {{"source": "Alice", "target": "Bob",
      "relation_type": "collaborated_with",
      "description": "Alice and Bob collaborated on a project.",
      "weight": 7, "source_chunk_ids": ["c1"]}},
    {{"source": "Alice", "target": "Acme Labs",
      "relation_type": "works_for",
      "description": "Alice is a researcher employed by Acme Labs.",
      "weight": 8, "source_chunk_ids": ["c1"]}}
  ]
}}
```

{few_shot_examples}---

TEXT:
```
{input_text}
```

Output (JSON only):"""


def render_extraction_prompt(
    *,
    window_chunks: "Sequence[Mapping[str, Any]]",
    entity_types: list[str] | tuple[str, ...],
    language: str,
    max_entities: int,
    max_relations: int,
    few_shot_locale: Optional[str] = None,
) -> str:
    """Fill in the extraction prompt template for one extraction window.

    A "window" is one or more consecutive chunks from the same document
    /parse_version that the graph extractor groups together for a
    single LLM call (task #30 §3.1.1). For ``len(window_chunks) == 1``
    the rendered prompt is structurally equivalent to the legacy
    single-chunk shape — there is exactly one boundary marker and the
    LLM still grounds every entity / relation in that single chunk —
    but the v2 prompt now requires ``source_chunk_ids`` on every
    record (task #30 §3.1.3 hard requirement #2). For ``len > 1`` the
    window text carries one ``[[chunk_id=<id> index=<n>]]`` marker per
    chunk and the prompt encourages cross-chunk relations + dedup.

    ``few_shot_locale`` (task #30 §3.1.3 default-off opt-in): when set
    to ``"zh"`` / ``"en"`` / ``"cross_chunk"`` the prompt embeds an
    extra few-shot example to bias the LLM toward the corresponding
    output style. Default ``None`` = no extra example (keeps the
    prompt token budget tight per Bryce Concern 3 / huangheng A2 5th
    co-scale const).

    Kept as a function (rather than inlined) so tests can snapshot the
    exact rendered string and catch accidental prompt regressions.
    """
    from aperag.indexing.entity_types import format_entity_types_for_prompt

    if not window_chunks:
        raise ValueError("render_extraction_prompt: window_chunks must be non-empty")

    chunk_blocks: list[str] = []
    chunk_ids: list[str] = []
    for index, chunk in enumerate(window_chunks):
        raw_id = chunk.get("chunk_id") or chunk.get("id") or ""
        chunk_id = str(raw_id)
        if not chunk_id:
            raise ValueError(f"render_extraction_prompt: window_chunks[{index}] missing chunk_id / id")
        text = str(chunk.get("text") or "")
        chunk_blocks.append(f"[[chunk_id={chunk_id} index={index}]]\n{text}")
        chunk_ids.append(chunk_id)

    input_text = "\n\n".join(chunk_blocks)
    allowed_chunk_ids_repr = ", ".join(repr(cid) for cid in chunk_ids)

    return ENTITY_RELATION_EXTRACTION.format(
        input_text=input_text,
        entity_types=format_entity_types_for_prompt(entity_types),
        language=language,
        max_entities=max_entities,
        max_relations=max_relations,
        allowed_chunk_ids=f"[{allowed_chunk_ids_repr}]",
        few_shot_examples=_few_shot_examples_block(few_shot_locale),
    )


# Few-shot example library (task #30 §3.1.3 default-off opt-in). The
# library is small on purpose — every example adds prompt tokens that
# trade off against the per-window cap (Bryce Concern 3 / huangheng A2
# 5th co-scale const). New locales should be small, focused, and only
# exercised when a real benchmark shows the locale wins.
_FEW_SHOT_EXAMPLES: Dict[str, str] = {
    "zh": """**Example** (Chinese):

Text:
```
[[chunk_id=c1 index=0]]
张伟在阿里云研究院担任高级研究员，与李娜共同负责图谱抽取项目。
```

Output:
```
{{
  "entities": [
    {{"name": "张伟", "entity_type": "Person",
      "description": "阿里云研究院的高级研究员，负责图谱抽取项目。",
      "source_chunk_ids": ["c1"]}},
    {{"name": "李娜", "entity_type": "Person",
      "description": "图谱抽取项目的合作研究员。",
      "source_chunk_ids": ["c1"]}},
    {{"name": "阿里云研究院", "entity_type": "Organization",
      "description": "张伟所在的研究机构。",
      "source_chunk_ids": ["c1"]}}
  ],
  "relations": [
    {{"source": "张伟", "target": "阿里云研究院",
      "relation_type": "works_for",
      "description": "张伟受雇于阿里云研究院。",
      "weight": 8, "source_chunk_ids": ["c1"]}},
    {{"source": "张伟", "target": "李娜",
      "relation_type": "collaborated_with",
      "description": "张伟与李娜共同负责图谱抽取项目。",
      "weight": 7, "source_chunk_ids": ["c1"]}}
  ]
}}
```

""",
    "cross_chunk": """**Example** (cross-chunk relation):

Text:
```
[[chunk_id=c1 index=0]]
Acme Labs is a research organization founded in 2010.

[[chunk_id=c2 index=1]]
Alice, a senior researcher, joined Acme Labs in 2020.
```

Output:
```
{{
  "entities": [
    {{"name": "Acme Labs", "entity_type": "Organization",
      "description": "A research organization founded in 2010.",
      "source_chunk_ids": ["c1", "c2"]}},
    {{"name": "Alice", "entity_type": "Person",
      "description": "A senior researcher who joined Acme Labs in 2020.",
      "source_chunk_ids": ["c2"]}}
  ],
  "relations": [
    {{"source": "Alice", "target": "Acme Labs",
      "relation_type": "works_for",
      "description": "Alice joined Acme Labs as a senior researcher in 2020.",
      "weight": 8, "source_chunk_ids": ["c1", "c2"]}}
  ]
}}
```

""",
}


def _few_shot_examples_block(locale: Optional[str]) -> str:
    """Return the rendered few-shot block for ``locale`` or ``""`` when
    the opt-in is off / the locale is unknown.

    A warning at the call site already covers unknown locales so the
    template-rendering path stays branch-free.
    """
    if not locale:
        return ""
    return _FEW_SHOT_EXAMPLES.get(locale, "")


__all__ = [
    "ENTITY_RELATION_EXTRACTION",
    "LLMCall",
    "build_collection_llm_callable",
    "render_extraction_prompt",
]
