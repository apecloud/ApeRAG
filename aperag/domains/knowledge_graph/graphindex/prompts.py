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

"""LLM prompts for graphindex v2.

Only one prompt template matters: ``ENTITY_RELATION_EXTRACTION``. It
asks the model for a **single JSON object** containing ``entities`` and
``relations`` arrays — no tuple-delimited parsing games, no multi-round
gleaning, no "keywords" side-output.

Why a single prompt:

* Fewer moving parts = fewer failure modes.
* Modern LLM providers all support ``response_format={"type":"json_object"}``,
  which guarantees syntactically valid JSON; parsing becomes a one-liner
  ``json.loads``.
* One prompt per task is easier to A/B test and swap out than the
  multi-stage extraction pipeline LightRAG v1 used.

The few-shot example is chosen for clarity, not coverage. If quality
regresses on a real dataset, tune the example rather than multiplying
prompts — the extraction pipeline treats the prompt as a single black box.
"""

from __future__ import annotations

# Wave 5 P1 chunk 1 (`aperag/indexing/llm.py` relocate per §K.9.1 item 3):
# ``ENTITY_RELATION_EXTRACTION`` + ``render_extraction_prompt`` 已 relocate
# 到 :mod:`aperag.indexing.llm`. 这里 re-export 保持 legacy callers 跨
# Wave 5 deprecation window 不破。一旦 retrieval/curation Phase 1
# close-out，整个 legacy module 删除。
from aperag.indexing.llm import ENTITY_RELATION_EXTRACTION, render_extraction_prompt  # noqa: F401

DESCRIPTION_SUMMARIZATION: str = """\
You are consolidating a knowledge-graph entry.

Below are {fragment_count} short descriptions of the same
{subject_kind}, extracted from different chunks of a document. Your job
is to produce ONE coherent summary in {language} that preserves every
fact mentioned.

**Rules**

1. Output ONLY the summary text. No preamble, no JSON, no markdown.
2. Target length: around {target_chars} characters (one or two
   paragraphs). Go longer if the fragments genuinely require it; never
   omit a fact to hit the target.
3. Merge duplicate facts into one sentence. Keep conflicting facts
   both — the downstream pipeline will flag contradictions, so do not
   silently pick a winner.
4. Do not add information that isn't in the fragments.
5. For a relation, keep the subject–predicate–object framing clear
   ("<source> <verb> <target> because ...").

**Subject**: {subject_label}

**Fragments** (separated by "---"):

---
{fragments_block}
---

Summary (plain text, {language}):"""


def render_summarization_prompt(
    *,
    subject_kind: str,
    subject_label: str,
    fragments: list[str] | tuple[str, ...],
    language: str,
    target_chars: int,
) -> str:
    """Render the description-summarization prompt.

    ``subject_kind`` is "entity" or "relation"; ``subject_label`` is the
    entity name or ``"<source> → <target>"`` for a relation. The
    fragments are inlined verbatim, separated by ``---`` lines so the
    model can tell them apart.
    """
    block = "\n---\n".join(f.strip() for f in fragments if f and f.strip())
    return DESCRIPTION_SUMMARIZATION.format(
        subject_kind=subject_kind,
        subject_label=subject_label,
        fragments_block=block,
        fragment_count=len([f for f in fragments if f and f.strip()]),
        language=language,
        target_chars=target_chars,
    )


KEYWORD_EXTRACTION: str = """\
---Role---

You are a helpful assistant tasked with identifying both high-level and
low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords.
High-level keywords focus on overarching concepts or themes.
Low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format (it will be parsed by a JSON parser,
  do not add any extra content in the output).
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

---Example---

Query: "How does international trade influence global economic stability?"

Output:
{{"high_level_keywords": ["international trade", "global economic stability", "economic impact"],
  "low_level_keywords": ["trade agreements", "tariffs", "currency exchange", "imports", "exports"]}}

---Real Data---

Query: {query}

Output (JSON only, keep the same language as the query):"""


def render_keyword_extraction_prompt(*, query: str) -> str:
    """Fill in the keyword extraction prompt template."""
    return KEYWORD_EXTRACTION.format(query=query)


__all__ = [
    "ENTITY_RELATION_EXTRACTION",
    "DESCRIPTION_SUMMARIZATION",
    "KEYWORD_EXTRACTION",
    "render_extraction_prompt",
    "render_summarization_prompt",
    "render_keyword_extraction_prompt",
]
