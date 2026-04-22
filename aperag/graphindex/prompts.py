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

ENTITY_RELATION_EXTRACTION: str = """\
You are an information-extraction assistant building a knowledge graph.

Read the TEXT below and return a single JSON object with two arrays:
``entities`` and ``relations``. Do not output anything outside the JSON
object.

**Rules**

1. Output language: {language}. Names keep their original script and
   case (e.g. English names capitalized, Chinese names unchanged).
2. Use only these entity types: {entity_types}. If no provided type
   fits, skip the entity rather than invent a new type.
3. Every entity needs a short, self-contained description in
   {language}. Do not add information that isn't in the text.
4. Every relation must reference entities by the exact ``name`` you put
   in the ``entities`` list. No self-loops (source != target).
5. ``weight`` is an integer 1-10 expressing how strongly the text
   supports the relation; default to 5 when unsure.
6. Cap: at most {max_entities} entities and {max_relations} relations.
   If the text contains more, prefer the most specific and the
   most-mentioned.
7. If the text has no extractable entities, return
   ``{{"entities": [], "relations": []}}``.

**JSON schema**

```
{{
  "entities": [
    {{"name": "<string>", "type": "<one of the allowed types>",
      "description": "<string>"}}
  ],
  "relations": [
    {{"source": "<entity name>", "target": "<entity name>",
      "description": "<string>", "weight": <int 1-10>}}
  ]
}}
```

**Example** (English, types=[person, organization]):

Text:
```
Alice, a researcher at Acme Labs, collaborated with Bob on the project.
```

Output:
```
{{
  "entities": [
    {{"name": "Alice", "type": "person",
      "description": "A researcher at Acme Labs."}},
    {{"name": "Bob", "type": "person",
      "description": "A collaborator on the project."}},
    {{"name": "Acme Labs", "type": "organization",
      "description": "Research organization where Alice works."}}
  ],
  "relations": [
    {{"source": "Alice", "target": "Bob",
      "description": "Alice and Bob collaborated on a project.",
      "weight": 7}},
    {{"source": "Alice", "target": "Acme Labs",
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
    """Fill in the extraction prompt template for one chunk batch.

    Kept as a simple function so tests can snapshot the exact rendered
    string and catch accidental prompt regressions.
    """
    return ENTITY_RELATION_EXTRACTION.format(
        input_text=input_text,
        entity_types=", ".join(entity_types),
        language=language,
        max_entities=max_entities,
        max_relations=max_relations,
    )


__all__ = ["ENTITY_RELATION_EXTRACTION", "render_extraction_prompt"]
