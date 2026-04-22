# Copyright 2026 ApeCloud, Inc.
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

"""LLM prompts for graph curation."""

from __future__ import annotations

import json
from typing import Any

MERGE_ADJUDICATION = """\
You are reviewing two knowledge-graph entities from the same collection.

Decide whether they refer to the same real-world entity.

Rules:

1. Output JSON only.
2. Be conservative. If the evidence is weak, answer `same_entity=false`.
3. `recommended_target_entity_id` must be one of the two provided ids
   when `same_entity=true`; otherwise return `null`.
4. Do not invent a new entity id or a new canonical node.
5. Use the deterministic signals as hints, not as truth.

JSON schema:
{{
  "same_entity": <true|false>,
  "confidence": <number between 0 and 1>,
  "reason": "<short explanation>",
  "recommended_target_entity_id": "<left id | right id | null>"
}}

Entity A:
{left_json}

Entity B:
{right_json}

Deterministic signals:
{signals_json}

Output JSON only:
"""


def render_merge_adjudication_prompt(
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    signals: dict[str, Any],
) -> str:
    return MERGE_ADJUDICATION.format(
        left_json=json.dumps(left, ensure_ascii=False, indent=2, sort_keys=True),
        right_json=json.dumps(right, ensure_ascii=False, indent=2, sort_keys=True),
        signals_json=json.dumps(signals, ensure_ascii=False, indent=2, sort_keys=True),
    )


__all__ = ["render_merge_adjudication_prompt"]
