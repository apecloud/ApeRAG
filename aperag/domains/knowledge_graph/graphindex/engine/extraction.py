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

"""Per-chunk entity / relation extraction via a single LLM call.

Input:  one ``Chunk`` + the config's entity types / language / caps.
Output: a list of ``Entity`` and ``Relation`` DTOs.

The extraction is **stateless per chunk**. That's a deliberate
constraint: it lets the indexer parallelise chunks freely and retry
single chunks on LLM failure without rebuilding global state. Entity
deduplication happens later in ``PostgresGraphStore.upsert_entities``,
not here.

Robustness notes:

* LLM provider guarantees ``response_format={"type":"json_object"}``
  syntactic validity, but not semantic correctness. We still validate
  with ``json.loads`` + light shape checks; anything broken returns
  ``([], [])`` so a single bad chunk doesn't fail the whole document.
* Empty responses (``{"entities": [], "relations": []}``) are valid —
  e.g. boilerplate chunks like page footers.
* Self-referencing relations or relations pointing at unknown entities
  are dropped silently (after logging) — the extraction prompt forbids
  both, but LLMs occasionally ignore that.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from typing import Any, Callable, Optional

from aperag.domains.knowledge_graph.graphindex.config import GraphIndexConfig
from aperag.domains.knowledge_graph.graphindex.dto import Chunk, Entity, Relation
from aperag.domains.knowledge_graph.graphindex.prompts import render_extraction_prompt

logger = logging.getLogger(__name__)

# Type of the LLM callable we accept. Kept narrow so callers can pass
# both the real ``CompletionService.agenerate`` and a simple test
# stub without subclassing anything.
LLMCall = Callable[[str], "Any"]  # async callable returning JSON-ish text


async def extract_from_chunk(
    *,
    chunk: Chunk,
    config: GraphIndexConfig,
    llm: LLMCall,
) -> tuple[list[Entity], list[Relation]]:
    """Run one LLM call to extract entities + relations for ``chunk``.

    Never raises for LLM-side problems (malformed JSON, unknown
    entity types, …) — all failure modes log a warning and return
    ``([], [])``. Raises only if ``llm`` itself raises in a non-recoverable
    way, and only after ``config.llm_max_retries`` attempts.
    """
    prompt = render_extraction_prompt(
        input_text=chunk.text,
        entity_types=config.entity_types,
        language=config.extraction_language,
        max_entities=config.max_entities_per_chunk,
        max_relations=config.max_relations_per_chunk,
    )

    raw: Optional[str] = None
    last_exc: Optional[BaseException] = None
    # Exponential backoff with jitter: 0s on first attempt, then
    # 0.5s, 1.0s, 2.0s, 4.0s (+ up to ~20% jitter). Upper-bounded at
    # 10s so the worst case stays well under a typical Celery task
    # soft-timeout. Without this we would hammer a 429-throttled or
    # mid-outage LLM endpoint through the full retry budget inside a
    # few milliseconds.
    max_attempts = config.llm_max_retries + 1
    for attempt in range(max_attempts):
        if attempt > 0:
            delay = min(10.0, 0.5 * (2 ** (attempt - 1)))
            delay *= 1.0 + random.random() * 0.2
            await asyncio.sleep(delay)
        try:
            raw = await llm(prompt)
            if isinstance(raw, str) and raw.strip():
                break
            logger.warning(
                "graphindex.extract: empty LLM response on chunk %s (attempt %d/%d)",
                chunk.chunk_id,
                attempt + 1,
                max_attempts,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "graphindex.extract: LLM error on chunk %s (attempt %d/%d): %s",
                chunk.chunk_id,
                attempt + 1,
                max_attempts,
                exc,
            )
    if not raw:
        if last_exc is not None:
            # Propagate so the indexer can decide whether to fail the
            # whole document or skip the chunk.
            raise last_exc
        return [], []

    parsed = _safe_json_parse(raw)
    if parsed is None:
        logger.warning(
            "graphindex.extract: non-JSON LLM output on chunk %s; dropping",
            chunk.chunk_id,
        )
        return [], []

    entities = _entities_from_parsed(parsed, chunk, config)
    relations = _relations_from_parsed(parsed, chunk, config, entities)
    return entities, relations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_json_parse(raw: str) -> Optional[dict]:
    r"""Best-effort JSON parse. Tolerates leading/trailing non-JSON noise
    (some LLMs prefix ``Output:`` or fence with ``\`\`\`json``)."""
    s = raw.strip()
    # Strip markdown fences if present.
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    # Find the first '{' and last '}' — everything outside is scaffolding.
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        obj = json.loads(s[first : last + 1])
        if isinstance(obj, dict):
            return obj
        return None
    except json.JSONDecodeError:
        return None


def normalize_entity_id(collection_id: str, name: str) -> str:
    """Stable, deterministic entity id.

    Uses a hash of ``(collection_id, lowercased_trimmed_name)`` so the
    same entity name written in multiple chunks collapses to a single
    row at upsert time. 12-char prefix of SHA-1 (48 bits) is ample for
    per-collection entity counts at expected ApeRAG scale; the
    birthday-collision threshold sits in the low-tens-of-millions
    range per collection, well above any realistic knowledge-graph
    we've seen. If per-collection entity counts ever approach 10M,
    widen this to 16 chars and run a collision audit.
    """
    key = f"{collection_id}::{name.strip().lower()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _entities_from_parsed(parsed: dict, chunk: Chunk, config: GraphIndexConfig) -> list[Entity]:
    raw_ents = parsed.get("entities") or []
    if not isinstance(raw_ents, list):
        return []

    allowed_types = set(config.entity_types)
    out: list[Entity] = []
    seen_names: set[str] = set()
    for item in raw_ents[: config.max_entities_per_chunk]:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        etype = (item.get("type") or "").strip().lower()
        description = (item.get("description") or "").strip()
        if not name or not etype:
            continue
        if etype not in allowed_types:
            # The prompt explicitly forbids inventing types; ignore
            # rather than guess what the LLM meant.
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        out.append(
            Entity(
                entity_id=normalize_entity_id(chunk.collection_id, name),
                collection_id=chunk.collection_id,
                name=name,
                type=etype,
                description=description,
                source_chunk_ids=(chunk.chunk_id,),
            )
        )
    return out


def _relations_from_parsed(
    parsed: dict,
    chunk: Chunk,
    config: GraphIndexConfig,
    entities: list[Entity],
) -> list[Relation]:
    raw_rels = parsed.get("relations") or []
    if not isinstance(raw_rels, list):
        return []

    name_to_id = {e.name: e.entity_id for e in entities}
    out: list[Relation] = []
    seen_pairs: set[tuple[str, str]] = set()
    for item in raw_rels[: config.max_relations_per_chunk]:
        if not isinstance(item, dict):
            continue
        src = (item.get("source") or "").strip()
        tgt = (item.get("target") or "").strip()
        desc = (item.get("description") or "").strip()
        weight = item.get("weight", 5)
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 5.0
        weight = max(1.0, min(10.0, weight))

        src_id = name_to_id.get(src)
        tgt_id = name_to_id.get(tgt)
        if not src_id or not tgt_id or src_id == tgt_id:
            continue
        pair = (src_id, tgt_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        out.append(
            Relation(
                collection_id=chunk.collection_id,
                source_id=src_id,
                target_id=tgt_id,
                description=desc,
                weight=weight,
                source_chunk_ids=(chunk.chunk_id,),
            )
        )
    return out


__all__ = ["extract_from_chunk", "normalize_entity_id", "LLMCall"]
