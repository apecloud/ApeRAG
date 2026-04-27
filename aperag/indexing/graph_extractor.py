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

"""Real LLM-driven graph extractor — Wave 4 T1.

Replaces the chunk 4b ``_no_op_extractor`` placeholder with an actual
LightRAG-style entity/relation extractor that calls the collection's
configured completion model for each chunk and parses the JSON output
into the new ``aperag.indexing.graph.EntityRecord`` /
``RelationRecord`` shapes.

The extractor is built per-collection so the LLM callable is closure-
bound to the collection's completion config (provider / model / api
key). The factory function returns an async closure with the
``GraphExtractor`` Protocol shape — the same shape
:class:`GraphModalityWorker` already accepts.

Failure semantics (per Wave 3 lesson #10 ship-incomplete-but-don't-
silently-lie):

* No completion model configured for the collection → factory raises
  :class:`aperag.indexing.worker_factory.WorkerFactoryError` so the
  graph row finalises FAILED with operator-facing diagnostics. Without
  this, the extractor would build but every dispatch would emit a
  cryptic LLM-side error.
* Per-chunk LLM call failures (malformed JSON, transient backend
  errors, etc.) log a warning and skip the chunk's entities/relations.
  The other chunks still contribute. This matches the legacy LightRAG
  extractor's failure semantics — one bad chunk does not poison the
  whole document.

Wave 5 follow-up (per architect msg=87e2b187 chunk 4d Option C
ruling): the legacy ``aperag/domains/knowledge_graph/graphindex/``
package is slated for elimination. The current implementation
imports ``render_extraction_prompt`` from the legacy ``prompts``
module + ``build_collection_llm_callable`` from the legacy
``integration`` module — these dependencies are intentional bridge
points and will be relocated to ``aperag/indexing/llm.py`` when the
Wave 5 cross-cutting refactor lands.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable, Mapping, Sequence

from aperag.indexing.graph import (
    EntityRecord,
    GraphExtractor,
    RelationRecord,
)

logger = logging.getLogger(__name__)


_DEFAULT_ENTITY_TYPES: tuple[str, ...] = (
    "organization",
    "person",
    "geo",
    "event",
    "product",
    "technology",
    "date",
    "category",
)
_DEFAULT_LANGUAGE = "en-US"
_DEFAULT_MAX_ENTITIES_PER_CHUNK = 32
_DEFAULT_MAX_RELATIONS_PER_CHUNK = 32
_DEFAULT_PER_CHUNK_TIMEOUT_SECONDS = 60.0


def build_collection_graph_extractor(collection: Any) -> GraphExtractor:
    """Construct a :class:`GraphExtractor` closure bound to
    ``collection``'s completion model.

    Read by :func:`aperag.indexing.worker_factory._build_graph_worker`
    when constructing a graph modality worker for an indexing dispatch.
    The closure is async and matches the
    ``Sequence[dict] -> Awaitable[(entities, relations)]`` shape
    :class:`aperag.indexing.graph.GraphModalityWorker` expects.

    Raises :class:`aperag.indexing.worker_factory.WorkerFactoryError`
    if the collection has no completion model configured (no LLM
    callable can be built) — the orchestrator finalises the graph row
    FAILED with the message rather than silently building a worker
    that will fail at dispatch time.
    """
    from aperag.indexing.llm import build_collection_llm_callable
    from aperag.indexing.worker_factory import WorkerFactoryError

    try:
        llm = build_collection_llm_callable(collection)
    except Exception as exc:  # noqa: BLE001 — wrap for orchestrator
        raise WorkerFactoryError(
            f"graph extractor: completion model not configured for collection "
            f"{getattr(collection, 'id', '<unknown>')}: {exc!r}; "
            f"set collection.config.enable_knowledge_graph=false or configure "
            f"the collection's completion model"
        ) from exc

    entity_types = tuple(_resolve_entity_types(collection))
    language = _resolve_language(collection)
    max_entities = _DEFAULT_MAX_ENTITIES_PER_CHUNK
    max_relations = _DEFAULT_MAX_RELATIONS_PER_CHUNK

    async def _extractor(chunks: Sequence[dict[str, Any]]) -> tuple[list[EntityRecord], list[RelationRecord]]:
        """Run the LLM extractor over every chunk in the dispatch."""
        if not chunks:
            return ([], [])

        entities: list[EntityRecord] = []
        relations: list[RelationRecord] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
            text = str(chunk.get("text") or "")
            if not text.strip():
                continue
            try:
                ents, rels = await _extract_one_chunk(
                    llm=llm,
                    text=text,
                    chunk_id=chunk_id,
                    entity_types=entity_types,
                    language=language,
                    max_entities=max_entities,
                    max_relations=max_relations,
                )
            except Exception:  # noqa: BLE001 — per-chunk failure isolation
                logger.exception(
                    "graph extractor: LLM call failed for chunk_id=%s in collection=%s; "
                    "skipping chunk's entities/relations",
                    chunk_id,
                    getattr(collection, "id", "<unknown>"),
                )
                continue
            entities.extend(ents)
            relations.extend(rels)
        return entities, relations

    return _extractor


# ---------------------------------------------------------------------
# Per-chunk extraction.
# ---------------------------------------------------------------------


async def _extract_one_chunk(
    *,
    llm: Callable[[str], Awaitable[str]],
    text: str,
    chunk_id: str,
    entity_types: tuple[str, ...],
    language: str,
    max_entities: int,
    max_relations: int,
) -> tuple[list[EntityRecord], list[RelationRecord]]:
    """Single-chunk extraction: render the prompt, call the LLM, parse
    the JSON response, return record lists.

    Wraps the LLM call in :func:`asyncio.wait_for` with the per-chunk
    timeout so a stuck LLM does not block the worker forever; on
    timeout we propagate :class:`asyncio.TimeoutError` to the caller
    which already logs + skips the chunk.
    """
    from aperag.indexing.llm import render_extraction_prompt

    prompt = render_extraction_prompt(
        input_text=text,
        entity_types=list(entity_types),
        language=language,
        max_entities=max_entities,
        max_relations=max_relations,
    )
    raw = await asyncio.wait_for(llm(prompt), timeout=_DEFAULT_PER_CHUNK_TIMEOUT_SECONDS)
    return _parse_extraction_response(raw=raw, chunk_id=chunk_id)


def _parse_extraction_response(
    *,
    raw: str,
    chunk_id: str,
) -> tuple[list[EntityRecord], list[RelationRecord]]:
    """Parse the LLM's JSON response into entity / relation records.

    The prompt asks for strict JSON with ``entities`` + ``relations``
    arrays; we accept either a fenced `````json ... ````` block or a bare JSON
    object so deployments that strip code-fences in their LLM
    middleware still work. Malformed payloads return ``([], [])``;
    individual records that fail to parse are logged + skipped so a
    single bad row does not drop the rest.
    """
    payload = _strip_code_fence(raw)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning(
            "graph extractor: chunk_id=%s response is not valid JSON; skipping entities/relations from this chunk",
            chunk_id,
        )
        return ([], [])

    if not isinstance(parsed, Mapping):
        logger.warning(
            "graph extractor: chunk_id=%s response is JSON but not an object; got %s",
            chunk_id,
            type(parsed).__name__,
        )
        return ([], [])

    entities: list[EntityRecord] = []
    for raw_entity in parsed.get("entities", []) or []:
        if not isinstance(raw_entity, Mapping):
            continue
        try:
            entities.append(_entity_from_dict(raw_entity, chunk_id=chunk_id))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "graph extractor: chunk_id=%s skipping malformed entity %r: %s",
                chunk_id,
                raw_entity,
                exc,
            )

    relations: list[RelationRecord] = []
    for raw_relation in parsed.get("relations", []) or []:
        if not isinstance(raw_relation, Mapping):
            continue
        try:
            relations.append(_relation_from_dict(raw_relation, chunk_id=chunk_id))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "graph extractor: chunk_id=%s skipping malformed relation %r: %s",
                chunk_id,
                raw_relation,
                exc,
            )

    return entities, relations


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n(.*)\n```\s*$", re.DOTALL)


def _strip_code_fence(raw: str) -> str:
    """Remove a wrapping markdown code fence if present so
    ``json.loads`` sees the inner JSON directly. Idempotent on bare
    JSON."""
    match = _FENCE_RE.match(raw)
    if match:
        return match.group(1)
    return raw.strip()


def _entity_from_dict(raw: Mapping[str, Any], *, chunk_id: str) -> EntityRecord:
    name = str(raw["name"]).strip()
    if not name:
        raise ValueError("entity name cannot be empty")
    entity_type = str(raw.get("type") or "")
    description = str(raw.get("description") or "")
    return EntityRecord(
        name=name,
        type=entity_type,
        description=description,
        source_chunk_ids=(chunk_id,) if chunk_id else (),
    )


def _relation_from_dict(raw: Mapping[str, Any], *, chunk_id: str) -> RelationRecord:
    source = str(raw["source"]).strip()
    target = str(raw["target"]).strip()
    if not source or not target:
        raise ValueError("relation source/target cannot be empty")
    rel_type = str(raw.get("type") or "")
    description = str(raw.get("description") or "")
    return RelationRecord(
        source=source,
        target=target,
        type=rel_type,
        description=description,
        source_chunk_ids=(chunk_id,) if chunk_id else (),
    )


# ---------------------------------------------------------------------
# Collection config readers — tolerant of the dict / pydantic-attr /
# JSON-string shapes ``Collection.config`` may take in the DB.
# ---------------------------------------------------------------------


def _resolve_entity_types(collection: Any) -> Sequence[str]:
    cfg = _resolve_config(collection)
    if cfg is None:
        return _DEFAULT_ENTITY_TYPES
    kg_config: Any = None
    if hasattr(cfg, "knowledge_graph_config"):
        kg_config = cfg.knowledge_graph_config
    elif isinstance(cfg, Mapping):
        kg_config = cfg.get("knowledge_graph_config")
    if kg_config is None:
        return _DEFAULT_ENTITY_TYPES
    if hasattr(kg_config, "entity_types"):
        types = kg_config.entity_types
    elif isinstance(kg_config, Mapping):
        types = kg_config.get("entity_types")
    else:
        types = None
    if not types:
        return _DEFAULT_ENTITY_TYPES
    return [str(t) for t in types]


def _resolve_language(collection: Any) -> str:
    cfg = _resolve_config(collection)
    if cfg is None:
        return _DEFAULT_LANGUAGE
    if hasattr(cfg, "language"):
        return str(cfg.language or _DEFAULT_LANGUAGE)
    if isinstance(cfg, Mapping):
        return str(cfg.get("language") or _DEFAULT_LANGUAGE)
    return _DEFAULT_LANGUAGE


def _resolve_config(collection: Any) -> Any:
    cfg = getattr(collection, "config", None)
    if cfg is None:
        return None
    if isinstance(cfg, str):
        try:
            return json.loads(cfg)
        except (TypeError, ValueError):
            return None
    return cfg


__all__ = ["build_collection_graph_extractor"]
