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
LLM-driven entity/relation extractor that calls the collection's
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
  The other chunks still contribute. This matches the prior reference
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

from aperag.indexing.entity_types import (
    merge_entity_type_values,
    normalize_entity_type,
    prompt_language_name,
)
from aperag.indexing.graph import (
    EntityRecord,
    GraphExtractor,
    RelationRecord,
)

logger = logging.getLogger(__name__)


_DEFAULT_LANGUAGE = "zh-CN"
_DEFAULT_MAX_ENTITIES_PER_CHUNK = 32
_DEFAULT_MAX_RELATIONS_PER_CHUNK = 32
_DEFAULT_PER_CHUNK_TIMEOUT_SECONDS = 60.0

# Per-document concurrency cap for LLM extraction calls. Mirrors the
# ``DEFAULT_LLM_CONCURRENCY = 4`` precedent in
# ``aperag/graph_curation/service.py`` so a single graph dispatch never
# bursts more than a small handful of LLM requests at once. Pre-fix
# this loop was strictly serial — a 1 000-chunk document spent 1-3 h
# on extraction. With concurrency 4 the same document finishes in
# ~25 % of the wall time without making the LLM provider's RPM cap
# any tighter than the worker pool already does.
_DEFAULT_EXTRACTOR_LLM_CONCURRENCY = 4

# Number of chunks processed serially before switching to the
# parallel ``asyncio.gather`` path. The serial bootstrap is what
# preserves the Wave 11 dynamic-entity-types feedback loop: each
# chunk's extracted ``entity_type``s propagate into the prompt of
# the next chunk, so brand-new types discovered in chunk[i] are
# available for chunk[i+1]. After this many chunks the active type
# list has typically saturated for the document; the remaining
# chunks run in parallel against the frozen type list. 20 was
# picked per huangheng spec msg=80b01696.
_BOOTSTRAP_CHUNK_COUNT = 20


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

    prompt_language = prompt_language_name(_resolve_language(collection))
    entity_types = tuple(_resolve_entity_types(collection))
    max_entities = _resolve_int_kg_config(collection, "max_entities_per_chunk", _DEFAULT_MAX_ENTITIES_PER_CHUNK)
    max_relations = _resolve_int_kg_config(collection, "max_relations_per_chunk", _DEFAULT_MAX_RELATIONS_PER_CHUNK)
    per_chunk_timeout = _resolve_float_kg_config(
        collection, "per_chunk_timeout_seconds", _DEFAULT_PER_CHUNK_TIMEOUT_SECONDS
    )

    async def _extractor(chunks: Sequence[dict[str, Any]]) -> tuple[list[EntityRecord], list[RelationRecord]]:
        """Run the LLM extractor over every chunk in the dispatch.

        Two-pass design (per huangheng msg=80b01696):

        1. **Bootstrap pass** — process the first
           :data:`_BOOTSTRAP_CHUNK_COUNT` chunks serially so each
           chunk's freshly-discovered entity types feed into the next
           chunk's prompt. This preserves the Wave 11 dynamic-entity-
           types feedback loop where the prompt's allowed-types list
           grows as the document is read.
        2. **Main pass** — run the remaining chunks through
           ``asyncio.gather`` bounded by an
           :class:`asyncio.Semaphore` of width
           :data:`_DEFAULT_EXTRACTOR_LLM_CONCURRENCY`. The active type
           list is frozen at the bootstrap end-state — by 20 chunks it
           has typically saturated for the document, and even if a
           later chunk introduces a brand-new type we accept the small
           recall hit in exchange for ~5x wall-clock speedup.

        Per-chunk failures are isolated in both passes (log + skip);
        one bad chunk never poisons the document's other entities and
        relations. A document with no chunks produces an empty result
        without making any LLM calls.
        """
        if not chunks:
            return ([], [])

        entities: list[EntityRecord] = []
        relations: list[RelationRecord] = []
        active_entity_types = list(entity_types)
        collection_id = getattr(collection, "id", "<unknown>")

        # ---- Pass 1: serial bootstrap (W11 dynamic-types feedback) ----
        bootstrap = chunks[:_BOOTSTRAP_CHUNK_COUNT]
        for chunk in bootstrap:
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
            text = str(chunk.get("text") or "")
            if not text.strip():
                continue
            try:
                ents, rels = await _extract_one_chunk(
                    llm=llm,
                    text=text,
                    chunk_id=chunk_id,
                    entity_types=tuple(active_entity_types),
                    language=prompt_language,
                    max_entities=max_entities,
                    max_relations=max_relations,
                    timeout_seconds=per_chunk_timeout,
                )
            except Exception:  # noqa: BLE001 — per-chunk failure isolation
                logger.exception(
                    "graph extractor: bootstrap LLM call failed for chunk_id=%s in collection=%s; "
                    "skipping chunk's entities/relations",
                    chunk_id,
                    collection_id,
                )
                continue
            entities.extend(ents)
            relations.extend(rels)
            active_entity_types = merge_entity_type_values(
                active_entity_types,
                [entity.entity_type for entity in ents],
            )

        # ---- Pass 2: parallel gather over remaining chunks ----
        remaining = chunks[_BOOTSTRAP_CHUNK_COUNT:]
        if not remaining:
            return entities, relations

        # Snapshot the active types post-bootstrap; the parallel pass
        # uses this frozen list, so concurrent chunks all see the same
        # prompt-side type universe.
        frozen_types = tuple(active_entity_types)
        semaphore = asyncio.Semaphore(_DEFAULT_EXTRACTOR_LLM_CONCURRENCY)

        async def _bounded_extract(
            chunk: Mapping[str, Any],
        ) -> tuple[list[EntityRecord], list[RelationRecord]]:
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
            text = str(chunk.get("text") or "")
            if not text.strip():
                return ([], [])
            async with semaphore:
                try:
                    return await _extract_one_chunk(
                        llm=llm,
                        text=text,
                        chunk_id=chunk_id,
                        entity_types=frozen_types,
                        language=prompt_language,
                        max_entities=max_entities,
                        max_relations=max_relations,
                        timeout_seconds=per_chunk_timeout,
                    )
                except Exception:  # noqa: BLE001 — per-chunk failure isolation
                    logger.exception(
                        "graph extractor: main LLM call failed for chunk_id=%s in collection=%s; "
                        "skipping chunk's entities/relations",
                        chunk_id,
                        collection_id,
                    )
                    return ([], [])

        results = await asyncio.gather(*(_bounded_extract(chunk) for chunk in remaining))
        for ents, rels in results:
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
    timeout_seconds: float,
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
    raw = await asyncio.wait_for(llm(prompt), timeout=timeout_seconds)
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
    # Keep accepting legacy ``type`` for custom prompts, but store the
    # canonical Wave 11 string field as ``entity_type``.
    entity_type = normalize_entity_type(raw.get("entity_type") or raw.get("type") or "")
    description = str(raw.get("description") or "")
    return EntityRecord(
        name=name,
        entity_type=entity_type,
        description=description,
        source_chunk_ids=(chunk_id,) if chunk_id else (),
    )


def _relation_from_dict(raw: Mapping[str, Any], *, chunk_id: str) -> RelationRecord:
    source = str(raw["source"]).strip()
    target = str(raw["target"]).strip()
    if not source or not target:
        raise ValueError("relation source/target cannot be empty")
    rel_type = str(raw.get("relation_type") or raw.get("type") or "")
    description = str(raw.get("description") or "")
    return RelationRecord(
        source=source,
        target=target,
        relation_type=rel_type,
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
        return []
    kg_config: Any = None
    if hasattr(cfg, "knowledge_graph_config"):
        kg_config = cfg.knowledge_graph_config
    elif isinstance(cfg, Mapping):
        kg_config = cfg.get("knowledge_graph_config")
    if kg_config is None:
        return []
    if hasattr(kg_config, "entity_types"):
        types = kg_config.entity_types
    elif isinstance(kg_config, Mapping):
        types = kg_config.get("entity_types")
    else:
        types = None
    if not types:
        return []
    return merge_entity_type_values((), types)


def _resolve_language(collection: Any) -> str:
    cfg = _resolve_config(collection)
    if cfg is None:
        return _DEFAULT_LANGUAGE
    if hasattr(cfg, "language"):
        return str(cfg.language or _DEFAULT_LANGUAGE)
    if isinstance(cfg, Mapping):
        return str(cfg.get("language") or _DEFAULT_LANGUAGE)
    return _DEFAULT_LANGUAGE


def _resolve_kg_config_value(collection: Any, field: str) -> Any:
    """Read ``collection.config.knowledge_graph_config.<field>`` tolerating
    the pydantic-attr / Mapping / JSON-string shapes the DB row may take.
    Returns ``None`` if any layer is missing — callers fall back to their
    default constant."""
    cfg = _resolve_config(collection)
    if cfg is None:
        return None
    kg_config: Any = None
    if hasattr(cfg, "knowledge_graph_config"):
        kg_config = cfg.knowledge_graph_config
    elif isinstance(cfg, Mapping):
        kg_config = cfg.get("knowledge_graph_config")
    if kg_config is None:
        return None
    if hasattr(kg_config, field):
        return getattr(kg_config, field)
    if isinstance(kg_config, Mapping):
        return kg_config.get(field)
    return None


def _resolve_int_kg_config(collection: Any, field: str, default: int) -> int:
    raw = _resolve_kg_config_value(collection, field)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "graph extractor: knowledge_graph_config.%s=%r is not an int; falling back to default %d",
            field,
            raw,
            default,
        )
        return default
    if value <= 0:
        logger.warning(
            "graph extractor: knowledge_graph_config.%s=%d must be positive; falling back to default %d",
            field,
            value,
            default,
        )
        return default
    return value


def _resolve_float_kg_config(collection: Any, field: str, default: float) -> float:
    raw = _resolve_kg_config_value(collection, field)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "graph extractor: knowledge_graph_config.%s=%r is not a float; falling back to default %s",
            field,
            raw,
            default,
        )
        return default
    if value <= 0:
        logger.warning(
            "graph extractor: knowledge_graph_config.%s=%s must be positive; falling back to default %s",
            field,
            value,
            default,
        )
        return default
    return value


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
