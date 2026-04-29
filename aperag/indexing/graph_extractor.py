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
from dataclasses import dataclass
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
_DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE = 1

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

# Maximum number of chunks dispatched per ``asyncio.gather`` batch
# in the main pass. Pre-fix the main pass called
# ``asyncio.gather(*[N coroutines])`` over the entire post-bootstrap
# remainder, so a 2 395-chunk document scheduled 2 375 simultaneous
# task objects on the event loop. Even with a Semaphore(4) limiting
# concurrent execution, the event loop still tracked the 2 375 task
# references and pumped them through the scheduler — at scale this
# wedged the BE process during a real user upload (Harry Potter txt,
# observed 2026-04-28). Bounded gather batches cap the coroutine
# fan-out per round so memory + scheduler pressure stays flat
# regardless of document size. 50 was picked per architect ratify
# msg=cec8b206 — produces ~48 batches for the 2 395-chunk doc and
# overlaps cleanly with the 4-way semaphore.
_MAIN_PASS_BATCH_SIZE = 50


@dataclass(frozen=True)
class _GraphChunkWindow:
    """Contiguous chunk group consumed by one graph extraction call.

    A1 deliberately keeps this as an internal graph-extractor shape:
    parser / vector / fulltext still consume the original chunk records.
    Prompt v2 and provenance validation are owned by A3; this class only
    centralises the window boundary contract for that later work.
    """

    chunks: tuple[Mapping[str, Any], ...]
    chunk_ids: tuple[str, ...]
    text: str

    @property
    def primary_chunk_id(self) -> str:
        """Legacy provenance fallback used until A3 parses source_chunk_ids."""
        return next((chunk_id for chunk_id in self.chunk_ids if chunk_id), "")


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
        # issue #1861 (task #14): graph extractor 是 builder 共享 caller 集合
        # 里唯一确定输出 JSON-only 的 caller, 显式开启 provider-side 强约束
        # ``response_format={"type":"json_object"}``. 共享 builder 默认仍是
        # None, 不影响 collection_regen / evaluation / summary worker /
        # graph_curation 等 prose-output caller.
        llm = build_collection_llm_callable(
            collection,
            response_format={"type": "json_object"},
        )
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
    graph_window_size = _resolve_int_kg_config(
        collection,
        "graph_extraction_window_size",
        _DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE,
    )
    graph_max_window_tokens = _resolve_optional_int_kg_config(collection, "graph_extraction_max_window_tokens")

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
           ``asyncio.gather`` in fixed-size batches of
           :data:`_MAIN_PASS_BATCH_SIZE`, bounded inside each batch
           by an :class:`asyncio.Semaphore` of width
           :data:`_DEFAULT_EXTRACTOR_LLM_CONCURRENCY`. The batched
           form bounds coroutine fan-out per round so a 2 000+ chunk
           document does not schedule thousands of simultaneous task
           references onto the event loop (which previously wedged
           the BE process at ~2 395 chunks). The active type list is
           frozen at the bootstrap end-state — by 20 chunks it has
           typically saturated for the document, and even if a later
           chunk introduces a brand-new type we accept the small
           recall hit in exchange for ~5x wall-clock speedup.

        Per-chunk failures are isolated in both passes (log + skip);
        one bad chunk never poisons the document's other entities and
        relations. A document with no chunks produces an empty result
        without making any LLM calls.
        """
        if not chunks:
            return ([], [])
        windows = _build_graph_chunk_windows(
            chunks,
            window_size=graph_window_size,
            max_window_tokens=graph_max_window_tokens,
        )
        if not windows:
            return ([], [])

        entities: list[EntityRecord] = []
        relations: list[RelationRecord] = []
        active_entity_types = list(entity_types)
        collection_id = getattr(collection, "id", "<unknown>")

        # ---- Pass 1: serial bootstrap (W11 dynamic-types feedback) ----
        bootstrap = windows[:_BOOTSTRAP_CHUNK_COUNT]
        for window in bootstrap:
            if not window.text.strip():
                continue
            try:
                ents, rels = await _extract_one_window(
                    llm=llm,
                    window=window,
                    entity_types=tuple(active_entity_types),
                    language=prompt_language,
                    max_entities=max_entities,
                    max_relations=max_relations,
                    timeout_seconds=per_chunk_timeout,
                )
            except Exception:  # noqa: BLE001 — per-chunk failure isolation
                logger.exception(
                    "graph extractor: bootstrap LLM call failed for window_chunk_ids=%s in collection=%s; "
                    "skipping window's entities/relations",
                    window.chunk_ids,
                    collection_id,
                )
                continue
            entities.extend(ents)
            relations.extend(rels)
            active_entity_types = merge_entity_type_values(
                active_entity_types,
                [entity.entity_type for entity in ents],
            )

        # ---- Pass 2: parallel gather over remaining chunks (chunked) ----
        remaining = windows[_BOOTSTRAP_CHUNK_COUNT:]
        if not remaining:
            return entities, relations

        # Snapshot the active types post-bootstrap; the parallel pass
        # uses this frozen list, so concurrent chunks all see the same
        # prompt-side type universe.
        frozen_types = tuple(active_entity_types)
        semaphore = asyncio.Semaphore(_DEFAULT_EXTRACTOR_LLM_CONCURRENCY)

        async def _bounded_extract(
            window: _GraphChunkWindow,
        ) -> tuple[list[EntityRecord], list[RelationRecord]]:
            if not window.text.strip():
                return ([], [])
            async with semaphore:
                try:
                    return await _extract_one_window(
                        llm=llm,
                        window=window,
                        entity_types=frozen_types,
                        language=prompt_language,
                        max_entities=max_entities,
                        max_relations=max_relations,
                        timeout_seconds=per_chunk_timeout,
                    )
                except Exception:  # noqa: BLE001 — per-chunk failure isolation
                    logger.exception(
                        "graph extractor: main LLM call failed for window_chunk_ids=%s in collection=%s; "
                        "skipping window's entities/relations",
                        window.chunk_ids,
                        collection_id,
                    )
                    return ([], [])

        # Process the remainder in fixed-size gather batches so the
        # event loop never holds more than ``_MAIN_PASS_BATCH_SIZE``
        # task references at once. The Semaphore still caps actual
        # concurrent LLM calls within each batch; the outer batching
        # is only there to bound coroutine fan-out.
        for batch_start in range(0, len(remaining), _MAIN_PASS_BATCH_SIZE):
            batch = remaining[batch_start : batch_start + _MAIN_PASS_BATCH_SIZE]
            batch_results = await asyncio.gather(*(_bounded_extract(chunk) for chunk in batch))
            for ents, rels in batch_results:
                entities.extend(ents)
                relations.extend(rels)

        return entities, relations

    return _extractor


def _build_graph_chunk_windows(
    chunks: Sequence[Mapping[str, Any]],
    *,
    window_size: int,
    max_window_tokens: int | None = None,
) -> list[_GraphChunkWindow]:
    """Group contiguous graph chunks into non-overlap extraction windows.

    Hard boundaries:
    - input order is preserved;
    - a window contains at most ``window_size`` chunks;
    - when present on adjacent chunks, ``document_id`` and ``parse_version``
      must match;
    - when present on adjacent chunks, section metadata must match;
    - an optional approximate token cap starts a new window before overflow.

    ``window_size=1`` returns one window per input chunk, preserving the
    old extractor's bootstrap/main-pass structure.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    token_cap = max_window_tokens if max_window_tokens and max_window_tokens > 0 else None
    windows: list[_GraphChunkWindow] = []
    current: list[Mapping[str, Any]] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = _estimate_graph_chunk_tokens(chunk)
        if current and (
            len(current) >= window_size
            or _graph_window_boundary_break(current[-1], chunk)
            or (token_cap is not None and current_tokens + chunk_tokens > token_cap)
        ):
            windows.append(_make_graph_chunk_window(current))
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += chunk_tokens

    if current:
        windows.append(_make_graph_chunk_window(current))
    return windows


def _make_graph_chunk_window(chunks: Sequence[Mapping[str, Any]]) -> _GraphChunkWindow:
    chunk_tuple = tuple(chunks)
    return _GraphChunkWindow(
        chunks=chunk_tuple,
        chunk_ids=tuple(_chunk_id(chunk) for chunk in chunk_tuple),
        text="\n\n".join(str(chunk.get("text") or "") for chunk in chunk_tuple if str(chunk.get("text") or "").strip()),
    )


def _chunk_id(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or "")


def _estimate_graph_chunk_tokens(chunk: Mapping[str, Any]) -> int:
    """Cheap deterministic token estimate for the A1 window cap.

    The parser's v2 fallback splitter is character based, while older
    document parsing paths may be tokenizer based. A1 only needs a stable
    upper-bound guard for grouping; A2 owns the final model prompt budget.
    """
    text = str(chunk.get("text") or "")
    if not text:
        return 0
    # CJK-heavy text tends toward one token per character; using
    # character count is conservative for the cap and dependency-free.
    return len(text)


def _chunk_metadata_value(chunk: Mapping[str, Any], key: str) -> Any:
    if key in chunk:
        return chunk.get(key)
    metadata = chunk.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    return None


def _graph_window_boundary_break(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    for key in ("document_id", "parse_version"):
        previous_value = _chunk_metadata_value(previous, key)
        current_value = _chunk_metadata_value(current, key)
        if previous_value is not None and current_value is not None and str(previous_value) != str(current_value):
            return True

    previous_section = _chunk_metadata_value(previous, "section_path") or _chunk_metadata_value(
        previous, "heading_anchor"
    )
    current_section = _chunk_metadata_value(current, "section_path") or _chunk_metadata_value(current, "heading_anchor")
    return bool(previous_section and current_section and str(previous_section) != str(current_section))


# ---------------------------------------------------------------------
# Per-chunk extraction.
# ---------------------------------------------------------------------


async def _extract_one_window(
    *,
    llm: Callable[[str], Awaitable[str]],
    window: _GraphChunkWindow,
    entity_types: tuple[str, ...],
    language: str,
    max_entities: int,
    max_relations: int,
    timeout_seconds: float,
    few_shot_locale: str | None = None,
) -> tuple[list[EntityRecord], list[RelationRecord]]:
    """Window extraction: render the v2 prompt over the window's chunks,
    call the LLM, parse the JSON response, return record lists.

    A1 introduced the :class:`_GraphChunkWindow` shape and dispatched
    every window through this entrypoint with the legacy single-text
    bridge. A3 replaces that bridge with boundary-marked prompt text +
    a parser that validates the LLM's ``source_chunk_ids`` against the
    window allowlist (spec §3.1.3 + §6.2):

    - ``window_size == 1`` — the prompt emits a single
      ``[[chunk_id=...]]`` block and the parser falls back to the
      lone allowed id when the LLM omits ``source_chunk_ids``,
      preserving the structural-equivalence contract that A1 already
      proves with its 35 integration tests.
    - ``window_size > 1`` — the prompt emits one boundary marker per
      chunk and the parser requires the LLM to populate
      ``source_chunk_ids`` against the allowlist; missing or
      out-of-allowlist ids are skipped + warned so a hallucinated
      chunk_id never poisons provenance.

    Wraps the LLM call in :func:`asyncio.wait_for` with the per-window
    timeout so a stuck LLM does not block the worker forever; on
    timeout we propagate :class:`asyncio.TimeoutError` to the caller
    which already logs + skips the window.
    """
    from aperag.indexing.llm import render_extraction_prompt

    if not window.chunks:
        return ([], [])

    allowed_chunk_ids = tuple(window.chunk_ids)
    if any(not cid for cid in allowed_chunk_ids):
        # Defensive: A1's _build_graph_chunk_windows already drops
        # chunks with no chunk_id, so this should be unreachable. If
        # it still fires (someone hand-builds a window in tests) we
        # cannot ground provenance — drop the window.
        logger.warning(
            "graph extractor: window contains a chunk with empty chunk_id; skipping window (window_size=%d)",
            len(window.chunks),
        )
        return ([], [])

    prompt = render_extraction_prompt(
        window_chunks=window.chunks,
        entity_types=list(entity_types),
        language=language,
        max_entities=max_entities,
        max_relations=max_relations,
        few_shot_locale=few_shot_locale,
    )
    raw = await asyncio.wait_for(llm(prompt), timeout=timeout_seconds)
    return _parse_extraction_response(
        raw=raw,
        allowed_chunk_ids=allowed_chunk_ids,
    )


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
    """Backward-compatible single-chunk wrapper around
    :func:`_extract_one_window`.

    Existing callers (and a few legacy tests) still pass a single
    ``text`` + ``chunk_id`` pair. We wrap them in a synthetic
    ``_GraphChunkWindow`` so the v2 prompt + parser invariant runs
    through one entrypoint.
    """
    window = _make_graph_chunk_window([{"chunk_id": chunk_id, "text": text}])
    return await _extract_one_window(
        llm=llm,
        window=window,
        entity_types=entity_types,
        language=language,
        max_entities=max_entities,
        max_relations=max_relations,
        timeout_seconds=timeout_seconds,
    )


# Cap on raw-response prefix written to the empty-result warn log —
# enough to recognise a model's error / empty payload pattern at a
# glance without bloating the log file. Bounded exactly the way Wave
# 7 task #11 narrative logs are bounded.
_EMPTY_RESULT_LOG_RAW_PREFIX_CHARS: int = 500


def _log_empty_extraction_if_applicable(
    *,
    raw: str,
    allowed_chunk_ids: tuple[str, ...],
    entities_count: int,
    relations_count: int,
) -> None:
    """Emit a single ``warning`` line when the LLM produced valid JSON
    that parsed cleanly but contained zero entities AND zero relations.

    Why this exists: Bryce 2026-04-29 incident (msg=358bb68f) — a
    user switched a collection's completion model to one whose
    instruction-following degraded against ApeRAG's extraction prompt
    (DeepSeek V4 Flash via OpenRouter returned ``{"entities":[],
    "relations":[]}`` for every chunk). The graph index status went
    ACTIVE per Wave 3 §F.1 semantics (no error → success), but the
    knowledge graph was silently empty for every doc indexed after
    the model swap. Diagnosing took several hours of grep + cross-
    reference because nothing in the logs flagged "extractor ran but
    produced nothing for THIS many chunks".

    Per-chunk warn level (not error) so it does not page on-call;
    each line carries the raw response prefix so an ops scan
    immediately reveals the model-prompt incompatibility pattern. If
    the same pattern recurs (every chunk in a doc empties out),
    grep ``graph extractor: empty extraction result`` returns N
    matches and the model swap timeline lines up with the failure
    window.
    """
    if entities_count > 0 or relations_count > 0:
        return
    raw_prefix = raw[:_EMPTY_RESULT_LOG_RAW_PREFIX_CHARS]
    logger.warning(
        "graph extractor: empty extraction result for chunk_ids=%s "
        "(LLM response parsed cleanly but produced 0 entities + 0 relations); "
        "raw response prefix (%d chars max): %r",
        list(allowed_chunk_ids),
        _EMPTY_RESULT_LOG_RAW_PREFIX_CHARS,
        raw_prefix,
    )


def _parse_extraction_response(
    *,
    raw: str,
    allowed_chunk_ids: tuple[str, ...],
) -> tuple[list[EntityRecord], list[RelationRecord]]:
    """Parse the LLM's JSON response into entity / relation records.

    The prompt asks for strict JSON with ``entities`` + ``relations``
    arrays; we accept either a fenced `````json ... ````` block or a bare JSON
    object so deployments that strip code-fences in their LLM
    middleware still work. Malformed payloads return ``([], [])``;
    individual records that fail to parse are logged + skipped so a
    single bad row does not drop the rest.

    Provenance invariant (task #30 §3.1.3 hard requirement #2 + parser
    invariant agreed in #indexing优化:f0614ea1): every record's
    ``source_chunk_ids`` must be a non-empty subset of the window's
    allowlist. ``window_size == 1`` is the legacy compat path — when
    the LLM omits ``source_chunk_ids`` we fall back to the single
    allowed id. ``window_size > 1`` requires the LLM to populate the
    field; missing / out-of-allowlist records are skipped + warned.
    """
    payload = _strip_code_fence(raw)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning(
            "graph extractor: chunk_ids=%s response is not valid JSON; skipping entities/relations from this window",
            list(allowed_chunk_ids),
        )
        return ([], [])

    if not isinstance(parsed, Mapping):
        logger.warning(
            "graph extractor: chunk_ids=%s response is JSON but not an object; got %s",
            list(allowed_chunk_ids),
            type(parsed).__name__,
        )
        return ([], [])

    allowed_set = frozenset(allowed_chunk_ids)
    fallback_chunk_id = allowed_chunk_ids[0] if len(allowed_chunk_ids) == 1 else None

    entities: list[EntityRecord] = []
    for raw_entity in parsed.get("entities", []) or []:
        if not isinstance(raw_entity, Mapping):
            continue
        try:
            entities.append(
                _entity_from_dict(
                    raw_entity,
                    allowed_chunk_ids=allowed_set,
                    fallback_chunk_id=fallback_chunk_id,
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "graph extractor: chunk_ids=%s skipping malformed entity %r: %s",
                list(allowed_chunk_ids),
                raw_entity,
                exc,
            )

    relations: list[RelationRecord] = []
    for raw_relation in parsed.get("relations", []) or []:
        if not isinstance(raw_relation, Mapping):
            continue
        try:
            relations.append(
                _relation_from_dict(
                    raw_relation,
                    allowed_chunk_ids=allowed_set,
                    fallback_chunk_id=fallback_chunk_id,
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "graph extractor: chunk_ids=%s skipping malformed relation %r: %s",
                list(allowed_chunk_ids),
                raw_relation,
                exc,
            )

    _log_empty_extraction_if_applicable(
        raw=raw,
        allowed_chunk_ids=allowed_chunk_ids,
        entities_count=len(entities),
        relations_count=len(relations),
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


def _resolve_source_chunk_ids(
    raw: Mapping[str, Any],
    *,
    allowed_chunk_ids: frozenset[str],
    fallback_chunk_id: str | None,
) -> tuple[str, ...]:
    """Read + validate ``source_chunk_ids`` against the window allowlist.

    Single-chunk window (``fallback_chunk_id`` is set) — missing field
    falls back to the single allowed id (legacy compat).

    Multi-chunk window (``fallback_chunk_id is None``) — the field
    must be a non-empty list of allowlisted strings; otherwise we
    raise ``ValueError`` so the caller skips + warns the record.

    Out-of-allowlist ids are silently dropped before the empty-check
    so an LLM that hallucinates a chunk_id does not poison provenance.
    """
    raw_ids = raw.get("source_chunk_ids")
    if raw_ids is None:
        if fallback_chunk_id is None:
            raise ValueError("source_chunk_ids is required for multi-chunk windows; the LLM omitted the field")
        return (fallback_chunk_id,)

    if not isinstance(raw_ids, (list, tuple)):
        raise ValueError(f"source_chunk_ids must be a list, got {type(raw_ids).__name__}")

    cleaned: list[str] = []
    seen: set[str] = set()
    for entry in raw_ids:
        cid = str(entry).strip()
        if not cid or cid not in allowed_chunk_ids or cid in seen:
            continue
        cleaned.append(cid)
        seen.add(cid)
    if not cleaned:
        if fallback_chunk_id is not None:
            return (fallback_chunk_id,)
        raise ValueError("source_chunk_ids has no values inside the window allowlist")
    return tuple(cleaned)


def _entity_from_dict(
    raw: Mapping[str, Any],
    *,
    allowed_chunk_ids: frozenset[str],
    fallback_chunk_id: str | None,
) -> EntityRecord:
    name = str(raw["name"]).strip()
    if not name:
        raise ValueError("entity name cannot be empty")
    # Keep accepting legacy ``type`` for custom prompts, but store the
    # canonical Wave 11 string field as ``entity_type``.
    entity_type = normalize_entity_type(raw.get("entity_type") or raw.get("type") or "")
    description = str(raw.get("description") or "")
    source_chunk_ids = _resolve_source_chunk_ids(
        raw,
        allowed_chunk_ids=allowed_chunk_ids,
        fallback_chunk_id=fallback_chunk_id,
    )
    return EntityRecord(
        name=name,
        entity_type=entity_type,
        description=description,
        source_chunk_ids=source_chunk_ids,
    )


def _relation_from_dict(
    raw: Mapping[str, Any],
    *,
    allowed_chunk_ids: frozenset[str],
    fallback_chunk_id: str | None,
) -> RelationRecord:
    source = str(raw["source"]).strip()
    target = str(raw["target"]).strip()
    if not source or not target:
        raise ValueError("relation source/target cannot be empty")
    rel_type = str(raw.get("relation_type") or raw.get("type") or "")
    description = str(raw.get("description") or "")
    source_chunk_ids = _resolve_source_chunk_ids(
        raw,
        allowed_chunk_ids=allowed_chunk_ids,
        fallback_chunk_id=fallback_chunk_id,
    )
    return RelationRecord(
        source=source,
        target=target,
        relation_type=rel_type,
        description=description,
        source_chunk_ids=source_chunk_ids,
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


def _resolve_optional_int_kg_config(collection: Any, field: str) -> int | None:
    raw = _resolve_kg_config_value(collection, field)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "graph extractor: knowledge_graph_config.%s=%r is not an int; ignoring override",
            field,
            raw,
        )
        return None
    if value <= 0:
        logger.warning(
            "graph extractor: knowledge_graph_config.%s=%d must be positive; ignoring override",
            field,
            value,
        )
        return None
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
