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

"""Per-process configuration for the graphindex module.

Design intent: **injected, not globally read**. The module's business
code must never call ``os.environ.get(...)`` directly — the app boot
path constructs one ``GraphIndexConfig`` instance from env and passes
it into the service. Tests construct their own config without touching
env, which keeps them hermetic.

This file deliberately does NOT expose a "global" GraphIndexConfig
singleton. If you need one, wire it at the service factory layer
(``service.py``), not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class GraphIndexConfig:
    """Everything the graphindex module needs to operate.

    Two groupings:
    * **ingest-time**: chunk shape + extraction LLM + entity types;
    * **query-time**: retrieval defaults (top_k, hop depth).

    Storage / embedding identity lives elsewhere — the service receives
    ``store: GraphStore`` and ``embedding_service`` directly via its
    constructor.
    """

    # ---- chunking ------------------------------------------------------
    chunk_token_size: int = 1200
    """Target tokens per chunk when splitting a document."""

    chunk_overlap_token_size: int = 100
    """Overlap size to keep context between adjacent chunks."""

    # ---- extraction ----------------------------------------------------
    entity_types: Sequence[str] = field(
        default_factory=lambda: (
            "organization",
            "person",
            "geo",
            "event",
            "product",
            "technology",
            "date",
            "category",
        )
    )
    """Closed-set entity type vocabulary the LLM is asked to tag with."""

    extraction_language: str = "en"
    """ISO-like language hint passed into the extraction prompt. Output
    prompts adapt tone/case accordingly but the JSON structure is stable."""

    max_chunks_per_batch: int = 8
    """How many chunks go into one extraction LLM request. Keep this low
    enough that the model's context window isn't strained and one hiccup
    doesn't blow up a whole document."""

    llm_max_retries: int = 2
    """Retry budget per-chunk-batch on malformed JSON / API failure."""

    llm_timeout_seconds: int = 180
    """Soft ceiling on a single extraction LLM call."""

    # ---- query ---------------------------------------------------------
    default_query_top_k: int = 10
    """Fallback ``top_k`` for ``query_context`` when the caller omits it."""

    default_query_max_hop: int = 1
    """How far to walk from anchor entities when assembling context. 1 hop
    is usually enough for RAG; 2 dramatically inflates result size."""

    # ---- safety --------------------------------------------------------
    max_entities_per_chunk: int = 40
    """Cap per extraction call so a rogue LLM response can't flood the
    graph with thousands of "entities"."""

    max_relations_per_chunk: int = 80
    """Same principle, for relations."""

    # ---- normalization / summarization --------------------------------
    summarize_at_fragments: int = 6
    """Trigger an LLM summary of an entity or relation's ``description``
    once the accumulated text has reached this many fragments (joined by
    ``\\n\\n``). LightRAG's analogue was ``force_llm_summary_on_merge``
    with default 10; we pick 6 because a high-frequency entity with 6
    supporting mentions already reads as a patchwork paragraph, and
    early summarization keeps retrieval prompts compact. Set to a very
    large number to effectively disable description summarization (not
    recommended — see ``max_description_chars`` fallback)."""

    max_description_chars: int = 4000
    """Hard cap on the stored ``description`` column. Two layers of
    protection stack:

    1. The write path normally calls ``summarize_at_fragments`` long
       before descriptions reach this size — the cap is just a safety
       rail against runaway growth (e.g. 100+ mentions of a single
       entity inside one big document).
    2. If the LLM summarizer is not wired in (``llm=None``), fails,
       or times out, the service falls back to truncating at a word
       boundary with a short "… [truncated]" marker. Truncation loses
       information; summarization is always preferred when available.

    Keep this well above ``summary_target_chars`` so a fresh summary
    never re-triggers truncation on the same write."""

    summary_target_chars: int = 800
    """Target length for LLM-produced summaries. The prompt asks the
    model to produce ~this many characters of coherent text covering
    the facts in every fragment, so after summarization a description
    sits well under ``max_description_chars`` and the fragment count
    resets to 1."""

    def __post_init__(self) -> None:
        if self.chunk_token_size <= 0:
            raise ValueError("chunk_token_size must be positive")
        if self.chunk_overlap_token_size < 0:
            raise ValueError("chunk_overlap_token_size must be >= 0")
        if self.chunk_overlap_token_size >= self.chunk_token_size:
            raise ValueError(
                "chunk_overlap_token_size must be < chunk_token_size; otherwise chunks would not make progress"
            )
        if not self.entity_types:
            raise ValueError("entity_types must contain at least one type")
        if self.summarize_at_fragments < 2:
            raise ValueError("summarize_at_fragments must be at least 2")
        if self.max_description_chars <= 0:
            raise ValueError("max_description_chars must be positive")
        if self.summary_target_chars <= 0:
            raise ValueError("summary_target_chars must be positive")
        if self.summary_target_chars >= self.max_description_chars:
            raise ValueError(
                "summary_target_chars must be less than max_description_chars; "
                "otherwise a fresh summary could re-trigger truncation"
            )
        # Normalize entity_types to a tuple for hash-safety on frozen dc.
        object.__setattr__(self, "entity_types", tuple(self.entity_types))


__all__ = ["GraphIndexConfig"]
