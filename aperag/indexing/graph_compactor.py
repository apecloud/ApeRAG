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

"""Graph index description compactor — Wave 7 task #2.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §K.12.6:
when an entity / relation accumulates many ``description_parts`` from
multiple documents, the joined text grows past what the embedder can
ingest cleanly and what the graph-DB column can hold. This component
compacts the joined parts into a single bounded summary that the
``GraphModalityWorker.sync()`` task #3 writes back into the new
``compacted_description`` field (task #1) before vector embedding.

Algorithm (mirrors legacy ``graphindex/service.py`` lines 158 / 456 /
500 with renamed surface and tighter knobs):

1. Join the parts with a paragraph separator.
2. If neither the fragment count nor the total character length crosses
   the threshold, return ``None`` (caller leaves the description as-is).
3. Otherwise call the LLM with a summarization prompt.
4. If the LLM call fails or returns empty, fall back to a hard
   character cap with a ``" … [truncated]"`` marker so descriptions
   never exceed the column budget even when the LLM is unreachable.

The component is a pure compute helper: no coupling to
``LineageGraphStore`` or any vector backend. It takes a list of
``str`` and returns either a compacted ``str`` or ``None``.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


LLMCall = Callable[[str], Awaitable[str]]


# Threshold knobs are module-level constants per architect spec
# (``msg=fc4b18ee``): not exposed via operator config so the
# private-deploy "免维护" guardrail holds.

MAX_DESCRIPTION_CHARS: int = 8000
"""Upper bound for the persisted compacted description.

Crossing this triggers compaction; the LLM target is well under this
so the post-compact text stays within the cap. The fallback truncator
also cuts at this length."""

SUMMARIZE_AT_FRAGMENTS: int = 8
"""When the joined ``parts`` list grows to this many fragments, compact
even if the total character count is still below the char cap. Catches
"high-frequency entity with many short mentions" cases."""

TARGET_SUMMARY_CHARS: int = 3000
"""Target length for the LLM summary prompt. The model is instructed to
land near this number; we still verify the response is at most
``MAX_DESCRIPTION_CHARS`` before accepting it."""

FALLBACK_TRUNCATE_MARK: str = " … [truncated]"
"""Marker appended to a fallback-truncated description so operators can
spot rows that hit the LLM-unavailable path."""

PART_SEPARATOR: str = "\n\n"


_SUMMARIZE_PROMPT_TEMPLATE: str = (
    "请将以下知识图谱节点 / 关系的多段描述压缩为一段连贯的中文总结，"
    "目标长度约 {target} 字以内。要求：\n"
    "1. 保留所有事实信息（实体、属性、时间、来源等），不要遗漏关键细节。\n"
    "2. 去重重复表述，合并语义相近的句子。\n"
    "3. 不要编造原文未出现的内容。\n"
    "4. 直接输出总结正文，不要加任何前言、标题或解释。\n\n"
    "原始描述片段：\n"
    "{joined}\n"
)


class GraphIndexCompactor:
    """Compact long description-part lists for one entity / relation.

    Caller usage (task #3 ``GraphModalityWorker.sync()``)::

        compactor = GraphIndexCompactor(llm)
        compacted = await compactor.compact_if_oversized(entity.parts_text)
        if compacted is not None:
            await store.set_compacted_description(entity_id, compacted)

    The component is stateless across calls; one instance per worker
    invocation is fine. ``llm`` is the same async callable shape used
    by the rest of the indexing pipeline (``aperag.indexing.llm.LLMCall``)
    so it can be wired from ``build_collection_llm_callable`` directly.
    """

    def __init__(self, llm: LLMCall) -> None:
        self._llm = llm

    async def compact_if_oversized(self, parts: list[str]) -> str | None:
        """Return a compacted description, or ``None`` if compaction is
        not warranted.

        Empty / whitespace-only parts are dropped before evaluation —
        an empty ``parts`` list returns ``None`` (nothing to compact).
        """
        cleaned = [p for p in parts if p and p.strip()]
        if not cleaned:
            return None

        joined = PART_SEPARATOR.join(cleaned)

        if len(cleaned) < SUMMARIZE_AT_FRAGMENTS and len(joined) < MAX_DESCRIPTION_CHARS:
            return None

        summary = await self._summarize_via_llm(joined)
        if summary is not None:
            return summary

        return self._fallback_truncate(joined)

    async def _summarize_via_llm(self, joined: str) -> str | None:
        """Call the LLM to produce a compacted summary.

        Returns ``None`` if the call fails, returns empty text, or the
        response exceeds the hard char cap (so ``compact_if_oversized``
        falls through to the truncator instead of writing an oversize
        row)."""
        prompt = _SUMMARIZE_PROMPT_TEMPLATE.format(
            target=TARGET_SUMMARY_CHARS,
            joined=joined,
        )
        try:
            raw = await self._llm(prompt)
        except Exception:
            logger.exception("graph_compactor: LLM summarization failed; falling back to truncation")
            return None

        summary = (raw or "").strip()
        if not summary:
            return None
        if len(summary) > MAX_DESCRIPTION_CHARS:
            logger.warning(
                "graph_compactor: LLM returned %d chars > cap %d; falling back to truncation",
                len(summary),
                MAX_DESCRIPTION_CHARS,
            )
            return None
        return summary

    @staticmethod
    def _fallback_truncate(joined: str) -> str:
        """Hard-cap the joined text at ``MAX_DESCRIPTION_CHARS`` and
        append the truncation marker.

        Prefers a word boundary in the last 64 characters of the budget
        so we don't cut mid-token; falls back to a hard slice if no
        whitespace is found in that window."""
        cap = MAX_DESCRIPTION_CHARS
        if len(joined) <= cap:
            return joined

        budget = cap - len(FALLBACK_TRUNCATE_MARK)
        if budget <= 0:
            return joined[:cap]

        window = joined[:budget]
        cut = window.rfind(" ", max(0, len(window) - 64))
        if cut <= 0:
            cut = len(window)
        return window[:cut].rstrip() + FALLBACK_TRUNCATE_MARK


__all__ = [
    "GraphIndexCompactor",
    "LLMCall",
    "MAX_DESCRIPTION_CHARS",
    "SUMMARIZE_AT_FRAGMENTS",
    "TARGET_SUMMARY_CHARS",
    "FALLBACK_TRUNCATE_MARK",
]
