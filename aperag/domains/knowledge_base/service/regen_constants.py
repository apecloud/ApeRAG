# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 10 §K.13 — collection auto-description regen constants.

Pinned thresholds + prompts + tool subset for the two-stage regen
pipeline (``collection_regen_service``). Kept in a separate module
so callers (reconciler hook, OpenAPI handlers, unit tests) can
import individual constants without dragging the full service.
"""

from __future__ import annotations

from datetime import timedelta

# ---------------------------------------------------------------------
# Trigger thresholds (per earayu2 msg=1b395cae 中保守 directive)
# ---------------------------------------------------------------------

#: Number of doc-changes since last summary regen that triggers an
#: immediate Stage 1 regen (bulk path).
BULK_THRESHOLD: int = 10

#: Minimum elapsed time since last regen before the deferred
#: small-change path will fire.
DEBOUNCE_WINDOW: timedelta = timedelta(minutes=60)

#: Minimum elapsed time since the most recent doc edit before the
#: deferred path will fire — protects against firing during active
#: editing sessions.
MIN_STALE_AGE: timedelta = timedelta(minutes=10)

# ---------------------------------------------------------------------
# Lease (cluster-level concurrency control via Collection row columns)
# ---------------------------------------------------------------------

#: Lease lifetime; must exceed worst-case Stage 1 latency (60s agent
#: timeout + safety margin). The reconciler reclaims expired leases on
#: its next sweep.
LEASE_TTL: timedelta = timedelta(seconds=900)

# ---------------------------------------------------------------------
# Stage 1 — Summary agent invocation
# ---------------------------------------------------------------------

#: Maximum agent multi-turn budget per regen.
SUMMARY_MAX_TURNS: int = 5

#: Maximum agent token budget per regen (input + output combined,
#: enforced at the agent runtime layer).
SUMMARY_MAX_TOKENS: int = 20000

#: Hard wall-clock timeout for Stage 1.
SUMMARY_TIMEOUT_SECONDS: int = 60

#: Read-only MCP tool subset the summary agent may call. Pinned per
#: huangheng Q1.4 (no write tools — description-gen agent must not
#: mutate collection state). Names match the registered tool ids in
#: ``aperag/mcp/tools/``.
SUMMARY_AGENT_TOOLS: tuple[str, ...] = (
    "list_collections",
    "vector_search",
    "fulltext_search",
    "graph_search",
    "query_graph_entities",
    "expand_graph_subgraph",
    "get_entity_detail",
    "read_document",
    "read_document_section",
    "read_document_outline",
    "read_document_chunk",
    "get_collection_metadata",
    "get_document_metadata",
)

#: Hard-coded Stage 1 system prompt. **Not** user-configurable — the
#: collection's LLM config is reused but the prompt itself is owned
#: by this service so deployments cannot break the regen contract by
#: editing prompts (per huangheng Q1.2).
SUMMARY_AGENT_SYSTEM_PROMPT: str = """\
你是 ApeRAG 的 collection summary 生成 agent。

任务: 用提供的 read-only 工具自由探索 collection 内容, 然后输出一段详细丰富的 summary。

可用工具 (read-only): list_collections, vector_search, fulltext_search, graph_search,
query_graph_entities, expand_graph_subgraph, get_entity_detail, read_document,
read_document_section, read_document_outline, read_document_chunk,
get_collection_metadata, get_document_metadata。**只读, 不修改**。

约束:
- 最多探索 5 轮 (multi-turn limit)
- 总 token 预算 20000
- 只输出最终 summary 文本 (no preamble, no markdown headings, plain prose)
- 客观描述内容, 不假设用户场景, 不夸大效果

输出语言 (根据探索到的 doc 主要语言决定):
- 中文 collection: 输出中文 summary, 长度 5000-10000 字, 详细介绍 collection 包含哪些主题、覆盖哪些场景、对哪些用户有价值
- 英文 collection: 输出英文 summary, 长度 3000-7000 词
- 混合: 用主语言, 末尾用一行附次要语言 mention

可探索的方向 (建议而非强制):
- 通过 list_collections 拿到当前 collection metadata (title / 创建时间 / doc 数)
- 通过 graph_search / query_graph_entities 探索知识图谱主要 entity / relation
- 通过 vector_search / fulltext_search 找代表性 doc 主题
- 通过 read_document_section / read_document_outline 深读 1-3 个代表性 doc
"""

# ---------------------------------------------------------------------
# Stage 1 Tier 2 — chunks.jsonl fallback (LLM-only, no agent multi-turn)
# ---------------------------------------------------------------------

#: Cap on total characters concatenated from chunks before truncation.
#: Sized so the resulting prompt fits comfortably under typical 32k /
#: 128k context windows after the system prompt + instructions.
CHUNKS_FALLBACK_MAX_CHARS: int = 24000

#: Maximum number of documents to sample chunks from. Sampling is
#: round-robin so a 1000-doc collection still gets diverse coverage.
CHUNKS_FALLBACK_MAX_DOCUMENTS: int = 12

#: Tier 2 prompt — single-shot summary from a stitched chunk corpus.
#: Mirror the agent prompt's voice + length contract so the FE never
#: sees a Tier-1 vs Tier-2 difference in tone or structure.
CHUNKS_FALLBACK_PROMPT_ZH: str = """\
你是 ApeRAG 的 collection summary 生成助手 (Tier 2 — agent 不可用时降级路径)。

下面是 collection ``{collection_title}`` 的代表性文档片段拼接结果。请根据这些片段
写出一段详细丰富的 summary。

要求:
- 长度: 5000-10000 字
- 客观介绍 collection 包含哪些主题、覆盖哪些场景、对哪些用户有价值
- 不假设用户场景, 不夸大效果
- 仅输出 summary 文本, 无 preamble, 无 markdown headings, plain prose

文档片段:
{chunks_text}
"""

CHUNKS_FALLBACK_PROMPT_EN: str = """\
You are ApeRAG's collection summary assistant (Tier 2 — fallback path used
when the agent path is unavailable).

Below are representative chunks stitched from documents in collection
``{collection_title}``. Write a detailed summary based on these chunks.

Requirements:
- Length: 3000-7000 words
- Objective coverage of themes, scenarios, and user value
- No speculation about user scenarios; no exaggerated claims
- Output summary text only, no preamble, no markdown headings, plain prose

Chunks:
{chunks_text}
"""

# ---------------------------------------------------------------------
# Stage 2 — Description derive
# ---------------------------------------------------------------------

#: Stage 2 derive prompt template. ``{summary}`` and ``{language}`` are
#: ``str.format`` placeholders.
DESCRIPTION_DERIVE_PROMPT_ZH: str = """\
请把以下 collection summary 浓缩为一段简短的 description (描述), 用于 UI 短列表展示。

要求:
- 长度: 200-500 字
- 抓住 collection 的核心主题 + 主要价值
- 客观描述, 不夸大
- 输出中文
- 只输出 description 文本, 无 preamble

Summary:
{summary}
"""

DESCRIPTION_DERIVE_PROMPT_EN: str = """\
Condense the following collection summary into a short description for UI list views.

Requirements:
- Length: 100-300 words
- Capture the core themes and primary value
- Objective tone, no exaggeration
- Output in English
- Description text only, no preamble

Summary:
{summary}
"""

#: Stage 2 timeout (single LLM call, much shorter than Stage 1).
DESCRIPTION_TIMEOUT_SECONDS: int = 30

# ---------------------------------------------------------------------
# Quality gate thresholds (per huangheng N4)
# ---------------------------------------------------------------------

#: Minimum total characters for a summary to be considered valid.
SUMMARY_MIN_CHARS: int = 200

#: Minimum total characters for a description to be considered valid.
DESCRIPTION_MIN_CHARS: int = 50

#: Substring blacklist — outputs containing any of these are rejected
#: as agent/LLM error responses.
INVALID_OUTPUT_FRAGMENTS: tuple[str, ...] = (
    "tool error",
    "无法生成",
    "请重试",
    "i cannot",
    "i can't",
    "i am unable",
)
