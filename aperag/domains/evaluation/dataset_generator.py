# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""AI auto-generate QA pairs for an evaluation dataset (preview path).

Architect lock: ``#evaluation msg=05c3ec83`` + ``msg=a9fb7efd``.

The flow is **preview-only** — no DB write here. The caller (typically
the FE) takes the returned ``GeneratedDatasetItem`` list, lets the user
review/edit/select, then POSTs the chosen rows to the existing
``/items`` append endpoint. That keeps low-quality LLM output out of the
dataset by default and matches the simple-stable "user controls
quality" guardrail.

Per-chunk shape (Weston msg=87124ce5 + msg=def9094f):

* Walk the collection's serving chunks.jsonl artifacts and pick
  ``count`` substantive chunks (>= 200 chars), favouring diversity
  by round-robin across documents.
* For each chunk fire one LLM call (parallelised under a small
  semaphore) asking for a single ``{question, expected_answer}`` JSON
  object in the resolved language.
* Each delivered item carries ``reference_context = chunk.text``
  truncated to :data:`_REFERENCE_CONTEXT_MAX_CHARS` (8 000) so the
  provenance is deterministic — Phase 3 LLM-as-judge can lift it
  straight off the dataset row instead of needing a backfill pass.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from sqlalchemy import and_, select

from aperag.config import get_async_session
from aperag.domains.evaluation.schemas import GeneratedDatasetItem
from aperag.domains.knowledge_base.db.models import Collection, Document

logger = logging.getLogger(__name__)


# Cap on the per-row provenance blob. Weston msg=87124ce5 — keep
# ``reference_context`` bounded so a 50 000-char chunk does not bloat
# every dataset row to the point that the eval surface is unusable.
_REFERENCE_CONTEXT_MAX_CHARS: int = 8000

# Generated questions can be over-eager LLMs. We over-sample chunks by
# this multiplier so we have material left after dedup / parse failures
# kick out duds, and still hit the requested ``count``.
_CHUNK_OVERSAMPLE: int = 2

# Bound concurrent LLM calls per request — mirror the graph-extractor
# semaphore (#1809) so a count=100 request does not stampede the model
# provider.
_LLM_CONCURRENCY: int = 4

# Per-chunk LLM timeout — keep below typical FE wait threshold so the
# operator gets feedback rather than a blank spinner.
_PER_CHUNK_LLM_TIMEOUT_SECONDS: float = 60.0

# Substantive-chunk threshold — chunks shorter than this are a poor
# basis for a QA pair (boilerplate / boilerplate-only headers). Mirror
# the Wave 10 chunks-fallback heuristic so two paths stay consistent.
_SUBSTANTIVE_CHUNK_MIN_CHARS: int = 200


_SUPPORTED_LANGUAGES: tuple[str, ...] = ("zh-CN", "en-US", "ja-JP", "ko-KR")
_DEFAULT_LANGUAGE: str = "zh-CN"


def _resolve_language(*, request_language: str | None, collection: Collection) -> str:
    if request_language and request_language in _SUPPORTED_LANGUAGES:
        return request_language
    # Fall back to the collection's configured language (Wave 10 §K.13
    # collection.config.language) before resorting to the product
    # default. earayu2 msg=e3498dc0: "默认跟随知识库语言".
    try:
        from aperag.schema.utils import parseCollectionConfig

        parsed = parseCollectionConfig(collection.config)
        if parsed and parsed.language and parsed.language in _SUPPORTED_LANGUAGES:
            return str(parsed.language)
    except Exception:  # noqa: BLE001 — config malformed → fall through
        logger.exception(
            "dataset_generator: parseCollectionConfig failed for %s — using default language",
            collection.id,
        )
    return _DEFAULT_LANGUAGE


_PROMPT_TEMPLATES: dict[str, str] = {
    "zh-CN": (
        "你是一名信息抽取助手，专门为知识库自动生成问答对，用于后续模型评测。\n"
        "请阅读下面的【内容】，输出**一对**严格 JSON：\n"
        '{{"question": "<问题文本>", "expected_answer": "<准确答案>"}}\n\n'
        "约束：\n"
        "1. 输出语言：中文。专有名词、法规名称、机构名等可以保留原文。\n"
        "2. 问题应能直接根据【内容】回答，不要凭空推断。\n"
        "3. 答案应当精炼（1-3 句），并完全可以从【内容】里得到支撑。\n"
        "4. 不要输出 JSON 以外的任何文字，不要使用 Markdown 代码块包裹。\n\n"
        "【内容】\n{chunk_text}\n\n"
        "JSON："
    ),
    "en-US": (
        "You are an information-extraction assistant. Generate exactly **one** "
        "QA pair from the CONTENT below for downstream evaluation.\n"
        "Output strict JSON with this shape (no Markdown fences, no extra text):\n"
        '{{"question": "<question text>", "expected_answer": "<concise answer>"}}\n\n'
        "Rules:\n"
        "1. Output language: English. Proper nouns may stay in their source script.\n"
        "2. The question must be directly answerable from CONTENT.\n"
        "3. The answer must be 1-3 sentences and fully supported by CONTENT.\n\n"
        "CONTENT:\n{chunk_text}\n\n"
        "JSON:"
    ),
    "ja-JP": (
        "あなたは情報抽出アシスタントです。下記の【コンテンツ】から評価用 QA ペアを"
        "**1 組**生成してください。\n"
        "次の形式で **JSON のみ** を出力してください（コードブロック禁止）：\n"
        '{{"question": "<質問>", "expected_answer": "<簡潔な回答>"}}\n\n'
        "制約：\n"
        "1. 出力言語：日本語。固有名詞は原語のままで構いません。\n"
        "2. 質問は【コンテンツ】から直接回答できる内容に限定してください。\n"
        "3. 回答は 1〜3 文の簡潔な文章で、完全に【コンテンツ】に裏付けられること。\n\n"
        "【コンテンツ】\n{chunk_text}\n\n"
        "JSON："
    ),
    "ko-KR": (
        "당신은 정보 추출 도우미입니다. 아래 【콘텐츠】를 바탕으로 평가용 QA 쌍을 "
        "**한 개**만 생성하세요.\n"
        "다음 형식의 **JSON만** 출력하세요(코드 블록 금지):\n"
        '{{"question": "<질문>", "expected_answer": "<간결한 답변>"}}\n\n'
        "규칙:\n"
        "1. 출력 언어: 한국어. 고유 명사는 원어를 유지해도 됩니다.\n"
        "2. 질문은 【콘텐츠】에서 바로 답할 수 있어야 합니다.\n"
        "3. 답변은 1-3 문장으로 간결하게, 【콘텐츠】에 근거해 작성하세요.\n\n"
        "【콘텐츠】\n{chunk_text}\n\n"
        "JSON:"
    ),
}


def _format_prompt(*, language: str, chunk_text: str, override: str | None) -> str:
    if override:
        # Caller-provided template — accept ``{chunk_text}`` and (optionally)
        # ``{language}`` placeholders. Fail closed on unsupported placeholders
        # so a typo does not produce a silently-empty prompt.
        try:
            return override.format(chunk_text=chunk_text, language=language)
        except (KeyError, IndexError):
            logger.warning(
                "dataset_generator: prompt_template override has unknown placeholders; "
                "using default template for language=%s",
                language,
            )
    template = _PROMPT_TEMPLATES.get(language) or _PROMPT_TEMPLATES[_DEFAULT_LANGUAGE]
    return template.format(chunk_text=chunk_text)


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(raw: str) -> str:
    match = _FENCE_RE.match(raw)
    return match.group(1).strip() if match else raw.strip()


def _parse_qa_pair(raw: str) -> tuple[str | None, str | None]:
    """Coerce the LLM's response into ``(question, expected_answer)`` or
    ``(None, None)`` if the payload is unusable."""
    payload = _strip_code_fence(raw)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return (None, None)
    if not isinstance(parsed, dict):
        return (None, None)
    question = parsed.get("question") or parsed.get("Question")
    answer = parsed.get("expected_answer") or parsed.get("answer") or parsed.get("Answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        return (None, None)
    question = question.strip()
    answer = answer.strip()
    if not question or not answer:
        return (None, None)
    return (question, answer)


async def _select_chunks(
    *,
    collection: Collection,
    desired_count: int,
) -> list[dict[str, Any]]:
    """Walk the collection's serving vector chunks.jsonl artifacts and
    pull ``desired_count`` substantive chunks, favouring diversity by
    round-robin across documents."""
    from aperag.indexing.models import DocumentIndex, IndexStatus, Modality
    from aperag.indexing.parser import read_chunks
    from aperag.objectstore.base import get_object_store

    async for session in get_async_session():
        doc_stmt = (
            select(Document.id)
            .where(
                and_(
                    Document.collection_id == collection.id,
                    Document.gmt_deleted.is_(None),
                )
            )
            .order_by(Document.id)
        )
        doc_ids = [row[0] for row in (await session.execute(doc_stmt)).all()]
        if not doc_ids:
            return []

        index_stmt = select(DocumentIndex.document_id, DocumentIndex.source_path).where(
            and_(
                DocumentIndex.document_id.in_(doc_ids),
                DocumentIndex.modality == Modality.VECTOR.value,
                DocumentIndex.status == IndexStatus.ACTIVE.value,
                DocumentIndex.is_serving.is_(True),
            )
        )
        path_by_doc = {doc_id: src for doc_id, src in (await session.execute(index_stmt)).all()}
        if not path_by_doc:
            return []

        store = get_object_store()
        chunks_per_doc: list[list[dict[str, Any]]] = []
        for doc_id in doc_ids:
            chunks_path = path_by_doc.get(doc_id)
            if not chunks_path:
                continue
            try:
                raw_chunks = await asyncio.to_thread(read_chunks, store, chunks_path)
            except Exception:  # noqa: BLE001 — one bad doc must not fail the request
                logger.exception(
                    "dataset_generator: read_chunks failed for %s (doc %s)",
                    chunks_path,
                    doc_id,
                )
                continue
            substantive = [
                c
                for c in raw_chunks
                if isinstance(c, dict)
                and isinstance(c.get("text"), str)
                and len(c["text"]) >= _SUBSTANTIVE_CHUNK_MIN_CHARS
            ]
            if substantive:
                chunks_per_doc.append(substantive)

        if not chunks_per_doc:
            return []

        # Round-robin pick across documents so a small dataset still
        # samples every doc rather than draining one document end to end.
        picked: list[dict[str, Any]] = []
        cursors = [0] * len(chunks_per_doc)
        while len(picked) < desired_count:
            advanced = False
            for doc_idx, doc_chunks in enumerate(chunks_per_doc):
                cur = cursors[doc_idx]
                if cur >= len(doc_chunks):
                    continue
                picked.append(doc_chunks[cur])
                cursors[doc_idx] = cur + 1
                advanced = True
                if len(picked) >= desired_count:
                    break
            if not advanced:
                break

        return picked
    return []


async def _generate_one(
    *,
    llm,
    chunk: dict[str, Any],
    language: str,
    prompt_override: str | None,
    semaphore: asyncio.Semaphore,
) -> GeneratedDatasetItem | None:
    chunk_text = str(chunk.get("text") or "")
    if not chunk_text:
        return None
    prompt = _format_prompt(language=language, chunk_text=chunk_text, override=prompt_override)
    async with semaphore:
        try:
            raw = await asyncio.wait_for(llm(prompt), timeout=_PER_CHUNK_LLM_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.info("dataset_generator: LLM timeout for chunk; skipping")
            return None
        except Exception:  # noqa: BLE001 — per-chunk failure isolation
            logger.exception("dataset_generator: LLM call failed for chunk; skipping")
            return None
    question, expected_answer = _parse_qa_pair(raw)
    if not question or not expected_answer:
        return None
    reference_context = chunk_text[:_REFERENCE_CONTEXT_MAX_CHARS]
    return GeneratedDatasetItem(
        question=question,
        expected_answer=expected_answer,
        reference_context=reference_context,
    )


async def generate_preview_items(
    *,
    collection: Collection,
    count: int,
    language: str | None,
    prompt_template: str | None,
    llm_factory=None,
) -> tuple[list[GeneratedDatasetItem], str]:
    """Produce ``count`` (or fewer) ``GeneratedDatasetItem`` objects from
    a collection's chunks. ``llm_factory`` defaults to the collection's
    configured completion model — tests pin a deterministic stub. Returns
    the items plus the resolved language so the caller can echo it on
    the response envelope."""
    resolved_language = _resolve_language(request_language=language, collection=collection)

    selected = await _select_chunks(collection=collection, desired_count=count * _CHUNK_OVERSAMPLE)
    if not selected:
        return ([], resolved_language)

    if llm_factory is None:
        from aperag.indexing.llm import build_collection_llm_callable

        llm_factory = build_collection_llm_callable
    llm = llm_factory(collection)

    semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)
    tasks = [
        _generate_one(
            llm=llm,
            chunk=chunk,
            language=resolved_language,
            prompt_override=prompt_template,
            semaphore=semaphore,
        )
        for chunk in selected
    ]
    results = await asyncio.gather(*tasks)

    items: list[GeneratedDatasetItem] = []
    seen_questions: set[str] = set()
    for item in results:
        if item is None:
            continue
        # Trivial dedup — case-insensitive question text. A future LLM-
        # judge embedding-similarity dedup is a Phase 3 follow-up; the
        # naive form keeps the simple-stable guardrail intact for MVP.
        key = item.question.casefold()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        items.append(item)
        if len(items) >= count:
            break

    return (items, resolved_language)


__all__ = [
    "generate_preview_items",
]
