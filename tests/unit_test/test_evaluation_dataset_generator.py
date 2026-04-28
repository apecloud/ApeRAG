# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the AI auto-generate QA pairs path
(``aperag/domains/evaluation/dataset_generator.py``).

Per architect lock ``#evaluation msg=05c3ec83`` / ``msg=a9fb7efd``: the
preview endpoint must surface ``{question, expected_answer,
reference_context}`` items, dedup by question, cap reference_context at
8 000 chars, and resolve language from request → collection.config →
default. The tests pin those contracts without a live LLM.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aperag.domains.evaluation import dataset_generator
from aperag.domains.evaluation.dataset_generator import (
    _REFERENCE_CONTEXT_MAX_CHARS,
    _format_prompt,
    _parse_qa_pair,
    _resolve_language,
    generate_preview_items,
)

# ---------------------------------------------------------------------
# _parse_qa_pair — shape coercion
# ---------------------------------------------------------------------


def test_parse_qa_pair_accepts_plain_json():
    raw = json.dumps({"question": "What is RAG?", "expected_answer": "Retrieval-Augmented Generation"})
    q, a = _parse_qa_pair(raw)
    assert q == "What is RAG?"
    assert a == "Retrieval-Augmented Generation"


def test_parse_qa_pair_accepts_fenced_json():
    raw = '```json\n{"question": "Q?", "expected_answer": "A"}\n```'
    q, a = _parse_qa_pair(raw)
    assert q == "Q?"
    assert a == "A"


def test_parse_qa_pair_accepts_legacy_keys():
    """Some models emit ``answer`` instead of ``expected_answer``."""
    raw = json.dumps({"question": "Q?", "answer": "A"})
    q, a = _parse_qa_pair(raw)
    assert q == "Q?"
    assert a == "A"


def test_parse_qa_pair_rejects_invalid_json():
    q, a = _parse_qa_pair("not even json — model rambled")
    assert q is None
    assert a is None


def test_parse_qa_pair_rejects_empty_fields():
    raw = json.dumps({"question": "  ", "expected_answer": ""})
    q, a = _parse_qa_pair(raw)
    assert q is None
    assert a is None


def test_parse_qa_pair_rejects_non_object_payload():
    raw = json.dumps([{"question": "Q?", "expected_answer": "A"}])  # array, not object
    q, a = _parse_qa_pair(raw)
    assert q is None
    assert a is None


# ---------------------------------------------------------------------
# _resolve_language — request → collection.config → default
# ---------------------------------------------------------------------


def _collection_with_language(language: str | None) -> SimpleNamespace:
    if language is None:
        config = "{}"
    else:
        config = json.dumps({"language": language})
    return SimpleNamespace(id="col_test", user="u", config=config)


def test_resolve_language_uses_request_when_supported():
    coll = _collection_with_language("zh-CN")
    assert _resolve_language(request_language="ja-JP", collection=coll) == "ja-JP"


def test_resolve_language_ignores_unsupported_request_value():
    coll = _collection_with_language("zh-CN")
    assert _resolve_language(request_language="klingon-KL", collection=coll) == "zh-CN"


def test_resolve_language_falls_back_to_collection_config():
    coll = _collection_with_language("ja-JP")
    assert _resolve_language(request_language=None, collection=coll) == "ja-JP"


def test_resolve_language_falls_back_to_default_when_collection_has_none():
    coll = _collection_with_language(None)
    assert _resolve_language(request_language=None, collection=coll) == "zh-CN"


def test_resolve_language_handles_unparseable_collection_config():
    coll = SimpleNamespace(id="col_test", user="u", config="{not json")
    # Falls through to default rather than raising.
    assert _resolve_language(request_language=None, collection=coll) == "zh-CN"


# ---------------------------------------------------------------------
# _format_prompt — language template + override
# ---------------------------------------------------------------------


def test_format_prompt_uses_zh_template_by_default():
    prompt = _format_prompt(language="zh-CN", chunk_text="蜜蜂养殖", override=None)
    assert "蜜蜂养殖" in prompt
    assert "中文" in prompt


def test_format_prompt_uses_en_template():
    prompt = _format_prompt(language="en-US", chunk_text="Apiary basics", override=None)
    assert "Apiary basics" in prompt
    assert "English" in prompt


def test_format_prompt_falls_back_to_default_for_unknown_language():
    prompt = _format_prompt(language="fr-FR", chunk_text="abeilles", override=None)
    # Default is zh-CN per dataset_generator._DEFAULT_LANGUAGE.
    assert "abeilles" in prompt


def test_format_prompt_override_substitutes_chunk_text():
    override = "TEMPLATE [{language}] ::: {chunk_text}"
    prompt = _format_prompt(language="en-US", chunk_text="X", override=override)
    assert prompt == "TEMPLATE [en-US] ::: X"


def test_format_prompt_override_with_unknown_placeholder_falls_back():
    """A typo in the override (``{question}`` instead of ``{chunk_text}``)
    must not produce a silently-empty prompt — fall back to the
    language default so the LLM still has the chunk in context."""
    override = "Bad template: {question}"
    prompt = _format_prompt(language="en-US", chunk_text="ChunkA", override=override)
    assert "ChunkA" in prompt


# ---------------------------------------------------------------------
# generate_preview_items — orchestration
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_preview_items_returns_empty_when_no_chunks(monkeypatch):
    coll = _collection_with_language("zh-CN")

    async def _no_chunks(*, collection, desired_count):
        return []

    monkeypatch.setattr(dataset_generator, "_select_chunks", _no_chunks)

    items, language = await generate_preview_items(
        collection=coll,
        count=5,
        language="zh-CN",
        prompt_template=None,
        llm_factory=lambda _c: (lambda _p: None),
    )
    assert items == []
    assert language == "zh-CN"


@pytest.mark.asyncio
async def test_generate_preview_items_dedupes_and_caps_reference_context(monkeypatch):
    coll = _collection_with_language("en-US")

    long_chunk = "a" * (_REFERENCE_CONTEXT_MAX_CHARS + 5_000)
    chunks = [
        {"text": long_chunk + "1", "id": "c1"},
        {"text": "second chunk text " + "b" * 250, "id": "c2"},
        {"text": "third chunk text " + "c" * 250, "id": "c3"},
    ]

    async def _select(*, collection, desired_count):
        return chunks[:desired_count]

    monkeypatch.setattr(dataset_generator, "_select_chunks", _select)

    # Two duplicate questions to verify dedup; one varied.
    responses = iter(
        [
            json.dumps({"question": "Same Q?", "expected_answer": "A1"}),
            json.dumps({"question": "Same Q?", "expected_answer": "A2"}),
            json.dumps({"question": "Different Q?", "expected_answer": "A3"}),
        ]
    )

    async def _llm(_prompt: str) -> str:
        return next(responses)

    items, language = await generate_preview_items(
        collection=coll,
        count=5,
        language=None,  # falls back to collection.config.language
        prompt_template=None,
        llm_factory=lambda _c: _llm,
    )
    assert language == "en-US"
    assert len(items) == 2
    assert {item.question for item in items} == {"Same Q?", "Different Q?"}
    # Reference context capped at the boundary so a 13K-char chunk
    # cannot bloat the row.
    for item in items:
        assert item.reference_context is not None
        assert len(item.reference_context) <= _REFERENCE_CONTEXT_MAX_CHARS


@pytest.mark.asyncio
async def test_generate_preview_items_skips_failed_llm_responses(monkeypatch):
    coll = _collection_with_language("zh-CN")

    chunks = [
        {"text": "chunk one " + "a" * 250, "id": "c1"},
        {"text": "chunk two " + "b" * 250, "id": "c2"},
        {"text": "chunk three " + "c" * 250, "id": "c3"},
    ]

    async def _select(*, collection, desired_count):
        return chunks

    monkeypatch.setattr(dataset_generator, "_select_chunks", _select)

    responses = iter(
        [
            "model rambled, not JSON",
            json.dumps({"question": "Valid Q?", "expected_answer": "Valid A"}),
            json.dumps({"question": "", "expected_answer": "no question"}),
        ]
    )

    async def _llm(_prompt: str) -> str:
        return next(responses)

    items, _ = await generate_preview_items(
        collection=coll,
        count=5,
        language="zh-CN",
        prompt_template=None,
        llm_factory=lambda _c: _llm,
    )
    assert len(items) == 1
    assert items[0].question == "Valid Q?"


@pytest.mark.asyncio
async def test_generate_preview_items_truncates_to_count(monkeypatch):
    coll = _collection_with_language("zh-CN")

    chunks = [{"text": f"chunk {i} " + "x" * 250, "id": f"c{i}"} for i in range(10)]

    async def _select(*, collection, desired_count):
        return chunks[:desired_count]

    monkeypatch.setattr(dataset_generator, "_select_chunks", _select)

    counter = iter(range(100))

    async def _llm(_prompt: str) -> str:
        i = next(counter)
        return json.dumps({"question": f"Q{i}", "expected_answer": f"A{i}"})

    items, _ = await generate_preview_items(
        collection=coll,
        count=3,
        language="zh-CN",
        prompt_template=None,
        llm_factory=lambda _c: _llm,
    )
    assert len(items) == 3
