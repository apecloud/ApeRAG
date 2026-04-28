# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 10 §K.13 Chunks C+D — collection_regen_service unit tests.

Pin the quality-gate + language-detection contracts. Lease, async DB
flow, and OpenAPI integration are covered separately by Chunk E
reconciler tests + Chunk G e2e narrative test (those require live DB
fixtures; this file is pure-Python unit coverage).
"""

from __future__ import annotations

from aperag.domains.knowledge_base.service.collection_regen_service import (
    _detect_language,
    is_valid_description,
    is_valid_summary,
)


# ---------------------------------------------------------------------
# Quality gates — summary
# ---------------------------------------------------------------------


def test_is_valid_summary_rejects_short_text():
    assert is_valid_summary("") is False
    assert is_valid_summary(None) is False
    assert is_valid_summary("short") is False
    # Just under the 200-char minimum.
    assert is_valid_summary("a" * 199) is False


def test_is_valid_summary_rejects_llm_error_responses():
    long_pad = " " + "a" * 250
    assert is_valid_summary(f"tool error: timeout{long_pad}") is False
    assert is_valid_summary(f"无法生成 summary: please retry{long_pad}") is False
    assert is_valid_summary(f"I cannot generate this{long_pad}") is False
    assert is_valid_summary(f"i can't help with that{long_pad}") is False


def test_is_valid_summary_requires_alphabetic_run():
    # 200 chars all-numeric → no 50-char alphabetic run.
    assert is_valid_summary("1234567890" * 25) is False


def test_is_valid_summary_accepts_well_formed_text():
    body = "This collection covers a wide range of topics from machine learning to"
    body += " distributed systems engineering and their practical applications. " * 5
    assert is_valid_summary(body) is True


def test_is_valid_summary_accepts_chinese_text():
    body = "本知识库涵盖机器学习、分布式系统工程及其实际应用等多个主题，" * 10
    assert is_valid_summary(body) is True


# ---------------------------------------------------------------------
# Quality gates — description
# ---------------------------------------------------------------------


def test_is_valid_description_rejects_short_text():
    assert is_valid_description(None) is False
    assert is_valid_description("") is False
    assert is_valid_description("short") is False
    assert is_valid_description("a" * 49) is False


def test_is_valid_description_rejects_llm_error_responses():
    pad = "a" * 60
    assert is_valid_description(f"tool error: timeout{pad}") is False
    assert is_valid_description(f"无法生成 description{pad}") is False


def test_is_valid_description_accepts_short_well_formed_text():
    text = "A concise description of this knowledge base for quick reference scanning."
    assert is_valid_description(text) is True


def test_is_valid_description_accepts_chinese_text():
    text = "这是一个简短的知识库描述，用于在 UI 列表中快速浏览查找。包含机器学习、分布式系统等多个主题，覆盖核心场景。"
    assert is_valid_description(text) is True


# ---------------------------------------------------------------------
# Language detection (per huangheng N1)
# ---------------------------------------------------------------------


def test_detect_language_chinese_majority():
    # 25 CJK chars vs ~10 latin — CJK majority.
    text = "这是一个中文知识库主要讲述机器学习相关的内容主题 plus some en"
    assert _detect_language(text) == "zh"


def test_detect_language_english_majority():
    text = "This is an English knowledge base 包含少量中文"
    assert _detect_language(text) == "en"


def test_detect_language_empty_defaults_english():
    assert _detect_language("") == "en"
    assert _detect_language("12345") == "en"


def test_detect_language_pure_chinese():
    text = "这是一个中文知识库"
    assert _detect_language(text) == "zh"


def test_detect_language_pure_english():
    text = "This is an English knowledge base"
    assert _detect_language(text) == "en"
