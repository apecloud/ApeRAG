# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 10 §K.13 supplementary #1 — mandatory test coverage.

Per huangheng N4 sediment + architect commitment (per Wave 10 design
doc §10): lease atomic / 3-tier fallback transitions / regen state
machine / API 400 reject / lazy-create branch + race / chunk picker.
The ``test_collection_regen_service.py`` companion file pins the
quality-gate + language-detection contracts; this file covers
orchestration.
"""

from __future__ import annotations

from typing import Any

import pytest

# Import order avoids the conversation.schemas ↔ agent_runtime.uimessage
# circular: load agent_runtime.schemas first so conversation.schemas can
# resolve ``File`` cleanly, then uimessage's ``_rebuild_chat_details``
# completes during conversation.schemas import.
import aperag.domains.agent_runtime.schemas  # noqa: F401
from aperag.domains.agent_runtime.uimessage import TextPart
from aperag.domains.knowledge_base.service import collection_regen_service
from aperag.domains.knowledge_base.service.collection_regen_service import (
    _extract_answer_text,
    _invoke_summary_agent,
    _pick_substantive_chunk_text,
    regen_description,
    regen_summary,
)

# ---------------------------------------------------------------------
# _pick_substantive_chunk_text — chunk picker
# ---------------------------------------------------------------------


def test_pick_substantive_chunk_text_prefers_long_chunks():
    chunks = [
        {"text": "short"},
        {"text": "x" * 250},
        {"text": "y" * 100},
    ]
    assert _pick_substantive_chunk_text(chunks) == "x" * 250


def test_pick_substantive_chunk_text_fallback_to_longest_when_none_substantive():
    chunks = [
        {"text": "short"},
        {"text": "medium length string"},
        {"text": "the unambiguously longest text under threshold"},
    ]
    # All under 200 chars — falls back to longest available text.
    assert _pick_substantive_chunk_text(chunks) == "the unambiguously longest text under threshold"


def test_pick_substantive_chunk_text_returns_none_for_empty_or_typeless_chunks():
    assert _pick_substantive_chunk_text([]) is None
    # No ``text`` field at all.
    assert _pick_substantive_chunk_text([{"meta": "x"}]) is None
    # ``text`` field but non-string.
    assert _pick_substantive_chunk_text([{"text": 123}]) is None


# ---------------------------------------------------------------------
# _extract_answer_text — UIMessage TextPart joiner
# ---------------------------------------------------------------------


def test_extract_answer_text_joins_text_parts_in_order():
    parts = [TextPart(text="first "), TextPart(text="second"), TextPart(text="")]
    # Last empty-text part filtered out by the helper's truthiness check.
    assert _extract_answer_text(parts) == "first second"


def test_extract_answer_text_skips_non_text_parts():
    from aperag.domains.agent_runtime.uimessage import TextPart

    class _Other:
        text = "not a textpart"

    parts = [TextPart(text="kept "), _Other(), TextPart(text="end")]
    assert _extract_answer_text(parts) == "kept end"


# ---------------------------------------------------------------------
# regen_summary state machine — lease busy / collection deleted / all-tiers-invalid
# ---------------------------------------------------------------------


class _FakeSession:
    """Minimal in-memory stand-in for ``AsyncSession`` used by
    ``regen_summary`` / ``regen_description`` orchestration tests.
    Records every ``execute`` call so assertions can inspect the
    side-effect order without touching a real DB.
    """

    def __init__(self, collection: Any | None) -> None:
        self.collection = collection
        self.executed: list[Any] = []
        self.commits = 0

    async def execute(self, stmt):
        self.executed.append(stmt)

        class _Result:
            def __init__(self, collection):
                self._collection = collection
                self.rowcount = 1  # _try_acquire_lease success

            def scalar_one_or_none(self):
                return self._collection

        return _Result(self.collection)

    async def commit(self):
        self.commits += 1


def _patch_session(monkeypatch: pytest.MonkeyPatch, session: _FakeSession) -> None:
    async def _gen(_engine=None):
        yield session

    monkeypatch.setattr(collection_regen_service, "get_async_session", _gen)


@pytest.mark.asyncio
async def test_regen_summary_returns_false_when_lease_busy(monkeypatch: pytest.MonkeyPatch):
    session = _FakeSession(collection=object())
    _patch_session(monkeypatch, session)

    async def _busy(_session, *, collection_id):
        return None  # someone else holds the lease

    monkeypatch.setattr(collection_regen_service, "_try_acquire_lease", _busy)

    released: list[str] = []

    async def _release(_session, *, collection_id, lease_token):
        released.append(collection_id)

    monkeypatch.setattr(collection_regen_service, "_release_lease", _release)

    assert await regen_summary("col_test") is False
    # Lease never acquired → release helper never invoked.
    assert released == []


@pytest.mark.asyncio
async def test_regen_summary_returns_false_when_collection_missing(monkeypatch: pytest.MonkeyPatch):
    session = _FakeSession(collection=None)
    _patch_session(monkeypatch, session)

    async def _ok(_session, *, collection_id):
        return "lease-token"

    monkeypatch.setattr(collection_regen_service, "_try_acquire_lease", _ok)

    released: list[str] = []

    async def _release(_session, *, collection_id, lease_token):
        released.append((collection_id, lease_token))

    monkeypatch.setattr(collection_regen_service, "_release_lease", _release)

    assert await regen_summary("col_test") is False
    # Lease was acquired so release MUST run in the finally branch.
    assert released == [("col_test", "lease-token")]


@pytest.mark.asyncio
async def test_regen_summary_falls_through_to_tier3_when_both_invokers_invalid(
    monkeypatch: pytest.MonkeyPatch,
):
    class _Coll:
        id = "col_test"
        gmt_deleted = None
        title = "t"
        user = "u"
        config = "{}"

    session = _FakeSession(collection=_Coll())
    _patch_session(monkeypatch, session)

    async def _ok(_session, *, collection_id):
        return "lease"

    async def _release(_session, *, collection_id, lease_token):
        return None

    monkeypatch.setattr(collection_regen_service, "_try_acquire_lease", _ok)
    monkeypatch.setattr(collection_regen_service, "_release_lease", _release)

    async def _agent_invalid(_collection):
        return None

    async def _fallback_invalid(_collection, *, llm):
        return None

    monkeypatch.setattr(collection_regen_service, "_invoke_summary_agent", _agent_invalid)
    monkeypatch.setattr(collection_regen_service, "_invoke_summary_chunks_fallback", _fallback_invalid)
    monkeypatch.setattr(
        collection_regen_service,
        "_default_llm_factory",
        lambda _c: lambda _p: None,
    )

    # All tiers fall through → transient skip → False (no DB write).
    assert await regen_summary("col_test") is False
    assert session.commits == 0  # only commits we counted came from execute mocks


@pytest.mark.asyncio
async def test_regen_summary_writes_back_when_tier1_succeeds(monkeypatch: pytest.MonkeyPatch):
    class _Coll:
        id = "col_test"
        gmt_deleted = None
        title = "t"
        user = "u"
        config = "{}"

    session = _FakeSession(collection=_Coll())
    _patch_session(monkeypatch, session)
    monkeypatch.setattr(collection_regen_service, "_try_acquire_lease", lambda *a, **k: _async_value("lease"))
    monkeypatch.setattr(collection_regen_service, "_release_lease", lambda *a, **k: _async_value(None))

    valid_summary = "x" * 250 + " " + "Lorem ipsum " * 30  # passes is_valid_summary

    async def _agent_ok(_c):
        return valid_summary

    async def _fallback_unused(_c, *, llm):
        raise AssertionError("fallback should not run when tier1 succeeds")

    monkeypatch.setattr(collection_regen_service, "_invoke_summary_agent", _agent_ok)
    monkeypatch.setattr(collection_regen_service, "_invoke_summary_chunks_fallback", _fallback_unused)
    monkeypatch.setattr(
        collection_regen_service,
        "_default_llm_factory",
        lambda _c: lambda _p: None,
    )

    assert await regen_summary("col_test") is True
    # Collection load + writeback (lease/release helpers are mocked away).
    assert len(session.executed) >= 2


@pytest.mark.asyncio
async def test_regen_summary_falls_through_tier1_to_tier2(monkeypatch: pytest.MonkeyPatch):
    class _Coll:
        id = "col_test"
        gmt_deleted = None
        title = "t"
        user = "u"
        config = "{}"

    session = _FakeSession(collection=_Coll())
    _patch_session(monkeypatch, session)
    monkeypatch.setattr(collection_regen_service, "_try_acquire_lease", lambda *a, **k: _async_value("lease"))
    monkeypatch.setattr(collection_regen_service, "_release_lease", lambda *a, **k: _async_value(None))

    async def _agent_invalid(_c):
        return None

    async def _fallback_ok(_c, *, llm):
        return "x" * 250 + " " + "Lorem ipsum " * 30

    monkeypatch.setattr(collection_regen_service, "_invoke_summary_agent", _agent_invalid)
    monkeypatch.setattr(collection_regen_service, "_invoke_summary_chunks_fallback", _fallback_ok)
    monkeypatch.setattr(
        collection_regen_service,
        "_default_llm_factory",
        lambda _c: lambda _p: None,
    )

    assert await regen_summary("col_test") is True


# ---------------------------------------------------------------------
# regen_description state machine — summary missing / quality gate
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regen_description_skips_when_summary_missing(monkeypatch: pytest.MonkeyPatch):
    class _Coll:
        id = "col_test"
        gmt_deleted = None
        summary = None
        config = "{}"

    session = _FakeSession(collection=_Coll())
    _patch_session(monkeypatch, session)
    monkeypatch.setattr(collection_regen_service, "_try_acquire_lease", lambda *a, **k: _async_value("lease"))
    monkeypatch.setattr(collection_regen_service, "_release_lease", lambda *a, **k: _async_value(None))

    assert await regen_description("col_test") is False


@pytest.mark.asyncio
async def test_regen_description_succeeds_with_valid_llm_output(monkeypatch: pytest.MonkeyPatch):
    class _Coll:
        id = "col_test"
        gmt_deleted = None
        summary = "本知识库覆盖大量主题，包含机器学习与分布式系统等内容。" * 5
        config = "{}"

    session = _FakeSession(collection=_Coll())
    _patch_session(monkeypatch, session)
    monkeypatch.setattr(collection_regen_service, "_try_acquire_lease", lambda *a, **k: _async_value("lease"))
    monkeypatch.setattr(collection_regen_service, "_release_lease", lambda *a, **k: _async_value(None))

    async def _llm(prompt: str) -> str:
        return (
            "这是一个简短的中文知识库描述，用于在 UI 列表中快速浏览查找。"
            "包含机器学习、分布式系统等多个主题，覆盖核心场景。"
        )

    monkeypatch.setattr(collection_regen_service, "_default_llm_factory", lambda _c: _llm)

    assert await regen_description("col_test") is True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


async def _async_value(value):
    return value


# ---------------------------------------------------------------------
# _invoke_summary_agent — Tier 1 collection.config.completion contract
# (per @earayu2 directive msg=107cbfa3: use the collection's LLM, not
# a default on the summary bot)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_summary_agent_returns_none_when_collection_has_no_completion(
    monkeypatch: pytest.MonkeyPatch,
):
    """Collection with no ``completion.model_id`` → Tier 1 short-circuits
    without touching the agent runtime. ``regen_summary`` then falls
    through to Tier 2."""
    from types import SimpleNamespace

    collection = SimpleNamespace(
        id="col_no_llm",
        user="user-1",
        title="Collection without LLM",
        config=(
            '{"source":"system","enable_vector":true,"enable_fulltext":true,'
            '"enable_knowledge_graph":false,"enable_summary":false,'
            '"enable_vision":false,"language":"zh-CN"}'
        ),
    )

    # If Tier 1 didn't short-circuit, ``get_or_create_summary_bot_for_user``
    # would be invoked next — fail loudly so the test pins the early
    # return contract.
    async def _should_not_be_called(*_a, **_kw):
        raise AssertionError("Tier 1 must short-circuit before bot/chat creation")

    monkeypatch.setattr(
        collection_regen_service,
        "get_or_create_summary_bot_for_user",
        _should_not_be_called,
    )

    assert await _invoke_summary_agent(collection) is None


@pytest.mark.asyncio
async def test_invoke_summary_agent_returns_none_when_completion_model_id_blank(
    monkeypatch: pytest.MonkeyPatch,
):
    """``completion`` block present but ``model_id`` empty → same
    short-circuit as the missing-completion case."""
    from types import SimpleNamespace

    collection = SimpleNamespace(
        id="col_blank_model",
        user="user-1",
        title="Collection with blank model_id",
        config=('{"source":"system","language":"zh-CN","completion":{"model_id":"","temperature":0.1}}'),
    )

    async def _should_not_be_called(*_a, **_kw):
        raise AssertionError("Tier 1 must short-circuit before bot/chat creation")

    monkeypatch.setattr(
        collection_regen_service,
        "get_or_create_summary_bot_for_user",
        _should_not_be_called,
    )

    assert await _invoke_summary_agent(collection) is None


@pytest.mark.asyncio
async def test_invoke_summary_agent_returns_none_on_unparseable_config(
    monkeypatch: pytest.MonkeyPatch,
):
    """A corrupt ``collection.config`` string must not propagate as a
    crash — it must be logged and degraded to ``None`` so Tier 2 can
    still run."""
    from types import SimpleNamespace

    collection = SimpleNamespace(
        id="col_bad_config",
        user="user-1",
        title="Collection with corrupt config",
        config="{not json",
    )

    async def _should_not_be_called(*_a, **_kw):
        raise AssertionError("Tier 1 must short-circuit before bot/chat creation")

    monkeypatch.setattr(
        collection_regen_service,
        "get_or_create_summary_bot_for_user",
        _should_not_be_called,
    )

    assert await _invoke_summary_agent(collection) is None
