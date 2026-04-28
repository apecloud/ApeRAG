# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Inline retry regression tests for ``_embed_batch_uncached``.

Pre-fix the embedding service called ``litellm.embedding`` once and
let any 429 / 5xx / timeout propagate to the orchestrator's row-level
retry path (5 attempts × 30/60/120/240/480s backoff). On a flaky
DashScope / OpenRouter rate-limit window this surfaced as the user-
visible "vector occasionally fails, then auto-retries to running"
UX (task #8 msg=02508e12).

Fix: short inline retry budget (3 attempts, exponential 1s/2s/4s base
+ jitter) absorbs short-lived hiccups without burning a full row-
level cycle. Non-retryable errors (auth / quota / config / validation)
propagate immediately.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aperag.llm.embed import embedding_service as embedding_module
from aperag.llm.embed.embedding_service import (
    _EMBED_INLINE_BASE_DELAY_SECONDS,
    _EMBED_INLINE_MAX_ATTEMPTS,
    EmbeddingService,
)
from aperag.llm.llm_error_types import (
    AuthenticationError,
    RateLimitError,
)


def _fake_litellm_response(batch_size: int, dim: int = 4):
    return {"data": [{"embedding": [0.0] * dim} for _ in range(batch_size)]}


def _build_service() -> EmbeddingService:
    return EmbeddingService(
        embedding_provider="alibabacloud",
        embedding_model="text-embedding-v3",
        embedding_service_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_service_api_key="sk-test-not-used",
        embedding_max_chunks_in_batch=8,
        caching=False,
    )


class _FakeRateLimit(Exception):
    """Stand-in for the litellm-side rate-limit error.

    ``wrap_litellm_error`` keys off the exception class name + message
    keywords. Naming this ``RateLimitError`` is enough for the wrap
    helper to map it to :class:`aperag.llm.llm_error_types.RateLimitError`,
    which is what :func:`is_retryable_error` recognises as retryable.
    """

    def __init__(self, message: str = "rate limit exceeded") -> None:
        super().__init__(message)


# Match the class name the wrap helper detects.
_FakeRateLimit.__name__ = "RateLimitError"


class _FakeAuthError(Exception):
    """Non-retryable: auth failure (401-style). Must propagate
    immediately without retry — retrying would just burn quota and
    the orchestrator's row-level slots."""

    def __init__(self, message: str = "invalid api key") -> None:
        super().__init__(message)


_FakeAuthError.__name__ = "AuthenticationError"


def test_embed_batch_uncached_retries_on_rate_limit_then_succeeds(monkeypatch):
    """First call hits 429; second call returns valid embeddings.
    The service must absorb the 429 inline and return the batch
    embeddings on the retry without bubbling to the caller."""
    service = _build_service()
    # Don't actually sleep in the test.
    monkeypatch.setattr(embedding_module.time, "sleep", lambda _delay: None)

    call_count = {"n": 0}

    def fake_embedding(**_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _FakeRateLimit()
        return _fake_litellm_response(batch_size=1)

    with patch.object(embedding_module.litellm, "embedding", side_effect=fake_embedding):
        result = service._embed_batch_uncached(["hello"])

    assert len(result) == 1
    assert call_count["n"] == 2, "must have retried exactly once after the 429"


def test_embed_batch_uncached_exhausts_budget_and_propagates_rate_limit(monkeypatch):
    """All attempts hit 429 → after the configured budget the
    wrapped :class:`RateLimitError` propagates so the orchestrator
    can still take its row-level retry."""
    service = _build_service()
    monkeypatch.setattr(embedding_module.time, "sleep", lambda _delay: None)

    call_count = {"n": 0}

    def fake_embedding(**_kwargs):
        call_count["n"] += 1
        raise _FakeRateLimit()

    with patch.object(embedding_module.litellm, "embedding", side_effect=fake_embedding):
        with pytest.raises(RateLimitError):
            service._embed_batch_uncached(["hello"])

    assert call_count["n"] == _EMBED_INLINE_MAX_ATTEMPTS, (
        f"expected exactly {_EMBED_INLINE_MAX_ATTEMPTS} attempts before propagating"
    )


def test_embed_batch_uncached_does_not_retry_non_retryable_errors(monkeypatch):
    """Auth / quota / config / validation errors are wasted retries —
    they propagate immediately without consuming the inline budget."""
    service = _build_service()
    sleep_calls: list[float] = []
    monkeypatch.setattr(embedding_module.time, "sleep", lambda delay: sleep_calls.append(delay))

    call_count = {"n": 0}

    def fake_embedding(**_kwargs):
        call_count["n"] += 1
        raise _FakeAuthError()

    with patch.object(embedding_module.litellm, "embedding", side_effect=fake_embedding):
        with pytest.raises(AuthenticationError):
            service._embed_batch_uncached(["hello"])

    assert call_count["n"] == 1, "non-retryable error must NOT trigger inline retry"
    assert sleep_calls == [], "no backoff sleep should fire for non-retryable error"


def test_embed_batch_uncached_uses_exponential_base_delay(monkeypatch):
    """Backoff schedule is exponential off ``_EMBED_INLINE_BASE_DELAY_SECONDS``
    with full-jitter on top. The minimum sleep on attempt N must be
    ``base * 2^(N-1)`` (jitter only adds, never subtracts)."""
    service = _build_service()
    sleep_calls: list[float] = []
    monkeypatch.setattr(embedding_module.time, "sleep", lambda delay: sleep_calls.append(delay))

    def fake_embedding(**_kwargs):
        raise _FakeRateLimit()

    with patch.object(embedding_module.litellm, "embedding", side_effect=fake_embedding):
        with pytest.raises(RateLimitError):
            service._embed_batch_uncached(["hello"])

    # 3 attempts → 2 sleeps (no sleep after the last failure).
    assert len(sleep_calls) == _EMBED_INLINE_MAX_ATTEMPTS - 1
    # Exponential base: attempt 1 retry sleeps at least base, attempt 2 retry
    # sleeps at least 2*base. Jitter only adds (uniform(0, delay)), so the
    # minimum is the base; the maximum is 2x the base.
    base = _EMBED_INLINE_BASE_DELAY_SECONDS
    assert base <= sleep_calls[0] <= 2 * base
    assert 2 * base <= sleep_calls[1] <= 4 * base
