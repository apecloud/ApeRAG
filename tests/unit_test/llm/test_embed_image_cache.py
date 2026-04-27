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

"""Unit tests for Wave 6 task #37: ``EmbeddingService.embed_image`` cache wiring.

Pins the contract that ``embed_image`` honours the canonical
application cache (``NAMESPACE_EMBEDDING``):

* identical ``(image_bytes, alt_text)`` calls hit the cache and skip the
  upstream LiteLLM call,
* the cache key shape is ``sha256(image_bytes)``-based, never embedding
  raw bytes (per `aperag/cache/README.md` policy),
* changing ``alt_text`` (or the image bytes) yields a distinct key,
* ``caching=False`` bypasses the cache and always calls the upstream.
"""

from __future__ import annotations

import hashlib

import pytest

from aperag.cache.application import ApplicationCachePolicy, SyncApplicationCache
from aperag.cache.key import build_cache_key
from aperag.llm.embed.embedding_service import EmbeddingService
from aperag.llm.llm_error_types import EmbeddingError, EmptyTextError


class _FakeSyncBackend:
    """Minimal in-memory backend matching ``SyncApplicationRedisCacheBackend``."""

    def __init__(self):
        self.store: dict[str, str | bytes] = {}
        self.get_calls = 0
        self.set_calls = 0

    def get(self, key: str):
        self.get_calls += 1
        return self.store.get(key)

    def mget(self, keys):
        return [self.get(key) for key in keys]

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.set_calls += 1
        self.store[key] = value

    def delete(self, *keys: str) -> int:
        for key in keys:
            self.store.pop(key, None)
        return len(keys)


def _make_cache() -> SyncApplicationCache:
    return SyncApplicationCache(
        backend=_FakeSyncBackend(),
        default_policy=ApplicationCachePolicy(namespace="embedding", ttl_seconds=60, max_value_bytes=4096),
    )


def _make_service(*, caching: bool = True) -> EmbeddingService:
    return EmbeddingService(
        embedding_provider="jina_ai",
        embedding_model="jina-embeddings-v4",
        embedding_service_url="https://api.jina.ai/v1",
        embedding_service_api_key="sk-test",
        embedding_max_chunks_in_batch=1,
        multimodal=True,
        caching=caching,
    )


@pytest.fixture
def cache(monkeypatch):
    fake = _make_cache()
    monkeypatch.setattr("aperag.llm.embed.embedding_service.get_sync_application_cache", lambda: fake)
    return fake


def _stub_litellm(monkeypatch, calls: list[bytes]):
    def _fake(self, *, image_bytes: bytes, alt_text: str):
        calls.append(image_bytes)
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(EmbeddingService, "_embed_image_via_litellm", _fake)


def test_embed_image_caches_identical_calls(cache, monkeypatch):
    """Second identical call must return from cache (no second compute)."""

    calls: list[bytes] = []
    _stub_litellm(monkeypatch, calls)

    service = _make_service()

    first = service.embed_image(b"\x89PNG\r\n\x1a\nfake-bytes", alt_text="cat")
    second = service.embed_image(b"\x89PNG\r\n\x1a\nfake-bytes", alt_text="cat")

    assert first == second == [0.1, 0.2, 0.3]
    assert len(calls) == 1, "second call should hit cache, not invoke LiteLLM"


def test_embed_image_distinct_alt_text_yields_distinct_keys(cache, monkeypatch):
    """Same image bytes + different alt_text must compute twice (independent cache rows)."""

    calls: list[bytes] = []
    _stub_litellm(monkeypatch, calls)

    service = _make_service()
    image_bytes = b"\x89PNG\r\n\x1a\nfake-bytes"

    service.embed_image(image_bytes, alt_text="cat")
    service.embed_image(image_bytes, alt_text="dog")

    assert len(calls) == 2, "alt_text change must miss the cache and re-compute"


def test_embed_image_distinct_bytes_yield_distinct_keys(cache, monkeypatch):
    """Different image bytes (even same alt_text) must compute twice."""

    calls: list[bytes] = []
    _stub_litellm(monkeypatch, calls)

    service = _make_service()

    service.embed_image(b"\x89PNG\r\n\x1a\nimage-A", alt_text="x")
    service.embed_image(b"\x89PNG\r\n\x1a\nimage-B", alt_text="x")

    assert len(calls) == 2, "different image bytes must produce a different cache key"


def test_embed_image_cache_key_uses_sha256_not_raw_bytes(monkeypatch):
    """Per `aperag/cache/README.md` raw bytes must never appear in the
    Redis key. The image fingerprint is a sha256 hex digest.
    """

    service = _make_service()
    image_bytes = b"sensitive-binary-payload"
    key_data = service._cache_key_for_image(image_bytes, alt_text="hint")

    assert key_data["file_hash"] == hashlib.sha256(image_bytes).hexdigest()
    assert key_data["kind"] == "image"
    assert key_data["multimodal"] is True
    # Build the actual Redis key and verify the raw bytes are absent.
    key = build_cache_key("embedding", key_data)
    assert "sensitive-binary-payload" not in key
    assert key.startswith("aperag:cache:v1:embedding:")


def test_embed_image_cache_disabled_always_calls_upstream(monkeypatch):
    """``caching=False`` must bypass the cache wiring entirely."""

    calls: list[bytes] = []
    _stub_litellm(monkeypatch, calls)

    service = _make_service(caching=False)

    service.embed_image(b"img", alt_text="x")
    service.embed_image(b"img", alt_text="x")

    assert len(calls) == 2, "caching=False must skip the cache and recompute every call"


def test_embed_image_rejects_non_multimodal():
    """Defense-in-depth: ``multimodal=False`` must raise before touching the cache."""

    service = EmbeddingService(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_service_url="https://api.openai.com/v1",
        embedding_service_api_key="sk-test",
        embedding_max_chunks_in_batch=1,
        multimodal=False,
        caching=True,
    )

    with pytest.raises(EmbeddingError):
        service.embed_image(b"any", alt_text="x")


def test_embed_image_rejects_empty_bytes():
    service = _make_service()
    with pytest.raises(EmptyTextError):
        service.embed_image(b"", alt_text="x")
