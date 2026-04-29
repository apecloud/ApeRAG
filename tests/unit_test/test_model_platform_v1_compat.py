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

"""Regression coverage for the pre-#1697 back-compat shim.

PR #1697 collapses the legacy ``llm_provider`` / ``model_service_provider``
/ ``llm_provider_models`` surface into the new
``model_provider`` / ``model_account`` / ``model`` schema. Two pieces of
external contract have to keep working through the cut:

* Blocker A (Weston msg=80e873c1) — ``/api/v1/embeddings`` is the
  permanent OpenAI-compat allowlist route; legacy callers still send
  ``{model, model_service_provider, custom_llm_provider}``. The new
  schema accepts both shapes — the triple is resolved server-side via
  ``ModelPlatformService``. New ``{model_id}`` callers must keep
  working too.

* Blocker C — collection / bot config blobs already in the database
  hold the legacy triple. Pydantic silently dropped extras after the
  schema cut, so existing rows would parse with ``model_id=None`` and
  the runtime resolver would 404. ``ModelSpec`` now stashes the triple
  for runtime resolution.

This file covers the parser + the resolver wire-up; the live HTTP
side is covered by hurl.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from aperag.db.ops import AsyncDatabaseOps
from aperag.domains.model_platform.db.models import ModelAccount
from aperag.domains.model_platform.schemas import EmbeddingRequest
from aperag.domains.model_platform.service.model_service import ModelPlatformService
from aperag.schema.common import ModelSpec
from aperag.utils.utils import utc_now


def test_embedding_request_accepts_new_model_id_shape():
    request = EmbeddingRequest.model_validate({"model_id": "mdl_123", "input": ["hi"]})
    assert request.model_id == "mdl_123"
    assert request.model is None
    assert request.model_service_provider is None
    assert request.custom_llm_provider is None


def test_embedding_request_accepts_legacy_triple_without_model_id():
    request = EmbeddingRequest.model_validate(
        {
            "input": "hi",
            "model": "text-embedding-v3",
            "model_service_provider": "alibabacloud",
            "custom_llm_provider": "openai",
        }
    )
    assert request.model_id is None
    assert request.model == "text-embedding-v3"
    assert request.model_service_provider == "alibabacloud"
    assert request.custom_llm_provider == "openai"


def test_model_spec_parses_new_model_id_shape():
    spec = ModelSpec.model_validate({"model_id": "mdl_default_chat", "temperature": 0.2})
    assert spec.model_id == "mdl_default_chat"
    assert spec.temperature == 0.2
    assert not spec.has_legacy_triple()


def test_model_spec_parses_legacy_triple_for_runtime_resolution():
    spec = ModelSpec.model_validate(
        {
            "model": "google/gemini-2.5-flash",
            "model_service_provider": "openrouter",
            "custom_llm_provider": "openrouter",
            "temperature": 0.3,
        }
    )
    assert spec.model_id is None
    assert spec.has_legacy_triple()
    assert spec.legacy_model == "google/gemini-2.5-flash"
    assert spec.legacy_provider == "openrouter"
    assert spec.legacy_custom_llm_provider == "openrouter"
    # Public dump must not leak legacy fields — they are runtime-only
    # and wire-side only the new ``model_id`` shape is canonical.
    dumped = spec.model_dump()
    assert "legacy_model" not in dumped
    assert "legacy_provider" not in dumped
    assert "legacy_custom_llm_provider" not in dumped


def test_model_spec_new_shape_takes_priority_over_legacy_triple():
    spec = ModelSpec.model_validate(
        {
            "model_id": "mdl_explicit",
            # Pretend a legacy caller also wrote the old fields.
            "model": "ignored",
            "model_service_provider": "ignored",
        }
    )
    assert spec.model_id == "mdl_explicit"
    assert not spec.has_legacy_triple()


# ---------------------------------------------------------------------------
# Weston blocker (msg=fcefbaf7) — ``query_model_account_api_key`` must NOT
# rank a more-recently-updated ``public`` row above the caller's own row.
# The documented contract is "fall back to public WHEN the user has no
# personal account" — a freshly-edited shared key must never silently
# shadow a user's own credential.
# ---------------------------------------------------------------------------


@pytest.fixture
async def _fallback_db_ops(monkeypatch):
    """In-memory ``AsyncDatabaseOps`` with just ``model_account`` created.

    Patches ``model_service.async_db_ops`` so
    ``ModelPlatformService.get_user_provider_api_key`` exercises the real
    repository query against the seeded rows.
    """

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: ModelAccount.__table__.create(c))
    sessionmaker_ = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session = sessionmaker_()
    db_ops = AsyncDatabaseOps(session=session)
    monkeypatch.setattr(
        "aperag.domains.model_platform.service.model_service.async_db_ops",
        db_ops,
    )
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()


async def test_user_personal_key_wins_over_newer_public_key(_fallback_db_ops):
    session = _fallback_db_ops
    now = utc_now()
    # Personal row updated 1h ago.
    session.add(
        ModelAccount(
            id="ma_user",
            user_id="alice",
            provider_type="jina",
            name="user-jina",
            display_name="alice's jina",
            base_url="https://api.jina.ai",
            encrypted_api_key="user_key",
            auth_config={},
            status="ACTIVE",
            extra={},
            gmt_created=now - timedelta(hours=1),
            gmt_updated=now - timedelta(hours=1),
        )
    )
    # Public row updated *just now* — newer than the personal row.
    session.add(
        ModelAccount(
            id="ma_public",
            user_id="public",
            provider_type="jina",
            name="public-jina",
            display_name="shared jina",
            base_url="https://api.jina.ai",
            encrypted_api_key="public_key",
            auth_config={},
            status="ACTIVE",
            extra={},
            gmt_created=now,
            gmt_updated=now,
        )
    )
    await session.commit()

    service = ModelPlatformService()
    api_key = await service.get_user_provider_api_key("alice", "jina", fallback_to_public=True)

    # Pre-fix this returned ``public_key`` because ORDER BY only sorted by
    # ``gmt_updated DESC`` — violating the documented "user has no personal
    # account → fallback to public" contract.
    assert api_key == "user_key"


async def test_public_key_returned_when_user_has_no_personal_account(_fallback_db_ops):
    """Sanity: the actual fallback path keeps working."""
    session = _fallback_db_ops
    now = utc_now()
    session.add(
        ModelAccount(
            id="ma_public_only",
            user_id="public",
            provider_type="jina",
            name="public-jina",
            display_name="shared jina",
            base_url="https://api.jina.ai",
            encrypted_api_key="public_key",
            auth_config={},
            status="ACTIVE",
            extra={},
            gmt_created=now,
            gmt_updated=now,
        )
    )
    await session.commit()

    service = ModelPlatformService()
    api_key = await service.get_user_provider_api_key("alice", "jina", fallback_to_public=True)

    assert api_key == "public_key"
