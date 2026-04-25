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

* Blocker A (Weston msg=80e873c1) — ``/api/v1/embeddings`` and
  ``/api/v1/rerank`` are permanent OpenAI-compat allowlist routes;
  legacy callers still send ``{model, model_service_provider,
  custom_llm_provider}``. The new schema accepts both shapes — the
  triple is resolved server-side via ``ModelPlatformService``. New
  ``{model_id}`` callers must keep working too.

* Blocker C — collection / bot config blobs already in the database
  hold the legacy triple. Pydantic silently dropped extras after the
  schema cut, so existing rows would parse with ``model_id=None`` and
  the runtime resolver would 404. ``ModelSpec`` now stashes the triple
  for runtime resolution.

This file covers the parser + the resolver wire-up; the live HTTP
side is covered by hurl.
"""

from __future__ import annotations

from aperag.domains.model_platform.schemas import EmbeddingRequest, RerankRequest
from aperag.schema.common import ModelSpec


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


def test_rerank_request_accepts_new_model_id_shape():
    request = RerankRequest.model_validate(
        {
            "model_id": "mdl_rerank",
            "query": "q",
            "documents": ["a", "b"],
        }
    )
    assert request.model_id == "mdl_rerank"


def test_rerank_request_accepts_legacy_triple():
    request = RerankRequest.model_validate(
        {
            "query": "q",
            "documents": ["a"],
            "model": "gte-rerank-v2",
            "model_service_provider": "alibabacloud",
        }
    )
    assert request.model_id is None
    assert request.model == "gte-rerank-v2"
    assert request.model_service_provider == "alibabacloud"


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
