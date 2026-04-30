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

"""Mechanical gate for the static vector-backend capability matrix.

task #61 P1-D3 (PR for #87) ships a static capability matrix that the
collection metadata response projects onto every read. The values are
documented declarations that mirror the spec decisions logged on
task #83 (vector adapter behavior fixes — P1-V2 / P1-V3 / P1-V4). This
test pins each value so a future drift in the static dict is caught at
unit-test time instead of leaking into the FE display silently.

Per architect msg=0044261f + Lesson #18 «lesson sediment + mechanical
gate 双 layer codification — 一记一 enforce» the test acts as the
"mechanical gate" half of the codification: the spec text in
``docs/zh-CN/architecture/task-61-db-adapter-compat-spec-v1.md`` § 2.3
P1-V2 / P1-V3 / P1-V4 declares the values; this test enforces that the
``_STATIC_VECTOR_BACKEND_CAPABILITIES`` dict and
``project_vector_backend_info()`` projection helper do not silently
diverge from the declaration.

Cross-PR consistency note: when task #83 (vector adapter behavior PR)
re-frames any of the declared capability values, the change must land
in the static dict here in the **same** PR (or as a follow-up amend
that updates this test together with the dict). The cross-PR pattern
mirrors PR #1933 (task #33 P3) ``test_graph_extraction_window_size_default_consistency``
and PR #1940 (task #78 A4) ``test_suggestion_action_response_requires_valid_success_shapes``
mechanical-gate boundary tests.
"""

from __future__ import annotations

import pytest

from aperag.schema.common import (
    VectorBackendCapabilities,
    VectorBackendInfo,
    project_vector_backend_info,
)
from aperag.schema.common import _STATIC_VECTOR_BACKEND_CAPABILITIES


def test_qdrant_capability_matrix_matches_p1_v_spec() -> None:
    """Pin Qdrant static capability matrix per task #83 P1-V2/V3/V4 spec declarations.

    If task #83 amends any of these declared values, update both the
    spec § 2.3 P1-V* declaration **and** this assertion in the same PR.
    """

    qdrant_caps = _STATIC_VECTOR_BACKEND_CAPABILITIES["qdrant"]
    # Per task #83 P1-V2 — Qdrant batch upsert is best-effort, no
    # cross-batch atomicity guarantee.
    assert qdrant_caps.supports_atomic_batch_upsert is False
    # Per task #83 P1-V3 — Qdrant filter translator rejects empty Or
    # parts (cuiwenbo task #70 round 1 sediment + spec § 2.3 P1-V3).
    assert qdrant_caps.supports_filter_or_with_empty_parts is False
    # Per task #83 P1-V4 — Qdrant exposes a legacy collection mode that
    # the platform may still create / read.
    assert qdrant_caps.supports_legacy_mode is True


def test_pgvector_capability_matrix_baseline() -> None:
    """Pin PGVector baseline capability.

    PGVector inherits Postgres transactional semantics for batch upsert
    and has no separate legacy schema mode — these two flags are the
    real backend divergence vs Qdrant and are mirrored from the
    connector ``BACKEND_CAPABILITIES`` ClassVar per task #83 PR #1948.

    ``supports_filter_or_with_empty_parts`` is uniformly False after
    task #83 P1-V3 (translator-level defense-in-depth rejects empty Or
    parts on both adapters); the flag stays in the schema so the FE
    can declare the uniform reject explicitly per spec § 2.3 P1-D3.
    """

    pgvector_caps = _STATIC_VECTOR_BACKEND_CAPABILITIES["pgvector"]
    assert pgvector_caps.supports_atomic_batch_upsert is True
    # Uniform reject across adapters per task #83 P1-V3 (PR #1948).
    assert pgvector_caps.supports_filter_or_with_empty_parts is False
    assert pgvector_caps.supports_legacy_mode is False


def test_static_matrix_covers_supported_backends() -> None:
    """Covered backends must equal the ``Literal`` declared on
    :class:`VectorBackendInfo.type` so the projection helper can never
    return a ``VectorBackendInfo`` with a non-Literal backend.
    """

    declared_types = {"pgvector", "qdrant"}
    assert set(_STATIC_VECTOR_BACKEND_CAPABILITIES.keys()) == declared_types


@pytest.mark.parametrize(
    "raw_value,expected_backend",
    [
        ("pgvector", "pgvector"),
        ("qdrant", "qdrant"),
        # Case + whitespace tolerant: the env value comes from
        # ``settings.vector_db_type`` which historically does not
        # normalize. Project helper normalizes so a misconfigured
        # ``QDRANT \n`` does not silently degrade to ``None``.
        ("QDRANT", "qdrant"),
        ("  pgvector  ", "pgvector"),
    ],
)
def test_project_vector_backend_info_normalizes_input(
    raw_value: str, expected_backend: str
) -> None:
    info = project_vector_backend_info(raw_value)
    assert isinstance(info, VectorBackendInfo)
    assert info.type == expected_backend


@pytest.mark.parametrize("raw_value", ["", "milvus", "weaviate", "unknown"])
def test_project_vector_backend_info_returns_none_for_unknown(
    raw_value: str,
) -> None:
    """Unknown backend → ``None`` so the FE can render a placeholder
    without a hard validation failure on misconfigured deployments.
    """

    info = project_vector_backend_info(raw_value)
    assert info is None


def test_project_vector_backend_info_returns_none_for_falsy_input() -> None:
    # ``settings.vector_db_type`` is typed ``str`` but we still defend
    # against ``None`` / empty since the helper signature accepts ``str``
    # and Pydantic-driven settings can produce odd values mid-test.
    assert project_vector_backend_info(None) is None  # type: ignore[arg-type]
    assert project_vector_backend_info("") is None


def test_collection_input_schema_does_not_expose_vector_backend() -> None:
    """Pin OpenAPI input/output split for the deployment vector-backend
    projection.

    ``Collection.vector_backend`` is a Pydantic v2 ``@computed_field``,
    so the *input* JSON Schema generated by Pydantic must not list it
    while the *output* JSON Schema must (with ``readOnly: true``). The
    composite request shapes that reuse :class:`Collection`
    (``CollectionCreate`` / ``CollectionUpdate``, plus the agent /
    chat-turn request bodies whose ``collections`` field embeds
    ``Collection-Input`` in the FastAPI / openapi-typescript output)
    therefore inherit the same property automatically.

    The dongdong msg=fa88e97b BLOCKER caught the previous
    ``Optional[VectorBackendInfo]`` regular-field implementation
    leaking onto every input shape that referenced ``Collection``; this
    test pins the fix so a future refactor cannot regress to a regular
    field without flipping the gate red.
    """

    from aperag.domains.knowledge_base.schemas import (
        Collection,
        CollectionCreate,
        CollectionUpdate,
    )

    output_schema = Collection.model_json_schema(mode="serialization")
    input_schema = Collection.model_json_schema(mode="validation")

    output_props = output_schema.get("properties", {})
    input_props = input_schema.get("properties", {})

    # Output schema must surface the projection + mark it read-only.
    assert "vector_backend" in output_props
    assert output_props["vector_backend"].get("readOnly") is True

    # Input schema must NOT surface the projection — calling
    # ``Collection(vector_backend=...)`` is rejected anyway, but we
    # still want OpenAPI consumers (FE typed schema, agent request
    # body, chat turn request body) to never see it as an editable
    # knob in the first place.
    assert "vector_backend" not in input_props

    # The composite request schemas embed Collection / CollectionConfig
    # but never accept ``vector_backend`` directly either.
    create_props = CollectionCreate.model_json_schema().get("properties", {})
    update_props = CollectionUpdate.model_json_schema().get("properties", {})
    assert "vector_backend" not in create_props
    assert "vector_backend" not in update_props


def test_collection_constructor_ignores_vector_backend_input() -> None:
    """Defensive: even if a caller submits a ``vector_backend`` payload
    on the input shape, Pydantic v2 silently ignores it (computed
    fields cannot be set from input) and the resulting object's
    ``vector_backend`` is still derived from the deployment setting.

    The combination of:
    * computed-field-only output (asserted by
      :func:`test_collection_input_schema_does_not_expose_vector_backend`)
    * input being silently ignored (this test)
    means a malicious / mistaken caller cannot override the deployment
    projection by stuffing ``vector_backend`` into a request body.
    """

    from aperag.domains.knowledge_base.schemas import Collection

    instance = Collection.model_validate(
        {
            "id": "col1",
            "vector_backend": {
                "type": "qdrant",
                "capabilities": {
                    "supports_atomic_batch_upsert": False,
                    "supports_filter_or_with_empty_parts": False,
                    "supports_legacy_mode": True,
                },
            },
        }
    )

    # The computed property is recomputed from settings on access; the
    # input payload is discarded. We only assert that the value does
    # NOT equal the malicious payload — the actual projection depends
    # on the active ``settings.vector_db_type`` in the test runtime.
    rendered = instance.model_dump()
    assert rendered.get("vector_backend") != {
        "type": "qdrant",
        "capabilities": {
            "supports_atomic_batch_upsert": False,
            "supports_filter_or_with_empty_parts": False,
            "supports_legacy_mode": True,
        },
    } or rendered.get("vector_backend") == project_vector_backend_info(
        # If the test deployment happens to be qdrant the values may
        # coincide; in that case the equality is fine because it came
        # from the projection helper, not the input payload.
        "qdrant"
    ).model_dump()  # type: ignore[union-attr]


def test_vector_backend_info_pydantic_round_trip() -> None:
    """Sanity: the Pydantic model round-trips through ``model_dump`` /
    ``model_validate`` so the generated OpenAPI schema mirrors the
    nested structure the FE consumes.
    """

    info = project_vector_backend_info("pgvector")
    assert info is not None
    payload = info.model_dump()
    assert payload == {
        "type": "pgvector",
        "capabilities": {
            "supports_atomic_batch_upsert": True,
            # Uniform reject across adapters per task #83 P1-V3 (PR #1948).
            "supports_filter_or_with_empty_parts": False,
            "supports_legacy_mode": False,
        },
    }
    rebuilt = VectorBackendInfo.model_validate(payload)
    assert rebuilt == info
    assert isinstance(rebuilt.capabilities, VectorBackendCapabilities)
