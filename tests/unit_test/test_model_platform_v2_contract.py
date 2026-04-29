from types import SimpleNamespace

from fastapi import FastAPI

from aperag.domains.model_platform.api.providers_v2_routes import router
from aperag.domains.model_platform.schemas import (
    ModelAccountCreate,
    ModelCapability,
    ModelCreate,
    ModelProvider,
    ModelUpdate,
    ModelUseScenario,
)
from aperag.domains.model_platform.service.model_service import (
    _model_extra_with_allowed_scenarios,
    allowed_scenarios_for_model,
    default_allowed_scenarios,
)
from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi


def _provider_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def test_model_platform_v2_routes_use_product_language():
    spec = _provider_v2_spec()
    paths = spec["paths"]

    assert "/api/v2/model-providers" in paths
    assert "/api/v2/model-accounts" in paths
    assert "/api/v2/model-accounts/{account_id}/validate" in paths
    assert "/api/v2/model-accounts/{account_id}/models" in paths
    assert "/api/v2/models" in paths
    assert "/api/v2/models/{model_id}" in paths
    assert "/api/v2/models/{model_id}/validate" in paths
    assert "/api/v2/model-uses" in paths
    assert "/api/v2/model-uses/{scenario}" in paths


def test_model_platform_v2_schemas_do_not_expose_litellm_dialects():
    spec = _provider_v2_spec()
    forbidden = {
        "completion_dialect",
        "embedding_dialect",
        "rerank_dialect",
        "custom_llm_provider",
        "model_service_provider",
    }

    for schema in spec["components"]["schemas"].values():
        assert forbidden.isdisjoint(set(schema.get("properties", {})))


def test_model_account_and_model_create_are_user_level_concepts():
    account = ModelAccountCreate(
        provider_type="dashscope",
        name="prod-dashscope",
        display_name="生产环境阿里百炼",
        base_url="https://dashscope.aliyuncs.com",
        api_key="sk-test",
    )
    model = ModelCreate(
        account_id="acct_1",
        provider_model_id="gte-rerank-v2",
        display_name="通义重排",
        capability=ModelCapability.RERANK,
    )

    assert account.provider_type == "dashscope"
    assert model.capability == ModelCapability.RERANK
    assert "dialect" not in account.model_dump()
    assert "custom_llm_provider" not in model.model_dump()


def test_model_provider_and_model_use_are_explicit():
    provider = ModelProvider(
        provider_type="openai_compatible",
        display_name="OpenAI Compatible",
        supported_capabilities=[ModelCapability.CHAT, ModelCapability.EMBEDDING],
    )

    assert provider.supported_capabilities == [ModelCapability.CHAT, ModelCapability.EMBEDDING]
    assert ModelUseScenario.AGENT_CHAT.value == "agent_chat"


def test_model_create_supports_multimodal_embedding_flag_defaults_false():
    """Wave 5 P2 chunk 3 (per §G.2.5.1 spec amend item 3): the
    ``supports_multimodal_embedding`` capability flag defaults to
    False so existing model rows / pre-Wave-5 callers see no
    behaviour change."""
    model = ModelCreate(
        account_id="acct_1",
        provider_model_id="text-embedding-3-large",
        display_name="OpenAI text-embedding-3-large",
        capability=ModelCapability.EMBEDDING,
    )
    assert model.supports_multimodal_embedding is False


def test_model_create_supports_multimodal_embedding_flag_can_be_set():
    """Operators register a real multimodal embedder (Voyage Multimodal /
    CLIP / Jina v3 / etc.) by setting this flag — it surfaces through
    the v2 ``/models`` route so a UI can render a checkbox."""
    model = ModelCreate(
        account_id="acct_1",
        provider_model_id="voyage-multimodal-3",
        display_name="Voyage Multimodal 3",
        capability=ModelCapability.EMBEDDING,
        supports_multimodal_embedding=True,
    )
    assert model.supports_multimodal_embedding is True


def test_model_v2_openapi_exposes_supports_multimodal_embedding():
    """The v2 routes' OpenAPI schema must surface the new capability
    flag so the API contract stays discoverable for UI clients."""
    spec = _provider_v2_spec()
    schemas = spec["components"]["schemas"]
    # All three model schemas (read / create / update) must carry
    # the field so a UI can read + modify it consistently.
    for schema_name in ("Model", "ModelCreate", "ModelUpdate"):
        assert schema_name in schemas, f"{schema_name} schema missing from v2 OpenAPI"
        properties = schemas[schema_name].get("properties", {})
        assert "supports_multimodal_embedding" in properties, (
            f"{schema_name} missing supports_multimodal_embedding capability flag"
        )


def test_model_v2_openapi_exposes_allowed_scenarios():
    """Scenario allowlists are an API-level model property even though the
    MVP stores them under ``Model.extra`` for DB compatibility."""
    spec = _provider_v2_spec()
    schemas = spec["components"]["schemas"]

    for schema_name in ("Model", "ModelCreate", "ModelUpdate"):
        assert schema_name in schemas, f"{schema_name} schema missing from v2 OpenAPI"
        properties = schemas[schema_name].get("properties", {})
        assert "allowed_scenarios" in properties, f"{schema_name} missing allowed_scenarios"

    parameters = spec["paths"]["/api/v2/models"]["get"].get("parameters", [])
    assert any(parameter.get("name") == "scenario" for parameter in parameters)
    assert any(parameter.get("name") == "capability" for parameter in parameters)


def test_allowed_scenarios_default_by_capability():
    assert default_allowed_scenarios(ModelCapability.CHAT) == [
        ModelUseScenario.AGENT_CHAT,
        ModelUseScenario.COLLECTION_COMPLETION,
        ModelUseScenario.BACKGROUND_TASK,
    ]
    assert default_allowed_scenarios(ModelCapability.EMBEDDING) == [
        ModelUseScenario.COLLECTION_EMBEDDING,
    ]
    assert default_allowed_scenarios(ModelCapability.RERANK) == []


def test_allowed_scenarios_missing_key_uses_default_but_empty_list_is_explicit():
    model_without_key = SimpleNamespace(capability=ModelCapability.CHAT, extra={"owner": "kept"})
    model_with_empty_allowlist = SimpleNamespace(
        capability=ModelCapability.CHAT,
        extra={"allowed_scenarios": []},
    )

    assert allowed_scenarios_for_model(model_without_key) == [
        ModelUseScenario.AGENT_CHAT,
        ModelUseScenario.COLLECTION_COMPLETION,
        ModelUseScenario.BACKGROUND_TASK,
    ]
    assert allowed_scenarios_for_model(model_with_empty_allowlist) == []


def test_model_create_and_update_accept_allowed_scenarios_without_extra_coupling():
    create = ModelCreate(
        account_id="acct_1",
        provider_model_id="qwen/qwen3-30b-a3b-instruct-2507",
        display_name="Qwen3 30B",
        capability=ModelCapability.CHAT,
        allowed_scenarios=[ModelUseScenario.COLLECTION_COMPLETION],
    )
    update = ModelUpdate(allowed_scenarios=[ModelUseScenario.BACKGROUND_TASK])

    assert create.model_dump()["allowed_scenarios"] == [ModelUseScenario.COLLECTION_COMPLETION]
    assert update.model_dump(exclude_unset=True)["allowed_scenarios"] == [ModelUseScenario.BACKGROUND_TASK]
    assert "allowed_scenarios" not in create.extra


def test_allowed_scenarios_patch_preserves_other_extra_keys():
    extra = _model_extra_with_allowed_scenarios(
        {"owner": "ops", "notes": {"tier": "prod"}},
        ModelCapability.CHAT,
        [ModelUseScenario.COLLECTION_COMPLETION],
    )

    assert extra["owner"] == "ops"
    assert extra["notes"] == {"tier": "prod"}
    assert extra["allowed_scenarios"] == [ModelUseScenario.COLLECTION_COMPLETION.value]
