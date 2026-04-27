from fastapi import FastAPI

from aperag.domains.model_platform.api.providers_v3_routes import router
from aperag.domains.model_platform.schemas import (
    ModelAccountCreate,
    ModelCapability,
    ModelCreate,
    ModelProvider,
    ModelUseScenario,
)
from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi


def _provider_v3_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v3")
    return filter_public_openapi(build_full_openapi_spec(app))


def test_model_platform_v3_routes_use_product_language():
    spec = _provider_v3_spec()
    paths = spec["paths"]

    assert "/api/v3/model-providers" in paths
    assert "/api/v3/model-accounts" in paths
    assert "/api/v3/model-accounts/{account_id}/validate" in paths
    assert "/api/v3/model-accounts/{account_id}/models" in paths
    assert "/api/v3/models" in paths
    assert "/api/v3/models/{model_id}" in paths
    assert "/api/v3/models/{model_id}/validate" in paths
    assert "/api/v3/model-uses" in paths
    assert "/api/v3/model-uses/{scenario}" in paths


def test_model_platform_v3_schemas_do_not_expose_litellm_dialects():
    spec = _provider_v3_spec()
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
    the v3 ``/models`` route so a UI can render a checkbox."""
    model = ModelCreate(
        account_id="acct_1",
        provider_model_id="voyage-multimodal-3",
        display_name="Voyage Multimodal 3",
        capability=ModelCapability.EMBEDDING,
        supports_multimodal_embedding=True,
    )
    assert model.supports_multimodal_embedding is True


def test_model_v3_openapi_exposes_supports_multimodal_embedding():
    """The v3 routes' OpenAPI schema must surface the new capability
    flag so the API contract stays discoverable for UI clients."""
    spec = _provider_v3_spec()
    schemas = spec["components"]["schemas"]
    # All three model schemas (read / create / update) must carry
    # the field so a UI can read + modify it consistently.
    for schema_name in ("Model", "ModelCreate", "ModelUpdate"):
        assert schema_name in schemas, f"{schema_name} schema missing from v3 OpenAPI"
        properties = schemas[schema_name].get("properties", {})
        assert "supports_multimodal_embedding" in properties, (
            f"{schema_name} missing supports_multimodal_embedding capability flag"
        )
