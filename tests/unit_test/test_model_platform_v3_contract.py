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
