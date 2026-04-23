from fastapi import FastAPI

from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi
from aperag.views.providers_v2 import router


def _provider_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def _json_ref(spec: dict, path: str, method: str, status: str = "200") -> str:
    return spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]["$ref"]


def test_provider_v2_routes_are_public_and_typed():
    spec = _provider_v2_spec()
    paths = spec["paths"]

    assert "/api/v2/providers/configuration" in paths
    assert "/api/v2/providers/available-models" in paths
    assert "/api/v2/default-models" in paths
    assert "/api/v2/providers" in paths
    assert "/api/v2/providers/{provider_name}" in paths
    assert "/api/v2/providers/{provider_name}/models" in paths
    assert "/api/v2/providers/{provider_name}/models/{api}/{model}" in paths

    assert _json_ref(spec, "/api/v2/providers/configuration", "get") == "#/components/schemas/LlmConfigurationResponse"
    assert _json_ref(spec, "/api/v2/providers/available-models", "post") == "#/components/schemas/ModelConfigList"
    assert _json_ref(spec, "/api/v2/default-models", "get") == "#/components/schemas/DefaultModelsResponse"
    assert _json_ref(spec, "/api/v2/default-models", "put") == "#/components/schemas/DefaultModelsResponse"
    assert _json_ref(spec, "/api/v2/providers", "post") == "#/components/schemas/LlmProvider"
    assert (
        _json_ref(spec, "/api/v2/providers/{provider_name}/models", "post") == "#/components/schemas/LlmProviderModel"
    )


def test_provider_v2_model_create_request_uses_path_provider():
    spec = _provider_v2_spec()

    request_ref = spec["paths"]["/api/v2/providers/{provider_name}/models"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    request_schema_name = request_ref.removeprefix("#/components/schemas/")
    request_schema = spec["components"]["schemas"][request_schema_name]

    assert request_schema_name == "LlmProviderModelCreateRequest"
    assert "provider_name" not in request_schema["properties"]
