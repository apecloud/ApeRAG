import re

from fastapi import FastAPI

from aperag.domains.model_platform.api.providers_v2_routes import router
from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi


def _provider_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def _json_ref(spec: dict, path: str, method: str, status: str = "200") -> str:
    return spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]["$ref"]


# v1 LLM provider / default-models / available-models routes are no longer mounted on main —
# the #26 final sweep is complete and only OpenAI-compat ``/api/v1/embeddings`` + ``/api/v1/rerank``
# remain under ``/api/v1`` (those live in ``llm_routes.py`` and are intentionally outside this
# baseline because they are the OpenAI-compat permanent allowlist, not v1 ghosts).
#
# Baseline is empty: the test now strictly forbids regrowth of any internal v1 LLM/provider
# CRUD paths (the OpenAI-compat allowlist is excluded by the path filter in
# ``_provider_v1_ghosts`` below — it only matches ``/api/v1/llm_*`` plus the two specific
# legacy default/available-models paths).
PROVIDER_V1_GHOST_PATHS: frozenset[str] = frozenset()


def _provider_v1_ghosts(spec: dict) -> set[str]:
    return {
        p
        for p in spec["paths"]
        if p.startswith("/api/v1/llm_") or p in {"/api/v1/default_models", "/api/v1/available_models"}
    }


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


def test_provider_v1_ghost_inventory_is_stable():
    """v1 provider ghost paths in the exported full+public spec must stay within the pinned baseline.

    Purpose is to prevent accidental regrowth: the #26 final sweep is expected to remove entries
    from :data:`PROVIDER_V1_GHOST_PATHS` (set will shrink), but nothing in future refactors should
    add new v1 llm/default-model/available-model routes.
    """
    from aperag.app import app

    full_spec = build_full_openapi_spec(app)
    public_spec = filter_public_openapi(full_spec)

    for spec_name, spec in (("full", full_spec), ("public", public_spec)):
        current = _provider_v1_ghosts(spec)
        unexpected = current - PROVIDER_V1_GHOST_PATHS
        assert not unexpected, (
            f"{spec_name} spec introduced new v1 provider ghost path(s) {sorted(unexpected)}; "
            "update PROVIDER_V1_GHOST_PATHS only if intentional"
        )


def test_provider_v2_delete_routes_contract():
    """Every DELETE route under providers_v2 must respect the command-vs-report contract:

    - must not declare both 200 and 204 success responses
    - if 204 is declared it must have no application/json body (pure command)
    """
    spec = _provider_v2_spec()
    checked = 0
    for path, operations in spec["paths"].items():
        op = operations.get("delete")
        if not op:
            continue
        responses = op.get("responses") or {}
        assert not ("200" in responses and "204" in responses), (
            f"DELETE {path} must not mix 200 and 204 success responses; pick one"
        )
        if "204" in responses:
            assert "content" not in (responses["204"] or {}), (
                f"DELETE {path} 204 response must not carry an application/json body"
            )
        else:
            assert "200" in responses, f"DELETE {path} must declare a 200 or 204 success response"
        checked += 1
    assert checked >= 1, "providers_v2 should expose at least one DELETE route to exercise this contract"


def test_provider_v2_all_write_request_bodies_omit_path_params():
    """Every POST/PUT/PATCH request body under providers_v2 must not redeclare path params."""
    spec = _provider_v2_spec()
    components = spec["components"]["schemas"]
    path_param_re = re.compile(r"\{([^{}]+)\}")

    checked = 0
    for path, methods in spec["paths"].items():
        path_params = set(path_param_re.findall(path))
        if not path_params:
            continue
        for method, operation in methods.items():
            if method not in {"post", "put", "patch"}:
                continue
            request_body = (operation or {}).get("requestBody") or {}
            json_schema = request_body.get("content", {}).get("application/json", {}).get("schema") or {}
            ref = json_schema.get("$ref")
            if not ref:
                continue
            schema_name = ref.removeprefix("#/components/schemas/")
            properties = set(components[schema_name].get("properties", {}).keys())
            overlap = path_params & properties
            assert not overlap, (
                f"{method.upper()} {path} request body {schema_name} duplicates path param(s) {sorted(overlap)}"
            )
            checked += 1

    assert checked >= 1, (
        f"Expected at least 1 write route with request body under providers_v2, but only inspected {checked}"
    )
