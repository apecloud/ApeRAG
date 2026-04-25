from fastapi import FastAPI

from aperag.domains.governance.api.quota_routes import router
from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi

REQUIRED_QUOTA_PATHS = {
    "/api/v2/quotas",
    "/api/v2/quotas/{user_id}",
    "/api/v2/quotas/{user_id}/recalculate",
    "/api/v2/system/default-quotas",
}


def _quota_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def _schema_refs(schema: dict) -> set[str]:
    refs: set[str] = set()
    if "$ref" in schema:
        refs.add(schema["$ref"])
    for value in schema.values():
        if isinstance(value, dict):
            refs |= _schema_refs(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    refs |= _schema_refs(item)
    return refs


def _json_schema(spec: dict, path: str, method: str, status: str = "200") -> dict:
    return spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]


def _request_schema(spec: dict, path: str, method: str) -> dict:
    return spec["paths"][path][method]["requestBody"]["content"]["application/json"]["schema"]


def test_quota_v2_public_paths_present_and_v1_paths_absent():
    spec = _quota_v2_spec()
    paths = set(spec["paths"])

    assert REQUIRED_QUOTA_PATHS <= paths
    assert "/api/v1/quotas" not in paths
    assert "/api/v1/quotas/{user_id}" not in paths
    assert "/api/v1/quotas/{user_id}/recalculate" not in paths
    assert "/api/v1/system/default-quotas" not in paths


def test_quota_v2_operation_ids_preserve_quotas_tag():
    spec = _quota_v2_spec()

    assert spec["paths"]["/api/v2/quotas"]["get"]["operationId"] == "quotas_get_quotas"
    assert spec["paths"]["/api/v2/quotas/{user_id}"]["put"]["operationId"] == "quotas_update_quota"
    assert (
        spec["paths"]["/api/v2/quotas/{user_id}/recalculate"]["post"]["operationId"] == "quotas_recalculate_quota_usage"
    )
    assert spec["paths"]["/api/v2/system/default-quotas"]["get"]["operationId"] == "quotas_get_system_default_quotas"
    assert spec["paths"]["/api/v2/system/default-quotas"]["put"]["operationId"] == "quotas_update_system_default_quotas"


def test_quota_v2_response_and_request_schemas_are_governance_owned():
    spec = _quota_v2_spec()

    quota_get_refs = _schema_refs(_json_schema(spec, "/api/v2/quotas", "get"))
    assert "#/components/schemas/UserQuotaInfo" in quota_get_refs
    assert "#/components/schemas/UserQuotaList" in quota_get_refs

    assert _request_schema(spec, "/api/v2/quotas/{user_id}", "put") == {
        "$ref": "#/components/schemas/QuotaUpdateRequest"
    }
    assert _json_schema(spec, "/api/v2/quotas/{user_id}", "put") == {"$ref": "#/components/schemas/QuotaUpdateResponse"}

    assert _json_schema(spec, "/api/v2/system/default-quotas", "get") == {
        "$ref": "#/components/schemas/SystemDefaultQuotasResponse"
    }
    assert _request_schema(spec, "/api/v2/system/default-quotas", "put") == {
        "$ref": "#/components/schemas/SystemDefaultQuotasUpdateRequest"
    }
    assert _json_schema(spec, "/api/v2/system/default-quotas", "put") == {
        "$ref": "#/components/schemas/SystemDefaultQuotasUpdateResponse"
    }


def test_live_app_quota_system_contract_is_v2_only():
    from aperag.app import app

    spec = filter_public_openapi(build_full_openapi_spec(app))
    paths = set(spec["paths"])

    assert REQUIRED_QUOTA_PATHS <= paths
    assert "/api/v1/quotas" not in paths
    assert "/api/v1/system/default-quotas" not in paths
