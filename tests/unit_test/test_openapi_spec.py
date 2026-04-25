import pytest
from fastapi import APIRouter, FastAPI

from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi


def test_custom_operation_id_uses_first_tag_and_function_name():
    router = APIRouter(tags=["sample-tag"])

    @router.get("/items")
    async def list_items():
        return []

    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router)

    spec = build_full_openapi_spec(app)

    assert spec["paths"]["/items"]["get"]["operationId"] == "sample_tag_list_items"


def test_public_openapi_filters_internal_prefixes():
    spec = {
        "openapi": "3.1.0",
        "paths": {
            "/api/v1/collections": {},
            "/api/v2/audit-logs": {},
            "/api/v2/audit-logs/{audit_id}": {},
            "/api/v2/config": {},
            "/api/v2/agent/chats/{chat_id}/turns": {},
        },
    }

    public_spec = filter_public_openapi(spec)

    assert "/api/v1/collections" in public_spec["paths"]
    assert "/api/v2/audit-logs" not in public_spec["paths"]
    assert "/api/v2/audit-logs/{audit_id}" not in public_spec["paths"]
    assert "/api/v2/config" not in public_spec["paths"]
    assert "/api/v2/agent/chats/{chat_id}/turns" in public_spec["paths"]


def test_audit_logs_hidden_from_public_openapi_for_live_app():
    """Live-app contract: ``/api/v2/audit-logs*`` must never appear in the
    public OpenAPI spec.

    Phase 8 task #50 (G4a) hard-cut the audit-logs router from
    ``/api/v1/audit-logs`` to ``/api/v2/audit-logs`` per canonical D7-2,
    and the hidden filter prefix in ``HIDDEN_FROM_PUBLIC_PATH_PREFIXES``
    must be kept in sync. This regression-guard pins that promise so a
    future refactor cannot silently leak audit-log paths into the public
    SDK surface.
    """
    from aperag.app import app

    full_spec = build_full_openapi_spec(app)
    public_spec = filter_public_openapi(full_spec)

    leaked = [path for path in public_spec.get("paths", {}) if "audit-logs" in path]
    assert not leaked, f"audit-logs paths must stay hidden from public OpenAPI per D7-2; leaked: {leaked}"


def test_build_full_openapi_spec_rejects_duplicate_operation_ids():
    app = FastAPI(generate_unique_id_function=lambda route: "duplicate")
    router = APIRouter()

    @router.get("/one", tags=["test"])
    def one():
        return {"ok": True}

    @router.get("/two", tags=["test"])
    def two():
        return {"ok": True}

    app.include_router(router)

    with pytest.raises(ValueError, match="Duplicate OpenAPI operationId"):
        build_full_openapi_spec(app)
