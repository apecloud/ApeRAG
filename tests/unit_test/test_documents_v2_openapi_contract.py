import re

from fastapi import FastAPI

from aperag.domains.knowledge_base.api.routes import router
from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi


def _documents_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def _json_ref(spec: dict, path: str, method: str, status: str = "200") -> str:
    return spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]["$ref"]


def _request_schema(spec: dict, path: str, method: str) -> dict:
    request_ref = spec["paths"][path][method]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    return spec["components"]["schemas"][request_ref.removeprefix("#/components/schemas/")]


# v1 documents ghost paths still wired in main pending #26 final sweep.
# Strictly document-scoped subpaths under /api/v2/collections/{collection_id}/...
DOCUMENTS_V1_GHOST_PATHS = frozenset(
    {
        "/api/v2/collections/{collection_id}/documents",
        "/api/v2/collections/{collection_id}/documents/confirm",
        "/api/v2/collections/{collection_id}/documents/fetch-url",
        "/api/v2/collections/{collection_id}/documents/staged",
        "/api/v2/collections/{collection_id}/documents/upload",
        "/api/v2/collections/{collection_id}/documents/{document_id}",
        "/api/v2/collections/{collection_id}/documents/{document_id}/download",
        "/api/v2/collections/{collection_id}/documents/{document_id}/object",
        "/api/v2/collections/{collection_id}/documents/{document_id}/preview",
        "/api/v2/collections/{collection_id}/documents/{document_id}/rebuild_indexes",
        "/api/v2/collections/{collection_id}/rebuild_failed_indexes",
    }
)


def _documents_v1_ghosts(spec: dict) -> set[str]:
    return {p for p in spec["paths"] if p in DOCUMENTS_V1_GHOST_PATHS}


def test_documents_v2_routes_are_public_and_typed():
    spec = _documents_v2_spec()
    paths = spec["paths"]

    required_paths = {
        "/api/v2/collections/{collection_id}/documents",
        "/api/v2/collections/{collection_id}/documents/staged",
        "/api/v2/collections/{collection_id}/documents/{document_id}",
        "/api/v2/collections/{collection_id}/documents/{document_id}/download",
        "/api/v2/collections/{collection_id}/documents/{document_id}/preview",
        "/api/v2/collections/{collection_id}/documents/{document_id}/object",
        "/api/v2/collections/{collection_id}/documents/{document_id}/rebuild_indexes",
        "/api/v2/collections/{collection_id}/rebuild_failed_indexes",
        "/api/v2/collections/{collection_id}/documents/upload",
        "/api/v2/collections/{collection_id}/documents/confirm",
        "/api/v2/collections/{collection_id}/documents/fetch-url",
    }

    assert required_paths <= set(paths)
    assert not any("/graph" in path or "/searches" in path for path in paths)

    assert _json_ref(spec, "/api/v2/collections/{collection_id}/documents", "post") == (
        "#/components/schemas/DocumentList"
    )
    assert _json_ref(spec, "/api/v2/collections/{collection_id}/documents", "get") == (
        "#/components/schemas/DocumentList"
    )
    assert _json_ref(spec, "/api/v2/collections/{collection_id}/documents/staged", "get") == (
        "#/components/schemas/StagedDocumentsResponse"
    )
    assert _json_ref(spec, "/api/v2/collections/{collection_id}/documents/{document_id}", "get") == (
        "#/components/schemas/Document"
    )
    assert _json_ref(spec, "/api/v2/collections/{collection_id}/documents", "delete") == (
        "#/components/schemas/DeleteDocumentsResponse"
    )
    assert _json_ref(spec, "/api/v2/collections/{collection_id}/documents/{document_id}/preview", "get") == (
        "#/components/schemas/DocumentPreview"
    )
    assert _json_ref(spec, "/api/v2/collections/{collection_id}/documents/{document_id}/rebuild_indexes", "post") == (
        "#/components/schemas/RebuildIndexesResponse"
    )
    assert _json_ref(spec, "/api/v2/collections/{collection_id}/rebuild_failed_indexes", "post") == (
        "#/components/schemas/RebuildIndexesResponse"
    )
    assert _json_ref(spec, "/api/v2/collections/{collection_id}/documents/upload", "post") == (
        "#/components/schemas/UploadDocumentResponse"
    )
    assert _json_ref(spec, "/api/v2/collections/{collection_id}/documents/confirm", "post") == (
        "#/components/schemas/ConfirmDocumentsResponse"
    )
    assert _json_ref(spec, "/api/v2/collections/{collection_id}/documents/fetch-url", "post") == (
        "#/components/schemas/FetchUrlResponse"
    )


def test_documents_v2_delete_and_request_bodies_are_path_canonical():
    spec = _documents_v2_spec()

    single_delete_responses = spec["paths"]["/api/v2/collections/{collection_id}/documents/{document_id}"]["delete"][
        "responses"
    ]
    assert "204" in single_delete_responses
    assert "200" not in single_delete_responses
    assert "content" not in single_delete_responses["204"]

    bulk_delete_schema = _request_schema(spec, "/api/v2/collections/{collection_id}/documents", "delete")
    assert "document_ids" in bulk_delete_schema["properties"]
    assert "collection_id" not in bulk_delete_schema["properties"]
    assert "document_id" not in bulk_delete_schema["properties"]

    rebuild_schema = _request_schema(
        spec,
        "/api/v2/collections/{collection_id}/documents/{document_id}/rebuild_indexes",
        "post",
    )
    assert "index_types" in rebuild_schema["properties"]
    assert "collection_id" not in rebuild_schema["properties"]
    assert "document_id" not in rebuild_schema["properties"]

    confirm_schema = _request_schema(spec, "/api/v2/collections/{collection_id}/documents/confirm", "post")
    assert "document_ids" in confirm_schema["properties"]
    assert "collection_id" not in confirm_schema["properties"]

    fetch_url_schema = _request_schema(spec, "/api/v2/collections/{collection_id}/documents/fetch-url", "post")
    assert "urls" in fetch_url_schema["properties"]
    assert "collection_id" not in fetch_url_schema["properties"]


def test_documents_v2_operation_ids_are_unique():
    spec = _documents_v2_spec()

    operation_ids = [
        operation["operationId"]
        for path_item in spec["paths"].values()
        for operation in path_item.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]

    assert len(operation_ids) == len(set(operation_ids))


def test_documents_v1_ghost_inventory_is_stable():
    """v1 documents ghost paths in the full+public spec must stay within the pinned baseline.

    The #26 final sweep is expected to remove entries from :data:`DOCUMENTS_V1_GHOST_PATHS`
    (set will shrink); nothing should introduce new v1 documents routes.
    """
    from aperag.app import app

    full_spec = build_full_openapi_spec(app)
    public_spec = filter_public_openapi(full_spec)

    for spec_name, spec in (("full", full_spec), ("public", public_spec)):
        current = _documents_v1_ghosts(spec)
        unexpected = current - DOCUMENTS_V1_GHOST_PATHS
        assert not unexpected, (
            f"{spec_name} spec introduced new v1 documents ghost path(s) {sorted(unexpected)}; "
            "update DOCUMENTS_V1_GHOST_PATHS only if intentional"
        )


def test_documents_v2_delete_routes_contract():
    """Every DELETE route under documents_v2 must respect the command-vs-report contract.

    Generalizes the single-route assertion in
    ``test_documents_v2_delete_and_request_bodies_are_path_canonical`` so future DELETE routes
    are covered automatically. The contract is:

    - a DELETE route must not declare both 200 and 204 success responses
    - if 204 is declared it must have no application/json body (pure command)
    - a 200 JSON DELETE (e.g. bulk delete returning per-item status) is allowed
    """
    spec = _documents_v2_spec()
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
    assert checked >= 1, "documents_v2 should expose at least one DELETE route to exercise this contract"


def test_documents_v2_all_write_request_bodies_omit_path_params():
    """Every POST/PUT/PATCH request body under documents_v2 must not redeclare path params.

    Generalizes the hardcoded list in
    ``test_documents_v2_delete_and_request_bodies_are_path_canonical`` so new write routes are
    covered automatically.
    """
    spec = _documents_v2_spec()
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
        f"Expected at least 1 write route with request body under documents_v2, but only inspected {checked}"
    )
