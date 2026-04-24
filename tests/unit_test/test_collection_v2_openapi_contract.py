import re

from fastapi import FastAPI

from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi
from aperag.views.collections_v2 import router


def _collection_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def _json_ref(spec: dict, path: str, method: str, status: str = "200") -> str:
    return spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]["$ref"]


# v1 collection CRUD/sharing ghost paths still wired in main pending #26 final sweep.
# Strictly collection-scoped (no /documents, /graphs, /searches, /export, /marketplace —
# those belong to other domain contract tests).
COLLECTION_V1_GHOST_PATHS = frozenset(
    {
        "/api/v1/collections",
        "/api/v1/collections/test-mineru-token",
        "/api/v1/collections/{collection_id}",
        "/api/v1/collections/{collection_id}/sharing",
        "/api/v1/collections/{collection_id}/summary/generate",
    }
)


def _collection_v1_ghosts(spec: dict) -> set[str]:
    return {p for p in spec["paths"] if p in COLLECTION_V1_GHOST_PATHS}


REQUIRED_PATHS = {
    "/api/v2/collections",
    "/api/v2/collections/{collection_id}",
    "/api/v2/collections/test-mineru-token",
    "/api/v2/collections/{collection_id}/summary/generate",
    "/api/v2/collections/{collection_id}/sharing",
}

# Non-204 JSON routes that must have a $ref response schema.
JSON_ROUTES = [
    ("/api/v2/collections", "post", "Collection"),
    ("/api/v2/collections", "get", "CollectionViewList"),
    ("/api/v2/collections/{collection_id}", "get", "Collection"),
    ("/api/v2/collections/{collection_id}", "put", "Collection"),
    ("/api/v2/collections/test-mineru-token", "post", "MineruTokenTestResponse"),
    (
        "/api/v2/collections/{collection_id}/summary/generate",
        "post",
        "CollectionSummaryTriggerResponse",
    ),
    ("/api/v2/collections/{collection_id}/sharing", "get", "SharingStatusResponse"),
]

# Command-style routes that must respond with 204 and have no JSON body.
NO_CONTENT_ROUTES = [
    ("/api/v2/collections/{collection_id}", "delete"),
    ("/api/v2/collections/{collection_id}/sharing", "post"),
    ("/api/v2/collections/{collection_id}/sharing", "delete"),
]


def test_collection_v2_public_paths_present():
    spec = _collection_v2_spec()
    paths = spec["paths"]
    for p in REQUIRED_PATHS:
        assert p in paths, f"missing public path: {p}"


def test_collection_v2_json_routes_use_explicit_response_schema():
    spec = _collection_v2_spec()
    for path, method, schema_name in JSON_ROUTES:
        ref = _json_ref(spec, path, method)
        assert ref == f"#/components/schemas/{schema_name}", (
            f"{method.upper()} {path} expected $ref to {schema_name}, got {ref}"
        )


def test_collection_v2_command_routes_return_204_without_body():
    spec = _collection_v2_spec()
    for path, method in NO_CONTENT_ROUTES:
        responses = spec["paths"][path][method]["responses"]
        assert "204" in responses, f"{method.upper()} {path} must declare 204 response"
        # 204 responses carry no content body.
        assert "content" not in responses["204"], f"{method.upper()} {path} 204 response must not carry a JSON body"
        # Command routes must not register a competing 200 JSON schema.
        assert "200" not in responses, f"{method.upper()} {path} must not mix 204 with a 200 JSON response"


def test_collection_v2_write_bodies_do_not_repeat_collection_id():
    """Write routes taking {collection_id} path param must not repeat it in the body."""
    spec = _collection_v2_spec()
    # Collect every write method that takes `collection_id` as a path param.
    offending = []
    for path, operations in spec["paths"].items():
        if "{collection_id}" not in path:
            continue
        for method, op in operations.items():
            if method not in {"post", "put", "patch"}:
                continue
            body = op.get("requestBody")
            if not body:
                continue
            json_schema = body.get("content", {}).get("application/json", {}).get("schema", {})
            ref = json_schema.get("$ref")
            if not ref:
                continue
            schema_name = ref.removeprefix("#/components/schemas/")
            props = spec["components"]["schemas"][schema_name].get("properties", {})
            if "collection_id" in props:
                offending.append((method.upper(), path, schema_name))
    assert not offending, f"write bodies must not repeat collection_id: {offending}"


def test_collection_v2_does_not_occupy_search_or_graph():
    """Retrieval ``/searches`` and knowledge_graph ``/graphs`` routes
    live in their own domain routers — the KB router owns collections
    and documents only.

    Phase 3 Step 5a merged ``collections_v2`` and ``documents_v2`` into
    a single KB domain router at
    ``aperag.domains.knowledge_base.api.routes.router`` (imported here
    through the ``aperag.views.collections_v2`` shim), so ``/documents``
    paths are now legitimate on this router; the former
    ``/documents`` exclusion has been dropped accordingly.
    """
    spec = _collection_v2_spec()
    for path in spec["paths"]:
        assert "/searches" not in path, f"KB router must not own search path: {path}"
        assert "/graphs" not in path, f"KB router must not own graph path: {path}"


def test_collection_v2_no_duplicate_operation_ids():
    spec = _collection_v2_spec()
    op_ids = []
    for operations in spec["paths"].values():
        for method, op in operations.items():
            if method in {"parameters", "summary", "description"}:
                continue
            op_id = op.get("operationId")
            if op_id is not None:
                op_ids.append(op_id)
    assert len(op_ids) == len(set(op_ids)), f"duplicate operationIds: {op_ids}"


def test_collection_v2_mineru_token_request_schema_is_typed():
    """Mineru token request must be a typed schema, not a naked dict."""
    spec = _collection_v2_spec()
    req = spec["paths"]["/api/v2/collections/test-mineru-token"]["post"]["requestBody"]
    ref = req["content"]["application/json"]["schema"]["$ref"]
    schema_name = ref.removeprefix("#/components/schemas/")
    assert schema_name == "MineruTokenTestRequest"
    props = spec["components"]["schemas"][schema_name]["properties"]
    assert "token" in props


def test_collection_v1_ghost_inventory_is_stable():
    """v1 collection CRUD/sharing ghost paths in the full+public spec must stay within the pinned baseline.

    The #26 final sweep is expected to remove entries from :data:`COLLECTION_V1_GHOST_PATHS`
    (set will shrink); nothing should introduce new v1 collection CRUD/sharing routes.
    """
    from aperag.app import app

    full_spec = build_full_openapi_spec(app)
    public_spec = filter_public_openapi(full_spec)

    for spec_name, spec in (("full", full_spec), ("public", public_spec)):
        current = _collection_v1_ghosts(spec)
        unexpected = current - COLLECTION_V1_GHOST_PATHS
        assert not unexpected, (
            f"{spec_name} spec introduced new v1 collection ghost path(s) {sorted(unexpected)}; "
            "update COLLECTION_V1_GHOST_PATHS only if intentional"
        )


def test_collection_v2_delete_routes_contract():
    """Every DELETE route under collections_v2 must respect the command-vs-report contract:

    - must not declare both 200 and 204 success responses
    - if 204 is declared it must have no application/json body (pure command)

    Generalizes ``test_collection_v2_command_routes_return_204_without_body`` to cover every
    DELETE route (new ones appear automatically).
    """
    spec = _collection_v2_spec()
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
    assert checked >= 1, "collections_v2 should expose at least one DELETE route to exercise this contract"


def test_collection_v2_all_write_request_bodies_omit_path_params():
    """Every POST/PUT/PATCH request body under collections_v2 must not redeclare path params.

    Generalizes ``test_collection_v2_write_bodies_do_not_repeat_collection_id`` to all path params
    (not just ``collection_id``) so future routes with additional path ids are covered automatically.
    """
    spec = _collection_v2_spec()
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
        f"Expected at least 1 write route with request body under collections_v2, but only inspected {checked}"
    )
