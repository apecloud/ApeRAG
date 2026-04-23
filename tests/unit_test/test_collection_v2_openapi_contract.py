from fastapi import FastAPI

from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi
from aperag.views.collections_v2 import router


def _collection_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def _json_ref(spec: dict, path: str, method: str, status: str = "200") -> str:
    return spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]["$ref"]


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
        assert "content" not in responses["204"], (
            f"{method.upper()} {path} 204 response must not carry a JSON body"
        )
        # Command routes must not register a competing 200 JSON schema.
        assert "200" not in responses, (
            f"{method.upper()} {path} must not mix 204 with a 200 JSON response"
        )


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


def test_collection_v2_does_not_occupy_documents_or_search_or_graph():
    """#11a scope must leave documents/search/graph routes to #11b / #12 / #13."""
    spec = _collection_v2_spec()
    for path in spec["paths"]:
        assert "/documents" not in path, f"collections_v2 must not own documents path: {path}"
        assert "/searches" not in path, f"collections_v2 must not own search path: {path}"
        assert "/graphs" not in path, f"collections_v2 must not own graph path: {path}"


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
