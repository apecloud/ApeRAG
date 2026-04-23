import re

from fastapi import FastAPI

from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi
from aperag.views.evaluation_v2 import router


def _evaluation_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def _json_ref(spec: dict, path: str, method: str, status: str = "200") -> str:
    return spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]["$ref"]


REQUIRED_PATHS = (
    "/api/v2/evaluation-datasets",
    "/api/v2/evaluation-datasets/{dataset_id}",
    "/api/v2/evaluation-datasets/{dataset_id}/items",
    "/api/v2/evaluation-datasets/{dataset_id}/items/{item_id}",
    "/api/v2/evaluation-runs",
    "/api/v2/evaluation-runs/{run_id}",
    "/api/v2/evaluation-runs/{run_id}/cancel",
    "/api/v2/evaluation-runs/{run_id}/items",
    "/api/v2/evaluation-runs/{run_id}/items/{item_id}/attempts",
    "/api/v2/evaluation-runs/{run_id}/items/{item_id}/retry",
)


def test_evaluation_v2_routes_are_public_and_typed():
    spec = _evaluation_v2_spec()
    paths = spec["paths"]

    for p in REQUIRED_PATHS:
        assert p in paths, f"missing public path {p}"

    assert _json_ref(spec, "/api/v2/evaluation-datasets", "post") == "#/components/schemas/EvaluationDatasetEnvelope"
    assert _json_ref(spec, "/api/v2/evaluation-datasets", "get") == "#/components/schemas/EvaluationDatasetListResponse"
    assert (
        _json_ref(spec, "/api/v2/evaluation-datasets/{dataset_id}/items", "get")
        == "#/components/schemas/EvaluationDatasetItemListResponse"
    )
    assert (
        _json_ref(spec, "/api/v2/evaluation-datasets/{dataset_id}/items", "post")
        == "#/components/schemas/EvaluationDatasetItemsAppendResponse"
    )
    assert _json_ref(spec, "/api/v2/evaluation-runs", "post") == "#/components/schemas/EvaluationRunEnvelope"
    assert (
        _json_ref(spec, "/api/v2/evaluation-runs/{run_id}", "get") == "#/components/schemas/EvaluationRunDetailResponse"
    )
    assert (
        _json_ref(spec, "/api/v2/evaluation-runs/{run_id}/cancel", "post") == "#/components/schemas/CancelRunResponse"
    )
    assert (
        _json_ref(spec, "/api/v2/evaluation-runs/{run_id}/items/{item_id}/retry", "post")
        == "#/components/schemas/EvaluationRunItemEnvelope"
    )


def test_evaluation_v2_operation_ids_are_unique():
    spec = _evaluation_v2_spec()
    op_ids: list[str] = []
    for methods in spec["paths"].values():
        for operation in methods.values():
            if isinstance(operation, dict) and operation.get("operationId"):
                op_ids.append(operation["operationId"])
    assert len(op_ids) == len(set(op_ids))


def test_full_app_spec_has_no_v1_evaluation_or_question_set_paths():
    from aperag.app import app

    full_spec = build_full_openapi_spec(app)
    public_spec = filter_public_openapi(full_spec)

    for spec_name, spec in (("full", full_spec), ("public", public_spec)):
        ghost = sorted(
            path
            for path in spec["paths"]
            if path.startswith("/api/v1/evaluations") or path.startswith("/api/v1/question-sets")
        )
        assert not ghost, f"{spec_name} spec must not expose v1 evaluation/question-set paths; found {ghost}"


def test_evaluation_v2_benchmark_and_version_paths_removed():
    """Negative gate: the simplified evaluation API must not leak the retired
    ``Benchmark`` / ``Dataset Version`` concepts into the public contract."""

    spec = _evaluation_v2_spec()
    leaked = sorted(
        path for path in spec["paths"] if path.startswith("/api/v2/benchmark-datasets") or "/versions" in path
    )
    assert not leaked, f"evaluation_v2 router must not expose benchmark/version paths; found {leaked}"

    components = spec.get("components", {}).get("schemas", {}) or {}
    forbidden_schemas = sorted(name for name in components if name.startswith("Benchmark"))
    assert not forbidden_schemas, f"evaluation_v2 spec must not export Benchmark* schemas; found {forbidden_schemas}"


def test_evaluation_run_create_requires_dataset_id_allows_optional_bot():
    """``EvaluationRunCreate`` pins the simplified run contract:

    - ``dataset_id`` is required;
    - ``bot_id`` is optional (service resolves the default bot);
    - legacy ``dataset_version_id`` must not appear.
    """

    spec = _evaluation_v2_spec()
    components = spec["components"]["schemas"]
    request_ref = spec["paths"]["/api/v2/evaluation-runs"]["post"]["requestBody"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    schema = components[request_ref.removeprefix("#/components/schemas/")]
    properties = schema.get("properties", {})
    required = set(schema.get("required", []) or [])

    assert "dataset_id" in properties
    assert "dataset_id" in required
    assert "bot_id" in properties
    assert "bot_id" not in required
    assert "dataset_version_id" not in properties


def test_evaluation_v2_delete_dataset_has_no_json_body():
    spec = _evaluation_v2_spec()
    responses = spec["paths"]["/api/v2/evaluation-datasets/{dataset_id}"]["delete"]["responses"]

    assert "204" in responses, "DELETE /evaluation-datasets/{dataset_id} must expose a 204 response"
    assert "200" not in responses, "DELETE /evaluation-datasets/{dataset_id} must not declare a 200 response"

    success_content = (responses["204"] or {}).get("content") or {}
    assert "application/json" not in success_content, (
        "DELETE /evaluation-datasets/{dataset_id} 204 response must not carry an application/json body"
    )


def test_evaluation_v2_all_write_request_bodies_omit_path_params():
    """Every POST/PUT/PATCH request body under ``evaluation_v2`` must not
    redeclare any path parameter as a body property."""

    spec = _evaluation_v2_spec()
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

    assert checked >= 2, (
        f"Expected at least 2 write routes with request bodies under evaluation_v2 router, but only inspected {checked}"
    )
