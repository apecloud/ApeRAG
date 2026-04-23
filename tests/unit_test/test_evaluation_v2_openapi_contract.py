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
    "/api/v2/benchmark-datasets",
    "/api/v2/benchmark-datasets/{dataset_id}",
    "/api/v2/benchmark-datasets/{dataset_id}/versions",
    "/api/v2/benchmark-datasets/{dataset_id}/versions/{version_id}",
    "/api/v2/benchmark-datasets/{dataset_id}/versions/{version_id}/cases",
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

    assert (
        _json_ref(spec, "/api/v2/benchmark-datasets", "post")
        == "#/components/schemas/BenchmarkDatasetEnvelope"
    )
    assert (
        _json_ref(spec, "/api/v2/benchmark-datasets", "get")
        == "#/components/schemas/BenchmarkDatasetListResponse"
    )
    assert (
        _json_ref(spec, "/api/v2/benchmark-datasets/{dataset_id}/versions", "post")
        == "#/components/schemas/BenchmarkDatasetVersionEnvelope"
    )
    assert (
        _json_ref(spec, "/api/v2/evaluation-runs", "post") == "#/components/schemas/EvaluationRunEnvelope"
    )
    assert (
        _json_ref(spec, "/api/v2/evaluation-runs/{run_id}", "get")
        == "#/components/schemas/EvaluationRunDetailResponse"
    )
    assert (
        _json_ref(spec, "/api/v2/evaluation-runs/{run_id}/cancel", "post")
        == "#/components/schemas/CancelRunResponse"
    )
    assert (
        _json_ref(spec, "/api/v2/evaluation-runs/{run_id}/items/{item_id}/retry", "post")
        == "#/components/schemas/EvaluationRunItemEnvelope"
    )


def test_evaluation_v2_write_request_bodies_omit_path_params():
    spec = _evaluation_v2_spec()
    components = spec["components"]["schemas"]

    create_version_ref = spec["paths"]["/api/v2/benchmark-datasets/{dataset_id}/versions"]["post"][
        "requestBody"
    ]["content"]["application/json"]["schema"]["$ref"]
    create_version_schema = components[create_version_ref.removeprefix("#/components/schemas/")]
    assert "dataset_id" not in create_version_schema.get("properties", {})


def test_evaluation_v2_operation_ids_are_unique():
    spec = _evaluation_v2_spec()
    op_ids: list[str] = []
    for methods in spec["paths"].values():
        for operation in methods.values():
            if isinstance(operation, dict) and operation.get("operationId"):
                op_ids.append(operation["operationId"])
    assert len(op_ids) == len(set(op_ids))
