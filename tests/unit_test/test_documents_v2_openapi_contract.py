from fastapi import FastAPI

from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi
from aperag.views.documents_v2 import router


def _documents_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def _json_ref(spec: dict, path: str, method: str, status: str = "200") -> str:
    return spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]["$ref"]


def _request_schema(spec: dict, path: str, method: str) -> dict:
    request_ref = spec["paths"][path][method]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    return spec["components"]["schemas"][request_ref.removeprefix("#/components/schemas/")]


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
