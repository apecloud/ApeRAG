from __future__ import annotations

import ast
from pathlib import Path

DOCUMENT_SERVICE = Path("aperag/domains/knowledge_base/service/document_service.py")
APP_LIFESPAN = Path("aperag/app.py")


def _method_source(path: Path, *, class_name: str, method_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef | ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(source, item) or ""
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _function_source(path: Path, *, function_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{function_name} not found in {path}")


def _called_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        match node.func:
            case ast.Name(id=name):
                names.add(name)
            case ast.Attribute(attr=name):
                names.add(name)
    return names


def test_document_delete_request_path_has_no_heavy_cleanup_calls():
    method = _method_source(DOCUMENT_SERVICE, class_name="DocumentService", method_name="_delete_document")

    forbidden_calls = (
        "cleanup_for_deleted_documents",
        "_delete_document_indexes",
        "delete_objects_by_prefix",
        "get_async_object_store",
    )
    calls = _called_names(method)
    for call_name in forbidden_calls:
        assert call_name not in calls


def test_deleted_document_indexes_helper_was_removed():
    source = DOCUMENT_SERVICE.read_text()

    assert "async def _delete_document_indexes" not in source
    assert "def _delete_document_indexes" not in source


def test_api_lifespan_does_not_build_cleanup_worker_factory():
    source = APP_LIFESPAN.read_text()

    assert "cleanup_worker_factory=None" in source
    calls = _called_names(_function_source(APP_LIFESPAN, function_name="combined_lifespan"))
    assert "build_for_cleanup_row" not in calls
