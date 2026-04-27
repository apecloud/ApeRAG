"""Code-first OpenAPI helpers for ApeRAG.

FastAPI routes and Pydantic models are the source of truth.  The helpers in
this module build exported OpenAPI artifacts for internal web clients and for
future public SDK consumers without relying on hand-maintained YAML specs.
"""

from __future__ import annotations

import copy
import inspect
import re
from collections import Counter
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute

# Paths under these prefixes stay available in the full internal spec but are
# intentionally removed from the public SDK/API spec.
HIDDEN_FROM_PUBLIC_PATH_PREFIXES: tuple[str, ...] = (
    "/api/v1/audit-logs",
    "/api/v1/config",
)


def _normalize_operation_part(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "default"


def custom_generate_unique_id(route: APIRoute) -> str:
    """Generate stable operationIds from route tag + function name.

    The previous YAML-first pipeline made operation naming a separate manual
    concern.  With code-first OpenAPI, route tags and function names become the
    canonical generated-client boundary, so the rule is explicit and testable.
    """

    tag = route.tags[0] if route.tags else "default"
    return f"{_normalize_operation_part(tag)}_{_normalize_operation_part(route.name)}"


def build_full_openapi_spec(app: FastAPI) -> dict[str, Any]:
    """Build the full internal OpenAPI schema from the live FastAPI app."""

    kwargs = {
        "title": app.title,
        "version": app.version,
        "openapi_version": app.openapi_version,
        "summary": app.summary,
        "description": app.description,
        "terms_of_service": app.terms_of_service,
        "contact": app.contact,
        "license_info": app.license_info,
        "routes": app.routes,
        "webhooks": app.webhooks.routes,
        "tags": app.openapi_tags,
        "servers": app.servers,
        "separate_input_output_schemas": app.separate_input_output_schemas,
        "external_docs": getattr(app, "openapi_external_docs", None),
    }
    accepted = set(inspect.signature(get_openapi).parameters)
    spec = get_openapi(**{key: value for key, value in kwargs.items() if key in accepted})
    _assert_unique_operation_ids(spec)
    return spec


def _assert_unique_operation_ids(spec: dict[str, Any]) -> None:
    operation_ids: list[str] = []
    for methods in (spec.get("paths") or {}).values():
        for operation in methods.values():
            if isinstance(operation, dict) and operation.get("operationId"):
                operation_ids.append(operation["operationId"])

    duplicates = sorted(operation_id for operation_id, count in Counter(operation_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate OpenAPI operationId values: {', '.join(duplicates)}")


def _path_hidden_from_public(path: str) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in HIDDEN_FROM_PUBLIC_PATH_PREFIXES)


def filter_public_openapi(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a public-facing spec with internal/admin/runtime paths removed."""

    out = copy.deepcopy(spec)
    paths = out.get("paths") or {}
    for path in [path for path in paths if _path_hidden_from_public(path)]:
        paths.pop(path, None)
    out["paths"] = paths
    return out
