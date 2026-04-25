# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase 9 D10.f (#98) — central registry of D10 tool annotations.

Per ``docs/modularization/d10-design-pack.md`` §D.5 the registry is a
**name → :class:`~aperag.mcp.capabilities.ToolAnnotation` map** that:

1. Drives the ``annotations=`` kwarg passed to ``@mcp_server.tool`` for
   each D10 tool — mirrored byte-for-byte into MCP ``list_tools``
   metadata.
2. Powers the SDK side Option A filter
   (:mod:`aperag.sdk.capability_filter`).

Each tool calls :func:`register` once at import time. The function
returns a plain ``dict`` so a tool registration line stays a one-liner::

    @mcp_server.tool(annotations=register("list_collections", ToolAnnotation(...)))
    async def list_collections(...): ...

The registry is read-only after process startup; callers MUST treat
:func:`get_all` and :func:`get` results as immutable.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from aperag.mcp.capabilities import ToolAnnotation

_REGISTRY: dict[str, ToolAnnotation] = {}


def register(tool_name: str, annotation: ToolAnnotation) -> dict[str, Any]:
    """Register ``annotation`` under ``tool_name`` and return its MCP dict.

    The returned ``dict`` is what FastMCP wants for the
    ``@mcp_server.tool(annotations=...)`` kwarg. Returning it inline
    keeps the call site a one-liner and guarantees the registry copy
    and the FastMCP-visible copy stay in sync (they come from the same
    :class:`ToolAnnotation` instance).

    Re-registering the same name with the **same annotation** is a
    no-op — that supports test reload / module hot-reload without
    blowing up. Re-registering with a *different* annotation raises
    :class:`ValueError` because that means two tools fought over the
    name.
    """
    if not tool_name or not isinstance(tool_name, str):
        raise ValueError("tool_name must be a non-empty string")
    if not isinstance(annotation, ToolAnnotation):
        raise TypeError(f"annotation must be ToolAnnotation, got {type(annotation).__name__}")

    existing = _REGISTRY.get(tool_name)
    if existing is not None and existing != annotation:
        raise ValueError(f"tool {tool_name!r} already registered with a different annotation")
    _REGISTRY[tool_name] = annotation
    return annotation.to_mcp_dict()


def get(tool_name: str) -> ToolAnnotation | None:
    """Return the annotation for ``tool_name`` or ``None`` if absent."""
    return _REGISTRY.get(tool_name)


def get_all() -> Mapping[str, ToolAnnotation]:
    """Return an immutable view of the full annotation registry.

    Returned value is a :class:`types.MappingProxyType` — mutating it
    raises :class:`TypeError`. Callers iterating must treat the
    contained :class:`ToolAnnotation` values as frozen (they are
    Pydantic models with ``frozen=True``).
    """
    return MappingProxyType(_REGISTRY)


def _reset_for_tests() -> None:
    """Drop all registered annotations.

    Test-only escape hatch — used by
    :mod:`tests.unit_test.mcp.test_capability_negotiation` to keep
    re-registration assertions hermetic. Production code MUST NOT call
    this (the registry is meant to be populated once at import time).
    """
    _REGISTRY.clear()


__all__ = [
    "get",
    "get_all",
    "register",
]
