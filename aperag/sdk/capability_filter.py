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

"""Phase 9 D10.f (#98) — Option A client-side capability filter.

Per ``docs/modularization/d10-design-pack.md`` §D.2 the canonical
capability negotiation is **client-side filter**: the MCP server exposes
the full tool surface annotated with :class:`ToolAnnotation`; each
client (Claude Code / Codex / Cursor / ApeRAG own-Agent) decides which
tools to surface to its LLM by looking at the annotation's
``capabilities`` map and matching against its own runtime capabilities.

This module is the canonical helper. It mirrors the §D.2 pseudocode::

    usable_tools = [
        t for t in all_tools
        if all(client_capabilities.get(req, False)
               for req, needed in t.annotations.capabilities.items()
               if needed)
    ]

with the addition of:

* :class:`FilterDecision` — explicit-not-silent reason payload (§D.3),
  so a client UI can show the user *why* a tool is hidden rather than
  silently dropping it.
* :func:`filter_tools` — the entry point. Pure function, no I/O.
* :func:`is_usable` — single-tool helper for callers that want to keep
  the tool in the list but tag it as unusable.

Server-side filtering (Option B per §D.4) is intentionally NOT
implemented — that escape hatch is reserved for narrow legal/compliance
scenarios outside D10 scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from aperag.mcp.capabilities import ToolAnnotation


@dataclass(frozen=True)
class FilterDecision:
    """Explicit-not-silent decision for a single tool.

    Carries the tool name, the verdict (``usable``), and — when
    excluded — the structured reason. Reasons:

    * ``"deprecated"`` — the tool is on a deprecation timeline. Clients
      MAY still surface it (e.g. for compatibility) but should prefer
      ``annotation.fallback_to`` when set.
    * ``"capability_required"`` — at least one capability key the tool
      needs is not granted by ``client_capabilities``. ``missing`` is
      the sorted tuple of those keys (§D.3 row "Capability missing").
    """

    tool_name: str
    usable: bool
    annotation: ToolAnnotation
    reason: str | None = None
    missing: tuple[str, ...] = field(default_factory=tuple)


def is_usable(
    annotation: ToolAnnotation,
    client_capabilities: Mapping[str, bool],
) -> tuple[bool, tuple[str, ...]]:
    """Return ``(usable, missing)`` for one annotation under a client.

    A capability key counts as "needed" only when the annotation marks
    it ``True`` — ``False`` entries are informational per §D.1. The
    client must grant every needed capability for the tool to be
    usable; granting is ``client_capabilities.get(key, False) is True``.
    Returns the sorted tuple of unmet needed keys (empty when usable).
    """

    missing = tuple(
        sorted(
            key for key, needed in annotation.capabilities.items() if needed and not client_capabilities.get(key, False)
        )
    )
    return (not missing, missing)


def filter_tools(
    annotations: Mapping[str, ToolAnnotation],
    client_capabilities: Mapping[str, bool],
    *,
    include_deprecated: bool = True,
) -> list[FilterDecision]:
    """Decide for each tool whether the client may surface it.

    ``annotations`` is a name → :class:`ToolAnnotation` map (typically
    :func:`aperag.mcp.tools._annotations.get_all`).
    ``client_capabilities`` is the client's runtime capability map —
    e.g. ``{"vision": True, "long_context": True, "graph_index":
    True, "fulltext_index": True, "web_access": False}`` for an
    air-gapped vision-capable LLM host.

    Returns one :class:`FilterDecision` per tool, in the same order as
    ``annotations`` iteration. Callers wanting just the usable subset
    can do ``[d for d in result if d.usable]``.

    ``include_deprecated`` defaults to ``True``: deprecated tools are
    still returned as ``usable=True`` (with ``reason="deprecated"`` so
    the UI can de-prioritize). Set ``include_deprecated=False`` to drop
    deprecated tools entirely — that flips them to
    ``usable=False, reason="deprecated"``.
    """

    decisions: list[FilterDecision] = []
    for name, annotation in annotations.items():
        usable, missing = is_usable(annotation, client_capabilities)
        if not usable:
            decisions.append(
                FilterDecision(
                    tool_name=name,
                    usable=False,
                    annotation=annotation,
                    reason="capability_required",
                    missing=missing,
                )
            )
            continue
        if annotation.deprecated:
            if include_deprecated:
                decisions.append(
                    FilterDecision(
                        tool_name=name,
                        usable=True,
                        annotation=annotation,
                        reason="deprecated",
                    )
                )
            else:
                decisions.append(
                    FilterDecision(
                        tool_name=name,
                        usable=False,
                        annotation=annotation,
                        reason="deprecated",
                    )
                )
            continue
        decisions.append(
            FilterDecision(
                tool_name=name,
                usable=True,
                annotation=annotation,
            )
        )
    return decisions


def usable_tool_names(
    annotations: Mapping[str, ToolAnnotation],
    client_capabilities: Mapping[str, bool],
    *,
    include_deprecated: bool = True,
) -> list[str]:
    """Convenience wrapper returning just the usable tool names."""

    return [
        decision.tool_name
        for decision in filter_tools(
            annotations,
            client_capabilities,
            include_deprecated=include_deprecated,
        )
        if decision.usable
    ]


def required_capabilities(
    annotations: Iterable[ToolAnnotation],
) -> tuple[str, ...]:
    """Aggregate every capability key any annotation marks ``True``.

    Useful for clients that want to advertise to their host *which*
    capabilities they need to honor in order to expose the full ApeRAG
    surface — e.g. for a CLI configuration prompt.
    """

    keys: set[str] = set()
    for annotation in annotations:
        for key, needed in annotation.capabilities.items():
            if needed:
                keys.add(key)
    return tuple(sorted(keys))


__all__ = [
    "FilterDecision",
    "filter_tools",
    "is_usable",
    "required_capabilities",
    "usable_tool_names",
]
