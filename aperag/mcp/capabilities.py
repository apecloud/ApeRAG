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

"""Phase 9 D10.f (#98) — capability negotiation Option A schema.

Per ``docs/modularization/d10-design-pack.md`` §D this module defines
the per-tool annotation schema that powers Option A (client-side filter)
capability negotiation:

* :class:`ToolAnnotation` — the canonical annotation envelope (§D.1).
  Each MCP tool registered in D10.c (8 read primitives) and D10.d
  (4 search primitives) carries one :class:`ToolAnnotation` value, which
  is also handed to ``@mcp_server.tool(annotations=...)`` via
  :meth:`ToolAnnotation.to_mcp_dict` so it shows up in MCP
  ``list_tools`` metadata for external Agents (Claude Code / Codex /
  Cursor).

* :data:`KNOWN_CAPABILITIES` — the closed set of capability keys that
  D10 currently uses (``vision``, ``long_context``, ``graph_index``,
  ``fulltext_index``, ``web_access``). Adding a new capability key
  requires a ``[D10 spec amendment]`` thread per §G hard gate.

The actual registry of tool name → annotation lives in
:mod:`aperag.mcp.tools._annotations`. Server-side filtering (Option B)
is intentionally NOT implemented — that escape hatch is reserved for a
future task per §D.4.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

# §D.1 closed set of capability keys. Adding a key here is a
# spec change — must go through a ``[D10 spec amendment]`` thread per
# §G hard gate before code in this module is touched.
KNOWN_CAPABILITIES: frozenset[str] = frozenset(
    {
        "vision",  # client can render / reason over images
        "long_context",  # client tolerates large content payloads
        "graph_index",  # collection has a graph index built
        "fulltext_index",  # collection has a fulltext index built
        "web_access",  # caller may reach the public internet
    }
)

# §D.1 closed set of "requires" prerequisites. These are NOT capability
# checks — they declare the operational pre-requisite the caller must
# satisfy (e.g. holding a valid auth principal that can reach
# collection-scoped data). Used today for documentation / SDK readability;
# the runtime enforcement still lives in D9 §2 authorization gate +
# D9 base resolve_authenticated_user.
KNOWN_REQUIRES: frozenset[str] = frozenset(
    {
        "collection_access",
        "web_access",
    }
)


class ToolAnnotation(BaseModel):
    """§D.1 per-tool annotation envelope.

    Mirrors the spec sample::

        @mcp_server.tool(
            annotations={
                "requires": ["collection_access"],
                "capabilities": {
                    "vision": False,
                    "long_context": False,
                    "graph_index": True,
                    "fulltext_index": True,
                    "web_access": True,
                },
                "deprecated": False,
                "deprecated_until": None,
                "fallback_to": None,
            },
        )

    Field semantics — locked here, do NOT add fields without an amendment:

    * ``requires`` — operational prerequisites the caller must satisfy
      (e.g. authenticated principal, web egress). The runtime gate is
      D9 §2; this list is for client-side documentation / discovery.
    * ``capabilities`` — capability ``True`` means the tool *requires*
      that capability of the caller (per §D.2 client-side filter
      pseudocode: ``if needed`` filter pulls from ``True`` entries). A
      ``False`` entry is informational ("this tool does NOT need
      vision") and never excludes a tool.
    * ``deprecated`` — when ``True`` the tool MUST also set either
      ``fallback_to`` or ``deprecated_until`` (or both). External
      clients should de-prioritize deprecated tools.
    * ``deprecated_until`` — ISO-8601 date string (``YYYY-MM-DD``)
      announcing when the tool will be removed. Free-form is rejected.
    * ``fallback_to`` — name of another registered tool that supersedes
      this one. Validated as non-empty when set; cross-name existence
      check is the registry's job (avoids circular import here).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    requires: tuple[str, ...] = Field(
        default=(),
        description="Operational prerequisites the caller must satisfy "
        "(client-readable; runtime enforcement is D9 §2).",
    )
    capabilities: Mapping[str, bool] = Field(
        default_factory=dict,
        description="Capability map. True = required of caller; False = "
        "informational. Keys must be in KNOWN_CAPABILITIES.",
    )
    deprecated: bool = Field(
        default=False,
        description="True when this tool is on a deprecation timeline.",
    )
    deprecated_until: str | None = Field(
        default=None,
        description="ISO-8601 date (YYYY-MM-DD) of planned removal.",
    )
    fallback_to: str | None = Field(
        default=None,
        description="Name of the tool that supersedes this one.",
    )

    @field_validator("requires")
    @classmethod
    def _validate_requires(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str) or not entry:
                raise ValueError("requires entries must be non-empty strings")
            if entry in seen:
                raise ValueError(f"requires contains duplicate entry: {entry!r}")
            if entry not in KNOWN_REQUIRES:
                raise ValueError(f"unknown requires key: {entry!r} (allowed: {sorted(KNOWN_REQUIRES)})")
            seen.add(entry)
        return value

    @field_validator("capabilities", mode="before")
    @classmethod
    def _validate_capabilities(cls, value: Any) -> Mapping[str, bool]:
        # ``mode="before"`` runs ahead of Pydantic's automatic coercion
        # so that ``"yes"`` / ``1`` / ``None`` raise instead of being
        # silently turned into ``True`` / ``False``. The capabilities
        # map is on the wire to external Agents — we want it to be
        # strict, JSON-bool only.
        if not isinstance(value, Mapping):
            raise ValueError("capabilities must be a mapping of str -> bool")
        for key, val in value.items():
            if not isinstance(key, str):
                raise ValueError(f"capability key must be str, got {type(key).__name__}")
            if key not in KNOWN_CAPABILITIES:
                raise ValueError(f"unknown capability key: {key!r} (allowed: {sorted(KNOWN_CAPABILITIES)})")
            if not isinstance(val, bool):
                raise ValueError(f"capability {key!r} value must be bool, got {type(val).__name__}")
        return dict(value)

    @field_validator("deprecated_until")
    @classmethod
    def _validate_deprecated_until(cls, value: str | None) -> str | None:
        if value is None:
            return None
        # Cheap, dependency-free YYYY-MM-DD shape check; the wire is
        # consumed by external clients so we want a hard format lock.
        if (
            len(value) != 10
            or value[4] != "-"
            or value[7] != "-"
            or not value[0:4].isdigit()
            or not value[5:7].isdigit()
            or not value[8:10].isdigit()
        ):
            raise ValueError(f"deprecated_until must be YYYY-MM-DD, got {value!r}")
        return value

    @field_validator("fallback_to")
    @classmethod
    def _validate_fallback_to(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value:
            raise ValueError("fallback_to must be a non-empty string when set")
        return value

    def to_mcp_dict(self) -> dict[str, Any]:
        """Render the annotation as the FastMCP ``annotations=`` kwarg.

        FastMCP's ``mcp_server.tool`` accepts ``ToolAnnotations | dict |
        None``. We pass a plain dict — JSON-serializable, mirrors the
        §D.1 spec sample byte-for-byte, and shows up unchanged in
        ``list_tools`` metadata for external Agents.
        """
        return {
            "requires": list(self.requires),
            "capabilities": dict(self.capabilities),
            "deprecated": self.deprecated,
            "deprecated_until": self.deprecated_until,
            "fallback_to": self.fallback_to,
        }


__all__ = [
    "KNOWN_CAPABILITIES",
    "KNOWN_REQUIRES",
    "ToolAnnotation",
]
