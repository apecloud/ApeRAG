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

"""Phase 9 D10.f (#98) — capability negotiation contract tests.

Covers ``docs/modularization/d10-design-pack.md`` §D end-to-end:

* §D.1 schema validity — known-only keys, deprecated_until format,
  fallback_to non-empty, capability values are bool.
* §D.2 client-side filter pseudocode — needed=True excludes when client
  lacks the capability; needed=False is informational.
* §D.3 explicit-not-silent — :class:`FilterDecision` carries a reason
  per excluded tool.
* §D.5 registry — every D10 tool that lands in the FastMCP registry
  via ``@mcp_server.tool(annotations=register(...))`` shows up in
  :func:`get_all`.

The test imports :mod:`aperag.mcp.server` once at module load so all
``@mcp_server.tool`` decorators run and the registry is populated; all
tests then operate on that single populated state. The
``_reset_for_tests`` escape hatch is used only by the few tests that
exercise re-registration.
"""

from __future__ import annotations

import importlib

import pytest

# Importing the server module triggers every @mcp_server.tool decorator
# defined in aperag/mcp/server.py and the four search_* modules — that
# is the only side-effect we rely on.
import aperag.mcp.server  # noqa: F401  (imported for side effects)
from aperag.mcp.capabilities import (
    KNOWN_CAPABILITIES,
    KNOWN_REQUIRES,
    ToolAnnotation,
)
from aperag.mcp.tools import _annotations
from aperag.sdk.capability_filter import (
    FilterDecision,
    filter_tools,
    is_usable,
    required_capabilities,
    usable_tool_names,
)

# ---------------------------------------------------------------------------
# §D.1 — annotation schema validity
# ---------------------------------------------------------------------------


class TestToolAnnotationSchema:
    def test_default_annotation_is_empty_and_frozen(self):
        a = ToolAnnotation()
        assert a.requires == ()
        assert dict(a.capabilities) == {}
        assert a.deprecated is False
        assert a.deprecated_until is None
        assert a.fallback_to is None
        # frozen=True
        with pytest.raises(Exception):
            a.deprecated = True  # type: ignore[misc]

    def test_known_capability_keys_accepted(self):
        a = ToolAnnotation(capabilities={key: True for key in KNOWN_CAPABILITIES})
        assert set(a.capabilities.keys()) == set(KNOWN_CAPABILITIES)

    def test_unknown_capability_key_rejected(self):
        with pytest.raises(ValueError, match="unknown capability key"):
            ToolAnnotation(capabilities={"telepathy": True})

    def test_capability_value_must_be_bool(self):
        with pytest.raises(ValueError, match="must be bool"):
            ToolAnnotation(capabilities={"vision": "yes"})  # type: ignore[dict-item]

    def test_known_requires_accepted(self):
        a = ToolAnnotation(requires=tuple(KNOWN_REQUIRES))
        assert set(a.requires) == set(KNOWN_REQUIRES)

    def test_unknown_requires_rejected(self):
        with pytest.raises(ValueError, match="unknown requires key"):
            ToolAnnotation(requires=("clearance_level_5",))

    def test_duplicate_requires_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            ToolAnnotation(requires=("collection_access", "collection_access"))

    def test_deprecated_until_format(self):
        ToolAnnotation(deprecated=True, deprecated_until="2026-12-31")
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            ToolAnnotation(deprecated=True, deprecated_until="Dec 31 2026")
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            ToolAnnotation(deprecated=True, deprecated_until="2026/12/31")

    def test_fallback_to_must_be_non_empty(self):
        ToolAnnotation(deprecated=True, fallback_to="vector_search")
        with pytest.raises(ValueError, match="non-empty"):
            ToolAnnotation(deprecated=True, fallback_to="")

    def test_to_mcp_dict_matches_spec_sample(self):
        a = ToolAnnotation(
            requires=("collection_access",),
            capabilities={"vision": False, "web_access": True},
            deprecated=False,
            deprecated_until=None,
            fallback_to=None,
        )
        # §D.1 sample shape — list-typed `requires`, dict-typed
        # `capabilities`, JSON-friendly None.
        assert a.to_mcp_dict() == {
            "requires": ["collection_access"],
            "capabilities": {"vision": False, "web_access": True},
            "deprecated": False,
            "deprecated_until": None,
            "fallback_to": None,
        }


# ---------------------------------------------------------------------------
# §D.5 — registry contract
# ---------------------------------------------------------------------------


# Built-in read/search/web tools must all carry annotations so any MCP
# client can make one consistent capability decision.
_EXPECTED_TOOLS: frozenset[str] = frozenset(
    {
        # D10.c read primitives
        "list_collections",
        "list_documents",
        "get_collection_metadata",
        "get_document_metadata",
        "read_document",
        "read_document_outline",
        "read_document_section",
        "read_document_chunk",
        # D10.d search primitives
        "vector_search",
        "graph_search",
        "fulltext_search",
        "web_search",
        "web_read",
    }
)


class TestRegistry:
    def test_all_d10_tools_registered(self):
        registered = set(_annotations.get_all().keys())
        # Allow for additional tools landing in future, but every
        # tool we ship in D10.f must be present.
        missing = _EXPECTED_TOOLS - registered
        assert missing == set(), f"D10 tools missing from annotation registry: {missing}"

    def test_registry_is_immutable_view(self):
        view = _annotations.get_all()
        with pytest.raises(TypeError):
            view["x"] = ToolAnnotation()  # type: ignore[index]

    def test_get_returns_same_annotation_as_get_all(self):
        view = _annotations.get_all()
        for name, ann in view.items():
            assert _annotations.get(name) is ann

    def test_get_unknown_returns_none(self):
        assert _annotations.get("nonexistent_tool") is None

    def test_capability_keys_in_registry_are_known(self):
        for name, ann in _annotations.get_all().items():
            for key in ann.capabilities:
                assert key in KNOWN_CAPABILITIES, f"tool {name!r} declares unknown capability {key!r}"
            for req in ann.requires:
                assert req in KNOWN_REQUIRES, f"tool {name!r} declares unknown requires entry {req!r}"

    def test_collection_search_tools_do_not_require_client_index_capabilities(self):
        """Graph/fulltext indexes are collection state, not client
        capabilities. Clients should not hide these tools just because
        the host lacks a local graph/fulltext feature flag.
        """

        ann = _annotations.get("graph_search")
        assert ann is not None
        assert "graph_index" not in ann.capabilities
        ann = _annotations.get("fulltext_search")
        assert ann is not None
        assert "fulltext_index" not in ann.capabilities

    def test_web_search_requires_web_access(self):
        ann = _annotations.get("web_search")
        assert ann is not None
        assert ann.capabilities.get("web_access") is True
        assert "web_access" in ann.requires

    def test_web_read_requires_web_access(self):
        ann = _annotations.get("web_read")
        assert ann is not None
        assert ann.capabilities.get("web_access") is True
        assert "web_access" in ann.requires

    def test_read_document_marks_long_context(self):
        ann = _annotations.get("read_document")
        assert ann is not None
        assert ann.capabilities.get("long_context") is True

    def test_smaller_read_primitives_do_not_require_long_context(self):
        # outline / section / chunk are explicitly bounded — they MUST
        # not opt their callers into a long-context capability filter.
        for name in ("read_document_outline", "read_document_section", "read_document_chunk"):
            ann = _annotations.get(name)
            assert ann is not None, name
            assert ann.capabilities.get("long_context") is False


# ---------------------------------------------------------------------------
# Registry register() semantics
# ---------------------------------------------------------------------------


class TestRegisterReregistrationGuard:
    def setup_method(self) -> None:
        # Take a snapshot, then hand the registry over for surgical
        # tests; teardown re-imports the server module to repopulate.
        self._snapshot = dict(_annotations.get_all())
        _annotations._reset_for_tests()

    def teardown_method(self) -> None:
        # Restore the production registry by re-running every
        # @mcp_server.tool decorator (single source of truth).
        _annotations._reset_for_tests()
        importlib.reload(importlib.import_module("aperag.mcp.tools.search_vector"))
        importlib.reload(importlib.import_module("aperag.mcp.tools.search_graph"))
        importlib.reload(importlib.import_module("aperag.mcp.tools.search_fulltext"))
        importlib.reload(importlib.import_module("aperag.mcp.tools.search_web"))
        importlib.reload(importlib.import_module("aperag.mcp.server"))
        # Sanity: every snapshot entry is back.
        restored = _annotations.get_all()
        for name in self._snapshot:
            assert name in restored

    def test_register_returns_mcp_dict(self):
        ann = ToolAnnotation(
            requires=("collection_access",),
            capabilities={"vision": False},
        )
        result = _annotations.register("__t1", ann)
        assert result == ann.to_mcp_dict()
        assert _annotations.get("__t1") is ann

    def test_re_register_same_annotation_is_idempotent(self):
        ann = ToolAnnotation(capabilities={"vision": True})
        first = _annotations.register("__t2", ann)
        second = _annotations.register("__t2", ann)
        assert first == second
        assert _annotations.get("__t2") is ann

    def test_re_register_different_annotation_raises(self):
        ann_a = ToolAnnotation(capabilities={"vision": True})
        ann_b = ToolAnnotation(capabilities={"vision": False})
        _annotations.register("__t3", ann_a)
        with pytest.raises(ValueError, match="already registered"):
            _annotations.register("__t3", ann_b)

    def test_register_rejects_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            _annotations.register("", ToolAnnotation())

    def test_register_rejects_non_annotation_value(self):
        with pytest.raises(TypeError, match="ToolAnnotation"):
            _annotations.register("__t4", {"requires": []})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# §D.2 client-side filter pseudocode
# ---------------------------------------------------------------------------


class TestFilterPseudocode:
    def test_needed_true_capability_excludes_lacking_client(self):
        ann = ToolAnnotation(capabilities={"web_access": True})
        usable, missing = is_usable(ann, {"web_access": False})
        assert usable is False
        assert missing == ("web_access",)

    def test_needed_true_capability_admits_granting_client(self):
        ann = ToolAnnotation(capabilities={"web_access": True})
        usable, missing = is_usable(ann, {"web_access": True})
        assert usable is True
        assert missing == ()

    def test_needed_false_capability_is_informational(self):
        # `vision: False` declares "this tool does not need vision".
        # A client that does NOT have vision must still be able to use
        # the tool — `False` entries never exclude.
        ann = ToolAnnotation(capabilities={"vision": False})
        usable, missing = is_usable(ann, {"vision": False})
        assert usable is True
        assert missing == ()

    def test_missing_client_capability_is_treated_as_False(self):
        # Per §D.2 pseudocode: `client_capabilities.get(req, False)`.
        ann = ToolAnnotation(capabilities={"web_access": True})
        usable, missing = is_usable(ann, {})  # no key at all
        assert usable is False
        assert missing == ("web_access",)

    def test_multiple_missing_caps_reported_sorted(self):
        ann = ToolAnnotation(capabilities={"web_access": True, "long_context": True, "vision": False})
        usable, missing = is_usable(ann, {"vision": True})
        assert usable is False
        assert missing == ("long_context", "web_access")  # sorted


# ---------------------------------------------------------------------------
# §D.3 explicit-not-silent (FilterDecision)
# ---------------------------------------------------------------------------


class TestFilterDecisions:
    def _registry(self) -> dict[str, ToolAnnotation]:
        return {
            "vector_search": ToolAnnotation(
                requires=("collection_access",),
                capabilities={"long_context": False},
            ),
            "graph_search": ToolAnnotation(
                requires=("collection_access",),
                capabilities={"long_context": False},
            ),
            "web_search": ToolAnnotation(
                requires=("web_access",),
                capabilities={"long_context": False, "web_access": True},
            ),
            "old_omnibus_search": ToolAnnotation(
                requires=("collection_access",),
                capabilities={"long_context": False},
                deprecated=True,
                deprecated_until="2026-12-31",
                fallback_to="vector_search",
            ),
        }

    def test_air_gapped_client_excludes_web_with_reason(self):
        decisions = filter_tools(
            self._registry(),
            {"long_context": True, "web_access": False},
        )
        by_name = {d.tool_name: d for d in decisions}
        assert by_name["web_search"].usable is False
        assert by_name["web_search"].reason == "capability_required"
        assert by_name["web_search"].missing == ("web_access",)

    def test_collection_index_state_does_not_hide_graph_search(self):
        decisions = filter_tools(
            self._registry(),
            {"long_context": True, "web_access": True},
        )
        by_name = {d.tool_name: d for d in decisions}
        assert by_name["graph_search"].usable is True
        assert by_name["graph_search"].reason is None
        # vector_search has no needed=True caps so it must remain usable
        assert by_name["vector_search"].usable is True
        assert by_name["vector_search"].reason is None

    def test_deprecated_tool_kept_with_reason_by_default(self):
        decisions = filter_tools(
            self._registry(),
            {"long_context": True, "web_access": True},
        )
        by_name = {d.tool_name: d for d in decisions}
        assert by_name["old_omnibus_search"].usable is True
        assert by_name["old_omnibus_search"].reason == "deprecated"

    def test_deprecated_tool_dropped_when_include_deprecated_false(self):
        decisions = filter_tools(
            self._registry(),
            {"long_context": True, "web_access": True},
            include_deprecated=False,
        )
        by_name = {d.tool_name: d for d in decisions}
        assert by_name["old_omnibus_search"].usable is False
        assert by_name["old_omnibus_search"].reason == "deprecated"

    def test_filter_decision_is_immutable(self):
        d = FilterDecision(
            tool_name="x",
            usable=True,
            annotation=ToolAnnotation(),
        )
        with pytest.raises(Exception):
            d.usable = False  # type: ignore[misc]

    def test_usable_tool_names_helper_returns_only_usable(self):
        names = usable_tool_names(
            self._registry(),
            {"long_context": True, "web_access": False},
        )
        # vector_search has no needed=True caps so it survives.
        # graph_search is gated by collection state at execution time,
        # not by client capability filtering.
        # old_omnibus_search is deprecated but still usable per default.
        # web_search is excluded by missing web access.
        assert set(names) == {"vector_search", "graph_search", "old_omnibus_search"}

    def test_required_capabilities_aggregates_only_true_keys(self):
        keys = required_capabilities(self._registry().values())
        # collection index availability is not a client capability.
        assert keys == ("web_access",)
