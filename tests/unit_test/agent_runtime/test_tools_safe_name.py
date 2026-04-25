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

"""Contract tests for D9 §A1+§A6 SafeToolName resolver (Phase 8 #75)."""

from __future__ import annotations

import re

import pytest

from aperag.domains.agent_runtime.tools.safe_name import (
    SAFE_NAME_RE,
    SafeNameRegistry,
    sanitize_tool_name,
)

# Provider-safe regex from the wire spec (D9 §A1) -- the actual at-rest
# tool part type literal must match this once the safe name is prefixed.
_TOOL_TYPE_RE = re.compile(r"^tool-[A-Za-z0-9_-]+$")


def test_sanitize_tool_name_preserves_provider_safe_chars():
    assert sanitize_tool_name("aperag_knowledge_base_search_collection") == ("aperag_knowledge_base_search_collection")


def test_sanitize_tool_name_substitutes_each_disallowed_char():
    # Per D9 §A6 the substitution is character-wise, NOT collapse-runs.
    assert sanitize_tool_name("foo.bar baz") == "foo_bar_baz"
    assert sanitize_tool_name("a..b") == "a__b"


def test_safe_name_regex_matches_only_provider_safe():
    assert SAFE_NAME_RE.search("a-1_") is None
    assert SAFE_NAME_RE.search("a.b") is not None
    assert SAFE_NAME_RE.search("a b") is not None


def test_resolve_assigns_naive_form_on_first_call():
    registry = SafeNameRegistry()
    result = registry.resolve("aperag-knowledge-base", "search_collection")
    assert result.safe_name == "aperag-knowledge-base_search_collection"
    assert result.collided is False
    # Assert wire-side ``tool-<name>`` would match the spec regex.
    assert _TOOL_TYPE_RE.match(f"tool-{result.safe_name}")


def test_resolve_is_idempotent_for_same_pair():
    registry = SafeNameRegistry()
    a = registry.resolve("aperag", "search")
    b = registry.resolve("aperag", "search")
    assert a.safe_name == b.safe_name


def test_resolve_appends_hash_suffix_on_collision():
    registry = SafeNameRegistry()
    # Two different MCP identities whose sanitised forms collide.
    a = registry.resolve("foo.bar", "x")  # naive: "foo_bar_x"
    b = registry.resolve("foo_bar", "x")  # naive: "foo_bar_x"
    assert a.safe_name == "foo_bar_x"
    assert a.collided is False
    assert b.collided is True
    # Hash suffix marker is the double underscore + 6 hex chars.
    assert b.safe_name.startswith("foo_bar_x__")
    assert re.match(r"^foo_bar_x__[0-9a-f]{6}$", b.safe_name)


def test_collision_suffix_is_stable_across_processes():
    # The hash suffix is sha256-derived; two registries should agree
    # on the disambiguated name for the same colliding pair.
    registry_a = SafeNameRegistry()
    registry_b = SafeNameRegistry()
    registry_a.resolve("foo.bar", "x")
    registry_b.resolve("foo.bar", "x")
    a = registry_a.resolve("foo_bar", "x")
    b = registry_b.resolve("foo_bar", "x")
    assert a.safe_name == b.safe_name


def test_reverse_lookup_returns_mcp_identity():
    registry = SafeNameRegistry()
    result = registry.resolve("aperag-knowledge-base", "search_collection")
    assert registry.reverse(result.safe_name) == (
        "aperag-knowledge-base",
        "search_collection",
    )


def test_reverse_lookup_returns_none_for_unknown():
    registry = SafeNameRegistry()
    assert registry.reverse("never-registered") is None


def test_metadata_returns_canonical_dict_for_wire():
    registry = SafeNameRegistry()
    result = registry.resolve("aperag-kb", "search")
    md = registry.metadata(result.safe_name)
    assert md == {"mcpServer": "aperag-kb", "mcpToolName": "search"}


def test_metadata_returns_empty_dict_for_unknown():
    registry = SafeNameRegistry()
    assert registry.metadata("unknown") == {}


def test_resolve_rejects_empty_inputs():
    registry = SafeNameRegistry()
    with pytest.raises(ValueError):
        registry.resolve("", "search")
    with pytest.raises(ValueError):
        registry.resolve("aperag", "")
