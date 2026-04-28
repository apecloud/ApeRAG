# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 10 §K.13 — agent-runtime tool subset enforcement tests.

Pin the ``_allowed_tool_names_for_bot`` mapping + ``_wrap_toolset_for_bot``
behaviour. The runtime layer (per design doc §4) is the canonical
enforcement point so the LLM cannot escape the read-only constraint
even if the system prompt is altered.
"""

from __future__ import annotations

from types import SimpleNamespace

from aperag.domains.agent_runtime.runtime import (
    _BOT_TYPE_ALLOWED_TOOLS,
    _allowed_tool_names_for_bot,
    _wrap_toolset_for_bot,
)


def test_summary_bot_subset_includes_thirteen_canonical_tools():
    summary_tools = _BOT_TYPE_ALLOWED_TOOLS.get("summary")
    assert summary_tools is not None
    # Every tool listed in the design doc §4 (13 read-only tools) must
    # be present. Adding new tools to the subset is intentional; this
    # test catches accidental drops or typos.
    expected = {
        "list_collections",
        "vector_search",
        "fulltext_search",
        "graph_search",
        "query_graph_entities",
        "expand_graph_subgraph",
        "get_entity_detail",
        "read_document",
        "read_document_section",
        "read_document_outline",
        "read_document_chunk",
        "get_collection_metadata",
        "get_document_metadata",
    }
    assert summary_tools == frozenset(expected)


def test_allowed_tool_names_for_summary_bot_returns_subset():
    bot = SimpleNamespace(type=SimpleNamespace(value="summary"))
    allowed = _allowed_tool_names_for_bot(bot)
    assert allowed is not None
    assert "vector_search" in allowed
    # Write tools are NOT permitted.
    assert "create_document" not in allowed
    assert "delete_document" not in allowed


def test_allowed_tool_names_for_agent_bot_returns_none_meaning_full_toolset():
    bot = SimpleNamespace(type=SimpleNamespace(value="agent"))
    assert _allowed_tool_names_for_bot(bot) is None


def test_allowed_tool_names_for_knowledge_bot_returns_none():
    bot = SimpleNamespace(type=SimpleNamespace(value="knowledge"))
    assert _allowed_tool_names_for_bot(bot) is None


def test_wrap_toolset_for_summary_bot_returns_filtered_toolset():
    from pydantic_ai.toolsets.filtered import FilteredToolset

    base_toolset = SimpleNamespace(name="base_mcp_toolset")
    bot = SimpleNamespace(type=SimpleNamespace(value="summary"))

    wrapped = _wrap_toolset_for_bot(base_toolset, bot)

    assert isinstance(wrapped, FilteredToolset)
    # The wrapped reference is the original toolset.
    assert wrapped.wrapped is base_toolset


def test_wrap_toolset_for_agent_bot_returns_original_toolset_unchanged():
    """Bots whose type is not in the subset map get the full toolset —
    no FilteredToolset wrapper added."""
    base_toolset = SimpleNamespace(name="base_mcp_toolset")
    bot = SimpleNamespace(type=SimpleNamespace(value="agent"))

    wrapped = _wrap_toolset_for_bot(base_toolset, bot)

    assert wrapped is base_toolset


def test_wrap_toolset_filter_function_uses_allowed_set():
    """The filter function returned by ``_wrap_toolset_for_bot`` must
    return ``True`` only for tools in the allowed set."""
    from pydantic_ai.toolsets.filtered import FilteredToolset

    base_toolset = SimpleNamespace()
    bot = SimpleNamespace(type=SimpleNamespace(value="summary"))
    wrapped = _wrap_toolset_for_bot(base_toolset, bot)
    assert isinstance(wrapped, FilteredToolset)

    filter_fn = wrapped.filter_func

    allowed = SimpleNamespace(name="vector_search")
    blocked = SimpleNamespace(name="create_collection")  # not in subset

    # ``filter_func`` is sync (per our setup); pass any sentinel
    # for the run context — it isn't inspected.
    assert filter_fn(None, allowed) is True
    assert filter_fn(None, blocked) is False
