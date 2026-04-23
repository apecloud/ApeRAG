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

"""MCP contract tests.

Pins the source-of-truth for the MCP ``search_collection`` /
``web_search`` / ``web_read`` tools so that Phase 2 / Phase 3
relocations cannot silently revert the URL or the response-parsing
schema back to the legacy aggregate.
"""

from __future__ import annotations

import ast
from pathlib import Path

from aperag.domains.retrieval import schemas as retrieval_schemas
from aperag.mcp import server as mcp_server


REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_SERVER_PATH = REPO_ROOT / "aperag" / "mcp" / "server.py"


def test_search_collection_schema_source():
    """``aperag.mcp.server`` must parse ``search_collection`` responses
    with the canonical ``SearchResult`` owned by the ``retrieval``
    domain, not with the legacy ``aperag.schema.view_models`` class.

    Guards against a refactor that "looks fine" (types are structurally
    compatible) but secretly re-establishes the aggregate dependency.
    """
    tree = ast.parse(MCP_SERVER_PATH.read_text())

    search_result_sources: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "SearchResult":
                    search_result_sources.add(node.module)

    assert search_result_sources == {"aperag.domains.retrieval.schemas"}, (
        "aperag/mcp/server.py must import SearchResult from "
        "aperag.domains.retrieval.schemas after the Phase 2 retrieval "
        "hard-cut; found instead: " + repr(sorted(search_result_sources))
    )

    # And the class the runtime actually uses must be the relocated one.
    assert mcp_server.SearchResult is retrieval_schemas.SearchResult


def test_search_collection_targets_v2_path():
    """The MCP wrapper must call ``/api/v2/collections/{id}/searches``
    so the legacy v1 path can be deleted by the Phase 2 hard-cut.

    Byte-level regex check: the tool body embeds the URL as an
    f-string; a v1 regression would show up as the literal substring
    ``/api/v1/collections/`` inside the ``search_collection`` region.
    """
    source = MCP_SERVER_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "search_collection":
            body_source = ast.get_source_segment(source, node) or ""
            assert "/api/v2/collections/" in body_source, (
                "search_collection must hit the v2 collections path; "
                "did the MCP wrapper regress to /api/v1/?"
            )
            assert "/api/v1/collections/" not in body_source, (
                "search_collection still references the legacy v1 "
                "collections path; remove the literal to keep the "
                "Phase 2 hard-cut honest."
            )
            return
    raise AssertionError("search_collection tool not found in aperag.mcp.server")
