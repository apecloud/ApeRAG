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

Originally pinned the ``search_collection`` URL + ``SearchResult``
import path during Phase 2's retrieval hard-cut so the omnibus
wrapper could not silently regress to ``/api/v1/`` or to the legacy
``aperag.schema.view_models`` aggregate.

D10.h (#100) hard-cuts ``search_collection`` outright (per
``earayu2 msg=9730bb6b`` no-users / no-data philosophy + amendment-#2
Drift #2/#3). The contract this file pins now is the *removal* —
``aperag.mcp.server`` must not re-introduce the omnibus tool, the
inline definition, or the legacy ``SearchResult`` import path. The
canonical search surface is the split tool family
(``aperag.mcp.tools.search_{vector,graph,fulltext,web}``) and is
covered by ``tests/unit_test/mcp/test_search_split.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from aperag.mcp import server as mcp_server

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_SERVER_PATH = REPO_ROOT / "aperag" / "mcp" / "server.py"


def test_legacy_search_collection_omnibus_stays_removed():
    """``aperag.mcp.server`` must not expose the omnibus
    ``search_collection`` runtime attribute or carry the inline
    ``async def`` after the D10.h cutover.
    """
    assert not hasattr(mcp_server, "search_collection"), (
        "search_collection must remain removed from aperag.mcp.server after the D10.h cutover."
    )

    tree = ast.parse(MCP_SERVER_PATH.read_text())
    inline_defs = [
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "search_collection"
    ]
    assert inline_defs == [], (
        "aperag/mcp/server.py must not define `async def search_collection`; "
        "the canonical search surface is `aperag.mcp.tools.search_*`."
    )


def test_search_result_legacy_import_stays_gone():
    """``SearchResult`` no longer needs to be imported into
    ``aperag.mcp.server`` because the omnibus wrapper that parsed
    backend responses is gone. A future regression that re-imports
    it would suggest the omnibus wrapper is sneaking back in.
    """
    tree = ast.parse(MCP_SERVER_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                assert alias.name != "SearchResult", (
                    "aperag/mcp/server.py must not re-import SearchResult; "
                    "the omnibus search wrapper that needed it was removed "
                    "in the D10.h cutover."
                )
