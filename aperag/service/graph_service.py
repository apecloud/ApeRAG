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

"""Legacy alias module for the knowledge-graph read service.

The concrete implementation relocated to
``aperag.domains.knowledge_graph.service`` by the Phase 2 hard-cut.
This module is kept as a thin re-export until the Phase 3 DB-split PR
retires the whole ``aperag.service.*`` aggregate. Existing consumers
like ``aperag/views/marketplace_collections.py`` and test fixtures in
``tests/unit_test/service/test_search_graph_contract.py`` keep working
without a mass edit in this PR.
"""

from __future__ import annotations

from aperag.domains.knowledge_graph.service import (
    GraphService,
    _adapt_edges,
    _adapt_nodes,
    graph_service,
)

__all__ = [
    "GraphService",
    "graph_service",
    "_adapt_edges",
    "_adapt_nodes",
]
