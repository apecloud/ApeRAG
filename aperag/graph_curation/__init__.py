# Copyright 2026 ApeCloud, Inc.
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

"""Graph curation workflow.

This module is deliberately separate from ``aperag.domains.knowledge_graph.graphindex``:

- ``graphindex`` owns graph truth and the merge primitive
- ``graph_curation`` owns suggestion discovery and review state
"""

from __future__ import annotations

__all__ = ["GraphCurationService", "graph_curation_service"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from aperag.graph_curation.service import GraphCurationService, graph_curation_service

    exports = {
        "GraphCurationService": GraphCurationService,
        "graph_curation_service": graph_curation_service,
    }
    return exports[name]
