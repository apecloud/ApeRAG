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

"""§D.3.5 :class:`LineageGraphStore` adapters — Wave 4 T8.

Wave 3 shipped the lineage-aware graph modality with only the
in-memory reference store; :func:`aperag.indexing.worker_factory.
_build_graph_worker` raises a ``WorkerFactoryError`` for any
collection that opts into graph indexing because the placeholder
is non-durable and silent-broken in production. Wave 4 lifts that
gate by wiring the three real backends ApeRAG already supports for
the legacy ``GraphStore`` Protocol — PostgreSQL / Neo4j / Nebula —
to the new lineage-SET-level Protocol.

The adapters live in this package (one module per backend) plus a
``factory`` helper that picks the right implementation based on
``collection.config.graph_backend_type``. The legacy
``aperag/domains/knowledge_graph/graphindex/storage/`` subpackage
is hard-cut deleted in the same PR so the codebase keeps exactly
one graph backend abstraction (architect msg=0aea9edf "no dual
store abstraction long-term").

The Postgres adapter is the reference implementation; Neo4j and
Nebula mirror its shape with per-backend storage primitives
(Cypher list-of-MAP property / nGQL JSON STRING tag respectively).
"""

from __future__ import annotations

__all__: list[str] = []
