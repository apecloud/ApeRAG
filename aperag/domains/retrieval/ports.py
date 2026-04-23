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

"""Cross-domain contracts owned by the ``retrieval`` domain.

Lesson 9a-quad (single-direction cross-domain bridge): protocols used as
dependency inversion points are owned by the CONSUMER domain. The
PROVIDER domain structurally satisfies them without importing anything
from the consumer, which keeps the dependency graph a DAG and prevents
accidental cycles.

Retrieval consumes graph search context from ``knowledge_graph`` while
executing the hybrid search pipeline. The ``GraphSearchContract``
protocol captures the **only** shape retrieval reads — a single
``query_context`` coroutine returning an object with a string ``text``
payload. ``knowledge_graph`` constructs its own graph-index service
instance elsewhere and that instance happens to match this protocol;
the two modules never need to cross-import at Python level.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GraphQueryContext(Protocol):
    """Minimal contract for the graph query context returned by
    ``GraphSearchContract.query_context``.

    Retrieval reads a single field, ``text``, which is concatenated
    into the hybrid-search recall set. Any other provider-specific
    state (source chunks, metadata, timing) is intentionally out of
    scope for this contract — retrieval must not introspect it.
    """

    text: str


@runtime_checkable
class GraphSearchContract(Protocol):
    """Knowledge-graph search contract consumed by ``retrieval``.

    Implementations live in the ``knowledge_graph`` domain (currently
    ``aperag.domains.knowledge_graph.graphindex.service.GraphIndexService``) but this domain
    does **not** import that module. The ``retrieval`` pipeline
    type-binds its dependency to this Protocol; the concrete service
    structurally satisfies it. That keeps the Phase 0 strict ban clean
    (no cross-domain imports) and prevents a future ``knowledge_graph``
    rewrite from breaking ``retrieval`` source-compat unless the
    contract itself changes.
    """

    async def query_context(
        self,
        *,
        collection_id: str,
        query: str,
        top_k: int,
    ) -> GraphQueryContext:
        """Return graph-sourced context text for the given query.

        An empty string ``text`` means the graph had no relevant
        evidence for the query — this is a **legitimate empty**, not
        an error, and the caller should treat it as "graph contributed
        nothing this round" rather than retry / fall back.
        """
        ...


__all__ = ["GraphQueryContext", "GraphSearchContract"]
