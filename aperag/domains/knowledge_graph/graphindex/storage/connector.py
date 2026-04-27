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

"""Dispatch adaptor: pick the right concrete ``GraphStore`` for the
configured backend.

Mirrors ``aperag/vectorstore/connector.py``: thin, declarative, one
``case`` branch per backend. Adding a new graph backend is adding one
import + instantiation here, nothing else.

Lazy imports keep heavy driver dependencies (``neo4j``, ``nebula3-python``)
out of the module-import graph until someone actually configures them.
"""

from __future__ import annotations

from typing import Any, Dict


class GraphStoreAdaptor:
    """Thin lazy-import dispatch around backend-specific graph stores.

    Usage::

        adaptor = GraphStoreAdaptor("neo4j", ctx={...})
        await adaptor.store.upsert_entities(...)
    """

    def __init__(self, graph_db_type: str, ctx: Dict[str, Any]) -> None:
        self.graph_db_type = graph_db_type
        self.ctx = ctx

        match graph_db_type:
            case "postgresql":
                from aperag.domains.knowledge_graph.graphindex.storage.postgres import PostgresGraphStore

                engine = ctx.get("engine")
                if engine is None:
                    raise ValueError("GraphStoreAdaptor: 'engine' is required in ctx for the postgresql backend")
                self.store = PostgresGraphStore(engine=engine)

            case "neo4j":
                from aperag.domains.knowledge_graph.graphindex.storage.neo4j import Neo4jGraphStore

                self.store = Neo4jGraphStore(
                    uri=ctx["neo4j_uri"],
                    username=ctx.get("neo4j_username", "neo4j"),
                    password=ctx.get("neo4j_password", ""),
                )

            case "nebula":
                from aperag.domains.knowledge_graph.graphindex.storage.nebula import NebulaGraphStore

                self.store = NebulaGraphStore(
                    hosts=ctx["nebula_hosts"],
                    username=ctx.get("nebula_username", "root"),
                    password=ctx.get("nebula_password", ""),
                    space_prefix=ctx.get("nebula_space_prefix", "aperag"),
                )

            case _:
                raise ValueError(f"unsupported graph_db_type: {graph_db_type!r}")


__all__ = ["GraphStoreAdaptor"]
