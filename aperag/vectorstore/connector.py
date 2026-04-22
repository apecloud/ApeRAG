# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dispatch adapter: pick the right concrete connector for the configured
backend. Keep this file tiny and declarative — adding a new backend is
one ``case`` branch."""

from typing import Any, Dict


class VectorStoreConnectorAdaptor:
    """Thin, lazy-import dispatch around backend-specific connectors.

    The lazy imports matter: each backend pulls in a heavy SDK
    (qdrant_client for Qdrant; pgvector + psycopg for pgvector). We load
    only what the configured ``vector_store_type`` asks for so CI jobs
    that only exercise one backend don't force the other backend's
    dependency graph.
    """

    def __init__(self, vector_store_type: str, ctx: Dict[str, Any], **kwargs: Any) -> None:
        self.ctx = ctx
        self.vector_store_type = vector_store_type

        match vector_store_type:
            case "qdrant":
                from aperag.vectorstore.qdrant_connector import QdrantVectorStoreConnector

                self.connector = QdrantVectorStoreConnector(ctx, **kwargs)
            case "pgvector":
                from aperag.vectorstore.pgvector_connector import PgvectorVectorStoreConnector

                self.connector = PgvectorVectorStoreConnector(ctx, **kwargs)
            case _:
                raise ValueError(f"unsupported vector store type: {vector_store_type!r}")
