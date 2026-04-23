# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""graphindex v2 — the only module ApeRAG business code should import for
knowledge-graph functionality.

Design rationale and scope decisions live in
``docs/zh-CN/design/graphindex_rewrite.md``.

Public exports:

* ``GraphIndexService``     — the facade.
* ``GraphIndexConfig``      — per-process configuration.
* ``PostgresGraphStore``    — the only shipped storage backend today.
* DTOs (``Chunk`` / ``Entity`` / ``Relation`` / ``GraphContext`` /
  ``KnowledgeGraph`` / ``IndexDocumentResult`` / ``DeleteDocumentResult``)
  — inputs and outputs of the service.

Everything else (``engine``, ``prompts``, ``models``, the storage base
Protocol) is implementation detail. Don't import from those paths in
business code.
"""

from aperag.domains.knowledge_graph.graphindex.config import GraphIndexConfig
from aperag.domains.knowledge_graph.graphindex.dto import (
    Chunk,
    DeleteDocumentResult,
    Entity,
    GraphContext,
    IndexDocumentResult,
    KnowledgeGraph,
    Relation,
)
from aperag.domains.knowledge_graph.graphindex.service import GraphIndexService
from aperag.domains.knowledge_graph.graphindex.storage import PostgresGraphStore

__all__ = [
    "GraphIndexService",
    "GraphIndexConfig",
    "PostgresGraphStore",
    "Chunk",
    "Entity",
    "Relation",
    "GraphContext",
    "KnowledgeGraph",
    "IndexDocumentResult",
    "DeleteDocumentResult",
]
