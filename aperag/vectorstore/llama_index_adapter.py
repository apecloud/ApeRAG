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

"""Adapters between LlamaIndex primitives and the backend-neutral DTOs.

Keeping these helpers in one place means only this module knows about
``BaseNode`` + ``TextNode``. Every backend stays ignorant of LlamaIndex,
which is what lets us swap Qdrant ↔ pgvector without touching the
indexer pipelines.
"""

from __future__ import annotations

from typing import Iterable, List

from llama_index.core.schema import BaseNode

from aperag.vectorstore.dto import VectorPoint

_FILTERABLE_METADATA_KEYS = ("indexer", "chat_id", "collection_id")


def node_to_vector_point(node: BaseNode, *, tenant_id: str | None = None) -> VectorPoint:
    """Convert a LlamaIndex node into the backend-neutral ``VectorPoint``.

    * ``id`` = ``node.node_id`` (LlamaIndex auto-generates UUIDs for
      TextNode, which is what every ApeRAG writer uses).
    * ``vector`` = ``node.embedding`` — must already be computed;
      upserting a node without an embedding is a caller bug.
    * ``payload`` carries the *flat* ``{text, metadata}`` shape that new
      data uses, AND mirrors query-time filterable keys
      (``indexer``/``chat_id``/``collection_id``) at the top level so the
      filter DSL — which still addresses flat payload keys — continues to
      hit new writes. Without this mirror, ``Eq("indexer", ...)`` and
      ``Eq("chat_id", ...)`` would silently miss every point written
      through this adapter (the keys would live only under
      ``metadata.*``). We deliberately do NOT flatten the rest of
      ``metadata``: only keys the filter contract already knows about are
      lifted. We also do NOT serialize the entire node into a
      ``_node_content`` string — that convention belonged to LlamaIndex's
      QdrantVectorStore and locks readers into LlamaIndex forever. The
      reader side (``dto.flatten_node_payload``) still understands
      ``_node_content`` for backward-read of pre-migration data.
    * ``metadata.collection_id`` is stamped with ``tenant_id`` when the
      caller didn't already set it, mirroring the defensive behavior
      ``embedding_utils`` used to rely on.
    """
    if node.embedding is None:
        raise ValueError(f"node {node.node_id!r} has no embedding; compute embeddings before upsert")

    metadata = dict(node.metadata or {})
    if tenant_id and not metadata.get("collection_id"):
        metadata["collection_id"] = tenant_id

    text: str
    try:
        text = node.get_content()
    except Exception:
        text = getattr(node, "text", "") or ""

    payload: dict = {"text": text, "metadata": metadata}
    for key in _FILTERABLE_METADATA_KEYS:
        value = metadata.get(key)
        if value is not None:
            payload[key] = value

    return VectorPoint(
        id=str(node.node_id),
        vector=list(node.embedding),
        payload=payload,
    )


def nodes_to_vector_points(nodes: Iterable[BaseNode], *, tenant_id: str | None = None) -> List[VectorPoint]:
    """Bulk version of ``node_to_vector_point``. Order-preserving."""
    return [node_to_vector_point(n, tenant_id=tenant_id) for n in nodes]


__all__ = ["node_to_vector_point", "nodes_to_vector_points"]
