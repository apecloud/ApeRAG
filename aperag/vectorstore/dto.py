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

"""Backend-neutral data types for the vector store abstraction.

What belongs here
-----------------
Anything that describes **what** we're storing / querying, expressed in
terms that any serious vector DB (Qdrant, pgvector, Milvus, ...) can
support. Concrete backends translate these DTOs into their native shapes.

What does NOT belong here
-------------------------
* ``qdrant_client`` / ``psycopg`` / ``pymilvus`` types. If a type here
  imports from a backend SDK, the abstraction is leaking.
* LlamaIndex ``BaseNode`` / ``TextNode``. Those are a layer above this
  module — callers build ``VectorPoint`` from a node if they want.

Every DTO here is a frozen dataclass on purpose:

* structural equality → unit tests pin exact shapes without setting up
  fakes;
* hashability → callers can dedupe filter trees / query requests before
  hitting the connector;
* immutability → no "someone mutated my query mid-flight" class of bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from aperag.vectorstore.filters import VectorFilter

# ---------------------------------------------------------------------------
# identity & shape
# ---------------------------------------------------------------------------

_VALID_DISTANCES = frozenset({"cosine", "dot", "euclid"})


@dataclass(frozen=True)
class VectorShape:
    """Identifies the "physical type" of a vector in a way every backend
    cares about.

    In multitenant layouts (Qdrant's ``aperag_vectors_<size>_<distance>`` or
    pgvector's corresponding table) the ``(size, distance)`` pair decides
    which physical shard the tenant's vectors live in. We normalize
    ``distance`` to a lowercase canonical form so that callers passing
    ``"Cosine"``, ``"COSINE"``, or ``"cosine"`` all map to the same shape.
    """

    size: int
    distance: str

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or self.size <= 0:
            raise ValueError(f"VectorShape.size must be a positive int, got {self.size!r}")
        canonical = str(self.distance).strip().lower()
        if canonical == "euclidian":
            canonical = "euclid"  # common typo we've seen in configs
        if canonical not in _VALID_DISTANCES:
            raise ValueError(f"VectorShape.distance must be one of {sorted(_VALID_DISTANCES)}, got {self.distance!r}")
        # Frozen dataclass: bypass __setattr__ protection for normalization.
        object.__setattr__(self, "distance", canonical)

    @property
    def canonical(self) -> str:
        """The lowercase canonical string used to form collection names.

        Kept as a property (not a raw attribute) so that future shape
        normalisations don't silently desync physical names.
        """
        return self.distance


@dataclass(frozen=True)
class TenantRef:
    """Per-tenant identity.

    Wrapping the tenant id in a dataclass (instead of passing bare strings)
    means every place that accepts "a tenant" is type-visible; a caller
    that accidentally passes an ApeRAG user id instead of a Collection id
    is a mypy error, not a silent cross-tenant bug.
    """

    id: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(f"TenantRef.id must be a non-empty str, got {self.id!r}")


# ---------------------------------------------------------------------------
# point — unit of write / retrieve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorPoint:
    """A single stored vector.

    * ``id`` is always ``str``. Qdrant supports UUID/int, pgvector uses
      UUID, Milvus supports varchar/int — we normalize at the connector
      boundary so callers don't branch on backend. All three accept UUID
      strings as ids, which is what ApeRAG writes.
    * ``vector`` is a plain ``list[float]`` — not numpy, not bytes. If a
      backend hands us arrays we convert at the boundary; callers never
      see numpy unless they specifically asked a backend for it.
    * ``payload`` is a free-form dict. LlamaIndex conventions like
      ``_node_content`` live **inside** the dict; they are not an
      abstraction-level concept. Writers typically put ``{"text": ...,
      "metadata": {...}}`` at the top level so all backends have the
      same read shape.
    """

    id: str
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("VectorPoint.id must be a non-empty str")
        if not isinstance(self.vector, list):
            raise TypeError(f"VectorPoint.vector must be list[float], got {type(self.vector).__name__}")
        if not isinstance(self.payload, dict):
            raise TypeError(f"VectorPoint.payload must be a dict, got {type(self.payload).__name__}")


# ---------------------------------------------------------------------------
# search request / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryRequest:
    """A single search intent.

    ``hints`` is the backend-specific escape hatch. Anything genuinely
    cross-backend (top_k, filter, threshold) is a first-class field;
    anything that only makes sense to one backend (e.g. Qdrant's
    ``hnsw_ef`` or pgvector's ``ef_search``) goes here as an opaque dict.
    The connector reads the hints it understands and silently ignores
    the rest — that keeps callers portable by default.

    ``with_vectors=True`` asks the connector to include the stored
    embedding in ``SearchHit.vector``. Most callers don't need it;
    requesting it costs real bytes over the wire.
    """

    embedding: List[float]
    top_k: int = 10
    flt: Optional[VectorFilter] = None
    score_threshold: Optional[float] = None
    with_vectors: bool = False
    hints: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.embedding, list) or not self.embedding:
            raise ValueError("QueryRequest.embedding must be a non-empty list[float]")
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(f"QueryRequest.top_k must be a positive int, got {self.top_k!r}")


@dataclass(frozen=True)
class SearchHit:
    """A single result from ``search()``.

    Intentionally minimal: id, score, payload, optional vector. Anything
    higher-level (ApeRAG's ``DocumentWithScore`` with its
    ``text`` / ``metadata`` split, plus LlamaIndex ``_node_content``
    reconstruction) is the caller's responsibility, implemented via the
    ``flatten_node_payload`` helper so neither the connector nor each
    call site has to know about LlamaIndex internals.

    ``score`` is the **normalized similarity** in ``[0, 1]`` where higher
    is better, regardless of which distance metric (cosine / euclid /
    dot) the underlying backend used. Adapters apply
    :func:`aperag.vectorstore.base.normalize_score` before constructing
    this DTO. The validator below enforces the range so a backend that
    forgets to normalize is caught at the boundary instead of silently
    polluting downstream score-threshold logic (task #61 P0-B).
    """

    id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None

    def __post_init__(self) -> None:
        # NaN comparisons all return False, so the explicit check catches
        # both "out of range" and "score is NaN" without branching.
        s = self.score
        if not isinstance(s, (int, float)):
            raise TypeError(f"SearchHit.score must be float in [0, 1], got {type(s).__name__}")
        if not (0.0 <= float(s) <= 1.0):
            raise ValueError(
                f"SearchHit.score must lie in the normalized [0, 1] range, got {s!r}; "
                "adapters must call aperag.vectorstore.base.normalize_score before "
                "constructing SearchHit (task #61 P0-B contract)."
            )


# ---------------------------------------------------------------------------
# payload helpers (LlamaIndex ↔ flat)
# ---------------------------------------------------------------------------


def flatten_node_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Qdrant/pgvector payload into a ``{text, metadata}`` shape.

    Old data (written via ``LlamaIndex.QdrantVectorStore.add(nodes)``) carries
    a ``_node_content`` string holding the serialised node. New data
    (written via ``VectorStoreConnector.upsert(points)``) carries ``text`` and
    ``metadata`` at the top level. To let every reader use a single code
    path, this helper returns a dict with guaranteed ``text`` and ``metadata``
    keys, preferring the flat form when both are present (new data wins over
    legacy blobs).

    Returns a **new** dict; the input is never mutated. ``payload`` is
    always returned inside the result under ``_raw`` so callers who want
    the original (e.g. for ``_node_content.relationships`` source discovery)
    can still reach it.
    """
    import json as _json

    if not isinstance(payload, dict):
        return {"text": None, "metadata": {}, "_raw": {}}

    text = payload.get("text")
    metadata = payload.get("metadata")

    node_content_raw = payload.get("_node_content")
    node_content: Optional[Dict[str, Any]] = None
    if isinstance(node_content_raw, str):
        try:
            parsed = _json.loads(node_content_raw)
            if isinstance(parsed, dict):
                node_content = parsed
        except _json.JSONDecodeError:
            node_content = None

    if text is None and node_content is not None:
        t = node_content.get("text")
        if isinstance(t, str):
            text = t

    if metadata is None and node_content is not None:
        m = node_content.get("metadata")
        if isinstance(m, dict):
            metadata = m

    if not isinstance(metadata, dict):
        # Last resort: strip obvious storage-layer fields and return
        # whatever scalar-ish fields are left. Matches the pre-refactor
        # fallback exactly, so records written by external tools still
        # round-trip.
        metadata = {k: v for k, v in payload.items() if k not in ("_node_content", "text")}

    # If the node_content carried relationships and metadata.source is
    # still missing, try to derive it — this mirrors what the old
    # qdrant_connector._convert_scored_point... did. Isolated here so no
    # connector needs to know about it.
    if node_content is not None and metadata.get("source") is None:
        rels = node_content.get("relationships")
        if isinstance(rels, dict):
            parent = rels.get("1")
            if isinstance(parent, dict):
                parent_meta = parent.get("metadata")
                if isinstance(parent_meta, dict):
                    src = parent_meta.get("source")
                    if isinstance(src, str) and src:
                        import os as _os

                        metadata = {**metadata, "source": _os.path.basename(src)}

    return {"text": text, "metadata": dict(metadata), "_raw": dict(payload)}


__all__ = [
    "VectorShape",
    "TenantRef",
    "VectorPoint",
    "QueryRequest",
    "SearchHit",
    "flatten_node_payload",
]


# ---------------------------------------------------------------------------
# backward-compat exports
# ---------------------------------------------------------------------------
# ``aperag.vectorstore.base`` used to export ``VectorPoint`` directly. Keep
# the import path working so callers don't have to churn all at once.
Scalar = Union[str, int, float, bool]  # re-exported from filters for convenience
