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

"""Backend-neutral filter DSL for vector store queries.

Why this module exists
----------------------
Before this layer, ``ContextManager`` in ``aperag/context/context.py``
constructed ``qdrant_client.models.Filter`` directly. Any new vector-DB
backend (pgvector, Milvus, …) would have had to either add its own
``if self.vectordb_type == "…"`` branch at every filter-building site, or
share Qdrant's Filter class — both of which are dead-ends for the
"zero-maintenance, private delivery" target we've set for ApeRAG.

This DSL is the contract between "what the business wants to filter by"
and "how each backend expresses it". It stays tiny on purpose: every node
here is an extra translator we have to maintain per backend.

Design rules (read before adding a node)
----------------------------------------
* **Minimalism over expressiveness.** Add a node only when an existing
  caller genuinely cannot express itself with the current set. Range
  queries, geo, fuzzy match — not needed today, so not here today.
* **Scalars only.** Values are ``str | int | float | bool``. Structured
  matching (list-contains, dict-subset) belongs in dedicated operators
  (not ``Eq`` with a list), and we don't have a use case for them yet.
* **No backend leakage.** No import from ``qdrant_client``, ``psycopg``,
  ``pymilvus`` etc. may exist in this file, ever.
* **Structural equality.** All nodes are frozen dataclasses so they're
  hashable and comparable, which makes tests trivially write-able and
  caches safe.

Composition helpers
-------------------
``all_of(...)`` / ``any_of(...)`` short-circuit 0- and 1-element cases so
callers don't have to litter their code with length checks. Passing
``None`` parts is explicitly allowed and filtered — this lets callers
conditionally build filters without nested ``if`` ladders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

Scalar = Union[str, int, float, bool]


@dataclass(frozen=True)
class Eq:
    """Exact payload-field equality. Null / missing fields do not match."""

    key: str
    value: Scalar


@dataclass(frozen=True)
class In:
    """Match any of ``values`` (i.e. SQL ``IN``). Empty values tuple matches
    nothing; callers should guard against constructing ``In(key, [])``."""

    key: str
    values: tuple[Scalar, ...]


def _in(key: str, values: Sequence[Scalar]) -> "In":
    """Convenience builder accepting any sequence, normalising to tuple."""
    return In(key=key, values=tuple(values))


@dataclass(frozen=True)
class IsEmpty:
    """The payload field is either absent or stored as null. Used today for
    backward-compatibility with pre-``indexer``-field points; every vector
    store with payloads supports this natively."""

    key: str


@dataclass(frozen=True)
class And:
    """All parts must match. Empty parts tuple is a logic bug and rejected
    at construction; use ``all_of()`` to get None-short-circuiting."""

    parts: tuple["VectorFilter", ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("And requires at least one part; use all_of() for short-circuit")


@dataclass(frozen=True)
class Or:
    """At least one part must match. Same empty-tuple rule as ``And``."""

    parts: tuple["VectorFilter", ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("Or requires at least one part; use any_of() for short-circuit")


@dataclass(frozen=True)
class Not:
    """Inverts the inner filter. Useful for exclusions (e.g. "not chat")."""

    inner: "VectorFilter"


VectorFilter = Union[Eq, In, IsEmpty, And, Or, Not]
"""Union of all DSL node types.

Exported separately so type hints elsewhere read ``VectorFilter``, not
``Union[...]``. The concrete types are importable individually for
pattern matching / isinstance checks in translators."""


def all_of(*parts: Optional[VectorFilter]) -> Optional[VectorFilter]:
    """AND the given parts, skipping ``None``s.

    * ``all_of()`` or ``all_of(None, None)`` → ``None``  (no filter)
    * ``all_of(Eq(...))`` → ``Eq(...)``                  (degenerate And)
    * ``all_of(a, b, c)`` → ``And(parts=(a, b, c))``
    """
    kept = tuple(p for p in parts if p is not None)
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    return And(parts=kept)


def any_of(*parts: Optional[VectorFilter]) -> Optional[VectorFilter]:
    """OR the given parts, skipping ``None``s. Same short-circuit rules as
    ``all_of``."""
    kept = tuple(p for p in parts if p is not None)
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    return Or(parts=kept)


__all__ = [
    "Scalar",
    "Eq",
    "In",
    "_in",
    "IsEmpty",
    "And",
    "Or",
    "Not",
    "VectorFilter",
    "all_of",
    "any_of",
]
