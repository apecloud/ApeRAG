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

"""Narrow contracts owned by the ``knowledge_graph`` domain.

The Phase 0 strict ban forbids code under ``aperag/domains/**`` from
importing ``aperag.db.models``. Several graph paths (the labels /
subgraph / curation flows) need to read a handful of fields from the
SQLAlchemy ``Collection`` row so the factory functions in
``aperag.domains.knowledge_graph.graphindex.integration`` can build a service instance. Rather
than cross the forbidden boundary or drag the whole SQLAlchemy surface
into the domain, this module declares a minimal ``Protocol`` that
pins only the fields the domain actually reads. The real ``Collection``
row structurally satisfies the protocol without either side importing
the other.

Lesson 9a-quad (single-direction cross-domain bridge): unlike
``retrieval.ports.GraphSearchContract`` (owned by a consumer, against a
provider in ``knowledge_graph``), ``CollectionRow`` is an *internal*
knowledge_graph port — it abstracts a legacy infrastructure type the
domain depends on, in preparation for the Phase 3 DB split where the
concrete ``Collection`` model will move under ``knowledge_base`` /
``collection`` canonically.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CollectionRow(Protocol):
    """Structural view of ``aperag.db.models.Collection`` that
    ``knowledge_graph`` is allowed to read.

    Only the fields the domain explicitly touches are declared — every
    other column on the legacy model stays invisible on purpose so a
    careless downstream caller cannot accidentally bind a larger
    surface than intended. The fields here mirror what the graphindex
    integration + curation paths currently read:

    * ``id`` — collection identifier for graph-index wiring;
    * ``user`` — owner user id for provider-key lookup;
    * ``title`` — included for logging / error messages only;
    * ``config`` — raw JSON-ish config dict parsed by
      ``aperag.schema.utils.parseCollectionConfig``. Typed as ``Any``
      because the legacy model stores it as a JSON column; the
      ``parseCollectionConfig`` helper hydrates it into a typed
      Pydantic object at the point of use.
    """

    id: str
    user: str
    title: str
    config: Any


__all__ = ["CollectionRow"]
