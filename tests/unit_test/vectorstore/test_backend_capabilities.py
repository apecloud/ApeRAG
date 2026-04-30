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

"""Static capability declaration tests (task #61 P1-V2 / P1-V4).

Each :class:`VectorStoreConnector` subclass declares its
``BACKEND_CAPABILITIES`` class-level attribute so callers
(API / FE / capability-aware optimizers) can read a machine-readable
declaration of what the adapter actually does — instead of guessing
from the backend name.

These tests pin the static values so:

1. A future refactor that drops a flag declaration on a concrete
   subclass fails fast (e.g. removing ``BACKEND_CAPABILITIES`` from
   :class:`PgvectorVectorStoreConnector` makes the flag undefined).
2. The cross-adapter capability matrix surfaces in code review as a
   single test file — consumers (cuiwenbo task #87 P1-D3 collection
   metadata Pydantic projection) read these values verbatim, so any
   change to the behaviour they describe must update both the adapter
   docstring + this test in the same PR.
"""

from __future__ import annotations

from aperag.vectorstore.base import VectorBackendCapabilities, VectorStoreConnector
from aperag.vectorstore.pgvector_connector import PgvectorVectorStoreConnector
from aperag.vectorstore.qdrant_connector import QdrantVectorStoreConnector

# ---------------------------------------------------------------------
# Shape — ensure both adapters declare the attribute and it's the
# right type.
# ---------------------------------------------------------------------


def test_pgvector_declares_backend_capabilities():
    caps = PgvectorVectorStoreConnector.BACKEND_CAPABILITIES
    assert isinstance(caps, VectorBackendCapabilities)


def test_qdrant_declares_backend_capabilities():
    caps = QdrantVectorStoreConnector.BACKEND_CAPABILITIES
    assert isinstance(caps, VectorBackendCapabilities)


def test_abstract_base_does_not_set_concrete_capabilities():
    """:class:`VectorStoreConnector` is abstract — only concrete
    subclasses declare a value. Keeping the base class assignment
    absent means a future subclass that forgets to declare gets a
    ``AttributeError`` at the call site, not a silent default."""
    # ``BACKEND_CAPABILITIES`` is a ``ClassVar`` annotation on the base
    # class without a value, so it doesn't actually exist on the base.
    assert "BACKEND_CAPABILITIES" not in VectorStoreConnector.__dict__


# ---------------------------------------------------------------------
# Capability matrix values — pinned by spec § 2.3 + task #83 P1-V*
# implementation. cuiwenbo task #87 P1-D3 reads these values for the
# collection metadata Pydantic projection, so changes here must be
# coordinated with that PR.
# ---------------------------------------------------------------------


def test_pgvector_supports_atomic_batch_upsert():
    """PGVector wraps the bulk INSERT ON CONFLICT in
    ``engine.begin()`` so a mid-batch failure rolls back the entire
    batch (task #61 P1-V2)."""
    assert PgvectorVectorStoreConnector.BACKEND_CAPABILITIES.supports_atomic_batch_upsert is True


def test_qdrant_does_not_support_atomic_batch_upsert():
    """Qdrant ``client.upsert(points, wait=True)`` is best-effort
    per-point — a mid-batch failure can leave some points written and
    others not (task #61 P1-V2). Callers needing atomic-batch
    semantics must chunk + verify."""
    assert QdrantVectorStoreConnector.BACKEND_CAPABILITIES.supports_atomic_batch_upsert is False


def test_pgvector_does_not_support_legacy_mode():
    """PGVector is multitenant-only — a per-tenant table layout would
    require dropping the shared-PG topology entirely (task #61 P1-V4)."""
    assert PgvectorVectorStoreConnector.BACKEND_CAPABILITIES.supports_legacy_mode is False


def test_qdrant_supports_legacy_mode():
    """Qdrant supports both legacy (``multitenant=False``,
    one-collection-per-tenant) and multitenant
    (``multitenant=True``, shared-collection + payload filter)
    layouts, controlled by the ``multitenant`` ctx flag (task #61
    P1-V4). New deployments default to multitenant."""
    assert QdrantVectorStoreConnector.BACKEND_CAPABILITIES.supports_legacy_mode is True
