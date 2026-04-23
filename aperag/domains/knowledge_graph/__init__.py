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

"""Canonical ``knowledge_graph`` domain.

Owns the read-side graph service (labels / subgraph / node merge),
the curation-workflow orchestration surface, and the HTTP handlers for
``/api/v2/collections/{id}/graphs*`` and
``/api/v2/collections/{id}/graph-curation`` (merge-suggestions).

``aperag.graph_curation.*`` and ``aperag.domains.knowledge_graph.graphindex.*`` remain the
infrastructure-level truth modules for SQL-heavy curation state and
the underlying graph storage primitives. They are **not** domain code
and the Phase 0 strict ban does not forbid domain modules from
importing them — the forbidden aggregates are only
``aperag.service.*``, ``aperag.schema.view_models``, and
``aperag.db.models``. Phase 3 (DB split) is expected to fold those
two modules into canonical ``knowledge_graph/**`` subpackages after
their SQLAlchemy dependency has been lifted behind a repository port.
"""
