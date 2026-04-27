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

"""Canonical ``retrieval`` domain.

Owns the hybrid-search pipeline (vector / fulltext / graph / summary /
vision recall + rerank) and the thin persistence-aware wrapper that
backs ``POST|GET|DELETE /api/v2/collections/{id}/searches*``.

Cross-domain contract is captured by ``aperag.domains.retrieval.ports``:
``GraphSearchContract`` is owned here (consumer) and satisfied by the
graph-index service instance constructed elsewhere in ``knowledge_graph``.
"""
