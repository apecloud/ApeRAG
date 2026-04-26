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

"""Legacy alias module for the hybrid-search pipeline service.

The concrete implementation relocated to
``aperag.domains.retrieval.pipeline`` by the Phase 2 hard-cut. This
module is kept as a thin re-export until the Phase 3 DB-split PR
retires it together with the rest of ``aperag.service.*`` so that
existing test callers (``tests/unit_test/test_es_p0_contract.py``,
``monkeypatch`` targets referencing
``aperag.service.search_pipeline_service.*``) keep working without a
mass edit in this PR.
"""

from __future__ import annotations

from aperag.domains.retrieval.pipeline import (
    SearchPipelineService,
    search_pipeline_service,
)

# Re-exports for ``monkeypatch.setattr("aperag.service.search_pipeline_service.extract_keywords", ...)``
# style test fixtures that reach into the old module path. The
# underlying helpers now live in their canonical modules; the names
# below are provided as writable aliases so ``monkeypatch`` keeps
# working for legacy tests.
from aperag.indexing.keyword_extract import extract_keywords  # noqa: F401
from aperag.utils.utils import generate_fulltext_index_name  # noqa: F401

__all__ = ["SearchPipelineService", "search_pipeline_service"]
