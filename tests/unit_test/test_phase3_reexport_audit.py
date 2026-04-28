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

"""Phase 3 G11 + G13 audit tests.

Phase 7 B5 (per ``docs/modularization/cleanup-inventory.md §B5``)
stripped the legacy ``aperag.db.models`` cross-domain re-export block;
every Phase 3 symbol now lives at its canonical
``aperag.domains.<d>.db.models`` path.

* **G11 (post-B5)** — ``test_phase3_classes_resolve_on_canonical_domain_paths``
  imports each of the 15 Phase 3 symbols from its canonical per-domain
  module AND asserts every matching table is registered on
  ``Base.metadata``. Registration drift still surfaces here before
  alembic autogen would turn a missing class into a ``drop_table``.

* **G13** — ``test_phase3_classes_have_single_definition_site`` scans
  every ``aperag/**/*.py`` for class definitions matching the 15
  Phase 3 names and asserts each appears exactly once. If a future
  refactor duplicates a class body (e.g. forgets to delete the legacy
  definition when moving the class to its domain module), SQLAlchemy
  would raise ``InvalidRequestError: Table ... is already defined``
  at import time; this test catches the dup much earlier with a clear
  pointer to the offending file pair.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from aperag.db.base import Base

# Ensure every per-domain model module loads so Base.metadata sees them.
for _domain in (
    "agent_runtime",
    "conversation",
    "evaluation",
    "governance",
    "identity",
    "knowledge_base",
    "knowledge_graph",
    "marketplace",
    "model_platform",
    "retrieval",
):
    importlib.import_module(f"aperag.domains.{_domain}.db.models")

# Wave 3 T3.1: ``DocumentIndex`` was moved out of the per-domain
# namespace (``aperag.domains.indexing.db.models``) into the new
# celery-redesign canonical location ``aperag.indexing.models``.
# Import it here so ``Base.metadata.tables['document_index']`` is
# populated for the table-presence assertion below.
importlib.import_module("aperag.indexing.models")

REPO_ROOT = Path(__file__).resolve().parents[2]

PHASE3_DB_CLASSES = (
    "Collection",
    # Wave 10 §K.13: ``CollectionSummary`` ORM hard-cut removed.
    "Document",
    "DocumentIndex",
    "SearchHistory",
    "GraphCurationRun",
    "GraphCurationSuggestion",
)

PHASE3_ENUMS = (
    "CollectionStatus",
    # Wave 10 §K.13: ``CollectionSummaryStatus`` removed alongside
    # ``CollectionSummary`` ORM hard-cut.
    "CollectionType",
    "DocumentStatus",
    "GraphCurationRunStatus",
    "GraphCurationSuggestionStatus",
)
# ``DocumentIndexStatus`` / ``DocumentIndexType`` were removed in Wave 3
# T3.1 alongside ``aperag/domains/indexing/db/models.py``. The new
# ``aperag/indexing/models.py`` exposes ``IndexStatus`` + ``Modality``
# instead, but those are not Phase-3-canonical Domain DB enums and live
# outside the ``aperag.domains.<d>.db.models`` namespace this audit
# enforces — so they are intentionally not added back here.

PHASE3_TABLES = (
    "collection",
    # Wave 10 §K.13: ``collection_summary`` table dropped via alembic
    # migration ``e1a2b3c4d5f6`` (Chunk B).
    "document",
    "document_index",
    "searchhistory",
    "graph_curation_runs",
    "graph_curation_suggestions",
)


# Each Phase 3 symbol now lives on exactly one canonical domain module
# (``aperag.domains.<d>.db.models``). Phase 7 B5 dropped the aggregate
# re-export block at ``aperag.db.models`` that historically carried the
# same names.
PHASE3_SYMBOL_TO_MODULE = {
    "Collection": "aperag.domains.knowledge_base.db.models",
    # Wave 10 §K.13: ``CollectionSummary`` removed.
    "Document": "aperag.domains.knowledge_base.db.models",
    # Wave 3 T3.1: the canonical ``DocumentIndex`` ORM lives in
    # ``aperag/indexing/models.py`` (the celery-redesign §F.1 row).
    # The legacy ``aperag/domains/indexing/db/models.py:DocumentIndex``
    # was hard-deleted in the same commit.
    "DocumentIndex": "aperag.indexing.models",
    "SearchHistory": "aperag.domains.retrieval.db.models",
    "GraphCurationRun": "aperag.domains.knowledge_graph.db.models",
    "GraphCurationSuggestion": "aperag.domains.knowledge_graph.db.models",
    "CollectionStatus": "aperag.domains.knowledge_base.db.models",
    # Wave 10 §K.13: ``CollectionSummaryStatus`` removed.
    "CollectionType": "aperag.domains.knowledge_base.db.models",
    "DocumentStatus": "aperag.domains.knowledge_base.db.models",
    "GraphCurationRunStatus": "aperag.domains.knowledge_graph.db.models",
    "GraphCurationSuggestionStatus": "aperag.domains.knowledge_graph.db.models",
}


def test_phase3_classes_resolve_on_canonical_domain_paths():
    """G11 (post-Phase-7-B5) audit: every Phase 3 DB class + enum must
    resolve via its canonical ``aperag.domains.<d>.db.models`` path and
    every Phase 3 table must be registered on ``Base.metadata`` (so
    Alembic autogen keeps seeing them).

    The 15-symbol list matches the canonical from the Phase 3
    design-lock Flag 4 (@符炫炜 msg=618c03fc). Changing the list is a
    canonical change that should be done in the design-lock thread,
    not here.
    """

    missing_symbols: list[str] = []
    for name, module_path in PHASE3_SYMBOL_TO_MODULE.items():
        module = importlib.import_module(module_path)
        if not hasattr(module, name):
            missing_symbols.append(f"{module_path}.{name}")

    missing_tables: list[str] = []
    for table in PHASE3_TABLES:
        if table not in Base.metadata.tables:
            missing_tables.append(table)

    assert not missing_symbols and not missing_tables, (
        "Phase 3 canonical-location audit failed after Phase 7 B5.\n"
        f"  missing domain-module symbols: {missing_symbols}\n"
        f"  missing Base.metadata tables: {missing_tables}"
    )


# Wave 3 T3.1: the legacy ``aperag/domains/indexing/db/models.py:
# DocumentIndex`` was deleted alongside the entire Celery indexing
# layer; the only remaining ``class DocumentIndex(Base):`` lives in
# ``aperag/indexing/models.py`` (table ``document_index``). The Wave
# 1/2 transitional dup allowlist is therefore intentionally empty —
# leaving the empty mapping (rather than dropping the symbol) keeps
# the call site below stable for any future short-lived duplicates.
WAVE_1_2_TEMPORARY_DUP_ALLOWLIST: dict[str, frozenset[str]] = {}


def test_phase3_classes_have_single_definition_site():
    """G13 no-duplicate-registration audit: each Phase 3 class body
    must live in exactly one ``class Foo(Base):`` site across the
    entire ``aperag/`` tree. A duplicate — caused e.g. by forgetting
    to delete the legacy definition after a physical move — would
    raise ``sqlalchemy.exc.InvalidRequestError`` on import, but the
    failure is miles away from the missed-delete site. Catching it
    here keeps the blast radius small.
    """

    # Only ORM (``class Foo(Base):``) and string-enum (``class Foo(str, Enum):``)
    # definitions count. Pydantic schemas in ``aperag/schema/view_models.py``
    # reuse names like ``Collection`` / ``Document`` on purpose — those are
    # different classes and should not trip this invariant.
    orm_pattern = re.compile(r"^class (\w+)\s*\(\s*Base\s*\)\s*:")
    enum_pattern = re.compile(r"^class (\w+)\s*\(\s*str\s*,\s*Enum\s*\)\s*:")

    duplicates: dict[str, list[str]] = {}
    for path in (REPO_ROOT / "aperag").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in source.splitlines():
            match = orm_pattern.match(line) or enum_pattern.match(line)
            if not match:
                continue
            name = match.group(1)
            if name in PHASE3_DB_CLASSES or name in PHASE3_ENUMS:
                duplicates.setdefault(name, []).append(path.relative_to(REPO_ROOT).as_posix())

    offenders = {
        name: paths
        for name, paths in duplicates.items()
        if len(paths) > 1 and frozenset(paths) != WAVE_1_2_TEMPORARY_DUP_ALLOWLIST.get(name)
    }
    missing = [name for name in (*PHASE3_DB_CLASSES, *PHASE3_ENUMS) if name not in duplicates]

    assert not offenders and not missing, (
        "Phase 3 single-definition invariant broken. Each symbol should "
        "live in exactly one file; a missing class means the physical "
        "move never happened, a duplicate means the legacy copy was "
        "left behind.\n"
        f"  duplicate definitions: {offenders}\n"
        f"  symbols with no definition: {missing}"
    )
