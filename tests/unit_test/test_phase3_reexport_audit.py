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

The Phase 3 design-lock (@符炫炜 msg=226b2584 + Flag 4 resolution
msg=618c03fc) requires that once the knowledge_base / indexing / graph /
retrieval DB modules split out of ``aperag.db.models``, the legacy
aggregate module keeps re-exporting every moved symbol so the 76
pre-Phase-3 callers see no breakage. This file pins that contract:

* **G11** — ``test_aperag_db_models_reexports_full_phase3_set`` asserts
  that all 15 Phase 3 symbols (7 DB classes + 8 lifecycle enums) are
  attribute-accessible on ``aperag.db.models`` AND the matching 7
  tables are registered on ``Base.metadata``. Re-export drift will
  surface here long before an alembic ``autogenerate`` turns a missing
  class into a ``drop_table`` operation on main.

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

import re
from pathlib import Path

import aperag.db.models as legacy_aggregate
from aperag.db.base import Base

REPO_ROOT = Path(__file__).resolve().parents[2]

PHASE3_DB_CLASSES = (
    "Collection",
    "CollectionSummary",
    "Document",
    "DocumentIndex",
    "SearchHistory",
    "GraphCurationRun",
    "GraphCurationSuggestion",
)

PHASE3_ENUMS = (
    "CollectionStatus",
    "CollectionSummaryStatus",
    "CollectionType",
    "DocumentStatus",
    "DocumentIndexStatus",
    "DocumentIndexType",
    "GraphCurationRunStatus",
    "GraphCurationSuggestionStatus",
)

PHASE3_TABLES = (
    "collection",
    "collection_summary",
    "document",
    "document_index",
    "searchhistory",
    "graph_curation_runs",
    "graph_curation_suggestions",
)


def test_aperag_db_models_reexports_full_phase3_set():
    """G11 re-export audit: every Phase 3 DB class + enum + table is
    still reachable through ``aperag.db.models`` after the physical
    move into the per-domain DB modules.

    The 15-symbol list matches the exact canonical from Phase 3
    design-lock Flag 4 (@符炫炜 msg=618c03fc). Changing the list is a
    canonical change that should be done in the design-lock thread,
    not here.
    """

    missing_symbols: list[str] = []
    for name in (*PHASE3_DB_CLASSES, *PHASE3_ENUMS):
        if not hasattr(legacy_aggregate, name):
            missing_symbols.append(name)

    missing_tables: list[str] = []
    for table in PHASE3_TABLES:
        if table not in Base.metadata.tables:
            missing_tables.append(table)

    assert not missing_symbols and not missing_tables, (
        "Phase 3 re-export shim in `aperag.db.models` lost one or more "
        "symbols, or `Base.metadata` lost one or more Phase 3 tables. "
        "Re-export must mirror the exact 7 DB + 8 enum + 7 table list "
        "defined by the Phase 3 design-lock (@符炫炜 msg=618c03fc).\n"
        f"  missing re-export symbols: {missing_symbols}\n"
        f"  missing Base.metadata tables: {missing_tables}"
    )


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

    offenders = {name: paths for name, paths in duplicates.items() if len(paths) > 1}
    missing = [name for name in (*PHASE3_DB_CLASSES, *PHASE3_ENUMS) if name not in duplicates]

    assert not offenders and not missing, (
        "Phase 3 single-definition invariant broken. Each symbol should "
        "live in exactly one file; a missing class means the physical "
        "move never happened, a duplicate means the legacy copy was "
        "left behind.\n"
        f"  duplicate definitions: {offenders}\n"
        f"  symbols with no definition: {missing}"
    )
