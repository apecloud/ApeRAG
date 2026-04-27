# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 7 task #10 close-out — grep-zero verification helper.

Per @Bryce's pre-check matrix (msg=11dd3b1b) + spec §K.12.10: once
task #10 deletes the legacy ``aperag/domains/knowledge_graph/
graphindex/`` package + drops the ``graphindex_*`` tables + cleans up
the residual call sites, the repository must contain **zero** hits
for a documented set of legacy patterns. This file pins those grep
contracts as runnable assertions so:

  * a future maintainer can `pytest` the close-out invariant locally;
  * CI catches any accidental re-introduction (e.g. a copy-paste from
    Wave-6-or-earlier docs / older branches);
  * task #10's PR has a binary, automatable "did the cleanup land?"
    gate beyond eyeballing 1000+ LOC of deletes.

Layered the same way the Wave 7 task #11 narrative scaffold (PR #1760)
is — module-level skip behind ``_TASK10_LEGACY_DELETED = False`` until
task #10 actually deletes the package, at which point the flag flips
True and these assertions become the close-out gate. Same PR that
flips the flag (or a same-author follow-up) is responsible for ensuring
all 10 patterns return 0.

Patterns enumerated per @Bryce's task #10 plan + §K.12.10 spec:

  1. ``from aperag.domains.knowledge_graph.graphindex`` — primary
     legacy import surface. Production callers (6 files / 11 sites
     per Bryce msg=11dd3b1b Pattern 1 v1) must all migrate.
  2. ``graphindex_nodes`` / ``graphindex_edges`` — table names. After
     alembic drop the only allowed mention is the migration script
     itself (excluded via path filter).
  3. ``import graphindex`` — bare ``import`` form (separate from
     ``from ... import``).
  4. ``_sync_entity_relation_vectors`` — Wave 6 vector-sync helper
     superseded by GraphModalityWorker.sync() Phase 3 (W7-3).
  5. ``_compact_oversized_descriptions`` — Wave 6 compactor entry
     superseded by GraphCompactor (W7-2).
  6. ``_summarize_description`` — Wave 6 summariser superseded by
     LLM unified-description path inside GraphCompactor.
  7. ``_fallback_truncate`` — Wave 6 fallback path superseded by
     GraphCompactor's graceful-degrade branch (renamed inside the new
     module — exception below allows the new name).
  8. ``_delete_removed_shadow_vectors`` — Wave 6 shadow-vector
     deletion superseded by Phase 3 snapshot-diff.
  9. ``query_context`` — Wave 6 retrieval entry superseded by
     ``GraphSearchService.compose_context`` (W7-5). Exception below
     allows the new ``compose_context`` mention itself.
 10. ``merge_entities`` — Wave 6 legacy method must NOT remain on
     ``GraphIndexService``; only allowed mention is on the new
     ``GraphCurationService.merge_entities`` (W7-6).

Each test runs the corresponding ``rg`` invocation and asserts that
the matched-line count, after subtracting the documented exception
paths, is 0.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------
# Gate. Until task #10 deletes the legacy package, every test would
# fail (because legacy code is intentionally still present). The flag
# flips True in the same PR that lands task #10's deletes.
# ---------------------------------------------------------------------

_TASK10_LEGACY_DELETED = False  # flip True in the task #10 close-out
# PR (Bryce). This file's contract: when the flag is True, every
# pattern below MUST return 0 hits (modulo the documented exception
# paths). When False, the file collects + skips so it never blocks
# pre-#10 PRs.

# Repository root, derived from this test file's location so the
# subprocess invocations work whether pytest is run from the repo
# root, the tests/ dir, or any other CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _have_ripgrep() -> bool:
    return shutil.which("rg") is not None


pytestmark = [
    pytest.mark.skipif(
        not _TASK10_LEGACY_DELETED,
        reason=(
            "task #10 (legacy graphindex package delete) has not landed yet; "
            "grep-zero contract starts when the close-out PR flips "
            "_TASK10_LEGACY_DELETED to True."
        ),
    ),
    pytest.mark.skipif(
        not _have_ripgrep(),
        reason="ripgrep (rg) binary not on PATH — needed for the grep-zero contract.",
    ),
]


def _rg_count(
    pattern: str,
    *,
    extra_paths: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> tuple[int, list[str]]:
    """Run ``rg`` and return ``(line_count, matched_lines)``.

    ``extra_paths`` (default: ``aperag/`` + ``tests/``) limits the
    search root. ``exclude_globs`` is forwarded to ``rg --glob`` to
    skip patterns like ``!**/migrations/**``. ``exclude_paths`` is a
    post-filter applied to the matched lines (path-prefix or
    substring match against the line) to drop documented exceptions
    that ``--glob`` cannot express precisely.
    """
    paths = extra_paths or ["aperag", "tests"]
    cmd: list[str] = ["rg", "--with-filename", "--line-number", "--no-heading", pattern]
    for glob in exclude_globs or []:
        cmd.extend(["--glob", glob])
    cmd.extend(paths)

    result = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # rg exit code: 0 = matches found, 1 = no matches, 2 = error.
    if result.returncode == 1:
        return 0, []
    if result.returncode != 0:
        raise RuntimeError(f"rg failed for pattern {pattern!r}: stderr={result.stderr!r}")

    lines = [line for line in result.stdout.splitlines() if line]
    if exclude_paths:
        kept = []
        for line in lines:
            if not any(exc in line for exc in exclude_paths):
                kept.append(line)
        lines = kept
    return len(lines), lines


def _assert_zero(
    pattern: str,
    *,
    exclude_globs: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    note: str = "",
) -> None:
    count, hits = _rg_count(
        pattern,
        exclude_globs=exclude_globs,
        exclude_paths=exclude_paths,
    )
    assert count == 0, (
        f"Wave 7 task #10 grep-zero violated for pattern {pattern!r}.\n"
        f"{note}\n"
        f"Matches ({count}):\n  " + "\n  ".join(hits)
    )


# ---------------------------------------------------------------------
# 10 grep-zero patterns. Each test names the pattern + the intended
# Wave 7 successor so a maintainer hitting a failure knows where the
# call should have moved to.
# ---------------------------------------------------------------------


def test_no_from_aperag_domains_knowledge_graph_graphindex_imports():
    """Pattern 1 — primary legacy import surface. Per Bryce
    msg=11dd3b1b Pattern 1 v1, 6 production files / 11 sites all
    migrate during task #10."""
    _assert_zero(
        r"from aperag\.domains\.knowledge_graph\.graphindex",
        note=(
            "Migrate to one of the new Wave 7 surfaces: "
            "GraphSearchService (retrieval reads), "
            "GraphCurationService (user merges), "
            "LineageEntityMerger (atomic merge orchestration), "
            "or LineageGraphStore (write path)."
        ),
    )


def test_no_graphindex_table_name_references():
    """Pattern 2 — ``graphindex_nodes`` / ``graphindex_edges`` table
    names. After the alembic drop in task #10's close-out migration,
    the only place these names may appear is the migration script
    itself."""
    _assert_zero(
        r"graphindex_(nodes|edges)",
        exclude_globs=["!aperag/migration/**"],
        note=(
            "After task #10's alembic drop migration, no production "
            "code may reference these table names. The migration "
            "script itself is excluded."
        ),
    )


def test_no_bare_import_graphindex():
    """Pattern 3 — bare ``import graphindex`` form. Catches any
    leftover module-level imports that the wider Pattern 1 v1 grep
    would miss (e.g. ``import aperag.domains.knowledge_graph.graphindex``)."""
    _assert_zero(
        r"^\s*import\s+aperag\.domains\.knowledge_graph\.graphindex",
        note=("Same migration targets as Pattern 1 — the bare import form is just a different code shape."),
    )


def test_no_legacy_sync_entity_relation_vectors():
    """Pattern 4 — Wave 6 helper superseded by
    GraphModalityWorker.sync() Phase 3 (W7-3, PR #1757)."""
    _assert_zero(
        r"_sync_entity_relation_vectors",
        note="Use GraphModalityWorker.sync() Phase 3 (Wave 7 W7-3).",
    )


def test_no_legacy_compact_oversized_descriptions():
    """Pattern 5 — Wave 6 compactor entry superseded by GraphCompactor
    (W7-2)."""
    _assert_zero(
        r"_compact_oversized_descriptions",
        note="Use GraphCompactor (Wave 7 W7-2).",
    )


def test_no_legacy_summarize_description():
    """Pattern 6 — Wave 6 summariser superseded by the LLM unified-
    description path inside GraphCompactor (W7-2)."""
    _assert_zero(
        r"_summarize_description",
        note=("Use GraphCompactor's LLM unified-description path (Wave 7 W7-2)."),
    )


def test_no_legacy_fallback_truncate():
    """Pattern 7 — Wave 6 fallback path superseded by
    GraphCompactor's graceful-degrade branch. The new module may
    expose a renamed helper; if so, document the new name as an
    explicit allowed substring under ``exclude_paths``.

    Today this is a strict zero — flip the exception list here only
    if task #10 (or a follow-up) intentionally renames-and-keeps the
    fallback under a new namespace."""
    _assert_zero(
        r"_fallback_truncate",
        note=(
            "Use GraphCompactor's graceful-degrade branch (Wave 7 "
            "W7-2). If renamed-and-kept, add the new path to the "
            "exception list."
        ),
    )


def test_no_legacy_delete_removed_shadow_vectors():
    """Pattern 8 — Wave 6 shadow-vector deletion superseded by Phase 3
    snapshot-diff (W7-3)."""
    _assert_zero(
        r"_delete_removed_shadow_vectors",
        note=("Use GraphModalityWorker.sync() Phase 3 snapshot-diff (Wave 7 W7-3)."),
    )


def test_no_legacy_query_context_except_compose_context():
    """Pattern 9 — Wave 6 retrieval entry superseded by
    ``GraphSearchService.compose_context``. The legacy method name
    is ``query_context``; the new method is ``compose_context``. We
    grep the old name explicitly so the new method's mention does
    not need an exception."""
    _assert_zero(
        r"\bquery_context\b",
        note=(
            "Use GraphSearchService.compose_context (Wave 7 W7-5). "
            "Note: the new method intentionally has a different name."
        ),
    )


def test_no_legacy_merge_entities_on_graphindex():
    """Pattern 10 — Wave 6 ``GraphIndexService.merge_entities`` must
    NOT remain. The new home is ``GraphCurationService.merge_entities``
    (W7-6) called via ``LineageEntityMerger`` (W7-6 chunk 2). Since
    the new method shares the name, this grep targets the legacy
    class binding only."""
    _assert_zero(
        r"GraphIndexService\.merge_entities",
        note=("Use GraphCurationService.merge_entities via LineageEntityMerger (Wave 7 W7-6)."),
    )
