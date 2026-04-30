"""Cross-source default value gate for ``graph_extraction_window_size`` (task #33 P3).

Background — task #30 B3 (PR #1925, merge ``43648f9``) locked
``graph_extraction_window_size`` default to ``2`` (sweet spot) per
``earayu2`` directive ``msg=adb0c366``. The lock landed in **four**
independent sources that all need to agree:

1. **Python const** — ``aperag/indexing/graph_extractor.py``
   ``_DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE``: the runtime fallback the
   extractor actually uses when a collection does not override the
   value.
2. **Pydantic Field** — ``aperag/schema/common.py``
   ``KnowledgeGraphConfig.graph_extraction_window_size``: the API
   schema's user-facing default (surfaced via the Field's ``examples``
   tuple and the description text).
3. **TS schema** — ``web/src/api-v2/schema.d.ts``: the auto-generated
   typed client schema's ``@example`` tag, consumed by frontend code.
4. **Spec doc** — ``docs/zh-CN/architecture/task-30-graph-chunk-window-spec-v1.md``
   § 3.1.1 + § 4.2 lock declaration: the architectural source of truth
   that PRs CR against.

PR #1925 itself surfaced the cross-source drift risk: the initial
commit only updated source 1 + 2 + 4. ``Weston`` ``msg=1b7d9bef``
BLOCKER 1 caught that ``schema.d.ts`` still carried the old default
``1``; ``huangheng`` ``msg=bf785b12`` NIT 1 caught § 3.1.1 line 85
still saying default ``1``. Both required a fix-forward commit. This
test gate exists so that next time only ``cicd-push`` lint+unit needs
to fail the PR — no manual cross-source inspection required, no
fix-forward round trip with reviewers acting as drift detectors.

Why a unit test (not a boundary test): ``tests/boundaries/`` is not
currently invoked by ``make test-unit`` / ``test-integration`` /
``cicd-push.yml`` (audit task #33 Layer 1 finding). ``tests/unit_test/``
runs on every push. Per simple-stable directive (earayu2
``msg=1224bec8``), the cheapest reliable gate is a unit test that runs
in the existing CI lane, not a new workflow file.

Scope discipline: this test pins the **default value parity** across
four sources only. It does not pin description text, override-recommendation
phrasing, or rationale wording — those evolve. If a future change
moves the default away from 2, this test will fail; the fix is to
update the Python const, Pydantic ``examples``, TS schema ``@example``,
and spec doc § 3.1.1 + § 4.2 in the same PR. The failure message
spells out which sources disagree so the operator does not have to
guess.

Sediment alignment: Lesson #13 v3 (cross-source default value
alignment, ``docs/zh-CN/architecture/task-17-cr-review-checklist.md``
§ 四) — this test is the codified gate Lesson #13 v3 was waiting for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPH_EXTRACTOR_PATH = REPO_ROOT / "aperag" / "indexing" / "graph_extractor.py"
SCHEMA_COMMON_PATH = REPO_ROOT / "aperag" / "schema" / "common.py"
TS_SCHEMA_PATH = REPO_ROOT / "web" / "src" / "api-v2" / "schema.d.ts"
SPEC_DOC_PATH = REPO_ROOT / "docs" / "zh-CN" / "architecture" / "task-30-graph-chunk-window-spec-v1.md"


def _python_const_default() -> int:
    """Read ``_DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE`` from ``graph_extractor.py``.

    Imports the module so the value matches what runtime actually uses;
    catches the case where the constant is shadowed / re-bound at import
    time.
    """
    from aperag.indexing.graph_extractor import (
        _DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE,
    )

    return _DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE


def _pydantic_field_default() -> int:
    """Read the ``examples`` value of ``KnowledgeGraphConfig.graph_extraction_window_size``.

    Pydantic ``Field(examples=[N])`` is the canonical default surfaced
    in OpenAPI + ``schema.d.ts`` ``@example`` generation, so it must
    match the Python const.
    """
    from aperag.schema.common import KnowledgeGraphConfig

    field = KnowledgeGraphConfig.model_fields["graph_extraction_window_size"]
    examples = field.examples
    assert examples, (
        "KnowledgeGraphConfig.graph_extraction_window_size Field is missing "
        "examples=[...]; the OpenAPI / TS schema default-value annotation "
        "depends on this. Add examples=[<canonical default>] in "
        "aperag/schema/common.py."
    )
    return int(examples[0])


_TS_SCHEMA_BLOCK_RE = re.compile(
    # Match the JSDoc block immediately preceding `graph_extraction_window_size?:`,
    # then capture the @example value inside that block.
    r"@example\s+(\d+)\s*\*/\s*\n\s*graph_extraction_window_size\?:",
    re.MULTILINE,
)


def _ts_schema_default() -> int:
    """Extract the ``@example`` integer attached to ``graph_extraction_window_size`` in ``schema.d.ts``.

    The TS schema is auto-generated from the Pydantic ``examples`` field,
    but it is committed to the repo (frontend imports it directly), so
    a stale regen drifts silently. This regex pins the lookup to the
    JSDoc block that immediately precedes the field declaration so we
    do not match the same ``@example 2`` in some other field block.
    """
    text = TS_SCHEMA_PATH.read_text(encoding="utf-8")
    match = _TS_SCHEMA_BLOCK_RE.search(text)
    assert match, (
        f"Could not locate JSDoc @example for graph_extraction_window_size in "
        f"{TS_SCHEMA_PATH}. Either the schema regen is missing the field, the "
        f"field name was renamed, or the JSDoc layout changed. Re-run the "
        f"OpenAPI -> TS schema regen and re-check this test."
    )
    return int(match.group(1))


_SPEC_LOCK_LINE_RE = re.compile(
    # Section 4.2 canonical lock declaration: **`graph_extraction_window_size = 2`**
    r"\*\*`graph_extraction_window_size\s*=\s*(\d+)`\*\*",
)
_SPEC_311_LINE_RE = re.compile(
    # Section 3.1.1 enumeration: **B3 lock default `2`**
    r"\*\*B3 lock default\s*`(\d+)`\*\*",
)


def _spec_doc_defaults() -> dict[str, int]:
    """Extract every locked default value from the task #30 spec doc.

    Two canonical lock sites in the spec:

    * § 3.1.1 — ``**B3 lock default `N`**`` in the schema-path enumeration
    * § 4.2  — ``**`graph_extraction_window_size = N`**`` in the lock
      chapter title line

    Both lines are part of the spec's locked-value contract. If either
    drifts away from the runtime default, ``CR 必对照架构文档`` (per
    earayu2 ``msg=f19f9fc5``) breaks down.
    """
    text = SPEC_DOC_PATH.read_text(encoding="utf-8")

    section_42_match = _SPEC_LOCK_LINE_RE.search(text)
    section_311_match = _SPEC_311_LINE_RE.search(text)

    assert section_42_match, (
        f"Could not locate § 4.2 lock line `**`graph_extraction_window_size = N`**` "
        f"in {SPEC_DOC_PATH}. Either the lock chapter was renamed/removed or the "
        f"markdown emphasis style changed. Restore the canonical lock line or "
        f"update this regex."
    )
    assert section_311_match, (
        f"Could not locate § 3.1.1 lock line `**B3 lock default `N`**` in "
        f"{SPEC_DOC_PATH}. Either § 3.1.1 was rewritten or the lock-shorthand "
        f"phrasing changed. Restore the canonical phrasing or update this regex."
    )

    return {
        "section_4_2_lock": int(section_42_match.group(1)),
        "section_3_1_1_enumeration": int(section_311_match.group(1)),
    }


def test_graph_extraction_window_size_default_consistent_across_sources():
    """All four sources of ``graph_extraction_window_size`` default must agree.

    Runs on every push via ``cicd-push.yml`` -> ``make test-unit``.

    This is the codified Lesson #13 v3 gate. It catches the same drift
    class that required two BLOCKER fix-forward rounds on PR #1925
    (Weston ``msg=1b7d9bef`` BLOCKER 1 + huangheng ``msg=bf785b12`` NIT
    1) — schema.d.ts and spec § 3.1.1 still carrying ``default 1`` while
    Python + Pydantic moved to ``2``.
    """
    python_const = _python_const_default()
    pydantic_examples = _pydantic_field_default()
    ts_example = _ts_schema_default()
    spec_defaults = _spec_doc_defaults()

    sources: dict[str, int] = {
        "python_const (aperag/indexing/graph_extractor.py)": python_const,
        "pydantic_examples (aperag/schema/common.py)": pydantic_examples,
        "ts_schema_example (web/src/api-v2/schema.d.ts)": ts_example,
        **{
            f"spec_doc § {k.replace('_', '.').replace('section.', '')} "
            f"(docs/zh-CN/architecture/task-30-graph-chunk-window-spec-v1.md)": v
            for k, v in spec_defaults.items()
        },
    }

    distinct = set(sources.values())
    assert len(distinct) == 1, (
        "graph_extraction_window_size default has drifted across sources.\n"
        "All four sources must declare the same integer (currently locked to 2 "
        "per task #30 B3 / earayu2 msg=adb0c366).\n"
        "Update ALL of the following in the same PR:\n"
        "  1. aperag/indexing/graph_extractor.py: _DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE\n"
        "  2. aperag/schema/common.py: KnowledgeGraphConfig.graph_extraction_window_size Field examples=[N]\n"
        "  3. web/src/api-v2/schema.d.ts: @example N (regenerate via OpenAPI -> TS pipeline)\n"
        "  4. docs/zh-CN/architecture/task-30-graph-chunk-window-spec-v1.md: § 3.1.1 + § 4.2 lock lines\n"
        "Future-default-change procedure (per spec § 4.2): ≥10 samples + ≥3 models "
        "no regression + PM + architect + earayu2 三方 confirm.\n"
        "\nObserved values per source:\n" + "\n".join(f"  - {name}: {value}" for name, value in sources.items())
    )


def test_graph_extraction_window_size_default_is_positive_integer():
    """Sanity check — the locked default must be a positive integer.

    ``window_size <= 0`` would break the assembler's ``len(chunks) //
    window_size`` math and the bootstrap formula; ``window_size`` is a
    chunk count, fractional values are not meaningful.
    """
    value = _python_const_default()
    assert isinstance(value, int) and value >= 1, (
        f"_DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE must be a positive integer, "
        f"got {value!r}. Negative or zero values break the window assembler "
        f"in aperag/indexing/graph_extractor.py."
    )


@pytest.mark.parametrize(
    "source_name, getter",
    [
        ("python_const", _python_const_default),
        ("pydantic_examples", _pydantic_field_default),
        ("ts_schema_example", _ts_schema_default),
    ],
)
def test_individual_source_extractor_does_not_raise(source_name, getter):
    """Each individual source extractor must succeed (no missing file / regex
    drift / Pydantic field rename).

    This separates "extractor broken" failures from "values drifted" failures
    so when CI turns red the operator immediately knows whether to update the
    test infrastructure or the schema.
    """
    value = getter()
    assert isinstance(value, int), (
        f"{source_name} extractor returned non-int {value!r}; the source format may have changed."
    )
