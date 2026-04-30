"""Boundary gate for task #31 A3: graph curation must be description-free.

Wave 5 task #5 split entity extraction into a fast facts lane and an
asynchronous vectors lane, and as part of that split the graph
extractor stopped emitting entity ``description`` text — entities now
land with ``description_parts=[]`` /
``compacted_description=None`` and the column is treated as
permanently NULL.

task #31 (graph node merge & background suggestion) inherits this
invariant. The dedup detection / scoring / snapshot / accept-apply
paths must therefore not read ``entity.description`` (or the legacy
``compacted_description`` / ``description_parts`` view it derives
from). Reading would either:

* always produce ``""`` (silently degrading dedup quality), or
* produce a stale fragment from a pre-Wave-5 row (silently leaking
  out-of-date text into suggestions / merged graph).

Per the task #31 spec § 3.1.5, this boundary gate enforces the
invariant by AST + grep over the two surfaces named in the spec:

* ``aperag/graph_curation/**``
* ``aperag/indexing/merge_candidate_detector.py``

Both must be description-read-free. Allowlist:

* ``EntityRecord(description=...)`` constructions (writing the L1
  graph store) — Wave 5 invariant requires the value passed there to
  be ``""``, but the *write* itself stays because the storage
  Protocol demands the field. The boundary checks that no read-form
  ``entity.description`` / ``.compacted_description`` /
  ``.description_parts`` access leaks back in.
* The ``CurationEntity.description`` field declaration in
  ``aperag/graph_curation/dto.py`` (declaration is a static type
  annotation, not a read).
* Comments / docstrings that *mention* ``description`` (the gate
  greps reads, not the word).

This pairs with the lesson sediment Lesson #14 (架构 invariant 删除
多轮迭代收尾) and Lesson #18-候选 (lesson + mechanical gate 双 layer):
the description-free invariant is documented in
``docs/zh-CN/architecture/task-31-graph-node-merge-spec-v1.md``
§ 3.1.5 + the Wave 5 sediment, and this test is the mechanical
enforcer.

**Lesson #18 候选 second-application demo trail (PR #1941)**: when
the spec § 3.1.5 ratify (符炫炜 + Bryce + ziang + huangzhangshu +
Weston multi-source review) listed exactly 6 detector / snapshot
call sites + 1 apply-path variant, every reviewer + the spec
author missed a 7th hidden read at
``aperag/graph_curation/service.py:845`` —
``text = entity.description or entity.name`` inside
``GraphCurationService._fetch_shadow_neighbors``. The boundary
gate caught it on first run (force-fix forward), turning
``reviewer-as-detector`` into ``CI-as-detector`` per the
Lesson #18 thesis. This is the canonical
**lesson sediment + mechanical gate 双 layer codification** value
demo: spec author + reviewers + AST scan together produced 0
false negatives only because the mechanical gate was paired with
the human-text lesson sediment. Lesson #12 v9 (first-principles
verify) is the human-side counterpart; Lesson #18 is the CI-side
counterpart.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_CURATION_ROOT = REPO_ROOT / "aperag" / "graph_curation"
MERGE_DETECTOR_FILE = REPO_ROOT / "aperag" / "indexing" / "merge_candidate_detector.py"

# Attribute names whose *read* would re-introduce description residue
# into the dedup / accept-apply paths. Includes the canonical
# ``description`` field plus the two derivation views (since
# ``CurationEntity.from_lineage`` no longer reads them, downstream
# modules in scope must not either).
FORBIDDEN_DESCRIPTION_READS = frozenset(
    {
        "description",
        "compacted_description",
        "description_parts",
    }
)

# Identifiers that legitimately *write* ``description`` (storage Protocol /
# DTO / dataclass declarations). These are not "reads" of an entity
# object's description and are explicitly allowed by the gate.
ALLOWED_RECORD_BUILDERS = frozenset(
    {
        "EntityRecord",  # storage Protocol — write only, value must be "" post Wave 5
        "DescriptionPart",  # dataclass — appears in legacy paths only
        "LegacyEntity",  # alias for CurationEntity inside merge_candidate_detector
        "CurationEntity",  # DTO declaration / construction
        "Entity",  # alias for CurationEntity inside service.py
    }
)


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _format_offender(path: Path, lineno: int, snippet: str) -> str:
    return f"  {path.relative_to(REPO_ROOT)}:{lineno}  →  {snippet}"


def _is_keyword_arg_to_record_builder(node: ast.Attribute) -> bool:
    """``EntityRecord(description=part.text)`` keeps ``part.text`` as a
    read of ``part.description`` if ``description`` is the suffix —
    but in practice the canonical legacy pattern is
    ``EntityRecord(description=<expr>)`` where the *attribute* name is
    just the keyword. We never want to flag the keyword side. AST
    keyword args are ``ast.keyword`` not ``ast.Attribute``; this
    function exists for completeness — keyword names are not
    ``Attribute`` nodes so this returns False by default. Kept as a
    structural reminder for future maintainers.
    """
    return False


# Variable / attribute names whose ``.compacted_description`` /
# ``.description`` / ``.description_parts`` access targets a NON-entity
# shape (legacy back-compat result objects, merge return DTOs). The
# Wave 5 invariant bans ENTITY description reads, not these aggregate
# result shapes that exist purely to ferry the legacy sync API
# response between layers.
NON_ENTITY_BASE_NAMES = frozenset(
    {
        # ``LineageMergeResult.compacted_description`` /
        # ``.unified_description`` carry the legacy ``merge_entities``
        # output for the sync ``handle_action()`` API. The async
        # accept-apply variant returns these as ``None`` / ``""`` so
        # reading them on a description-free path is a no-op — the
        # legacy path's read of ``merge_result.compacted_description``
        # is the only consumer and is preserved per spec § 3.1.5.
        "merge_result",
        # Generic suffix-pattern for legacy result objects.
        # NOTE: matched as exact base name only — anything more
        # selective requires semantic analysis the boundary doesn't do.
    }
)


def _description_read_offenders_in(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    source_lines = source.splitlines()
    offenders: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in FORBIDDEN_DESCRIPTION_READS:
            continue
        # Skip writes (``foo.description = ...``) — Store(ctx) means
        # this is an assignment target, not a read.
        if isinstance(getattr(node, "ctx", None), ast.Store):
            continue
        # Skip non-entity result shapes (legacy back-compat). The
        # Wave 5 invariant bans entity description reads; aggregate
        # result objects (LineageMergeResult etc.) carry their own
        # ``compacted_description`` / ``unified_description`` fields
        # that legacy sync API consumers rely on.
        if isinstance(node.value, ast.Name) and node.value.id in NON_ENTITY_BASE_NAMES:
            continue
        snippet = source_lines[node.lineno - 1].strip() if node.lineno - 1 < len(source_lines) else "<unavailable>"
        offenders.append((node.lineno, snippet))
    return offenders


def test_graph_curation_modules_do_not_read_entity_description() -> None:
    """``aperag/graph_curation/**`` (excl. ``lineage_merge.py`` legacy
    path) must not read ``entity.description`` /
    ``.compacted_description`` / ``.description_parts``.

    Wave 5 description-NULL invariant (task #31 A3, spec § 3.1.5 +
    § 5.2.a): the dedup detection / candidate scoring / snapshot
    surface in ``graph_curation/`` no longer derives signals from
    descriptions. ``lineage_merge.merge_entities`` (legacy
    description-bearing variant) is excluded by file allowlist
    because the spec preserves it for manual API back-compat — the
    new accept-apply variant
    (``merge_entities_apply_description_free``) is gated by a
    dedicated assertion below.

    ⚠️ ``dto.py`` is **in scope** (per huangzhangshu BLOCKER on PR
    #1941, msg=2deb5407): spec § 3.1.5 lists
    ``CurationEntity.from_lineage`` as one of the 6 description-free
    call sites, so the gate must catch future regressions that
    re-introduce ``entity.compacted_description`` /
    ``entity.description_parts`` reads inside ``from_lineage``.
    Dataclass field *declarations* (``description: str = ""``) are
    ``ast.AnnAssign`` nodes, and constructor *keyword args*
    (``cls(description="")``) are ``ast.keyword`` nodes — neither is
    an ``ast.Attribute`` access on an entity object, so the AST
    walker does not false-positive on them. The boundary catches
    *reads* of the form ``entity.description`` / ``.compacted_description``
    / ``.description_parts`` only.
    """

    offenders: list[str] = []
    for path in _python_files(GRAPH_CURATION_ROOT):
        # Allowlist:
        #   * lineage_merge.py — legacy description-bearing manual API path
        #     (the new accept-apply worker uses
        #     merge_entities_apply_description_free which is enforced
        #     by `test_lineage_merge_apply_description_free_does_not_read_entity_description`)
        # NB: ``dto.py`` is intentionally NOT excluded — see docstring.
        if path.name == "lineage_merge.py":
            continue
        for lineno, snippet in _description_read_offenders_in(path):
            offenders.append(_format_offender(path, lineno, snippet))

    assert not offenders, (
        "Wave 5 description-NULL invariant violated (task #31 A3, spec § 3.1.5).\n"
        "The dedup detection / candidate scoring / snapshot surface in "
        "aperag/graph_curation/ must not read entity.description / "
        ".compacted_description / .description_parts — Wave 5 graph extractor "
        "no longer emits description text, so reading here either silently "
        "degrades scoring (always-empty) or leaks stale fragments. Offenders:\n" + "\n".join(offenders)
    )


def test_merge_candidate_detector_does_not_read_entity_description() -> None:
    """``aperag/indexing/merge_candidate_detector.py`` must not read
    ``entity.description`` / ``.compacted_description`` /
    ``.description_parts``.

    Wave 5 description-NULL invariant (task #31 A3, spec § 3.1.5):
    the sync auto-detect detector path produces suggestions written
    to the L1 graph_curation_suggestions store; reading description
    here would persist either always-empty rows (degrading recall)
    or stale fragments (leaking pre-Wave-5 text into reviewer view).
    The vector recall query must use ``entity.name +
    entity.entity_type`` instead, mirroring how the graph_vectors
    worker writes the entity vector.
    """

    offenders: list[str] = []
    for lineno, snippet in _description_read_offenders_in(MERGE_DETECTOR_FILE):
        offenders.append(_format_offender(MERGE_DETECTOR_FILE, lineno, snippet))

    assert not offenders, (
        "Wave 5 description-NULL invariant violated (task #31 A3, spec § 3.1.5).\n"
        "MergeCandidateDetector must not read entity.description / "
        ".compacted_description / .description_parts; the embedding query "
        "should use entity.name + entity.entity_type instead. Offenders:\n" + "\n".join(offenders)
    )


def test_lineage_merge_apply_description_free_does_not_read_entity_description() -> None:
    """The async accept-apply variant
    ``LineageEntityMerger.merge_entities_apply_description_free`` must
    not read entity descriptions.

    Wave 5 description-NULL invariant (task #31 A3, spec § 3.1.5):
    the new async accept-apply worker uses this variant. The legacy
    ``merge_entities`` method is preserved for manual API back-compat
    (allowlisted from this gate by isolating just the variant); the
    variant itself must skip every LLM unified description /
    compactor / sentinel description write / vector embed step that
    the legacy path performs.
    """

    source = (REPO_ROOT / "aperag" / "graph_curation" / "lineage_merge.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    source_lines = source.splitlines()

    target_method = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name == "merge_entities_apply_description_free"
        ):
            target_method = node
            break
    assert target_method is not None, (
        "merge_entities_apply_description_free not found on LineageEntityMerger — "
        "task #31 A3 (spec § 3.1.5) requires this variant for the async accept-apply "
        "worker. Either restore the variant or update this boundary test to track its "
        "current name."
    )

    offenders: list[str] = []
    for node in ast.walk(target_method):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in FORBIDDEN_DESCRIPTION_READS:
            continue
        if isinstance(getattr(node, "ctx", None), ast.Store):
            continue
        snippet = source_lines[node.lineno - 1].strip() if node.lineno - 1 < len(source_lines) else "<unavailable>"
        offenders.append(f"  lineage_merge.py:{node.lineno}  →  {snippet}")

    assert not offenders, (
        "merge_entities_apply_description_free must skip every description-bearing "
        "step (LLM unified / compactor / sentinel description write / vector embed). "
        "Reading entity.description / .compacted_description / .description_parts "
        "would re-introduce the legacy path. Offenders:\n" + "\n".join(offenders)
    )


def test_dto_module_is_in_boundary_scope() -> None:
    """Sanity check: ``aperag/graph_curation/dto.py`` MUST be in the
    AST-scan scope of
    :func:`test_graph_curation_modules_do_not_read_entity_description`.

    Per spec § 3.1.5 item 4, ``CurationEntity.from_lineage`` is one
    of the 6 description-free call sites. The boundary gate must
    therefore catch any future regression that re-introduces
    ``entity.compacted_description`` / ``entity.description_parts``
    reads inside ``from_lineage``. Whole-file excluding ``dto.py``
    would silently disable this protection
    (per huangzhangshu BLOCKER on PR #1941, msg=2deb5407).

    Synthetic regression check: simulate a re-introduced read inside
    a temporary AST node and assert that the offender detector
    surfaces it. We don't actually mutate ``dto.py`` on disk — we
    construct a fake AST module containing the forbidden pattern and
    feed it through the same matcher used by the file-scoped gate.
    """
    # Build a fake `from_lineage` body that re-introduces
    # `entity.compacted_description` read.
    fake_source = (
        "def from_lineage(entity):\n"
        "    description = entity.compacted_description or ''\n"  # forbidden
        "    return description\n"
    )
    tree = ast.parse(fake_source)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in FORBIDDEN_DESCRIPTION_READS:
            continue
        if isinstance(getattr(node, "ctx", None), ast.Store):
            continue
        if isinstance(node.value, ast.Name) and node.value.id in NON_ENTITY_BASE_NAMES:
            continue
        offenders.append(node.attr)

    assert offenders, (
        "AST gate failed to flag a re-introduced `entity.compacted_description` read — "
        "this would let `dto.py` regress without the boundary catching it."
    )
    assert "compacted_description" in offenders


def test_dto_field_declaration_is_not_a_false_positive() -> None:
    """The ``description: str = ""`` dataclass annotation in
    ``CurationEntity`` must NOT trip the gate.

    AST shape: ``ast.AnnAssign`` with ``target=ast.Name("description")``
    — not ``ast.Attribute``. Same for the constructor keyword arg
    ``description=""`` (``ast.keyword``).
    """
    # Confirm the live ``dto.py`` is in scope and produces zero
    # offenders today (the explicit positive control sister of
    # ``test_graph_curation_modules_do_not_read_entity_description``).
    dto_path = REPO_ROOT / "aperag" / "graph_curation" / "dto.py"
    assert dto_path.exists()
    offenders = _description_read_offenders_in(dto_path)
    assert not offenders, (
        "`dto.py` should produce zero AST-form description reads after "
        "task #31 A3. If this fails, either `from_lineage` regressed "
        "(actual offender) or the AST walker is mis-classifying field "
        "annotations / keyword args as reads (false positive — fix the "
        "walker, do NOT allowlist dto.py whole-file).\n"
        f"Offenders: {offenders}"
    )


def test_lineage_merge_apply_description_free_does_not_call_llm_or_compactor() -> None:
    """Variant must not invoke LLM unified description / Compactor /
    vector embed write. AST gate over the variant body.

    Forbidden call surfaces (per spec § 3.1.5 «不调 LLM unified
    description / compactor / __curation_merge__ description part /
    vector embedding»):

    * ``self._llm`` / ``self._unified_description`` — LLM unified
      description prompt.
    * ``self._compactor.compact_if_oversized`` — GraphIndexCompactor
      pass.
    * ``self._upsert_vector_point`` / ``self._delete_vector_point`` —
      vector embed write / delete (orphan vectors GC'd by task #11
      lane instead).
    """

    source = (REPO_ROOT / "aperag" / "graph_curation" / "lineage_merge.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    source_lines = source.splitlines()

    target_method = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name == "merge_entities_apply_description_free"
        ):
            target_method = node
            break
    assert target_method is not None

    forbidden_attrs = {"_llm", "_unified_description", "_compactor", "_upsert_vector_point", "_delete_vector_point"}
    offenders: list[str] = []

    for node in ast.walk(target_method):
        # Match `self.<forbidden_attr>` whether bare or as a call target.
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in forbidden_attrs
        ):
            snippet = source_lines[node.lineno - 1].strip() if node.lineno - 1 < len(source_lines) else "<unavailable>"
            offenders.append(f"  lineage_merge.py:{node.lineno}  self.{node.attr}  →  {snippet}")
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
            and node.value.attr == "_compactor"
        ):
            # `self._compactor.compact_if_oversized(...)` — chained call
            snippet = source_lines[node.lineno - 1].strip() if node.lineno - 1 < len(source_lines) else "<unavailable>"
            offenders.append(f"  lineage_merge.py:{node.lineno}  self._compactor.{node.attr}  →  {snippet}")

    assert not offenders, (
        "merge_entities_apply_description_free must not invoke LLM unified description / "
        "Compactor pass / vector embed write — Wave 5 description-NULL invariant + "
        "task #31 A3 spec § 3.1.5 explicit «不调» list. Offenders:\n" + "\n".join(offenders)
    )
