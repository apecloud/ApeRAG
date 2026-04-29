"""Process-split DI parity boundary gate.

Task #17 (PR #1884) split the ApeRAG runtime into two processes: the
FastAPI API process (``aperag/app.py``) and the indexing-worker
process (``aperag/cli/indexing_worker.py``). The two entry points
must wire the **same** set of cross-domain Protocol DI seams, or the
worker process crashes the moment it executes any code path (parse /
summary regen / agent runtime / quota check) that calls
``_get_*_ops()``.

The hot-fix moved the 10 setter calls into a single helper
``aperag.bootstrap.wire_cross_domain_di_seams``; both processes now
call that helper. This boundary test enforces three properties:

1. ``aperag/app.py`` and ``aperag/cli/indexing_worker.py`` both call
   ``wire_cross_domain_di_seams`` at the module/entrypoint level.
2. The helper itself wires every domain Protocol setter that the API
   process used to wire inline. AST-walks ``aperag/bootstrap/__init__.py``
   for ``set_*_ops`` callable patterns.
3. FastAPI-only setup (``configure_fastapi`` / ``register_exception_handlers``)
   never leaks into the worker entry point.

Per Lesson #11 v5 (entry-point migration sub-check) +
Weston 三分类框架 (必须 port / process-level 决策 / FastAPI-only 排除).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PY = REPO_ROOT / "aperag" / "app.py"
CLI_PY = REPO_ROOT / "aperag" / "cli" / "indexing_worker.py"
BOOTSTRAP_PY = REPO_ROOT / "aperag" / "bootstrap" / "__init__.py"

# 类 3 (FastAPI-only) — must NOT appear in the worker entry point.
FASTAPI_ONLY_CALLS = frozenset(
    {
        "configure_fastapi",
        "register_exception_handlers",
    }
)


def _collect_called_canonical_names(path: Path) -> set[str]:
    """Return the canonical (originally-imported) names of every
    callable invoked at module level OR inside any function whose name
    matches ``main`` / ``_amain`` / ``lifespan`` / ``combined_lifespan``.

    The "canonical" name is the imported symbol — e.g. for
    ``from foo import bar as _foo_bar`` followed by ``_foo_bar(...)``
    we record ``"bar"``. This makes the assertion robust against
    different files using different local aliases for the same setter.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name

    interesting_function_names = {"main", "_amain", "lifespan", "combined_lifespan"}

    def _is_interesting_node(candidate: ast.AST) -> bool:
        if isinstance(candidate, ast.AsyncFunctionDef | ast.FunctionDef):
            return candidate.name in interesting_function_names
        return False

    called: set[str] = set()

    def _visit_call(call: ast.Call) -> None:
        func = call.func
        if isinstance(func, ast.Name) and func.id in aliases:
            called.add(aliases[func.id])

    # Module-level statements first.
    for stmt in tree.body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                _visit_call(sub)

    # Then walk into the entry-point lifespan / main functions —
    # `aperag/app.py` keeps router wiring inside module-level but
    # leaves room for future setter calls from inside lifespan, and
    # `aperag/cli/indexing_worker.py` does its DI wiring inside
    # ``_amain``.
    for node in ast.walk(tree):
        if _is_interesting_node(node):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    _visit_call(sub)

    return called


def _collect_setter_names_from_bootstrap() -> set[str]:
    """Find every ``set_*_ops`` invocation inside the bootstrap helper.

    These are the canonical setter names that ``app.py`` used to call
    inline; the helper now owns them. A worker entry point that calls
    ``wire_cross_domain_di_seams`` therefore reaches the same set.
    """
    tree = ast.parse(BOOTSTRAP_PY.read_text(encoding="utf-8"))

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name

    setters: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            canonical = aliases.get(node.func.id, node.func.id)
            if canonical.startswith("set_") and canonical.endswith("_ops"):
                setters.add(canonical)
    return setters


def test_app_and_cli_worker_both_call_wire_cross_domain_di_seams():
    app_calls = _collect_called_canonical_names(APP_PY)
    cli_calls = _collect_called_canonical_names(CLI_PY)
    helper = "wire_cross_domain_di_seams"
    assert helper in app_calls, (
        f"aperag/app.py must call {helper}() at module level so the API process "
        "wires every cross-domain DI seam before serving requests."
    )
    assert helper in cli_calls, (
        f"aperag/cli/indexing_worker.py must call {helper}() at startup so the "
        "indexing-worker process wires the same set of seams as the API process. "
        "PR #1884 (task #17) regressed this; e2e-http-provider hit "
        "RuntimeError: PromptTemplateOps not wired in the worker container."
    )


def test_bootstrap_helper_wires_at_least_the_known_critical_seams():
    """Failing this test means the helper stopped wiring a seam that
    was wired before the hot-fix. Add the new setter to the helper or
    update the canonical list with a clear migration note.
    """
    wired = _collect_setter_names_from_bootstrap()
    expected_subset = {
        # knowledge_base
        "set_marketplace_ops",
        "set_marketplace_collection_ops",
        "set_search_pipeline_ops",
        "set_quota_ops",
        # agent_runtime / model_platform prompt CRUD
        "set_prompt_template_ops",
        "set_prompt_crud_ops",
        # identity init hooks
        "set_bot_init_ops",
        "set_chat_init_ops",
        "set_quota_init_ops",
    }
    missing = expected_subset - wired
    assert not missing, (
        "aperag/bootstrap/__init__.py:wire_cross_domain_di_seams "
        f"stopped calling {sorted(missing)}. Either re-wire it or update "
        "this test with the migration note explaining why the seam is gone."
    )


def test_fastapi_only_setup_does_not_leak_into_cli_worker():
    cli_calls = _collect_called_canonical_names(CLI_PY)
    leaked = FASTAPI_ONLY_CALLS & cli_calls
    assert not leaked, (
        f"aperag/cli/indexing_worker.py must not call {sorted(leaked)} — those are "
        "FastAPI-only setup hooks. The worker process has no FastAPI app instance; "
        "leaking these into the CLI is a Weston 类 3 violation."
    )
