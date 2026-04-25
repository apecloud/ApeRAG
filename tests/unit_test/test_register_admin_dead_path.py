"""Phase 8 task #65 (G5c) — guard against regrowth of the
``/api/v1/test/register_admin`` dev-only bootstrap shortcut.

The path was deleted from the backend in earlier Phase 8 cleanup; this
PR also removed the only remaining callers (pytest fixture, bootstrap
shell, protocol doc). The canonical replacement is
``POST /api/v2/auth/register`` — on a fresh database the first user is
automatically promoted to admin by the existing register flow
(``aperag/domains/identity/api/auth_routes.py::register_view`` plus the
``on_after_register`` hook in ``user_manager.py``), so test bootstrap
gets an admin user via the same code path production deployments use.

This test pins that promise: nothing under ``aperag/``,
``tests/e2e_pytest/``, ``tests/e2e_http/``, or ``docs/`` may reference
the dead ``/api/v1/test/register_admin`` path again.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEAD_PATHS = ("/api/v1/test/register_admin",)

# Scan these trees only — bootstrap callers + any leftover docs.
SCAN_ROOTS = (
    REPO_ROOT / "aperag",
    REPO_ROOT / "tests" / "e2e_pytest",
    REPO_ROOT / "tests" / "e2e_http",
    REPO_ROOT / "tests" / "unit_test",
    REPO_ROOT / "docs",
)

# Extensions to scan. Skip binaries / generated artefacts.
SCAN_SUFFIXES = {".py", ".sh", ".hurl", ".md", ".ts", ".tsx", ".yaml", ".yml", ".json"}


def test_no_module_references_dead_register_admin_path():
    offenders: dict[str, set[str]] = {}
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in SCAN_SUFFIXES:
                continue
            # Skip this guard file itself — it documents the dead path.
            if path.resolve() == Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            hits = {p for p in DEAD_PATHS if p in text}
            if hits:
                offenders[str(path.relative_to(REPO_ROOT))] = hits

    assert not offenders, (
        "`/api/v1/test/register_admin` was deleted in Phase 8 task #65 (G5c). "
        "Use the canonical `POST /api/v2/auth/register` instead — on a fresh DB "
        "the first registered user is auto-promoted to admin by the existing "
        "register flow. Offenders:\n  "
        + "\n  ".join(f"{path}: {sorted(hits)}" for path, hits in sorted(offenders.items()))
    )
