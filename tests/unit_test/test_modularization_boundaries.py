"""Modularization Phase 0 boundary tests.

These tests pin the `#模块化重构` v2 destructive-first baseline so that
subsequent Phase 1/2/... PRs can prove they are actually shrinking the
legacy surface rather than just shuffling code. Conventions:

- Allowlist fixtures under ``tests/boundaries/*.txt`` store repo-relative
  paths, one per line, sorted. A fixture must only ever shrink in later
  PRs (never grow), until it reaches zero at the phase milestone.
- Backend domain boundary is a strict ban (no allowlist). It scans
  ``aperag/domains/**`` (currently non-existent by design) and rejects
  any reverse import of the legacy aggregate modules. When the first
  domain is extracted, the test starts enforcing canonical-only imports
  automatically.

See ``docs/modularization/README.md`` for the full target baseline.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOUNDARIES_DIR = REPO_ROOT / "tests" / "boundaries"
WEB_SRC = REPO_ROOT / "web" / "src"


def _load_allowlist(name: str) -> list[str]:
    raw = (BOUNDARIES_DIR / name).read_text().splitlines()
    return sorted({line.strip() for line in raw if line.strip()})


def _iter_web_source_files() -> list[Path]:
    return sorted(
        path
        for path in WEB_SRC.rglob("*")
        if path.is_file()
        and path.suffix in {".ts", ".tsx"}
        and "node_modules" not in path.parts
        and "__tests__" not in path.parts
    )


LEGACY_API_IMPORT_RE = re.compile(r"""from\s+['"]@/api['/]""")
RAW_SCHEMA_IMPORT_RE = re.compile(r"""from\s+['"]@/api-v2/schema['"]""")
# Direct construction or consumption of the low-level HTTP client in the
# Next.js ``app/`` tree. Only identifiers with explicit module semantics
# are matched so we do not trip on prose / comments / unrelated names.
DIRECT_CLIENT_RE = re.compile(
    r"""\b(?:defaultApi|apiClient|browserApiClient|createServerApiClient)\b"""
)

LEGACY_AGGREGATE_MODULES = (
    "aperag.service",
    "aperag.schema.view_models",
    "aperag.db.models",
)


# ---------- FE boundary tests ----------


def test_web_no_legacy_api_import_outside_allowlist():
    """`@/api` legacy SDK may only be imported from files pinned in
    ``tests/boundaries/web_legacy_api_allowlist.txt``.

    Phase 1c deletes ``web/src/api/*`` entirely and the allowlist must
    reach zero. Each intermediate Phase 1b PR must shrink the list for
    the domain it migrates. A PR that adds a new legacy import (or
    reintroduces a removed one) fails here before review.
    """
    allowlist = set(_load_allowlist("web_legacy_api_allowlist.txt"))
    offenders: list[str] = []
    missing: list[str] = []

    current: set[str] = set()
    for path in _iter_web_source_files():
        if not LEGACY_API_IMPORT_RE.search(path.read_text()):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        current.add(rel)
        if rel not in allowlist:
            offenders.append(rel)

    missing = sorted(allowlist - current)

    assert not offenders, (
        "New files import the legacy `@/api` SDK outside the allowlist. "
        "Migrate them to `@/features/<domain>/{client,server}-api` or "
        "update `tests/boundaries/web_legacy_api_allowlist.txt` with a "
        "deliberate entry + PR-body justification:\n  "
        + "\n  ".join(sorted(offenders))
    )
    assert not missing, (
        "Allowlist entries no longer exist in the tree — remove them "
        "from `tests/boundaries/web_legacy_api_allowlist.txt` so the "
        "shrinking contract stays honest:\n  " + "\n  ".join(missing)
    )


def test_web_raw_schema_import_limited_to_typed_adapters():
    """``@/api-v2/schema`` is the raw OpenAPI type surface and must only
    be imported by the typed adapter layer. Everything else reads types
    from ``features/<domain>/types``.

    The allowlist is an **exact lock** (no shrink, no grow). If a new
    typed adapter file is legitimately added, the allowlist must be
    updated in the same PR and justified in the PR body.
    """
    expected = set(_load_allowlist("web_raw_schema_allowlist.txt"))
    actual: set[str] = set()
    for path in _iter_web_source_files():
        if RAW_SCHEMA_IMPORT_RE.search(path.read_text()):
            actual.add(path.relative_to(REPO_ROOT).as_posix())

    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    assert not extra, (
        "`@/api-v2/schema` may only be imported from the typed adapter "
        "allowlist. If this is a new typed adapter, update "
        "`tests/boundaries/web_raw_schema_allowlist.txt` and explain in "
        "the PR body:\n  " + "\n  ".join(extra)
    )
    assert not missing, (
        "Allowlist entries no longer exist in the tree — remove them "
        "from `tests/boundaries/web_raw_schema_allowlist.txt`:\n  "
        + "\n  ".join(missing)
    )


def test_web_app_routes_use_feature_adapters_only():
    """Next.js route files under ``web/src/app/**`` must fetch data
    through ``features/<domain>/{server,client}-api`` rather than
    constructing the low-level HTTP client directly.

    ``features/*`` and ``lib/api/typed/*`` legitimately consume the
    low-level client — scope is restricted to ``app/**`` so those
    canonical adapters are not affected.
    """
    allowlist = set(_load_allowlist("web_route_data_allowlist.txt"))
    app_root = WEB_SRC / "app"
    offenders: list[str] = []
    current: set[str] = set()

    for path in app_root.rglob("*"):
        if not (path.is_file() and path.suffix in {".ts", ".tsx"}):
            continue
        if not DIRECT_CLIENT_RE.search(path.read_text()):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        current.add(rel)
        if rel not in allowlist:
            offenders.append(rel)

    missing = sorted(allowlist - current)
    assert not offenders, (
        "Route files under `web/src/app/**` must route data access "
        "through `features/<domain>/{server,client}-api` rather than "
        "construct the low-level HTTP client directly. Either migrate "
        "the caller or update "
        "`tests/boundaries/web_route_data_allowlist.txt`:\n  "
        + "\n  ".join(sorted(offenders))
    )
    assert not missing, (
        "Allowlist entries no longer exist — remove them from "
        "`tests/boundaries/web_route_data_allowlist.txt`:\n  "
        + "\n  ".join(missing)
    )


# ---------- Backend boundary test ----------


def _iter_domain_py_files() -> list[Path]:
    root = REPO_ROOT / "aperag" / "domains"
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and path.name != "__init__.py"
    )


def _imported_modules(source: str) -> set[str]:
    """Return the fully-qualified module roots referenced by ``import`` /
    ``from ... import`` statements in ``source``. Only top-level imports
    are considered; relative imports stay local to the domain.

    ``if TYPE_CHECKING:`` guarded imports are **not** excluded — the
    modularization baseline forbids a canonical domain from binding its
    type contract to a legacy aggregate even if the import is erased at
    runtime. Domains that need an auth / user context must express it as
    a narrow local ``Protocol`` (or a domain-owned contract), not as a
    type-only reference to ``aperag.db.models.User``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative import, stays inside the domain package.
                continue
            if node.module:
                modules.add(node.module)
    return modules


def test_aperag_domains_never_import_legacy_aggregate_modules():
    """Strict ban: any code under ``aperag/domains/<domain>/**`` must not
    import ``aperag.service.*``, ``aperag.schema.view_models``, or
    ``aperag.db.models`` — the three god aggregate modules the v2
    modularization is dismantling.

    When ``aperag/domains/`` does not yet exist this test passes
    trivially (there is nothing to scan). The moment the first canonical
    domain is extracted in Phase 2, the scan turns on automatically and
    protects the new boundary from accidental cross-layer imports.
    Legacy code outside ``aperag/domains/`` is intentionally not
    inspected — historical debt is tracked in
    ``docs/modularization/`` rather than enforced by the test.
    """
    offenders: list[str] = []
    for path in _iter_domain_py_files():
        modules = _imported_modules(path.read_text())
        for legacy in LEGACY_AGGREGATE_MODULES:
            if any(module == legacy or module.startswith(legacy + ".") for module in modules):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()} imports "
                    f"{legacy}.*"
                )

    assert not offenders, (
        "Canonical domain code imports a legacy aggregate module — "
        "replace the import with a contract / domain repository "
        "exposed through `aperag/domains/<owner>/contracts` or a "
        "domain-local model. Offenders:\n  " + "\n  ".join(sorted(offenders))
    )


def _iter_domain_api_py_files() -> list[Path]:
    """Enumerate `aperag/domains/<d>/api/**/*.py` route modules. These
    are the files whose HTTP handler signatures express the outward
    auth / dependency-injection contract of a canonical domain."""
    root = REPO_ROOT / "aperag" / "domains"
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file()
        and path.name != "__init__.py"
        and "api" in path.relative_to(root).parts
    )


def _annotation_is_any(node: ast.expr | None) -> bool:
    """Return True if ``node`` is the bare ``Any`` annotation (``Any``
    or ``typing.Any``)."""
    if node is None:
        return True  # Missing annotation is as bad as ``Any``.
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "Any"
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
    ):
        return True
    return False


def _depends_callee_name(default: ast.expr | None) -> str | None:
    """If ``default`` is ``Depends(<name>)``, return ``<name>``. Else
    return ``None``. Used to locate auth-style dependency parameters on
    route handlers."""
    if not isinstance(default, ast.Call):
        return None
    func = default.func
    if not (isinstance(func, ast.Name) and func.id == "Depends"):
        return None
    if not default.args:
        return None
    dep = default.args[0]
    if isinstance(dep, ast.Name):
        return dep.id
    if isinstance(dep, ast.Attribute):
        return dep.attr
    return None


AUTH_DEPENDENCY_NAMES = frozenset(
    {
        # Anything that looks like "the current authenticated user"
        # dependency. Add more names here when Phase 4 identity
        # introduces a canonical `current_user` contract.
        "required_user",
        "current_user",
    }
)


def test_aperag_domains_auth_dependency_is_not_any():
    """An ``aperag/domains/<d>/api/**`` HTTP handler that binds
    ``Depends(required_user)`` (or another auth dependency) must not
    annotate the parameter as ``Any`` / ``typing.Any`` / missing. The
    domain is the public contract surface and it must express the
    capability it relies on — typically a narrow local ``Protocol``
    (e.g. ``AuthenticatedUser``) until Phase 4 identity promotes the
    contract to a domain-owned type.

    The Phase 0 strict ban already prevents direct ``aperag.db.models``
    imports inside ``aperag/domains/**``; this test closes the
    complementary loophole: strict-ban compliance must not be bought
    with a type-safety regression to ``Any``.
    """
    offenders: list[str] = []
    for path in _iter_domain_api_py_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            for arg, default in zip(args.args[-len(args.defaults):], args.defaults):
                dep_name = _depends_callee_name(default)
                if dep_name is None or dep_name not in AUTH_DEPENDENCY_NAMES:
                    continue
                if _annotation_is_any(arg.annotation):
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()}::"
                        f"{node.name}({arg.arg}) — "
                        f"Depends({dep_name}) parameter must declare a "
                        f"typed contract (Protocol / domain type), not "
                        f"Any / missing"
                    )

    assert not offenders, (
        "Canonical domain API handler uses `Any` / missing annotation "
        "on an auth dependency. Introduce a narrow local `Protocol` "
        "that expresses the fields the domain actually reads (usually "
        "just `id`), or import a canonical domain-owned auth context. "
        "Never strip the type:\n  " + "\n  ".join(sorted(offenders))
    )
