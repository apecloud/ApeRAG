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
DIRECT_CLIENT_RE = re.compile(r"""\b(?:defaultApi|apiClient|browserApiClient|createServerApiClient)\b""")

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
        "deliberate entry + PR-body justification:\n  " + "\n  ".join(sorted(offenders))
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
        "from `tests/boundaries/web_raw_schema_allowlist.txt`:\n  " + "\n  ".join(missing)
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
        "`tests/boundaries/web_route_data_allowlist.txt`:\n  " + "\n  ".join(sorted(offenders))
    )
    assert not missing, (
        "Allowlist entries no longer exist — remove them from "
        "`tests/boundaries/web_route_data_allowlist.txt`:\n  " + "\n  ".join(missing)
    )


# ---------- Backend boundary test ----------


def _iter_domain_py_files() -> list[Path]:
    root = REPO_ROOT / "aperag" / "domains"
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if path.is_file() and path.name != "__init__.py")


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
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()} imports {legacy}.*")

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
        if path.is_file() and path.name != "__init__.py" and "api" in path.relative_to(root).parts
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
            for arg, default in zip(args.args[-len(args.defaults) :], args.defaults):
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


# ---------- Retrieval <-> knowledge_graph one-way bridge ----------


_RETRIEVAL_FORBIDDEN_KG_IMPORTS = (
    # Retrieval must go through `retrieval.ports.GraphSearchContract`;
    # it is not allowed to static-import the knowledge_graph domain
    # service or its schemas.
    "aperag.domains.knowledge_graph.service",
    "aperag.domains.knowledge_graph.schemas",
    # Retrieval must not import the graph-curation or graphindex
    # integration modules directly — those are knowledge_graph
    # territory. The pipeline uses a narrow local factory helper to
    # reach graphindex, but only through the `GraphSearchContract`
    # protocol; explicitly forbid any broader imports here.
    "aperag.graph_curation",
    "aperag.graphindex",
)

_KG_FORBIDDEN_RETRIEVAL_IMPORTS = (
    # Lesson 9a-quad: the Protocol is owned by the *consumer*. The
    # provider domain (knowledge_graph) must not static-import the
    # consumer's ports module because doing so would reintroduce the
    # circular dependency the one-way bridge was built to avoid.
    "aperag.domains.retrieval.ports",
    "aperag.domains.retrieval.schemas",
    "aperag.domains.retrieval.service",
    "aperag.domains.retrieval.pipeline",
)


def _iter_domain_py_files_for(domain: str) -> list[Path]:
    root = REPO_ROOT / "aperag" / "domains" / domain
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if path.is_file() and path.name != "__init__.py")


def test_retrieval_kg_protocol_boundary_is_one_way():
    """Lessons 9a-quad + G3 / G10: the ``retrieval`` and
    ``knowledge_graph`` domains talk through Protocols owned by the
    consumer, never through direct static imports.

    * ``retrieval/**`` must not import any ``knowledge_graph`` service
      / schemas module, nor the ``aperag.graph_curation`` /
      ``aperag.graphindex`` packages wholesale (the one exception is
      the narrow local graphindex factory call that still resolves
      to a ``GraphSearchContract`` Protocol).
    * ``knowledge_graph/**`` must not import the ``retrieval`` ports
      or service / schemas (which would re-establish the cycle).

    Scoped to ``aperag/domains/**`` only — infrastructure code (e.g.
    ``aperag.graphindex.*``) is free to import whatever it needs.
    """
    offenders: list[str] = []

    for path in _iter_domain_py_files_for("retrieval"):
        # Local factory helper import is allowed (it reaches into
        # graphindex but only returns a ``GraphSearchContract`` typed
        # reference) — we therefore forbid only the *top-level*
        # ``aperag.graphindex`` / ``aperag.graph_curation`` names but
        # allow ``aperag.graphindex.integration`` because the pipeline
        # needs the singleton factory. Specifically list the forbidden
        # packages without their submodules.
        modules = _imported_modules(path.read_text())
        for forbidden in _RETRIEVAL_FORBIDDEN_KG_IMPORTS:
            hit = next(
                (module for module in modules if module == forbidden or module.startswith(forbidden + ".")),
                None,
            )
            if hit is None:
                continue
            # Whitelist the narrow graphindex submodules the pipeline
            # legitimately uses via the ``GraphSearchContract`` bridge.
            if forbidden == "aperag.graphindex" and hit in {
                "aperag.graphindex.integration",
            }:
                continue
            offenders.append(
                f"{path.relative_to(REPO_ROOT).as_posix()} imports {hit} "
                f"(forbidden by retrieval ↔ knowledge_graph one-way bridge)"
            )

    for path in _iter_domain_py_files_for("knowledge_graph"):
        modules = _imported_modules(path.read_text())
        for forbidden in _KG_FORBIDDEN_RETRIEVAL_IMPORTS:
            if any(module == forbidden or module.startswith(forbidden + ".") for module in modules):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()} imports {forbidden} "
                    f"(forbidden by retrieval ↔ knowledge_graph one-way bridge)"
                )

    assert not offenders, (
        "retrieval ↔ knowledge_graph cross-domain bridge must stay "
        "one-way: the Protocol is owned by the consumer, the provider "
        "structurally satisfies it. Replace the direct import with a "
        "Protocol-typed dependency or move the call to the owning "
        "domain. Offenders:\n  " + "\n  ".join(sorted(offenders))
    )


# ---------- Legacy route residue check ----------


_LEGACY_ROUTE_PATTERNS = (
    # Retrieval / KG routes that must not survive under the residual
    # ``aperag.views.*`` modules after the Phase 2 hard-cut.
    re.compile(r"""@router\.(?:post|get|delete|put|patch)\(\s*["'][^"']*/collections/\{[^/]+\}/searches"""),
    re.compile(r"""@router\.(?:post|get|delete|put|patch)\(\s*["'][^"']*/collections/\{[^/]+\}/graphs/labels"""),
    re.compile(
        r"""@router\.(?:post|get|delete|put|patch)\(\s*["'][^"']*/collections/\{[^/]+\}/graphs(?!/export/kg-eval)"""
    ),
    re.compile(r"""@router\.(?:post|get|delete|put|patch)\(\s*["'][^"']*/collections/\{[^/]+\}/graph-curation"""),
)


def test_no_legacy_retrieval_or_graph_routes_remain():
    """After the Phase 2 hard-cut, ``aperag/views/collections.py`` and
    ``aperag/views/graph.py`` must not contain any router decorator
    that still owns a retrieval / knowledge_graph path.

    The only explicitly allowed graph route left in ``views/graph.py``
    is the ``GET /collections/{id}/graphs/export/kg-eval`` 410-Gone
    shim (kept for out-of-tree callers hitting the deleted LightRAG
    endpoint). The regex above carves out that exception with a
    negative lookahead.
    """
    paths = [
        REPO_ROOT / "aperag" / "views" / "collections.py",
        REPO_ROOT / "aperag" / "views" / "graph.py",
    ]
    offenders: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        source = path.read_text()
        for pattern in _LEGACY_ROUTE_PATTERNS:
            for match in pattern.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                snippet = match.group(0).strip()
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{line}: {snippet}")

    assert not offenders, (
        "Residual legacy retrieval / knowledge_graph route decorator "
        "survived the Phase 2 hard-cut. Move the handler into the "
        "canonical domain (`aperag/domains/retrieval/api/routes.py` "
        "or `aperag/domains/knowledge_graph/api/routes.py`) and "
        "delete the legacy entry. Offenders:\n  " + "\n  ".join(offenders)
    )


# ---------- Phase 3 knowledge_base ↔ legacy services one-way bridge ----------


_KB_CONSUMER_OWNED_PROTOCOL_MODULES = (
    # KB owns the Protocol surfaces that marketplace / search_pipeline /
    # quota legacy services structurally satisfy. Provider-side import
    # of the consumer's ports.py would re-introduce the circular
    # dependency the DI pattern was built to avoid (lesson 9a-quad).
    "aperag.domains.knowledge_base.ports",
)

# The legacy provider services that Phase 3 Step 5b2c / 5a wire into KB's
# consumer-owned Protocol slots. They satisfy ``MarketplaceOps`` /
# ``MarketplaceCollectionOps`` / ``SearchPipelineOps`` / ``QuotaOps``
# structurally; they must never import KB's ports.py.
_KB_LEGACY_PROVIDER_SERVICES = (
    "aperag/service/marketplace_service.py",
    "aperag/service/marketplace_collection_service.py",
    "aperag/service/search_pipeline_service.py",
    "aperag/service/quota_service.py",
)


def test_knowledge_base_protocol_boundary_is_consumer_owned():
    """Lesson 9a-quad applied to Phase 3 KB domain: the KB domain owns
    the Protocols (``MarketplaceOps`` / ``MarketplaceCollectionOps`` /
    ``SearchPipelineOps`` / ``QuotaOps`` / ``AuthenticatedUser`` in
    ``aperag/domains/knowledge_base/ports.py``); the legacy provider
    services at ``aperag/service/`` structurally satisfy them. The
    providers must never import the consumer's ports module — doing so
    would re-establish the cycle the consumer-owned Protocol pattern
    was built to break.

    Scope is intentionally scoped to the four known provider modules.
    If Phase 4 marketplace / Phase 5 quota move the implementation
    under ``aperag/domains/`` the list shrinks; the domain G1 ban
    already covers the ``aperag/domains/**`` side.
    """
    offenders: list[str] = []
    for rel_path in _KB_LEGACY_PROVIDER_SERVICES:
        path = REPO_ROOT / rel_path
        if not path.exists():
            continue
        modules = _imported_modules(path.read_text())
        for forbidden in _KB_CONSUMER_OWNED_PROTOCOL_MODULES:
            if any(module == forbidden or module.startswith(forbidden + ".") for module in modules):
                offenders.append(
                    f"{rel_path} imports {forbidden} (forbidden by lesson 9a-quad: "
                    "consumer owns the Protocol, provider structurally satisfies it)"
                )

    assert not offenders, (
        "A legacy provider service imported the KB consumer-owned "
        "Protocol module. Drop the import; structural satisfaction is "
        "all that's required (the concrete class just needs matching "
        "method signatures). Offenders:\n  " + "\n  ".join(sorted(offenders))
    )


def test_knowledge_base_di_wire_up_populated_after_app_import():
    """Phase 3 Step 5b2c canonical: ``aperag/app.py`` module-scope
    wire-up must populate all four KB consumer-owned Protocol DI slots
    before any FastAPI handler runs.

    The four getters (``_get_marketplace_ops`` /
    ``_get_marketplace_collection_ops`` / ``_get_search_pipeline_ops``
    / ``_get_quota_ops``) raise ``RuntimeError`` on unwired state, so
    a forgotten or re-ordered startup wire-up would fail loudly —
    this test makes it fail at CI time instead of at first request.

    The sibling-import pattern (``document_service`` reuses
    ``collection_service``'s accessors) means a single wire-up also
    covers document_service; asserting the four module-level globals
    here is therefore sufficient.
    """
    # Import ``aperag.app`` fresh so the module-scope wire-up fires
    # (subsequent imports are a no-op because Python caches the
    # loaded module in ``sys.modules``).
    import aperag.app  # noqa: F401
    import aperag.domains.knowledge_base.service.collection_service as kb_cs

    missing = [
        name
        for name in (
            "_marketplace_ops",
            "_marketplace_collection_ops",
            "_search_pipeline_ops",
            "_quota_ops",
        )
        if getattr(kb_cs, name, None) is None
    ]
    assert not missing, (
        "Knowledge-base consumer-owned Protocol DI slot(s) unwired "
        "after ``import aperag.app``. Check the startup section of "
        "``aperag/app.py`` (Phase 3 Step 5b2c). Missing: " + ", ".join(missing)
    )


# ---------- Phase 4 identity/governance/model_platform/marketplace gates (G15/G16/G17) ----------


# Phase 4 G15 — non-identity domains must never import the identity
# Role enum; use string-literal compare (``user.role == "admin"``)
# instead. The canonical is explicit about AST-level import ban with
# no literal allowlist (msg=6d2ae86a + msg=896584ee).
_G15_G16_BANNED_ROLE_USER_SOURCES = (
    "aperag.db.models",
    "aperag.domains.identity.db.models",
)

_G15_G16_SCOPE_DOMAINS = (
    "marketplace",
    "governance",
    "model_platform",
    "knowledge_base",
    "retrieval",
    "indexing",
    "knowledge_graph",
    "conversation",
    "agent_runtime",
    "evaluation",
    "web_access",
)


def _iter_domain_files_for_g15_g16() -> list[Path]:
    files: list[Path] = []
    for domain in _G15_G16_SCOPE_DOMAINS:
        root = REPO_ROOT / "aperag" / "domains" / domain
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.is_file() and path.name != "__init__.py":
                files.append(path)
    return sorted(files)


def test_phase4_consumer_domains_never_import_role_enum():
    """G15: non-identity domains must never ``from aperag.db.models
    import Role`` or ``from aperag.domains.identity.db.models import
    Role``. Consumers compare ``user.role == "admin"`` by literal
    string against per-domain ``AuthenticatedUser`` / ``UserView``
    Protocols whose ``role`` attribute is typed ``str``.

    Identity domain itself is exempt — internal ``Role`` usage is
    canonical. Literal-value allowlist is a soft convention enforced
    by reviewer CR, not by this test (msg=6d2ae86a canonical).
    """
    offenders: list[str] = []
    for path in _iter_domain_files_for_g15_g16():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in _G15_G16_BANNED_ROLE_USER_SOURCES:
                imported = {alias.name for alias in node.names}
                if "Role" in imported:
                    offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()} imports Role from {node.module}")
    assert not offenders, (
        "Non-identity domain imports the ``Role`` enum. Use literal "
        'compare (``user.role == "admin"``) against the per-domain '
        "Protocol's ``role: str`` attribute instead.\n  " + "\n  ".join(sorted(offenders))
    )


def test_phase4_consumer_domains_never_import_user_orm_class():
    """G16: non-identity domains must never ``from
    aperag.db.models import User`` or ``from
    aperag.domains.identity.db.models import User``. Route handlers
    and services should depend on the per-domain
    ``AuthenticatedUser(Protocol)`` (lesson 9a-ter) instead.

    Identity domain is exempt — it owns ``User`` and its own
    ``UserManager`` needs the ORM class.
    """
    offenders: list[str] = []
    for path in _iter_domain_files_for_g15_g16():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in _G15_G16_BANNED_ROLE_USER_SOURCES:
                imported = {alias.name for alias in node.names}
                if "User" in imported:
                    offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()} imports User from {node.module}")
    assert not offenders, (
        "Non-identity domain imports the ``User`` ORM class. Use "
        "the per-domain ``AuthenticatedUser(Protocol)`` or a narrow "
        "``UserView(Protocol)`` contract instead (lesson 9a-ter).\n  " + "\n  ".join(sorted(offenders))
    )


def test_phase4_di_critical_wirings_at_app_startup():
    """G17: runtime smoke — after ``import aperag.app`` the
    ``CRITICAL_WIRINGS`` registry of (module, attribute) pairs must
    all resolve to non-``None`` Protocol instances. This catches
    forgotten / re-ordered startup wire-up at CI time instead of at
    first request. msg=896584ee canonical: do not rely on AST setter
    naming scan (fragile) — runtime state is what actually matters.
    """
    import aperag.app  # noqa: F401 — triggers module-scope wire-up
    from aperag.domains.identity.service import user_manager as identity_user_manager
    from aperag.domains.knowledge_base.service import collection_service as kb_collection_service

    CRITICAL_WIRINGS = [
        # Phase 3 knowledge_base DI slots (Step 5b2c).
        (kb_collection_service, "_marketplace_ops"),
        (kb_collection_service, "_marketplace_collection_ops"),
        (kb_collection_service, "_search_pipeline_ops"),
        (kb_collection_service, "_quota_ops"),
        # Phase 4 identity DI slots (Step 4-S7d).
        (identity_user_manager, "_bot_init_ops"),
        (identity_user_manager, "_chat_init_ops"),
        (identity_user_manager, "_quota_init_ops"),
    ]
    missing = [f"{module.__name__}.{attr}" for module, attr in CRITICAL_WIRINGS if getattr(module, attr, None) is None]
    assert not missing, (
        "Critical DI wire-up missing after ``import aperag.app`` — "
        "check the startup section of ``aperag/app.py``. Missing: " + ", ".join(missing)
    )


def test_phase5_di_critical_wirings_at_app_startup():
    """G18 alt: runtime smoke for the permanent consumer-owned Protocol
    DI slots — after ``import aperag.app`` the listed ``_<name>_ops``
    slots must resolve to non-``None`` instances.

    The registry carries exactly the DI slots whose providers are
    standalone-infrastructure modules with no natural domain home
    (``quota_service``, ``prompt_template_service``). Every
    domain-moved provider is reached via a direct sibling /
    cross-domain import.

    ``dispatch_fn`` in ``aperag.domains.evaluation.worker`` is
    intentionally **not** listed — it is a module-level test-injection
    seam, not a Protocol+DI slot.
    """
    import aperag.app  # noqa: F401 — triggers module-scope wire-up
    from aperag.domains.agent_runtime import runtime as agent_runtime_runtime
    from aperag.domains.conversation.service import bot_service as conversation_bot_service

    PHASE5_CRITICAL_WIRINGS = [
        # bot_service ↔ quota_service (standalone-infra, permanent seam).
        (conversation_bot_service, "_quota_ops"),
        # runtime.py ↔ prompt_template_service (standalone-infra,
        # permanent seam).
        (agent_runtime_runtime, "_prompt_template_ops"),
    ]
    missing = [
        f"{module.__name__}.{attr}" for module, attr in PHASE5_CRITICAL_WIRINGS if getattr(module, attr, None) is None
    ]
    assert not missing, (
        "Phase 5 critical DI wire-up missing after ``import aperag.app`` — "
        "check the startup section of ``aperag/app.py``. Missing: " + ", ".join(missing)
    )


def test_phase5_domain_routes_never_use_pep_563_future_annotations():
    """Lesson 9a-quatuordec codification: FastAPI route modules must
    not use ``from __future__ import annotations``. PEP 563
    stringifies the ``-> Response`` return annotation, and the
    FastAPI ``is_body_allowed_for_status_code(204)`` check at route
    registration dereferences the annotation by value — stringified
    annotations trip the assertion and the route fails to register.

    Phase 3 step 5a discovered the interaction and
    ``docs/modularization/breaking-changes/phase3-knowledge_base.md``
    lesson 9a-quatuordec recorded it. This test enforces the
    discipline across **every** ``aperag/domains/**/api/routes.py``
    module and the two-router conversation module it pulls in.
    """

    def _has_future_annotations(path: Path) -> bool:
        """AST-based check — ignores docstring / comment mentions."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                if any(alias.name == "annotations" for alias in node.names):
                    return True
        return False

    offenders: list[str] = []
    for route_file in (REPO_ROOT / "aperag" / "domains").rglob("api/routes.py"):
        if _has_future_annotations(route_file):
            offenders.append(route_file.relative_to(REPO_ROOT).as_posix())
    # Phase 4 model_platform splits into two router modules — include both.
    for extra in (
        REPO_ROOT / "aperag" / "domains" / "model_platform" / "api" / "llm_routes.py",
        REPO_ROOT / "aperag" / "domains" / "model_platform" / "api" / "providers_v2_routes.py",
    ):
        if extra.exists() and _has_future_annotations(extra):
            offenders.append(extra.relative_to(REPO_ROOT).as_posix())
    assert not offenders, (
        "FastAPI route modules must not declare ``from __future__ import "
        "annotations`` — PEP 563 breaks ``is_body_allowed_for_status_code`` "
        "for 204 handlers (lesson 9a-quatuordec). Offenders:\n  " + "\n  ".join(sorted(offenders))
    )


# ---------- Phase 1 FE closeout gates (G12-G17) ----------


FEATURES_DIR = WEB_SRC / "features"

BROWSER_CLIENT_IMPORT_RE = re.compile(r"""from\s+['"]@/lib/api/typed/browser['"]""")
SERVER_CLIENT_IMPORT_RE = re.compile(r"""from\s+['"]@/lib/api/typed/server['"]""")


def test_features_client_api_imports_browser_only():
    """G12: ``features/*/client-api.ts`` must consume
    ``@/lib/api/typed/browser`` exclusively — never the server client
    (which depends on ``next/headers::cookies()`` and breaks at browser
    runtime)."""
    offenders: list[str] = []
    for path in FEATURES_DIR.glob("*/client-api.ts"):
        text = path.read_text()
        if SERVER_CLIENT_IMPORT_RE.search(text):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, (
        "`features/*/client-api.ts` must not import the server typed "
        "client. Use `@/lib/api/typed/browser` instead:\n  " + "\n  ".join(sorted(offenders))
    )


def test_features_server_api_imports_server_only():
    """G13: symmetric — ``features/*/server-api.ts`` consumes the
    server typed client (``next/headers::cookies()`` context), never
    the browser client."""
    offenders: list[str] = []
    for path in FEATURES_DIR.glob("*/server-api.ts"):
        text = path.read_text()
        if BROWSER_CLIENT_IMPORT_RE.search(text):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, (
        "`features/*/server-api.ts` must not import the browser typed "
        "client. Use `@/lib/api/typed/server` instead:\n  " + "\n  ".join(sorted(offenders))
    )


def test_features_types_is_single_source_for_domain_consumers():
    """G14: domain types must flow through ``features/<d>/types``.
    Component / route / adapter files outside the
    ``web_raw_schema_allowlist.txt`` may not import
    ``@/api-v2/schema`` directly — they read types from the owning
    feature.

    This anchors the single-source-of-truth rule at the module level
    and complements ``test_web_raw_schema_import_limited_to_typed_adapters``
    by making the rationale explicit."""
    allowed = set(_load_allowlist("web_raw_schema_allowlist.txt"))
    offenders: list[str] = []
    for path in _iter_web_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in allowed:
            continue
        if RAW_SCHEMA_IMPORT_RE.search(path.read_text()):
            offenders.append(rel)
    assert not offenders, (
        "Domain consumers must read types from `@/features/<d>/types`, "
        "not `@/api-v2/schema`. Only the typed-adapter allowlist may "
        "import raw schema directly:\n  " + "\n  ".join(sorted(offenders))
    )


def test_no_legacy_sdk_directory():
    """G15: Phase 1c deletes ``web/src/api/`` entirely. A reintroduction
    re-creates the dual-SDK problem the modularization is dismantling."""
    legacy_dir = WEB_SRC / "api"
    assert not legacy_dir.exists(), (
        f"Legacy SDK directory reintroduced at {legacy_dir}. "
        "Route new callers through `@/features/<domain>/{client,server}-api`."
    )


def test_no_legacy_lib_api_low_level_wrappers():
    """G16: the low-level ``web/src/lib/api/client.ts`` and
    ``server.ts`` wrappers are replaced by ``lib/api/typed/{browser,server}.ts``
    as part of Phase 1c. Callers that still needed the untyped path
    were the reason the legacy SDK stuck around; deleting these closes
    the fallback."""
    for name in ("client.ts", "server.ts"):
        target = WEB_SRC / "lib" / "api" / name
        assert not target.exists(), (
            f"Legacy low-level wrapper reintroduced at {target}. "
            "Use the typed client from `@/lib/api/typed/{browser,server}` "
            "via a `features/<domain>/{client,server}-api.ts` adapter."
        )


HIDDEN_API_OWNERS = {
    # Backend path prefix → allowed owner directory under ``web/src``.
    # These paths are excluded from the public OpenAPI spec via
    # ``HIDDEN_FROM_PUBLIC_PATH_PREFIXES`` in ``aperag/openapi_spec.py``
    # and therefore cannot be typed through ``openapi-fetch``. Raw
    # ``fetch()`` is a deliberate boundary exception (lesson 9a-ter)
    # confined to the owning feature adapter. Phase 4 governance
    # (decisions R + S) unhides both and promotes them to typed
    # wrappers; at that point this gate's entry for the promoted path
    # is removed (or the gate is dropped if the set becomes empty).
    "/api/v1/audit-logs": "features/audit/",
    "/api/v1/config": "features/auth/",
}


def test_hidden_api_raw_fetch_confined_to_owning_features():
    """G17 (multi-hidden-path variant per msg=6e0c542f): raw-fetch
    references to backend paths hidden from the public OpenAPI must be
    confined to the feature adapter that owns the domain.

    Phase 4 governance (decisions R + S in msg=659a98da) unhides audit
    + config and promotes them to typed wrappers; when both are
    unhidden this gate either loses its entries or is dropped."""
    offenders: list[str] = []
    for path in _iter_web_source_files():
        text = path.read_text()
        rel = path.relative_to(REPO_ROOT).as_posix()
        web_src_rel = path.relative_to(WEB_SRC).as_posix()
        for prefix, owner in HIDDEN_API_OWNERS.items():
            if prefix not in text:
                continue
            if not web_src_rel.startswith(owner):
                offenders.append(f"{rel} references {prefix}")
    assert not offenders, (
        "Hidden-path raw fetch reference found outside the owning "
        "feature adapter. Move the call to the owning feature or wait "
        "for the Phase 4 typed wrapper promotion. Offenders:\n  " + "\n  ".join(sorted(offenders))
    )
