# ApeRAG Backend Architecture — Post-Phase-6 Current State

> **Baseline**: `origin/main @ 28a9f531` (Phase 6 / PR #1635 merged, task #20 Done).
> **Status**: Authoritative source-of-truth for the modularization end-state. Replaces historical roadmap docs as the current-state reference.
> **Scope**: Documents **what exists today** — current domain layout, canonical rules, boundary gates, runtime seams, shim lifecycle. Historical plan/roadmap content lives in `README.md` / `roadmap.md` / `target-domain-map.md` for reference only. Future candidates are listed in Section 8 and must not be read as current state.

---

## 1. Executive summary

The modularization re-architected the ApeRAG backend from a monolithic `aperag.service.*` / `aperag.db.models` / `aperag.schema.view_models` layout into a per-domain layout under `aperag/domains/**` with enforced boundary gates and permanent DI seams for shared infrastructure.

### Phase 0→6 delivery

| Phase | PR | Merge | Scope |
| --- | --- | --- | --- |
| 0–2 | (multiple) | merged | Design-lock + retrieval / knowledge_graph / web_access hard-cut |
| 3 (#16) | #1629 | `3b208d80` | `knowledge_base` + `indexing` pilot split + DB layer base + boundary gates G1/G11/G13/G14 |
| 4 (#25) | #1633 | `023fffd7` | `identity` + `governance` + `model_platform` + `marketplace` split + G15/G16/G17 gates + identity DI adapters |
| 5 (#26) | #1634 | `c0a60d48` | `conversation` + `agent_runtime` + `evaluation` split + G18 alt + G19 gates + consumer-owned Protocol `QuotaOps` / `ChatCollectionServiceOps` / `PromptTemplateOps` |
| 6 (#20) | #1635 | `28a9f531` | Final cleanup: retire dead `ChatDocumentOps` Protocol, retire `ChatCollectionServiceOps` seam (sibling direct import), promote `_TERMINAL_RUN_STATUSES` to `EvaluationRunStatus.is_terminal()` classmethod, introduce `identity_user_ops.set_chat_collection` facade, fix `PromptTemplateOps` / `QuotaOps` as **standalone-infra permanent seams** |

### Steady-state facts (at `28a9f531`)

- **12 backend domains** under `aperag/domains/**`: `identity` · `governance` · `model_platform` · `marketplace` · `knowledge_base` · `indexing` · `retrieval` · `knowledge_graph` · `conversation` · `agent_runtime` · `evaluation` · `web_access`.
- **20 boundary tests** codified in `tests/unit_test/test_modularization_boundaries.py` covering **G1–G19**.
- **Two separate runtime-wiring registries** (see Section 5): G17 (Phase 3+4) holds **7 entries** covering KB consumer-owned Protocol seams + the three identity `*InitOps` adapters and remains fully active; G18 alt (Phase 5) was collapsed over the Phase 5/6 run to its final **2 permanent entries** — `conversation.bot_service._quota_ops` and `agent_runtime.runtime._prompt_template_ops`, both standalone-infra providers without a natural domain home. Both registries are separately enforced at `import aperag.app` time.
- **HTTP API byte-stable** throughout: `scripts/export_openapi.py --check` passed every squash-merge boundary.
- **54 CR review cycles** across Phase 3+4+5+6 (14+17+18+5), all no-blocker.

---

## 2. Domain map

Each backend domain lives under `aperag/domains/<domain>/`. The per-domain directory contract (not every domain uses every slot):

- `db/models.py` — SQLAlchemy ORM classes + Enum types owned by this domain.
- `schemas.py` — Pydantic schemas owned by this domain; bound onto `aperag.schema.view_models` through the dual-hook pattern (see Section 3).
- `ports.py` — consumer-owned `Protocol` classes declaring the shape this domain needs from its cross-domain collaborators; implementations live elsewhere and are wired at startup.
- `service/` or `service.py` — service-layer business logic; imports only this domain's own DB/schemas and cross-domain deps via ports or direct imports (see Section 3).
- `api/routes.py` (and any sibling `*_routes.py` for prefix-split cases) — FastAPI route module. Router var name is domain-specific; `aperag/views/*.py` holds router-re-export shims.

### 2.1 Domain inventory

| Domain | DB (Enum + ORM) | Schemas | Service modules | Consumer ports (own) | API routes |
| --- | --- | --- | --- | --- | --- |
| **identity** | `Role` · `User` · `OAuthAccount` | 13 | `user_manager`, `identity_user_ops` | `AuthenticatedUser`, `BotInitOps`, `ChatInitOps`, `QuotaInitOps` | — (auth wiring in `aperag/views/auth.py` + shim re-exports) |
| **governance** | `ApiKeyStatus` · `AuditResource` · `ApiKey` · `AuditLog` | 6 | `api_key_service`, `audit_service` | `AuthenticatedUser`, `UserView` | `api/routes.py` |
| **model_platform** | `APIType` · `LLMProvider` · `LLMProviderModel` | 26 | `default_model_service`, `llm_available_model_service`, `llm_provider_service` | `AuthenticatedUser` | `api/llm_routes.py` + `api/providers_v2_routes.py` (2-router split because `/api/v1` and `/api/v2` prefixes coexist) |
| **marketplace** | `CollectionMarketplaceStatusEnum` · `CollectionMarketplace` · `UserCollectionSubscription` | 3 | `marketplace_service`, `marketplace_collection_service` | `AuthenticatedUser` | `api/routes.py` |
| **knowledge_base** | `CollectionStatus` · `CollectionSummaryStatus` · `CollectionType` · `DocumentStatus` · `Collection` · `CollectionSummary` · `Document` | 24 | `collection_service` (consumer), `collection_summary_service`, `document_service` | `AuthenticatedUser`, `MarketplaceOps`, `MarketplaceCollectionOps`, `SearchPipelineOps`, `QuotaOps` | `api/routes.py` |
| **indexing** | `DocumentIndexType` · `DocumentIndexStatus` · `DocumentIndex` | 0 | functional modules (`manager`, `document_parser`, `vector_index`, `fulltext_index`, `graph_index`, `summary_index`, `vision_index`) | `CollectionIndexingView`, `IndexingTrigger` | — |
| **retrieval** | `SearchHistory` | 10 | `service.py` (monolithic) + `pipeline.py` | `GraphQueryContext`, `GraphSearchContract` | `api/routes.py` |
| **knowledge_graph** | `GraphCurationRunStatus` · `GraphCurationSuggestionStatus` · `GraphCurationRun` · `GraphCurationSuggestion` | 15 | `service.py` + `graphindex/*` (11 submodules) | `CollectionRow` | `api/routes.py` (+ a 410-Gone legacy shim on `aperag/views/graph.py`) |
| **conversation** | `BotStatus` · `BotType` · `ChatStatus` · `ChatPeerType` · `TurnFeedbackType` · `TurnFeedbackTag` · `Bot` · `Chat` · `TurnFeedback` | 20 | `bot_service` (consumer), `chat_collection_service`, `chat_document_service`, `chat_service`, `chat_title_service`, `turn_feedback_service` | `KnowledgeBaseCollectionView`, `AuthenticatedUser`, `QuotaOps` | `api/routes.py` |
| **agent_runtime** | `AgentTurnStatus` · `AgentEventActor` · `AgentArtifactType` · `AgentTurn` · `AgentTimelineEvent` · `AgentArtifact` | 13 | `runtime` (consumer, hosts the `_prompt_template_ops` slot), `services`, `storage` | `AuthenticatedUser`, `PromptTemplateOps` | `api/routes.py` |
| **evaluation** | `EvaluationDatasetSourceType` · `EvaluationRunStatus` · `EvaluationRunItemStatus` · `EvaluationRunItemAttemptStatus` · `EvaluationJudgeMode` · `EvaluationDataset` · `EvaluationDatasetItem` · `EvaluationRun` · `EvaluationRunItem` · `EvaluationRunItemAttempt` | 24 | `services`, `worker` (hosts `dispatch_fn` test-injection seam — intentionally **not** a DI slot, see Section 5), `tasks`, `judges`, `constants` + `db/repositories/evaluation_v2.py` | `AuthenticatedUser`; `ChatSessionOps` and `AgentTurnDispatchOps` are **dead Protocol literals** (zero runtime callers — see footnote and Section 8 F14) | `api/routes.py` |
| **web_access** | — (owns no entities) | 7 | — (functional sub-packages: `reader/`, `search/`, `utils/`) | — | `api/routes.py` |

> **Footnote (evaluation).** `aperag/domains/evaluation/ports.py` also carries class definitions for `ChatSessionOps` and `AgentTurnDispatchOps`, but these are **dead Protocol literals** with zero runtime references. They were seeded in Phase 5 5-S1 on the assumption that `chat_service` and `agent_runtime` would stay legacy; after the Phase 5 5-S4b / 5-S5b domain moves plus the `9162ec4` rebase, `evaluation.worker::dispatch_fn` switched to late-import direct cross-domain access and the seams were never wired. Their class bodies remain only because Phase 6 scope (msg=92bbdadb) locked to five specific cleanup entries; they are a candidate for mechanical deletion in a future sweep (Section 8, F14) following the precedent of Phase 6 entry 4 (`ChatDocumentOps` deletion).

### 2.2 Top-level infrastructure modules (not domain-owned)

| File / Dir | Role |
| --- | --- |
| `aperag/app.py` | FastAPI app factory; module-scope DI wire-up for all seams — 7 Phase 3+4 slots + 2 Phase 5+6 permanent slots (Section 5) |
| `aperag/config.py` | Settings / environment / database URL |
| `aperag/exception_handlers.py`, `aperag/exceptions.py` | FastAPI exception integration |
| `aperag/openapi_spec.py` | OpenAPI schema generation + `HIDDEN_FROM_PUBLIC_PATH_PREFIXES` registry |
| `aperag/db/base.py` | SQLAlchemy declarative `Base`; shared across all per-domain `db/models.py` so alembic autogenerate sees one metadata registry (see lesson in Phase 3 breaking-changes doc) |
| `aperag/db/ops.py` | Async DB query helpers |
| `aperag/db/models.py` | **Re-export shim** for pre-migration callers (see Section 6) |
| `aperag/db/repositories/*.py` | Repository layer — mix of domain-owned and legacy aggregate repos |
| `aperag/schema/common.py` | **Shared primitive module** — pure Pydantic value objects consumed by ≥2 domains. See Section 3.2 for the strict admission criteria. |
| `aperag/schema/view_models.py` | **Re-export shim with dual-hook blocks** for pre-migration callers (see Section 3.3 + Section 6) |
| `aperag/schema/utils.py` | Schema utilities |
| `aperag/llm/` | LiteLLM integration (completion / embedding / rerank / caching / tracking). Shared infrastructure, not a domain. |
| `aperag/utils/` | `audit_decorator`, constants, date/pagination helpers, spider, LLM response handling |
| `aperag/service/*.py` | **Re-export shims** for pre-migration callers + a handful of legacy-only services (Section 6) |
| `aperag/views/*.py` | Router re-export shims + a handful of non-domain legacy views (Section 6) |
| `aperag/agent_runtime/`, `aperag/evaluation_v2/` | **Legacy top-level packages** retained as re-export shims to keep pre-migration imports alive (Section 6) |

### 2.3 Shared primitive module — `aperag/schema/common.py`

`aperag/schema/common.py` is a **shared primitive module**, **not another aggregation layer**.

Admission criterion — a type lives in `common.py` if and only if:

1. It is a **primitive Pydantic shape** — pure value object: pagination envelope, chunk payload, model spec, a config subtree. No domain-specific semantics, no ORM dependency, no service-facing facade.
2. It is **consumed by ≥2 domains** with semantics that do not belong to any single one.

If a type is consumed by exactly one domain, it belongs in that domain's `schemas.py` — never in `common.py`.

Current `common.py` entries (8) and their cross-domain justification:

- `ModelSpec` — shared by knowledge_base (embedding/completion config), model_platform, conversation (bot config).
- `KnowledgeGraphConfig` / `IndexPrompts` — shared by knowledge_base (collection config) + indexing (reconciler prompts).
- `CollectionConfig` — KB's public config shape, consumed by knowledge_base + source ingestion + bots + retrieval.
- `PageResult` / `PaginatedResponse` — pagination envelope, reused by list endpoints across 6+ domains.
- `Chunk` / `VisionChunk` — search/chunking primitives, shared by knowledge_base + retrieval + indexing + vision.

New additions must pass the same criterion. This gate is enforced by CR (not AST automation) to prevent `common.py` from drifting back into a `view_models.py`-style catch-all.

### 2.4 `indexing` / `retrieval` / `knowledge_graph` (high-level structure — deeper SME input still needed)

**Note**: the following is a **high-level structural summary only**. Phase 3 Step 2/3/5a/6/7/8 landed the depth of these domains, and implement-level detail (Protocol method signatures, DI wire-up timing, `graphindex/*` reconcile loop, Nebula space lifecycle, indexing reconciler state machine) should be backfilled by the landing owner or a future deep-dive SME. The entries here cover architecture-level relationships only.

- **`indexing`** — reconciler + per-index-type workers (vector / fulltext / graph / summary / vision). Owns `DocumentIndex` ORM + `DocumentIndexType` / `DocumentIndexStatus` enums. Consumers (like `knowledge_base.collection_service`) drive it through `SearchPipelineOps` (consumer-owned) and indexing exposes `CollectionIndexingView` / `IndexingTrigger` for retrieval/KB wiring.
- **`retrieval`** — search pipeline orchestration + chunk aggregation + reranking. Consumer in the retrieval↔knowledge_graph relationship: retrieval owns `GraphQueryContext` / `GraphSearchContract`, provider-side (knowledge_graph) structurally satisfies them.
- **`knowledge_graph`** — entity / relation ORM + Nebula Graph client + `graphindex` reconciler. Provider for retrieval's `GraphSearchContract`. Owns `CollectionRow` (an internal port abstracting the legacy `Collection` shape the graphindex integration depends on).

### 2.5 `conversation` intra-domain dependency topology

`conversation` hosts six services; all intra-domain dependencies are sibling direct imports (no Protocol seams needed inside a single domain). The call direction at steady state:

- `bot_service` and `turn_feedback_service` are leaves — they do not consume any other conversation service.
- `chat_title_service` reads `chat_service` for chat metadata and calls `model_platform.default_model_service` (cross-domain direct) for the default-model lookup.
- `chat_collection_service` calls `knowledge_base.collection_service` + `knowledge_base.schemas.CollectionCreate` (cross-domain direct), `model_platform.llm_available_model_service` for embedding-model selection, and `identity.service.identity_user_ops.set_chat_collection` for the User write (Section 3.4).
- `chat_document_service` consumes `chat_collection_service` (sibling direct — Phase 6 entry 2 retired the `ChatCollectionServiceOps` Protocol seam in favor of this import), plus `knowledge_base.service.document_service` + `knowledge_base.schemas.Document` (cross-domain direct).
- `chat_service` is the top-level CRUD service and exports `chat_service_global` as a module-level singleton plus a `ChatRow` ORM alias so that the ORM row and the Pydantic `Chat` response schema can coexist in the same module without name collision.

Cross-domain consumers of this domain: `agent_runtime.runtime` (late-imports `chat_document_service`) and `evaluation.worker` (late-imports `chat_service_global` inside `dispatch_fn`). Both use late-import to keep module-import-time clean of evaluation → agent_runtime → conversation cycles.

### 2.6 Domain-edge relationship map

The cross-domain read/write relationships at steady state (edges labelled with the mechanism):

```
identity ←── (AuthenticatedUser Protocol, read-only)      ← every other domain
identity ─── (identity_user_ops.set_chat_collection facade)→ called by conversation.chat_collection_service

governance ─── (audit/api_key endpoints)                  (no cross-domain inbound; consumed by admin UI)
governance.quota_service (legacy standalone-infra)        ─── (QuotaOps Protocol)→ knowledge_base.collection_service, conversation.bot_service

model_platform ─── (default_model_service direct import)  ← conversation.chat_title_service
model_platform ─── (llm_available_model_service direct)   ← conversation.chat_collection_service

marketplace ─── (MarketplaceOps / MarketplaceCollectionOps Protocols, structurally satisfied)
                                                           → knowledge_base.collection_service

knowledge_base ─── (Collection / Document direct cross-domain imports)
                → consumed by conversation.chat_collection_service, retrieval.pipeline, agent_runtime.runtime, web_access routes

indexing ─── (CollectionIndexingView / IndexingTrigger)    ↔ knowledge_base / retrieval

retrieval ─── (one-way Protocol bridge, lesson 9a-quad)    → knowledge_graph
knowledge_graph: must never import retrieval.ports / retrieval.service / retrieval.schemas (G10/G3 enforced)

conversation.chat_service_global ─── direct cross-domain   ← evaluation.worker
conversation.chat_document_service ─── direct cross-domain ← agent_runtime.runtime

agent_runtime.runtime ─── (_prompt_template_ops DI slot)   ← standalone infra: aperag.service.prompt_template_service
agent_runtime (runtime/services/schemas) ─── direct        ← evaluation.worker

evaluation ─── (no cross-domain provider role; consumed by admin UI; worker has test-injection dispatch_fn seam)

web_access ─── (no entities; consumes KB Collection for web-search enriched queries)
```

---

## 3. Canonical rules

Four rules governed every cross-domain seam across Phase 0→6. Future work must uphold them.

### 3.1 Direct import vs `Protocol + DI`

Every cross-domain data access takes **one** of two forms:

**(a) Direct cross-domain import** — when the provider has been moved into `aperag/domains/**`:

```python
# consumer in agent_runtime.runtime:
from aperag.domains.conversation.service.chat_document_service import chat_document_service
await chat_document_service.has_documents_in_chat(chat_id, user)
```

G1 bans `aperag.service.*`, `aperag.db.models`, `aperag.schema.view_models`; it does **not** ban `aperag.domains.<other>.*`. Sibling (same-domain) imports are always direct.

**(b) Consumer-owned `Protocol` + module-scope DI slot** — when the provider has **not** moved into `aperag/domains/**`:

- Consumer declares a narrow `Protocol` in its own `ports.py`.
- Consumer provides a module-level slot, setter, and accessor: `_ops: Optional[XOps] = None`, `set_x_ops(ops) -> None`, `_get_x_ops() -> XOps` (the accessor raises `RuntimeError` with an actionable message if unwired, to fail at startup rather than during the first request).
- Concrete provider structurally satisfies the Protocol (no provider-side import of the consumer's `ports.py`).
- `aperag/app.py` wires the legacy singleton (or an adapter when signature shapes differ) into the slot at module scope, before routes start serving.

This is **consumer-owned Protocol** (lesson 9a-quad): consumer declares what it needs; the provider is decoupled from the contract; the test_modularization_boundaries `test_knowledge_base_protocol_boundary_is_consumer_owned` enforces that legacy providers never `import` the consumer's ports.

### 3.2 The two subclasses of `Protocol + DI`

The `Protocol + DI` seams in the code today fall into exactly two subclasses, with different meanings:

**(A) legacy-not-moved-yet** — transitional. A provider that is expected to relocate in a future phase; the seam exists to decouple the consumer from the legacy `aperag.service.*` location in the meantime. Example (historical): Phase 5's `ChatCollectionServiceOps` seam existed because `chat_collection_service` had not yet moved; when it did (Phase 5 5-S4f), the seam was retired in Phase 6 entry 2 in favor of a sibling direct import.

**(B) standalone-infra permanent** — the provider has no natural domain home (cross-cutting, primarily infrastructure), and the `Protocol + DI` seam is its canonical, permanent integration point. Two seams live here today (the only two active `CRITICAL_WIRINGS`; Section 5):

- `QuotaOps` — consumers: `knowledge_base.collection_service`, `conversation.bot_service`. Provider: `aperag.service.quota_service` (governance-adjacent infra, consumed cross-cuttingly).
- `PromptTemplateOps` — consumer: `agent_runtime.runtime`. Provider: `aperag.service.prompt_template_service` (cross-cutting across agent_runtime runtime execution, conversation bot-config resolution, indexing prompt resolution, and the user-facing `/prompts` REST surface).

A new `Protocol + DI` seam is created **only** for legacy-not-moved-yet (class A) or standalone-infra (class B). Any other cross-domain consumption uses direct import.

### 3.3 Dual-hook Scenario A — view_models re-export without triggering G1

The Pydantic schema migration had to preserve `from aperag.schema.view_models import <DomainSchema>` for pre-migration callers while moving the canonical definition into `aperag/domains/<d>/schemas.py`. G1 forbids domain code from importing `aperag.schema.view_models`, which rules out a naive `import`-based re-export from the domain side. The two-way binding below is the solution ("Scenario A"), first used in Phase 3 Step 4b for knowledge_base and then reused as the canonical dual-hook template for Phase 4 (4 domains), Phase 5 (conversation + agent_runtime carve), and Phase 6 cleanup:

```python
# aperag/schema/view_models.py — end of file
try:
    from aperag.domains.knowledge_base.schemas import (  # noqa: E402, F401
        Collection, CollectionView, ..., MineruTokenTestResponse,
    )
except ImportError:
    pass
```

```python
# aperag/domains/knowledge_base/schemas.py — end of file
def _bind_view_models_reexports() -> None:
    import sys
    _vm = sys.modules.get("aperag.schema.view_models")
    if _vm is None:
        return
    for _name in __all__:
        setattr(_vm, _name, globals()[_name])

_bind_view_models_reexports()
```

The key trick is `sys.modules.get("aperag.schema.view_models")` — a runtime string lookup, not an `import` statement — so AST-based G1 scanning does not trip. At runtime, the pattern converges regardless of load order:

- If `view_models` loads first, its `try:` block imports from the domain module; the domain's `_bind_view_models_reexports` then sees `view_models` already in `sys.modules` and `setattr`s the canonical class objects back.
- If the domain module loads first, `_bind_view_models_reexports` finds `view_models` not yet in `sys.modules` and early-returns; when `view_models` loads later, its `try:` block imports from the now-loaded domain module.

In both orders the invariant `view_models.X is aperag.domains.<d>.schemas.X` holds (single class-object identity).

As of `28a9f531`, `aperag/schema/view_models.py` ends with 6 dual-hook `try:` blocks (knowledge_base, identity, governance, marketplace, model_platform, conversation), plus a 7th block for Phase 5 agent_runtime's `AgentMessage` carve.

### 3.4 Per-domain `AuthenticatedUser(Protocol)` vs the `User` ORM (lesson 9a-ter and 9a-sexdec)

Domain route handlers and service functions must not bind to the concrete `User` ORM class — they depend on a per-domain `AuthenticatedUser(Protocol)` that only exposes the fields that handler actually reads (at minimum `id: str`, optionally `role: str`). G16 enforces this: an AST import ban for `User` outside the identity domain. `required_user` at FastAPI dependency time returns a concrete `User` row, which structurally satisfies each domain's local `AuthenticatedUser`.

For **write** access to the `User` row from another domain (the single case that exists today is `conversation.chat_collection_service` updating `User.chat_collection_id` when a chat is promoted to a collection), the hierarchy is:

1. **Terminal canonical** — an identity-owned facade. `aperag/domains/identity/service/identity_user_ops.py` exposes narrow async functions like `set_chat_collection(session, user_id, collection_id)` that wrap `session.get(User, …)` + attribute assignment inside the caller's transaction. Consumers import the facade by name, never the ORM class. This is what Phase 6 entry 1 delivered.
2. **Acceptable fallback** — inline `sqlalchemy.text("UPDATE users SET … WHERE id = :uid")` with explicit parameter binding, single call site, same-transaction semantics. Accepted as a transitional workaround (Phase 5 5-S4f did this before the facade existed) but non-canonical for the final state.
3. **Forbidden** — `session.get(User, …)` or `session.query(User)` in any non-identity domain.

This hierarchy is codified as lesson 9a-sexdec.

---

## 4. Boundary gates

Every boundary gate is codified as a pytest test in `tests/unit_test/test_modularization_boundaries.py` (backend scope — the FE G1–G6/G12–G17 gates in the same file cover `web/` and are outside this document's scope). The pytest run is part of `make test-unit`, which is a required CI check; every gate also corresponds to a `noqa`-hostile error message so that CR catches violations at review time, not at deploy time.

### 4.1 Backend gate catalog

| Gate | Test function | Purpose | Mechanism |
| --- | --- | --- | --- |
| **G1** | `test_aperag_domains_never_import_legacy_aggregate_modules` | Strict ban: `aperag/domains/**` must not import `aperag.service.*`, `aperag.schema.view_models`, or `aperag.db.models`. Enforces the domain-owned canonical path. | AST `ast.walk` + `ast.ImportFrom` scan over every `aperag/domains/**/*.py` non-`__init__` file; matches against `LEGACY_AGGREGATE_MODULES` tuple. |
| **G4** | `test_aperag_domains_auth_dependency_is_not_any` | Auth-dependency parameters (`required_user` / `current_user`) in domain API handlers must not be typed `Any` / untyped; they must bind to a `Protocol` (typically `AuthenticatedUser`). | AST annotation scan over handler function defs under `aperag/domains/**/api/**/*.py`. |
| **G10 / G3** | `test_retrieval_kg_protocol_boundary_is_one_way` | `retrieval` may consume narrow slices of `knowledge_graph` via the `graphindex.integration` port, but must not import `knowledge_graph.service` / `knowledge_graph.schemas` / `aperag.graph_curation` / `aperag.graphindex`. `knowledge_graph` must never import `retrieval.ports` / `retrieval.schemas` / `retrieval.service` / `retrieval.pipeline`. | AST import scan over `aperag/domains/retrieval/**` + `aperag/domains/knowledge_graph/**`. |
| **G14** | `test_no_legacy_retrieval_or_graph_routes_remain` | After Phase 2 hard-cut, no retrieval or kg route decorators may live in `aperag/views/collections.py` or `aperag/views/graph.py` (except the single 410-Gone shim on `/collections/{id}/graphs/export/kg-eval`). | Regex match on `_LEGACY_ROUTE_PATTERNS` over the two legacy view files. |
| **KB consumer-owned** | `test_knowledge_base_protocol_boundary_is_consumer_owned` | KB's `ports.py` is consumer-owned: legacy providers (`marketplace_service`, `marketplace_collection_service`, `search_pipeline_service`, `quota_service`) must **not** import `aperag.domains.knowledge_base.ports`. Enforces that Protocols travel in the direction consumer → contract → structurally-satisfying provider, never the reverse. | AST import scan on the four legacy provider files. |
| **KB DI smoke** | `test_knowledge_base_di_wire_up_populated_after_app_import` | After `import aperag.app`, the four KB DI slots on `collection_service` (`_marketplace_ops`, `_marketplace_collection_ops`, `_search_pipeline_ops`, `_quota_ops`) must resolve non-`None`. | Runtime `getattr` on module globals. |
| **G15** | `test_phase4_consumer_domains_never_import_role_enum` | Non-identity domains must not `from aperag.db.models import Role` or `from aperag.domains.identity.db.models import Role`. Role remains identity-internal; consumers compare against the string literal (`user.role == "admin"`) via the per-domain `AuthenticatedUser.role: str`. | AST `ImportFrom` scan over all domains except `identity`. |
| **G16** | `test_phase4_consumer_domains_never_import_user_orm_class` | Non-identity domains must not import the `User` ORM class. Reads go through `AuthenticatedUser(Protocol)` / `UserView(Protocol)`; writes go through the identity-owned facade (lesson 9a-sexdec). | AST `ImportFrom` scan over all domains except `identity`. |
| **G17** | `test_phase4_di_critical_wirings_at_app_startup` | Runtime smoke: after `import aperag.app`, seven Phase 3+4 DI slots must resolve non-`None` — four KB slots (`_marketplace_ops`, `_marketplace_collection_ops`, `_search_pipeline_ops`, `_quota_ops`) and three identity slots (`_bot_init_ops`, `_chat_init_ops`, `_quota_init_ops`). Catches forgotten or re-ordered startup wire-up at CI time. Runtime-state check, not AST setter-naming scan. | Runtime `getattr` on module globals. |
| **G18 alt** | `test_phase5_di_critical_wirings_at_app_startup` | Runtime smoke for the permanent consumer-owned Protocol DI slots: `conversation.bot_service._quota_ops` and `agent_runtime.runtime._prompt_template_ops`. These are the standalone-infra seams (Section 3.2 class B). `dispatch_fn` in `evaluation.worker` is intentionally **not** listed — it is a module-level test-injection seam, not a Protocol + DI slot. | Runtime `getattr` on module globals. |
| **G19** | `test_phase5_domain_routes_never_use_pep_563_future_annotations` | `aperag/domains/**/api/routes.py` and the two `model_platform` split router modules must not declare `from __future__ import annotations`. FastAPI response-model handling at `status_code=204` combined with PEP 563 string-form annotations produced a silent drift in Phase 3 Step 5a (lesson 9a-quatuordec); codifying the ban ensures the trap stays closed. | AST `ImportFrom` scan for `__future__` with `annotations` name over every route module under `aperag/domains/**/api/`. |

The `G11` / `G13` labels referenced in some historical docs map to the FE legacy-SDK and FE server-api / client-api split checks earlier in the same file; they guard the frontend boundary, not the backend domain boundary.

### 4.2 Gate lineage

- Phase 3 introduced G1 / G4 / G10 / G14 / KB consumer-owned / KB DI smoke.
- Phase 4 introduced G15 / G16 / G17, driven by the decision to move `identity` without exposing `User` or `Role` to any consumer domain.
- Phase 5 introduced G18 alt / G19. G18 alt specifically generalized G17's "runtime wiring must be populated" pattern to the Phase 5 consumer-owned seams. G19 codified lesson 9a-quatuordec after the PEP-563 regression.

---

## 5. Runtime seams — `CRITICAL_WIRINGS` steady state

Two separate runtime-wiring registries live side by side at `import aperag.app` time. Be precise: the "final 2 permanent" phrase refers to the **Phase 5** G18 alt registry; the Phase 3+4 **G17** registry still holds **7** live entries. Collapsing the two into a single "2-entry" summary is incorrect.

### 5.1 The Phase 5 permanent two-entry registry (G18 alt)

At steady state (post-Phase-6), the `PHASE5_CRITICAL_WIRINGS` registry in `test_phase5_di_critical_wirings_at_app_startup` contains exactly **two** entries:

1. `(aperag.domains.conversation.service.bot_service, "_quota_ops")` — structurally satisfied by `aperag.service.quota_service.quota_service`. Wired in `aperag/app.py` at startup.
2. `(aperag.domains.agent_runtime.runtime, "_prompt_template_ops")` — satisfied by `_PromptTemplateOpsAdapter(...)` wrapping `aperag.service.prompt_template_service.prompt_template_service`. Wired in `aperag/app.py` at startup.

Both providers are **standalone-infra** (Section 3.2 class B) — they have no natural domain home because their consumers span multiple domains and their semantics do not belong to any single one. Their `Protocol + DI` seam is canonical, not transitional; no future phase is expected to retire them.

The Phase 3+4 G17 registry (seven entries: four KB + three identity) remains active and separately verified by `test_phase4_di_critical_wirings_at_app_startup`. Those seven entries bridge legacy concrete providers through DI adapters so that identity's `UserManager.on_after_register` and KB's `collection_service` can be wired at startup without the domains importing the legacy services directly.

### 5.2 Why G18 alt collapsed to two (but G17 did not)

During Phase 5 the preliminary Phase 5 registry was projected to hold as many as five consumer-owned seams (`QuotaOps`, `ChatCollectionServiceOps`, `DefaultModelOps`, `ChatDocumentOps`, `PromptTemplateOps`). As each of the collaborators completed a domain move (`default_model_service` in Phase 4, `chat_document_service` in Phase 5 5-S4d, `chat_collection_service` in Phase 5 5-S4f), the rule of Section 3.1 applied: a domain-moved provider is reached via direct import, so the corresponding Phase 5 seam retired. Phase 6 entry 2 retired `ChatCollectionServiceOps` in favor of a sibling direct import after 5-S4f put `chat_collection_service` in the same `conversation` domain. Phase 6 entry 3 classified the remaining two as standalone-infra permanent. G18 alt ended at two.

G17 did not collapse in the same way. Its seven entries cover two distinct categories:

- **KB consumer-owned Protocol slots (four entries)** — `_marketplace_ops`, `_marketplace_collection_ops`, `_search_pipeline_ops`, `_quota_ops` on `knowledge_base.collection_service`. `_quota_ops` and `_marketplace_ops` sit on providers whose standalone-infra classification was locked in Phase 6 entry 3 (`quota_service` / `prompt_template_service`-equivalent positioning). `_search_pipeline_ops` is provided by `aperag.service.search_pipeline_service`, a legacy provider whose long-term classification (standalone-infra permanent vs Phase 7+ extraction into a domain) was **not** decided during Phase 6 and is listed in Section 8 for future resolution. `_marketplace_collection_ops` is provided by `marketplace.marketplace_collection_service` (which did move in Phase 4), but the slot is still wired at startup because the KB collection_service keeps a narrow consumer surface through the Protocol; Phase 6 did not revisit this seam (see Section 8 for the candidate simplification).
- **Identity `*InitOps` adapters (three entries)** — `_bot_init_ops`, `_chat_init_ops`, `_quota_init_ops` on `identity.user_manager`. These are consumed by `UserManager.on_after_register` to kick off default bot creation, chat collection initialization, and quota seeding when a new user registers. The providers were legacy at Phase 4 time; after Phase 5 the `bot_service` / `chat_collection_service` moved into `conversation` and `quota_service` stayed as standalone-infra. The three `app.py` adapters (`_BotInitOpsAdapter` / `_ChatInitOpsAdapter` / `_QuotaInitOpsAdapter`) still resolve their internals through `aperag.service.bot_service` / `aperag.service.chat_collection_service` / `aperag.service.quota_service` (shim-resolved to the domain singletons where applicable), but the adapters themselves are the identity-side contract and stay wired at startup. Retirement would require collapsing the three Protocols, which is a new design decision — not a sequel to Phase 5's domain-move rule.

So G17 is stable at seven entries and has no collapse pending; G18 alt is stable at two permanent entries by the Phase 6 classification. The phrase "2 permanent" in Section 1 and elsewhere in this document always refers to G18 alt.

Two patterns that look like seams but are intentionally outside both registries:

- **`dispatch_fn` in `aperag.domains.evaluation.worker`** — a module-level function reference used for test injection (tests monkeypatch it to stub out the real turn dispatcher). It is not a `Protocol + DI` slot — it does not follow the slot/setter/accessor shape, does not need app-scope wire-up, and stays functional in test fixtures. Listing it in `CRITICAL_WIRINGS` would falsely signal that the runtime needs it populated at startup (it does not — the default implementation is a proper function, not a `None` slot).
- **Phase 4 identity adapters** (`_BotInitOpsAdapter` / `_ChatInitOpsAdapter` / `_QuotaInitOpsAdapter` in `aperag/app.py`) — these wrap the concrete services to expose the `BotInitOps` / `ChatInitOps` / `QuotaInitOps` Protocol surfaces required by `UserManager.on_after_register`. They are verified by the Phase 4 G17 registry, not the Phase 5 G18 alt registry.

---

## 6. Legacy shim lifecycle

To keep pre-migration callers (tests, third-party code, in-flight branches) working, every moved symbol keeps a re-export shim at its legacy import path. The shims fall into four categories; none of them are removed at Phase 6 — they remain to preserve API compatibility and are candidates for a future `Phase 7+` hard-delete sweep (Section 8). This section documents the current state so the shim map can be re-audited before that sweep.

### 6.1 `aperag/service/*.py` — service-layer shims

19 files re-export the class and singleton from `aperag/domains/<d>/service/<s>.py`:

```
aperag/service/api_key_service.py              → aperag.domains.governance.service.api_key_service
aperag/service/audit_service.py                → aperag.domains.governance.service.audit_service
aperag/service/bot_service.py                  → aperag.domains.conversation.service.bot_service
aperag/service/chat_collection_service.py      → aperag.domains.conversation.service.chat_collection_service
aperag/service/chat_document_service.py        → aperag.domains.conversation.service.chat_document_service
aperag/service/chat_service.py                 → aperag.domains.conversation.service.chat_service
aperag/service/chat_title_service.py           → aperag.domains.conversation.service.chat_title_service
aperag/service/collection_service.py           → aperag.domains.knowledge_base.service.collection_service
aperag/service/collection_summary_service.py   → aperag.domains.knowledge_base.service.collection_summary_service
aperag/service/default_model_service.py        → aperag.domains.model_platform.service.default_model_service
aperag/service/document_service.py             → aperag.domains.knowledge_base.service.document_service
aperag/service/graph_service.py                → aperag.domains.knowledge_graph.service
aperag/service/llm_available_model_service.py  → aperag.domains.model_platform.service.llm_available_model_service
aperag/service/llm_provider_service.py         → aperag.domains.model_platform.service.llm_provider_service
aperag/service/marketplace_collection_service.py → aperag.domains.marketplace.service.marketplace_collection_service
aperag/service/marketplace_service.py          → aperag.domains.marketplace.service.marketplace_service
aperag/service/prompt_template_service.py      ← standalone-infra permanent seam (Phase 6 entry 3 canonical, not a shim)
aperag/service/quota_service.py                ← standalone-infra permanent seam (Phase 6 entry 3 canonical, not a shim)
aperag/service/search_pipeline_service.py      ← legacy provider for KB `SearchPipelineOps`; classification (standalone-infra permanent vs Phase 7+ extraction) pending future decision
aperag/service/turn_feedback_service.py        → aperag.domains.conversation.service.turn_feedback_service
```

And the following `aperag/service/*.py` files are legacy-only (no domain extraction has happened): `chat_completion_service.py`, `evaluation_service.py`, `export_service.py`, `question_set_service.py`, `setting_service.py`, `test_mcp_agent.py`. Whether any of these become Phase 7+ extraction candidates is in Section 8.

### 6.2 `aperag/views/*.py` — router re-export shims

13 view files re-export a domain router under the pre-migration name:

```
aperag/views/agent_runtime.py              → aperag.domains.agent_runtime.api.routes.router
aperag/views/api_key.py                    → aperag.domains.governance.api.routes.router
aperag/views/audit.py                      → aperag.domains.governance.api.routes.router
aperag/views/bots_v2.py                    → aperag.domains.conversation.api.routes.bots_router
aperag/views/chat.py                       → aperag.domains.conversation.api.routes.chat_router
aperag/views/collections_v2.py             → aperag.domains.knowledge_base.api.routes.router
aperag/views/documents_v2.py               → aperag.domains.knowledge_base.api.routes.router
aperag/views/evaluation_v2.py              → aperag.domains.evaluation.api.routes.router
aperag/views/llm.py                        → aperag.domains.model_platform.api.llm_routes.router
aperag/views/marketplace_collections.py    → aperag.domains.marketplace.api.routes.router
aperag/views/marketplace.py                → aperag.domains.marketplace.api.routes.router
aperag/views/providers_v2.py               → aperag.domains.model_platform.api.providers_v2_routes.router
```

Plus `aperag/views/auth.py` (legacy fastapi-users wiring + OAuth routers; not yet a domain — the `UserManager` and `required_user` / `optional_user` helpers live here by design). And the legacy views retained for non-domain endpoints: `chat_documents.py`, `collections.py` (hosts a 410-Gone shim), `config.py`, `export.py`, `graph.py` (hosts a 410-Gone shim), `main.py`, `openai.py`, `prompts.py`, `quota.py`, `settings.py`, `test.py`, `utils.py`.

### 6.3 `aperag/db/models.py` + `aperag/schema/view_models.py`

`aperag/db/models.py` re-exports every Phase 3/4/5-moved symbol through per-domain import blocks; the `identity` block is special-cased to be module-top because `Invitation.role` binds `Role` at class-body evaluation time, whereas governance / marketplace / model_platform / knowledge_base / indexing / retrieval / knowledge_graph / conversation / agent_runtime / evaluation are all end-of-file blocks. This yields 53 re-exported symbols (15 Phase 3 + 13 Phase 4 + 25 Phase 5) plus the pre-existing non-migrated classes (the legacy v1 evaluation classes, `Invitation`, etc.).

`aperag/schema/view_models.py` uses the dual-hook pattern (Section 3.3) via 6 end-of-file `try:` blocks (knowledge_base, identity, governance, marketplace, model_platform, conversation) plus a 7th block for `AgentMessage`. Pre-migration imports (`from aperag.schema.view_models import <SchemaName>`) keep working unchanged.

### 6.4 Legacy top-level packages

- `aperag/agent_runtime/` retains `__init__.py`, `runtime.py`, `schemas.py`, `services.py`, `storage.py` as re-export shims so imports like `aperag.agent_runtime.runtime.agent_runtime_manager` still resolve. The canonical implementation lives in `aperag/domains/agent_runtime/`.
- `aperag/evaluation_v2/` retains `__init__.py`, `constants.py`, `judges.py`, `schemas.py`, `services.py`, `tasks.py`, `worker.py` as re-export shims. The canonical implementation lives in `aperag/domains/evaluation/`.
- `aperag/db/repositories/evaluation_v2.py` retains a shim for the evaluation repo mixin whose canonical home is now under `aperag/domains/evaluation/db/repositories/`.

---

## 7. Historical index

For the full rationale of each phase's hard-cuts and the decisions that produced today's layout, see the per-phase breaking-changes documents; this architecture doc deliberately does not repeat their tables.

- `docs/modularization/breaking-changes/phase-template.md` — template for new breaking-changes docs.
- `docs/modularization/breaking-changes/phase2-retrieval-knowledge_graph.md` — Phase 2 hard-cut of retrieval + knowledge_graph.
- `docs/modularization/breaking-changes/phase2-web_access.md` — Phase 2a hard-cut of web_access.
- `docs/modularization/breaking-changes/phase3-knowledge_base.md` — Phase 3 knowledge_base pilot + indexing split + the original G1 / G11 / G13 / G14 gates + the dual-hook Scenario A origin.
- `docs/modularization/breaking-changes/phase4-identity-governance-model_platform-marketplace.md` — Phase 4 identity + governance + model_platform + marketplace + the introduction of G15 / G16 / G17.
- `docs/modularization/breaking-changes/phase5-conversation-agent-eval.md` — Phase 5 conversation + agent_runtime + evaluation + G18 alt + G19 + the `ChatDocumentOps` retirement rationale.

The earlier modularization roadmap and the pre-Phase-2 target domain map live in:

- `docs/modularization/README.md` — historical overview + phase timeline.
- `docs/modularization/roadmap.md` — original phase plan.
- `docs/modularization/target-domain-map.md` — pre-Phase-2 canonical target domain structure.
- `docs/modularization/gate-checklist.md` — gate status tracker, updated phase-by-phase.
- `docs/modularization/hurl-coverage-matrix.md` — HTTP endpoint coverage.
- `docs/modularization/fe-legacy-sdk-inventory.md` — frontend legacy SDK deprecation status.

Those three roadmap docs are authoritative for *history* and *plan*; for *current state*, prefer this document.

---

## 8. Future candidates

Items that are known work but are deliberately **not** part of the current state. None of these are scheduled as a commitment; each would be a separate `Phase 7+` task with its own PM dispatch. Listing them here prevents rediscovery and keeps the current-state / future-work boundary clean. Presentation follows the architect audit (task #27 thread): the three high-value candidates get a paragraph each; the rest are compressed into a short catalog.

### 8.1 High-value candidates

**F1. Cross-domain integration test coverage.** Boundary tests today catch static import + DI-wire-up drift, but nothing verifies cross-domain runtime flows end-to-end (identity → conversation → agent_runtime → evaluation, or knowledge_base → retrieval → knowledge_graph). A focused Phase 7+ task could add three to four integration tests exercising the canonical flows against live domain services (no mocks), serving as regression coverage for the Section 3 canonical rules.

**F2. `aperag/service/*.py` legacy shim hard-delete audit.** 19 of the 22 files under `aperag/service/` are re-export shims whose canonical implementation now lives under `aperag/domains/<d>/service/`. Three (`quota_service`, `prompt_template_service`, `search_pipeline_service`) are standalone-infra canonical and stay. After a caller-by-caller audit (including `aperag/app.py` wiring, legacy views, tests, hurl fixtures, alembic migrations), a single batch-delete PR can drop the remainders; Section 6 documents the full map.

**F3. `aperag/views/*.py` migration or retirement map.** 26 files / ~2200 LOC remain. 13 are router re-export shims (Section 6.2); the others are mixes of still-active non-domain endpoints (`auth.py`, `config.py`, `main.py`, `openai.py`, `prompts.py`, `quota.py`, `settings.py`), retained 410-Gone compatibility (`collections.py`, `graph.py`), and legacy views with no domain home assigned yet (`chat_documents.py`, `export.py`, `test.py`). A Phase 8+ task should produce a migration map (which views stay as infrastructure-only, which migrate to a domain, which retire) before any hard-delete pass.

### 8.2 Compressed catalog

- **F4. `_enum_column` helper consolidation** (effort S, value L-M). Nine byte-identical copies across per-domain `db/models.py` → consolidate into `aperag/db/common.py` (flagged in Phase 3 Step 2).
- **F5. Per-domain `AuthenticatedUser(Protocol)` consolidation** (effort M, value M). 10+ local declarations; collapsing into a shared `aperag/domains/identity/contracts/` Protocol weakens the consumer-owned principle (Section 3.1) — judgement call.
- **F6. `aperag/app.py` DI wire-up extraction** (effort L, value M). 43+ imports, 4 inline adapter classes, 18+ setter calls; extracting into a dedicated `aperag/wiring.py` or `aperag/di/setup.py` reduces `app.py` complexity.
- **F7. `aperag/schema/view_models.py` residual legacy re-exports** (effort M, value M). 15 Phase 3/5 re-export symbols plus call sites in `aperag/app.py` + legacy views + migrations; would be batched with F3.
- **F8. G-gate-to-test mapping doc** (effort S, value M). Explicit table tying G1–G19 to the 20 boundary tests; useful if F1 lands so integration coverage can slot in beside gate enforcement. G3 / G10 currently lack dedicated integration tests.
- **F9. `aperag/agent_runtime/` + `aperag/evaluation_v2/` top-level shim hard-delete** (effort S, value M). Residual legacy packages are only imported by a handful of internal call sites; verify + batch delete.
- **F10. `aperag/platform/` layering** (effort S, value L). Shared-infra modules today live scattered under `aperag/{db,llm,objectstore,vectorstore,docparser,trace,mcp,context,concurrent_control,query}`; a future organizational pass could collect them under a canonical `aperag/platform/` subtree without any semantic change.
- **F11. `web_access` domain depth** (effort TBD, value L). `web_access` today has schemas and routes but no entities or service layer. A future phase could decide whether `reader/` and `search/` warrant the full domain contract or stay as functional modules.
- **F12. `/api/v1` vs `/api/v2` prefix unification** (effort M, value L). `model_platform` and `conversation` both carry a two-router split for prefix coexistence; a future cleanup could collapse once `/api/v1` is deprecated.
- **F13. Deeper SME write-up for `indexing` / `retrieval` / `knowledge_graph`** (effort M, value M). Section 2.4 is high-level; the Phase 3 landing owner or a future deep-dive SME could backfill Protocol method signatures, DI wire-up timing, the `graphindex` reconcile loop, Nebula space lifecycle, and the indexing reconciler state machine.
- **F14. Dead Protocol class literal sweep** (effort S, value L). `aperag.domains.evaluation.ports` still declares `ChatSessionOps` and `AgentTurnDispatchOps` with zero runtime callers (`rg 'ChatSessionOps\|AgentTurnDispatchOps' aperag/ | grep -v ports.py` returns empty). Same shape as `ChatDocumentOps` before Phase 6 entry 4 deleted it. Trivial mechanical cleanup.
- **F15. `aperag.service.search_pipeline_service` classification decision** (effort S design / M–L execute, value M). The provider is currently wired through KB's `_search_pipeline_ops` Protocol + DI slot; an explicit design-lock is needed on whether it graduates to **standalone-infra permanent** (like `quota_service` and `prompt_template_service`) or is extracted into a new `search_pipeline` / `retrieval` sub-domain. Decision unblocks both the seam's steady-state classification and the F2 shim hard-delete audit.

---

*Document baseline: `origin/main @ 28a9f531`. Last reviewed: post-Phase-6 merge (2026-04-24). If the repo state diverges from the sections above, either this document or the repo is out of date — fix whichever is easier to fix and do not rely on a stale copy.*
