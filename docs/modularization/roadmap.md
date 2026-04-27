# Modularization roadmap

This roadmap is the execution source of truth for the remaining ApeRAG
modularization work after the Phase 0 baseline and the first Phase 1 FE
batches. It complements `target-domain-map.md` and `gate-checklist.md`:

- `target-domain-map.md` defines the final domain ownership.
- `gate-checklist.md` defines merge gates.
- This file defines execution order, PR slicing, dependencies, and
  explicit non-goals.

Current baseline: `main @ 6a89865f` after PR #1613.

| Boundary fixture | Current count | Target |
| --- | ---: | ---: |
| `web_legacy_api_allowlist.txt` | 43 | 0 before Phase 1c completes |
| `web_raw_schema_allowlist.txt` | 11 | exact canonical typed-adapter lock |
| `web_route_data_allowlist.txt` | 19 | 0 before Phase 1c completes |

## Operating rules

- Keep the top-level Slock channel quiet. Execution plans, review,
  blocker discussion, and merge closure live in the owning task thread.
- Keep only one or two execution windows active. Use this roadmap for
  visibility; do not bulk-create dozens of tasks.
- Every PR must be independently reviewable, reversible by `git revert`,
  and must state its allowlist delta or explicitly state why it has none.
- If a PR touches OpenAPI paths, it must update generated schema,
  relevant hurl coverage, and a breaking-change table.
- If a FE adapter returns a typed schema component, fallback values must
  satisfy the component's required shape. Never use `{}` as a generic
  empty state.
- Type-propagation cascades are allowed only when an upstream typed source
  change directly causes a downstream type error, the downstream diff is
  limited to type source / nullability narrowing, and any file with
  remaining legacy imports stays in the allowlist.
- Domain ownership follows API and data semantics, not directory names.
  Cross-domain calls must go through canonical feature/domain contracts.
- Bounded redesign is allowed inside the current domain slice. A
  modularization PR may rewrite old code when that produces a cleaner
  canonical boundary, provided the behavior contract is explicit, tests
  are added or updated, and the PR does not expand into another phase or
  domain.

## Bounded redesign policy

Small modularization PRs are not limited to mechanical file moves or
import replacement. They may refactor, redesign, or rewrite legacy code
inside the current slice when the change is needed to land the target
domain boundary cleanly.

Allowed examples:

- replace legacy enum classes with schema-derived types and typed consts,
- replace ambiguous adapter fallbacks with typed empty-list or fail-fast
  behavior,
- collapse a legacy service/helper into the new domain contract when the
  old shape would force a permanent shim,
- rewrite a route shell to expose the canonical contract directly.

Hard limits:

- Behavior changes must be explicit in the PR body. OpenAPI response
  shape, hurl payload, DB migration, FE contract, and user-visible
  behavior changes must be listed.
- Tests must cover the new boundary: unit, typed-contract, boundary,
  hurl, or migration tests depending on the touched surface.
- Redesign must stay within the current domain / task scope. Reviewers
  may request a split if the rewrite crosses into another phase.
- Opportunistic refactors must be directly adjacent to the current
  migration boundary: the same feature adapter, migrated caller, route
  shell, or typed error/fallback rule. Broad DRY cleanup across already
  merged domains must be a separate cleanup PR, not hidden inside a
  domain migration.
- Shared helpers should be introduced only after the repeated behavior
  is stable across endpoints. Do not abstract over distinct semantics
  such as `throw`, `notFound`, and typed empty-list fallback until the
  helper can preserve each required response shape.
- Temporary bridge or shim code must name the phase or PR that deletes
  it. Long-term compatibility shims are not accepted by default.
- If a behavior shift is intentional and not strictly compatible, it
  must be recorded in the relevant breaking-change table or PR notes.

## Phase timeline

| Phase | Scope | Primary owners | Estimated PRs | Dependencies | Hard gates | Explicit non-goals |
| --- | --- | --- | ---: | --- | --- | --- |
| 1b remaining | Finish FE caller migration batches that can be wrapped by current typed schema. | FE owner + FE reviewers | 6-10 | Phase 0 fixtures; current typed API clients | Shrink legacy and route-data allowlists with each real migration; scoped typed-contract tests | No backend path rename; no DB split; no shared-component hard move |
| 1c | Delete generated FE legacy SDK and low-level route callers. | FE owner + architecture review | 1-2 | All Phase 1b reachable callers migrated | `web/src/api/*` deleted; legacy and route-data allowlists empty | No new domain behavior |
| 2 | Hard-cut `web_access`, `knowledge_graph`, and `retrieval` API/domain ownership. | Backend owner + FE mirror owner for graph/search | 3-5 | Phase 0 strict ban; Phase 1 FE typed adapter pattern | `aperag/domains/**` strict-ban clean; OpenAPI/hurl/breaking tables updated | No physical DB model split; no `knowledge_base` rewrite |
| 3 | Split `knowledge_base` and `indexing`; pilot DB model split. | Backend owner + FE document/collection owner | 5-8 | Phase 2 graph/retrieval boundaries stable | Alembic/import-cycle gates; document/collection hurl; index status contract | No conversation or control-plane cleanup |
| 4 | Control-plane domains: `identity`, `governance`, `model_platform`, `marketplace`. | Backend owner + FE owner | 6-9 | Phase 1 FE adapters; Phase 3 DB split lessons | Public/internal OpenAPI split; admin hurl; auth/quota/audit/config breaking tables | No agent runtime or chat refactor |
| 5 | `conversation`, `agent_runtime`, and `evaluation` cleanup. | Backend owner + FE owner | 5-8 | Phase 2/3 retrieval and indexing contracts stable | SSE/event/artifact compatibility or breaking table; chat/evaluation hurl | No shared infra rewrites unrelated to these domains |
| 6 | Final cleanup: shared FE components, neutral infra, stale shims, docs. | Architecture + owners | 4-7 | Phases 1-5 merged | Zero stale shims; docs and inventories current; full smoke green | No new product behavior |

## Active execution windows

| Task | Current work | Status | Review lane |
| --- | --- | --- | --- |
| #4 | Phase 1b FE legacy SDK batches. Current next batch: `collection` Option B. | In progress | FE contract, allowlists, typed fallback shape |
| #5 | Phase 2 PR-2a `web_access` backend/API hard-cut. | In progress | strict-ban imports, OpenAPI v2 path, hurl, MCP |
| #10 | Roadmap source-of-truth. | Doc-only review | roadmap completeness and accuracy |

## Phase 1b remaining FE rollout

The table is based on `main @ 6a89865f`. Counts must be rechecked after
each merge because allowlists are exact fixtures.

| Batch | Scope | Expected allowlist impact | Owner | Notes / gates |
| --- | --- | --- | --- | --- |
| `collection` Option B | `collection-form.tsx`, `collection-list.tsx`, `collection-header.tsx`, `export-dialog.tsx` | legacy `43 -> 39`, route-data `19 -> 18`, raw unchanged | FE owner | Extend existing collection typed-contract test. Defer `feature-visibility.ts` and `tools.ts` to document/indexing/retrieval. `ExportTaskResponse` empty body must throw. |
| `document` | document table/status/upload/url-import callers plus marketplace document subcomponents where appropriate | TBD | FE owner | Depends on typed document/indexing surface. Keep `DocumentIndex` ownership aligned with Phase 3 `indexing`. |
| `audit` | workspace/admin audit logs | no Phase 1b action | Governance owner later | Deferred to Phase 4 because `/api/v1/audit-logs` is hidden from public OpenAPI. Do not expose it publicly in Phase 1b. |
| `admin identity/governance` | users, quota settings, parser/config settings, app/auth layouts | TBD | FE + backend owner | Split by `identity` and `governance`; do not mix auth with quota/config. |
| `chat shell` | `components/chat/*`, `user-avatar.tsx` | TBD | FE + conversation owner | Requires conversation/chat ownership split. No wholesale component directory move before ownership inventory. |
| `low-level lib` | `web/src/lib/api/client.ts`, `web/src/lib/api/server.ts` | final route-data cleanup | FE owner | Delete or replace only when all app callers use feature adapters. |

## Phase 2 backend/domain rollout

| PR | Domain | Backend scope | FE scope | OpenAPI / hurl | DB scope | Non-goals |
| --- | --- | --- | --- | --- | --- | --- |
| 2a | `web_access` | Move `views/web.py`, `websearch/search`, `websearch/reader`, `websearch/utils`, and `Web*` schemas into `aperag/domains/web_access/**`. Mount routes under `/api/v2`. Update MCP URLs and schema imports. | No FE caller or feature adapter. OpenAPI schema regen may update `web/src/api-v2/schema.d.ts` only for web path rename. | Add authenticated deterministic `18_web_access_http.hurl`; add `phase2-web_access.md`; remove `/api/v1/web/*`. | none | No `knowledge_graph` or `retrieval`; no provider-dependent success path in deterministic hurl. |
| 2b | `knowledge_graph` | Move graph routes/schemas and graph domain shell under `aperag/domains/knowledge_graph/**` without violating strict-ban. DB-heavy service/model work may be bridged or deferred. | Create/update `features/graph/*`; migrate graph route callers. | Update `14_graph_http.hurl`; add breaking table; regen OpenAPI. | Logical owner only; physical `GraphCurationRun` / `GraphCurationSuggestion` split deferred unless DB split is explicitly in scope. | No retrieval or document/indexing work. |
| 2c | `retrieval` | Move collection-scoped search route shell/schemas/query planning under `aperag/domains/retrieval/**`; avoid importing legacy aggregates inside domains. | Create/update `features/search/*`; migrate search route callers. | Add retrieval hurl; add breaking table; regen OpenAPI. | Logical owner `SearchHistory`; physical split deferred. | Do not move `chats/{chat_id}/search`; that belongs to `conversation`. |

## Phase 3 knowledge base, indexing, and DB split

| Workstream | Scope | Dependencies | Gates |
| --- | --- | --- | --- |
| `knowledge_base` routes/services | Collection and document lifecycle routes move to canonical domain boundaries. | Phase 2 retrieval/graph no longer coupled to collection service internals. | Collection/document hurl green; FE `features/collection` and `features/document` updated. |
| `indexing` domain | `DocumentIndex`, rebuild/status state machine, scheduling contracts. | DB model ownership decision. | Index status projection contract; no direct `DocumentIndex` leakage into `knowledge_base`. |
| DB model split pilot | Move first domain SQLAlchemy models under `aperag/domains/<d>/models/` with neutral session/base in platform. | Strict import cycle audit; Alembic dry-run. | `Base.metadata.tables` diff reviewed; Alembic autogenerate reviewed; downgrade/destructive notes. |
| Async DB facade cleanup | Replace `AsyncDatabaseOps` facade usage with domain repositories/contracts. | DB split pilot. | No new `aperag.db.models` imports inside domains; repository contract tests. |

## Phase 4 control-plane rollout

| Domain | Scope | FE mirrors | Key breaking decisions |
| --- | --- | --- | --- |
| `identity` | Auth, user, invitation, API key ownership. | `features/identity`, existing `features/api-key` | Auth public/internal paths; admin user surface; `app-provider.tsx` auth debt. |
| `governance` | Quota, audit, settings, config. | `features/quota`, `features/audit`, `features/settings` | Split `/api/v1/quotas` into user/admin surfaces; decide audit public/internal OpenAPI. |
| `model_platform` | Providers, default models, prompt templates. | `features/providers`, `features/prompt` | Provider credential/admin semantics; prompt response component and enum debt. |
| `marketplace` | Collection marketplace ownership and subscriptions. | `features/marketplace` | Marketplace document/preview concrete response typing and knowledge-base contract. |

## Phase 5 conversation, agent runtime, and evaluation

| Domain | Scope | Gates |
| --- | --- | --- |
| `conversation` | Bots, chats, feedback, chat-scoped search, `components/chat/*` ownership split. | Chat hurl and typed-contract tests; no accidental `agent_runtime` ownership bleed. |
| `agent_runtime` | Keep top-level runtime canonical or explicitly document future domain path. | SSE/event/artifact shape stable or breaking table; compatibility tests green. |
| `evaluation` | Remove legacy v1 evaluation/question-set references after v2 coverage is complete. | Evaluation v2 hurl, FE feature tests, docs updated. |

## Phase 6 cleanup

| Area | Scope | Exit gate |
| --- | --- | --- |
| FE shared cleanup | Move `components/shared` / shell components only after ownership inventory. | No domain-specific API call in shared shell; imports documented. |
| Neutral infra | Move only true platform code into `aperag/platform` / `web/src/lib`. | No business rule in platform/lib. |
| Shim deletion | Delete temporary re-exports, stale adapters, old docs, and old fixtures. | `rg` for known legacy import/path patterns returns zero or documented exceptions. |
| Documentation | Update this roadmap, inventories, hurl matrix, and breaking tables. | Docs match merged code and CI gates. |

## Cross-phase dependency graph

| Dependency | Reason |
| --- | --- |
| Phase 1b collection before Phase 1c | Legacy SDK cannot be deleted until collection callers are migrated or explicitly deferred to later domain batches. |
| Phase 2 graph/search before Phase 3 knowledge_base split | Collection service currently carries graph/search coupling; Phase 3 needs those boundaries stable first. |
| DB split after at least one strict-ban domain lands | The first domain hard-cut validates import direction before SQLAlchemy model relocation. |
| Governance audit after public/internal OpenAPI decision | Audit is hidden from public spec today; Phase 1b must not expose it accidentally. |
| Conversation cleanup after retrieval split | Chat-scoped search must be intentionally separated from collection-scoped retrieval. |

## Roadmap maintenance

Update this file when a phase task is created, a phase is merged, or a
planned non-goal becomes in-scope. Each update should be doc-only unless
it is part of a phase PR whose runtime changes require the roadmap to
change in the same commit.
