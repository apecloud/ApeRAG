# Phase 7 Legacy Cleanup Inventory

**Baseline**: `origin/main @ 8946d867` (post-Phase 0→6 modularization + FE rewrite main branches merged).
**Owner**: 符炫炜 (主架构师) — maintains this file as SSoT; PM (@架构师) drives batch scheduling; coding lanes execute per batch.
**Scope**: identify backend + frontend hard-delete candidates remaining after Phase 0→6 modularization and the FE rewrite project.

Sibling docs: [`architecture.md`](./architecture.md) (post-Phase-6 canonical), [`../frontend-rewrite/mismatch-registry.md`](../frontend-rewrite/mismatch-registry.md) (FE ghost features + gates).

---

## 0. Aggregate impact

| Surface | Deletable now | Audit first | Must preserve |
|---------|--------------:|------------:|--------------:|
| Backend `aperag/service/` | 18 shim files / 455 LOC | 5 ACTIVE_LEGACY / 1 860 LOC | 3 PERMANENT / 1 058 LOC |
| Backend `aperag/views/` | 12 shim + 1 empty `__init__` / 327 LOC | 13 ACTIVE_LEGACY / 1 910 LOC | — |
| Backend `aperag/agent_runtime/` | 5 shim / 127 LOC | — | — |
| Backend `aperag/evaluation_v2/` | 7 shim / 125 LOC | — | — |
| Backend `aperag/db/models.py` re-export block | ~70 LOC in one file | retain local models | `Invitation` top-level `Role` import **must stay** |
| Backend `aperag/schema/view_models.py` re-export blocks | ~160 LOC across 7 blocks | — | local schemas stay |
| Backend misc dirs (`auth/`, `context/`, `graphindex/`, `graph_curation/`, `index/`, `query/`, `source/`, `concurrent_control/`, empty `chat/`, `graph/`, `misc/`) | 3 empty dirs (removable `__init__`) | all non-empty are ACTIVE shared infra (~7 900 LOC) | — |
| Frontend `web/src/api/` (old openapi-gen client) | 202 files / 24 531 LOC DEAD | — | — |
| Frontend `web/src/lib/api/{client,server}.ts` | 2 files / 105 LOC DEAD | — | `lib/api/typed/*` stays |
| Frontend `web/src/app/globals.default.css` | 123 LOC DEAD | — | — |
| Frontend `web/src/components/ui/*` unused | 6 files / 954 LOC UNUSED_EXPORT | — | active shadcn stays |
| Frontend feature API unused exports | 15 functions across 8 files | trim in-place | — |
| Frontend `package.json` | 7 deps removable (`react-pdf`, `@dnd-kit/*` ×4, `tw-animate-css`, `@stepperize/react`) | — | — |

Back-end quick numbers: **43 files / ~1 034 LOC** deletable in shim/unused batches; **18 files / 3 770 LOC** need audit; **3 files / 1 058 LOC** PERMANENT.
Front-end quick numbers: **211 files / ~25 713 LOC** DEAD or SUPERSEDED at zero product risk.

---

## 1. PERMANENT — must NOT delete

### 1.1 Backend standalone-infra seams (cross-domain Protocol + DI)

| Path | LOC | Reason |
|------|----:|--------|
| `aperag/service/quota_service.py` | 392 | Cross-domain quota Protocol (`app.py` lines 101–112); 4 consumers (KB / conversation / identity / agent_runtime) |
| `aperag/service/prompt_template_service.py` | 625 | Cross-domain prompt Protocol (`app.py` lines 114–145); consumers in agent_runtime + conversation |
| `aperag/service/search_pipeline_service.py` | 41 | Retrieval search pipeline re-export anchor; test fixtures monkeypatch this path |

These map 1:1 to `architecture.md` Section F18 "2 permanent CRITICAL_WIRINGS + 1 standalone-infra".

### 1.2 Backend `Invitation` class-body Role binding

`aperag/db/models.py` line 36:
```python
from aperag.domains.identity.db.models import Role   # keep — Invitation class body depends on this at load time
```
`Invitation.role = Column(EnumColumn(Role), ...)` executes at import time; G15 (non-identity domains forbid Role import) is explicitly exempt here. Do not move `Invitation` without moving the `Role` import.

### 1.3 Backend G17 / G18-alt CRITICAL_WIRINGS

`tests/unit_test/test_modularization_boundaries.py` G17 (Phase 3+4: 7 entries) + G18-alt (Phase 5: 2 entries) assert runtime smoke of DI wire-up. Any shim deletion must still leave these 9 wirings intact; see `app.py` lines 101–145.

### 1.4 Backend misc shared-infra dirs (keep, not shim)

All non-empty under `aperag/`:
- `aperag/auth/authentication.py` — fastapi-users backend config (272 LOC)
- `aperag/concurrent_control/` — Redis + threading lock manager (831 LOC, 36 import sites)
- `aperag/context/` — request context (167 LOC)
- `aperag/graphindex/` — KG indexing infra (1 976 LOC, 36 sites)
- `aperag/graph_curation/` — graph curation helper (1 057 LOC, 14 sites)
- `aperag/index/` — multi-index manager (3 591 LOC, 5 sites)
- `aperag/query/` — query builder (51 LOC, 9 sites)
- `aperag/source/` — document source plugins (186 LOC, 9 sites)

These are shared infra, not shims; they stay. Future domain absorption is a Phase 8+ candidate.

---

## 2. Backend hard-delete candidates — batched

### Batch B1 — `aperag/service/` pure re-export shims (18 files, 455 LOC)

All files are single-line `from aperag.domains.<d>.service.<s> import *` (or narrow equivalents).

```
api_key_service.py            → domains.governance.service.api_key_service
audit_service.py              → domains.governance.service.audit_service
bot_service.py                → domains.conversation.service.bot_service
chat_collection_service.py    → domains.conversation.service.chat_collection_service
chat_document_service.py      → domains.conversation.service.chat_document_service
chat_service.py               → domains.conversation.service.chat_service
chat_title_service.py         → domains.conversation.service.chat_title_service
collection_service.py         → domains.knowledge_base.service.collection_service
collection_summary_service.py → domains.knowledge_base.service.collection_summary_service
default_model_service.py      → domains.model_platform.service.default_model_service
document_service.py           → domains.knowledge_base.service.document_service
graph_service.py              → domains.knowledge_graph.service
llm_available_model_service.py → domains.model_platform.service.llm_available_model_service
llm_provider_service.py       → domains.model_platform.service.llm_provider_service
marketplace_collection_service.py → domains.marketplace.service.marketplace_collection_service
marketplace_service.py        → domains.marketplace.service.marketplace_service
prompt_template_service.py    — PERMANENT (see §1.1)
quota_service.py              — PERMANENT (see §1.1)
search_pipeline_service.py    — PERMANENT (see §1.1)
turn_feedback_service.py      → domains.conversation.service.turn_feedback_service
```

Delete gate: `grep -R "from aperag.service\.\(api_key_service\|audit_service\|…\)" aperag tests` returns zero hits after consumer migration; then rm the file + rerun G1-G19.

### Batch B2 — `aperag/views/` pure router re-export shims (12 files, 313 LOC) + 1 empty `__init__` (14 LOC)

```
agent_runtime.py   → domains.agent_runtime.api.routes.router
api_key.py         → domains.governance.api.routes (router not mounted from shim)
audit.py           → domains.governance.api.routes
bots_v2.py         → domains.conversation.api.routes.bots_router
chat.py            → domains.conversation.api.routes.chat_router
collections_v2.py  → domains.knowledge_base.api.routes
documents_v2.py    → domains.knowledge_base.api.routes
evaluation_v2.py   → domains.evaluation.api.routes
llm.py             → domains.model_platform.api.llm_routes
marketplace.py     → domains.marketplace.api.routes
marketplace_collections.py → domains.marketplace.api.routes
providers_v2.py    → domains.model_platform.api.providers_v2_routes
__init__.py        empty (14 LOC)
```

Delete-safety: `aperag/app.py` already mounts the canonical domain routers directly (lines 227–251). Shim views are imported by no live mount; they only linger in tests / legacy docs.

Delete gate: same grep pattern on `aperag.views.*`, then rm + G1-G19.

### Batch B3 — `aperag/agent_runtime/` (5 files, 127 LOC)

All 5 files are Phase 5 Step 5-S5b shim re-exports pointing at `aperag.domains.agent_runtime.*`. Known callers:
- `aperag.evaluation_v2.worker` (also a shim, deleted in B4)
- `aperag.views.agent_runtime` (shim, deleted in B2)
- a handful of test modules

Delete gate: after B2/B4 merges + test imports updated, rm entire `aperag/agent_runtime/`; G17 / G18-alt CRITICAL_WIRINGS still pass because wire-up already uses `aperag.domains.agent_runtime.*`.

### Batch B4 — `aperag/evaluation_v2/` (7 files, 125 LOC)

Same profile as B3; re-exports `aperag.domains.evaluation.*`. Callers: `aperag.views.evaluation_v2` (B2 shim), Celery task registry, tests.

### Batch B5 — `aperag/db/models.py` re-export block (~70 LOC in-file)

Strip lines 395–462 re-export block. Retain:
- Local models: `ConfigModel`, `UserQuota`, `ModelServiceProvider`, `QuestionType`, `EvaluationStatus`, `EvaluationItemStatus`, `ExportTaskStatus`, `QuestionSet`, `Question`, `Evaluation`, `EvaluationItem`, `Setting`, `ExportTask`, `PromptTemplate`, `Invitation`
- `Invitation` + top-level `Role` import (§1.2) — unchanged
- `Base` (Alembic metadata anchor) — unchanged

Delete gate: all `from aperag.db.models import <Bot | Chat | User | ApiKey | LLMProvider | Collection | Document | …>` re-routed to `from aperag.domains.<d>.db.models import …`; Alembic autogen diff must be empty.

### Batch B6 — `aperag/schema/view_models.py` dual-hook blocks

7 blocks (lines ~700–889): knowledge_base / identity / governance / marketplace / model_platform / conversation / agent_runtime re-exports via `sys.modules.get(...)` string lookup to avoid G1 AST scan. Delete in this order (lowest consumer count first):

| Block | Domain | Schemas | Approx consumer sites |
|-------|--------|---------|----------------------:|
| 7 | agent_runtime | 1 | low |
| 4 | marketplace | 3 | low |
| 3 | governance | 6 | low |
| 2 | identity | 12 | medium |
| 5 | model_platform | 20+ | medium |
| 1 | knowledge_base | 9 | medium |
| 6 | conversation | 21 | high (Bot/Chat/User/ApiKey) |

Each sub-batch: migrate callers to `aperag.domains.<d>.schemas`, verify `X is aperag.schema.view_models.X is aperag.domains.<d>.schemas.X` still holds (dual-hook invariant), then rm that block.

Retain local schemas (`FailResponse`, `Settings`, `ParserHealthItem`, `PromptDetail`, etc.).

### Batch B7 — empty dirs under `aperag/`

`aperag/chat/`, `aperag/graph/`, `aperag/misc/` contain 0 meaningful content (empty package markers). Remove.

---

## 3. Backend ACTIVE_LEGACY — audit before delete (18 files, 3 770 LOC)

### 3.1 Services with real logic (5 files, 1 860 LOC)

| File | LOC | Audit question |
|------|----:|----------------|
| `chat_completion_service.py` | 468 | Backs `/v1/chat/completions` OpenAI-compat endpoint. Does conversation domain fully own this, or does this remain a standalone streaming adapter? |
| `evaluation_service.py` | 514 | Celery task lifecycle + polling. Compare with `aperag.domains.evaluation.services`; determine which layer owns Celery vs HTTP. |
| `export_service.py` | 199 | Export task CSV pipeline. Candidate to move under `domains.knowledge_base.service.export_service`. |
| `question_set_service.py` | 197 | Evaluation question-set CRUD. Candidate move to `domains.evaluation`. |
| `setting_service.py` | 86 | User settings. Candidate move to `domains.governance` (ties with quota/audit) or split to identity. |
| `test_mcp_agent.py` | 332 | Mixed test utility + live service. Likely belongs under `tests/` fixtures; confirm and relocate. |

### 3.2 Views with real HTTP handlers (13 files, 1 910 LOC)

| File | LOC | Mount | Audit question |
|------|----:|-------|----------------|
| `auth.py` | 500 | `/api/v1/auth` | fastapi-users + OAuth (GitHub/Google/Authing/Logto). Identity domain has routes for basic auth; decide whether OAuth migration is in Phase 7 or stays here. |
| `collections.py` | 108 | `/api/v1/collections` | Upload / staged / fetch-url flow. Domain `/api/v2/collections/…/documents/upload` already exists (knowledge_base domain). Confirm v2 coverage vs intentionally retained v1. |
| `config.py` | 55 | `/api/v1/config` | `auth-config` endpoint. Identity/governance candidate. |
| `export.py` | 69 | `/api/v1/export` | 3 routes paired with `export_service` (§3.1). |
| `graph.py` | 155 | `/api/v1/graph` | Graph curation suggestions + status. Overlaps with `domains.knowledge_graph.api.routes`. |
| `main.py` | 219 | `/api/v1/bots, /chats, /feedbacks` | Bot / chat / feedback CRUD. Conversation domain has v2 equivalents; decide if v1 is a client contract to keep. |
| `openai.py` | 83 | `/v1/chat/completions` | OpenAI-compat entry. Depends on `chat_completion_service`. |
| `prompts.py` | 205 | `/api/v1/user-prompts` | Prompt customization + mineru token. Depends on PERMANENT `prompt_template_service`. |
| `quota.py` | 234 | `/api/v1/quota` | Quota admin routes. Uses PERMANENT `quota_service`. |
| `settings.py` | 67 | `/api/v1/settings` | Parser health + update settings. Uses `setting_service` (§3.1). |
| `test.py` | 60 | dev-only | reset-demo + token probe. |
| `utils.py` | 134 | helper | OAuth method detection — imported by `auth.py` + `config.py`. |
| `chat_documents.py` | 21 | not mounted | Looks like a stub, confirm unused and delete. |

Audit rule: for each route, either (a) confirm a `/api/v2/` (or other) canonical equivalent already exists and the FE has migrated → remove the v1 route, or (b) mark as **keep-v1** in this inventory with rationale (client contract stability, OpenAI-compat, etc.).

### 3.3 Test files referencing legacy paths

- OpenAPI-contract tests hitting `aperag.views.<shim>` — keep until B2 merges, then point at domain routers or delete.
- Tests using `monkeypatch.setattr("aperag.service.…", …)` — re-target to `aperag.domains.<d>.service…` before B1 delete.

---

## 4. Frontend hard-delete candidates

### 4.1 Batch F1 — `web/src/api/` old openapi-gen client (202 files, 24 531 LOC)

Pre-`api-v2` OpenAPI generated client. All callers have migrated to `web/src/api-v2/schema.d.ts` + `openapi-fetch`. Remaining grep hits only inside `web/src/lib/api/{client,server}.ts` which are themselves dead (F2).

Delete gate: one grep for `from '@/api/` or `from 'src/api/` across `web/src/` returns zero hits outside F1/F2.

### 4.2 Batch F2 — `web/src/lib/api/client.ts` + `server.ts` (2 files, 105 LOC)

Old wrappers. `web/src/lib/api/typed/{browser,server,errors,index}.ts` is the active replacement (openapi-fetch-based). Delete the two old files; keep typed/*.

### 4.3 Batch F3 — `web/src/app/globals.default.css` (1 file, 123 LOC)

Dead pre-rewrite backup of oklch tokens before the amber/Manrope migration. Not imported.

### 4.4 Batch F4 — unused shadcn UI components (6 files, 954 LOC)

Zero imports across `web/src/`:
- `components/ui/menubar.tsx` (276)
- `components/ui/combobox.tsx` (274)
- `components/ui/navigation-menu.tsx` (168)
- `components/ui/pagination.tsx` (127)
- `components/ui/resizable.tsx` (56)
- `components/ui/avatar.tsx` (53)

Delete gate: confirm not re-exported by `components/ui/index.ts` (if present), confirm `components.json` doesn't list as required, then rm.

### 4.5 Batch F5 — unused feature exports (15 symbols across 8 files)

Trim in-place, do not delete files:
- `features/admin/client-api.ts`: `listUserQuotas`
- `features/bot/client-api.ts`: `toTitleLanguage`, `TitleGenerateInput`, `buildTitleGenerateRequest`, `updateBot`, `deleteBot`
- `features/collection/client-api.ts`: `triggerCollectionSummary`
- `features/document/client-api.ts`: `ListDocumentsOptions`, `deleteDocuments`, `buildDocumentDownloadUrl`
- `features/document/server-api.ts`: `ListDocumentsServerOptions`
- `features/evaluation/client-api.ts`: `updateEvaluationDataset`, `updateEvaluationDatasetItem`
- `features/evaluation/server-api.ts`: `listEvaluationRunItemAttempts`
- `features/knowledge-graph/client-api.ts`: `mergeGraphNodes`

### 4.6 Batch F6 — `web/package.json` unused dependencies (7 packages)

Zero `import` sites across `web/src/`:
- `react-pdf ^10.1.0`
- `@dnd-kit/core ^6.3.1`
- `@dnd-kit/modifiers ^9.0.0`
- `@dnd-kit/sortable ^10.0.0`
- `@dnd-kit/utilities ^3.2.2`
- `tw-animate-css ^1.3.6`
- `@stepperize/react ^5.1.6`

Remove from `package.json`, `yarn install`, confirm `yarn build` passes.

### 4.7 Frontend style residue (no-delete, refactor in-place)

- 3 inline `var(--primary)` sites: `quota-radial-chart.tsx`, `app/layout.tsx`, `documents-table.tsx` — align with the amber token set (keep `var(--primary)` since it now resolves to `#C96442`, but audit that each is the intended brand accent vs a leftover blue expectation).
- `--chart-1`…`--chart-5` CSS vars defined in `globals.css` but no TS/TSX references — remove or wire into `ENTITY_PALETTE`/recharts usages; leave for F-style follow-up.

### 4.8 Do-not-delete frontend

- `web/src/api-v2/schema.d.ts` (11 313 LOC, auto-generated) — active.
- `web/src/lib/api/typed/*` — active.
- All chat components (`chat/*`) — active post-L2 rewrite.
- All i18n namespaces — every `page_*.json` maps to a live route; no orphans.
- `web/src/app/docs/*` — active MDX renderer (unless product decides to remove `/docs` route; treat as product decision, not cleanup).
- Framework-required files (`middleware.ts`, `mdx-components.tsx`, services/cookies, hooks/use-mobile, middlewares/apiProxy).

---

## 5. Gates

Every Phase 7 delete PR must pass all of:

```bash
# Backend boundary gates
uv run pytest tests/unit_test/test_modularization_boundaries.py -x -q

# OpenAPI byte stability
uv run python scripts/export_openapi.py --check

# Python lint
uv run ruff check aperag/ tests/

# Frontend lint (when touching web/)
cd web && yarn lint
```

Also recommended per batch:
- Full `uv run pytest tests/` for B5/B6 (risk high — import identity changes).
- Manual `yarn dev` smoke for F1/F2 (no visual regression expected since only removing unused code).
- `grep -R "from aperag\.service\." aperag tests` / `grep -R "from aperag\.schema\.view_models" aperag tests` / `grep -R "from '@/api/" web/src` returning zero before rm.

---

## 6. Batch → lane proposal (PM input)

Delivery model inherits the successful pattern from the FE rewrite: fast-merge discipline + 最小 review + Weston blocker-level CR 补位 + 10 min PM cadence.

Owner preference (per lane SME + msg=54d910eb / msg=be8d6f21 prep):

| Batch | Scope | Recommended owner | Notes |
|-------|-------|-------------------|-------|
| B1 | `aperag/service/` 18 shim delete + caller migration | Opus lane (Bryce or chenyexuan split by domain) | Can split Phase 3 vs Phase 4/5 sub-batches |
| B2 | `aperag/views/` 12 shim delete + `__init__` | Opus lane (Bryce for Phase 4 views, chenyexuan for Phase 5 views) | Low risk; routers already direct-mount |
| B3 | `aperag/agent_runtime/*` 5 shim delete | chenyexuan (Phase 5 SME) | After B2 agent_runtime shim is gone |
| B4 | `aperag/evaluation_v2/*` 7 shim delete | chenyexuan (Phase 5/6 SME) | Touch Celery task imports |
| B5 | `aperag/db/models.py` 53 re-export cleanup (preserve `Invitation` + `Role` import) | Bryce (Phase 4 SME) | Highest DB risk; verify Alembic autogen clean |
| B6 | `aperag/schema/view_models.py` 7 sub-batches (low→high consumer count) | split by domain SME (Bryce Phase 4 blocks, chenyexuan Phase 5 block, cuiwenbo Phase 3 block) | dual-hook identity invariant; run full pytest |
| B7 | empty `aperag/{chat,graph,misc}/` removal | any lane | trivial |
| B-audit | ACTIVE_LEGACY §3 audit + per-route decision | Bryce + chenyexuan + cuiwenbo by Phase | Output: rm routes with domain equivalents; keep-v1 list with rationale |
| F1 | `web/src/api/` 202-file delete | huangheng (FE L1 SME) | zero code risk once F2 also removed |
| F2 | `web/src/lib/api/{client,server}.ts` delete | huangheng | paired with F1 |
| F3 | `globals.default.css` delete | huangheng | trivial |
| F4 | unused shadcn ui delete | huangheng | verify `components.json` + `components/ui/index.ts` |
| F5 | trim 15 unused feature exports | any FE lane | low risk |
| F6 | trim 7 unused deps | huangheng + run `yarn install` | CI build confirms |

GPT lanes (dongdong / weihong) absorb overflow or `B-audit` leg work once Opus lanes are saturated.

---

## 7. Execution discipline

1. **Canonical SSoT = this file.** Every batch PR references it in commit body (`per docs/modularization/cleanup-inventory.md §Bx`).
2. **PERMANENT list (§1) is strictly off-limits.** Every coding lane self-checks the file being deleted is not in §1.
3. **Every batch PR ends with `Ghost-check:` line** reusing the FE rewrite convention (`docs/frontend-rewrite/mismatch-registry.md` §6). Example:
   ```
   Ghost-check: none — batch B1 sub-batch 2; no PERMANENT §1.1 files touched; G17/G18-alt unchanged.
   ```
4. **10 min PM cadence.** `@架构师` drives checkpoints; coding lanes post status + gate results per batch thread.
5. **Weston blocker-level minimal CR** continues; Opus 可 restored → prefer Opus CR for high-risk batches (B5/B6), Weston still covers when Opus 限流.
6. **Fix-forward discipline.** Small style/residue cleanup (F4.7 chart vars, inline `var(--primary)` spot-check) 作为 follow-up tiny patches，不阻塞主批次。
7. **Roll-back guarantee.** Every batch PR is small (≤ ~500 LOC diff, often much smaller) + squash-merged with commit body linking to this SSoT; revert path is trivial if a regression surfaces.
