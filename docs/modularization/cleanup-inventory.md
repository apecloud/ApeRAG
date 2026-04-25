# Phase 7 Legacy Cleanup Inventory (v1 — hard-cut)

**Baseline**: `origin/main @ 97aa4bf8` (post-Phase 0→6 modularization + FE rewrite main branches merged + v0 inventory).
**Owner**: 符炫炜 (主架构师) — canonical + 禁删白名单 + inventory SSoT；PM (@架构师) drives cadence + task split + owner；@Weston blocker-level minimal CR；coding lanes 优先 Opus。
**Scope**: 一次性 hard-cut 前后端老代码 → 新架构 (Phase 0→6 domains + FE rewrite tokens)。允许 breaking changes，验收是 deployable end-state。

Sibling docs: [`architecture.md`](./architecture.md) (post-Phase-6 canonical); [`../frontend-rewrite/mismatch-registry.md`](../frontend-rewrite/mismatch-registry.md) (FE ghost features + gates).

---

## 0. 口径变更（v0 → v1）

per earayu2 msg=78fdb6fc / msg=be4c5c19 / msg=ac0a9243:

- **破坏性 hard-cut**：默认删全部 shim / 老入口 / 兼容层；**不再保留兼容双轨**。
- **大 PR**：目标压成 **1-2 个 PR**（默认：`PR-BE destructive backend cleanup` + `PR-FE destructive frontend cleanup`），不再按 batch-by-batch 小迭代。
- **验收标准** = 主干可部署 + 可测试 + 可运行。盖过以前的 "保守兼容 + 慢慢迁" 策略。
- **禁删白名单仍严格保护**（§1）— hard-cut 不是乱删，是 surgical。
- **每候选文件 3 类**：`直接删` / `迁完再删` / `白名单禁删`（v0 里只分 SHIM/ACTIVE_LEGACY/PERMANENT，v1 重新归类到这 3 档）。

---

## 1. 禁删白名单（3 层 distinction — per Bryce msg=647b2b17 canonical catch + PM msg=c179775c lock）

**规则**：coding lane 动任何 `aperag/service/*.py` 或 `aperag/db/models.py` Invitation 上下文 之前，必须先对这个清单 self-check。

### 1.1 Layer A — G18 alt CRITICAL_WIRINGS Protocol + DI seams（2 条，live runtime wire-up）

| Path | Consumer wire-up | 删了就炸 |
|------|------------------|---------|
| `aperag/service/quota_service.py` | `conversation.bot_service._quota_ops` (`app.py` lines 101–112) | `test_phase5_di_critical_wirings_at_app_startup` G18 alt |
| `aperag/service/prompt_template_service.py` | `agent_runtime.runtime._prompt_template_ops` (`app.py` lines 114–145) | 同上 |

### 1.2 Layer B — Cross-domain standalone-infra（1 条，非 G18 alt 但仍禁删）

| Path | 为什么禁删 | 删了就炸 |
|------|------------|---------|
| `aperag/service/search_pipeline_service.py` | retrieval / KB / indexing 多域消费；**非 re-export shim**，本身就是 implementation；test fixtures monkeypatch 直接挂它 | `tests/unit_test/test_es_p0_contract.py:10` `from aperag.service.search_pipeline_service import SearchPipelineService` + 若干 monkeypatch |

**重要**：Layer A + Layer B 加起来 = **3 个 `aperag/service/*.py` permanent files**。任何 coding lane 按 "grep shim re-export 清零" 扫时，看到 Layer B 这个文件有 `class` 实现，**不许错判成 "残留老 service 需迁到 retrieval domain"** — 它是 canonical 保留为 standalone 的 cross-domain infra，与 G17 Phase 4 identity `UserInitOps` 性质一致。

### 1.3 Layer C — Special-case load-time binding

| Path | 为什么禁删 |
|------|------------|
| `aperag/db/models.py` line 36 `from aperag.domains.identity.db.models import Role` | `Invitation` class body line 211 `role = Column(EnumColumn(Role), ...)` 在 import-time 执行；G15 `Role` import ban 对 `aperag/db/models.py` 不适用 — canonical exempt |

### 1.4 Layer D — G17 Phase 3+4 CRITICAL_WIRINGS

7 条 Phase 3+4 DI wirings（identity 3 + KB 4），`test_phase3_p4_di_critical_wirings_at_app_startup` 保障。删掉任何 wire-up 在 `app.py` 内都会炸。具体入口见 `architecture.md` Section F17。

### 1.5 Layer E — `aperag/` 非 shim shared infra 目录（keep 全部，不属 Phase 7 cleanup）

`aperag/auth/` `aperag/concurrent_control/` `aperag/context/` `aperag/graphindex/` `aperag/graph_curation/` `aperag/index/` `aperag/query/` `aperag/source/` — 这些是 cross-domain shared infra（非 canonical domain code，非 shim），共 ~7 900 LOC；Phase 8+ domain absorption candidate，**本轮不动**。

---

## 2. 前端 frontend destructive cleanup（`PR-FE`）

**Owner**：huangheng (FE L1 SME) 优先（per msg=8e203455 standby + msg=8a4fcf27 Opus 优先）。
**Scope**：211 files / ~25 713 LOC DEAD/SUPERSEDED，一次性删。零产品风险。

### 2.1 直接删（bundle 进 PR-FE）

| Group | 数量 | LOC | 说明 |
|-------|-----:|----:|------|
| `web/src/api/*` 整个目录 | 202 files | 24 531 | 老 openapi-gen client，被 `web/src/api-v2/` 完全替代；消费者已迁 |
| `web/src/api/openapi.merged.yaml` | 1 | 0 | 生成 spec，无消费 |
| `web/src/lib/api/client.ts` + `server.ts` | 2 | 105 | 老 wrapper；`lib/api/typed/*` 是 active 替代 |
| `web/src/app/globals.default.css` | 1 | 123 | pre-rewrite oklch 备份；无 import |
| `web/src/components/ui/menubar.tsx` | 1 | 276 | 0 callers |
| `web/src/components/ui/combobox.tsx` | 1 | 274 | 0 callers |
| `web/src/components/ui/navigation-menu.tsx` | 1 | 168 | 0 callers |
| `web/src/components/ui/pagination.tsx` | 1 | 127 | 0 callers |
| `web/src/components/ui/resizable.tsx` | 1 | 56 | 0 callers |
| `web/src/components/ui/avatar.tsx` | 1 | 53 | 0 callers |
| `package.json` 删 7 deps | — | — | `react-pdf` + `@dnd-kit/{core,modifiers,sortable,utilities}` + `tw-animate-css` + `@stepperize/react` |

### 2.2 trim in-place（同 PR-FE commit）

15 unused feature exports across 8 files — 保留文件本身，只删函数签名/类型：
- `features/admin/client-api.ts`: `listUserQuotas`
- `features/bot/client-api.ts`: `toTitleLanguage`, `TitleGenerateInput`, `buildTitleGenerateRequest`, `updateBot`, `deleteBot`
- `features/collection/client-api.ts`: `triggerCollectionSummary`
- `features/document/client-api.ts`: `ListDocumentsOptions`, `deleteDocuments`, `buildDocumentDownloadUrl`
- `features/document/server-api.ts`: `ListDocumentsServerOptions`
- `features/evaluation/client-api.ts`: `updateEvaluationDataset`, `updateEvaluationDatasetItem`
- `features/evaluation/server-api.ts`: `listEvaluationRunItemAttempts`
- `features/knowledge-graph/client-api.ts`: `mergeGraphNodes`

### 2.3 FE 白名单（禁删）

- `web/src/api-v2/schema.d.ts` (auto-generated)，`web/src/lib/api/typed/*` active；
- `web/src/lib/design-tokens.ts` — 所有 exports（`COLORS` / `ENTITY_PALETTE` / `ENTITY_LABELS_*` / `RADIUS` / `SHADOW` / `FONTS` / `LAYOUT` / `CANVAS_DARK` / `entityTypeToPaletteKey` / `EntityType`）即便某些 TS export 暂无 callsite，**仍保留作为 reserved canonical**（类似 BE search_pipeline_service 的 "当前 grep unused 但设计 reserved"）；
- 所有 chat / graph / workspace 新视觉 components；
- 所有 i18n namespace (27 × 2 locales)；
- Framework-required (`middleware.ts` / `mdx-components.tsx` / `hooks/use-mobile` / `services/cookies` / `middlewares/apiProxy`).

### 2.4 FE 可选收尾（非阻塞，可 PR-FE 内含或 follow-up）

- 3 处 inline `var(--primary)` 仍然 valid（`--primary` 已指向 `#C96442`）— 确认即可，不改；
- `--chart-1`…`--chart-5` CSS vars 无 TS/TSX 引用但定义保留 — reserved for future chart；
- `web/src/app/docs/*` 保留（MDX renderer active）。

### 2.5 FE PR gate

```bash
cd web && yarn install   # package.json 删 7 deps 后
cd web && yarn lint      # ESLint 0 errors
cd web && yarn build     # production build 过
grep -R "from '@/api/" web/src                  # → 0 hits
grep -R "from '@/lib/api/client'" web/src       # → 0 hits
grep -R "from '@/lib/api/server'" web/src       # → 0 hits
```

---

## 3. 后端 backend destructive cleanup（`PR-BE`）

**Owner**：Bryce (Phase 4 SME — identity / governance / model_platform / marketplace + db.models + schema/view_models) + chenyexuan (Phase 5/6 SME — conversation / agent_runtime / evaluation) + cuiwenbo (Phase 3 SME — KB / indexing / retrieval / knowledge_graph) 协作 on 1 PR。
**PM** (@架构师) 决定主 author + co-author 分工。
**Scope**：43 直接删 shim + 3 770 LOC ACTIVE_LEGACY audit → hard-cut，一次性合进 main。

### 3.1 直接删（bundle 进 PR-BE）

**A. `aperag/service/` 18 pure re-export shims** — 455 LOC
```
api_key_service.py        → rm
audit_service.py          → rm
bot_service.py            → rm
chat_collection_service.py → rm
chat_document_service.py  → rm
chat_service.py           → rm
chat_title_service.py     → rm
collection_service.py     → rm
collection_summary_service.py → rm
default_model_service.py  → rm
document_service.py       → rm
graph_service.py          → rm
llm_available_model_service.py → rm
llm_provider_service.py   → rm
marketplace_collection_service.py → rm
marketplace_service.py    → rm
prompt_template_service.py    — KEEP (Layer A)
quota_service.py              — KEEP (Layer A)
search_pipeline_service.py    — KEEP (Layer B)
turn_feedback_service.py  → rm
```
同批 rewrite consumer imports: `from aperag.service.<name>` → `from aperag.domains.<d>.service.<name>`.

**B. `aperag/views/` 12 pure router re-export shims + 1 empty `__init__`** — 327 LOC
```
agent_runtime.py / api_key.py / audit.py / bots_v2.py / chat.py /
collections_v2.py / documents_v2.py / evaluation_v2.py / llm.py /
marketplace.py / marketplace_collections.py / providers_v2.py / __init__.py → rm
```
`app.py` 已经直接 mount domain routers (lines 227–251)；shim 是死的。

**C. `aperag/agent_runtime/` 整个包 5 files / 127 LOC** → rm
- `__init__.py` / `runtime.py` / `schemas.py` / `services.py` / `storage.py`
- 同批 rewrite callers 到 `aperag.domains.agent_runtime.*`

**D. `aperag/evaluation_v2/` 整个包 7 files / 125 LOC** → rm
- 同批 rewrite Celery task imports + test fixtures 到 `aperag.domains.evaluation.*`

**E. `aperag/db/models.py` re-export block** (lines 395–462) → strip
- 保留：`ConfigModel` / `UserQuota` / `ModelServiceProvider` / `QuestionType` / `EvaluationStatus` / `EvaluationItemStatus` / `ExportTaskStatus` / `QuestionSet` / `Question` / `Evaluation` / `EvaluationItem` / `Setting` / `ExportTask` / `PromptTemplate` / `Invitation` / `Base`
- 保留：line 36 `from aperag.domains.identity.db.models import Role`（Layer C special-case）
- 同批 rewrite callers: `from aperag.db.models import <Bot|Chat|User|ApiKey|LLMProvider|Collection|Document|…>` → `from aperag.domains.<d>.db.models import …`

**F. `aperag/schema/view_models.py` 7 dual-hook blocks** (lines ~700–889) → strip
- 同批 rewrite callers 到 `aperag.domains.<d>.schemas`
- 保留 local schemas: `FailResponse` / `Settings` / `ParserHealthItem` / `PromptDetail` / `UserPromptsResponse` / `Prompts` / `UpdateUserPromptsRequest` 等 + `PromptSupportTier` 等本地 enum
- dual-hook identity invariant (`X is view_models.X is domain.schemas.X`) 在 rewrite 完成后自然失效 — 接受这个破坏（hard-cut 口径），只保证 callers 全切完

**G. 空目录 `aperag/chat/` + `aperag/graph/` + `aperag/misc/`** → rm

### 3.2 迁完再删（bundle 进 PR-BE — hard-cut ACTIVE_LEGACY decisions）

对每个 ACTIVE_LEGACY 文件，以下为 canonical 迁移决策。PM 可以在 PR-BE 内拆 co-author 分工，但结论一次性落。

| File | LOC | canonical decision |
|------|----:|--------------------|
| `aperag/service/chat_completion_service.py` | 468 | **迁到** `aperag/domains/conversation/service/chat_completion_service.py`（OpenAI-compat 归 conversation 域），然后 rm 老 path + rewire `aperag/views/openai.py` |
| `aperag/service/evaluation_service.py` | 514 | **迁到** `aperag/domains/evaluation/service/evaluation_task_service.py`（Celery task lifecycle 归 evaluation 域），然后 rm |
| `aperag/service/export_service.py` | 199 | **迁到** `aperag/domains/knowledge_base/service/export_service.py`，然后 rm |
| `aperag/service/question_set_service.py` | 197 | **迁到** `aperag/domains/evaluation/service/question_set_service.py`，然后 rm |
| `aperag/service/setting_service.py` | 86 | **迁到** `aperag/domains/governance/service/setting_service.py`（与 quota / audit 同域），然后 rm |
| `aperag/service/test_mcp_agent.py` | 332 | **迁到** `tests/fixtures/mcp_agent.py`（本来是 test util），然后 rm |
| `aperag/views/auth.py` | 500 | **迁到** `aperag/domains/identity/api/auth_routes.py`（OAuth + fastapi-users 归 identity 域的新 sub-router），挂 `/api/v2/auth`；rm 老 view |
| `aperag/views/collections.py` (v1 upload) | 108 | v2 已覆盖 (`POST /api/v2/collections/{id}/documents/upload`)；**直接 rm**，FE 确认没再消费 v1 |
| `aperag/views/config.py` | 55 | **迁到** `aperag/domains/identity/api/config_routes.py`，挂 `/api/v2/config`；rm 老 view |
| `aperag/views/export.py` | 69 | **迁到** `aperag/domains/knowledge_base/api/export_routes.py`，挂 `/api/v1/export` 保留 |
| `aperag/views/graph.py` | 155 | **迁到** `aperag/domains/knowledge_graph/api/curation_routes.py`，挂 `/api/v1/graph` 保留 |
| `aperag/views/main.py` | 219 | bot / chat / feedback v1 CRUD — v2 已覆盖 (`/api/v2/bots`, `/api/v2/chats`, `/api/v2/feedback`)；**直接 rm** 老路径；FE 若消费 v1 需要同批迁到 v2 |
| `aperag/views/openai.py` | 83 | **迁到** `aperag/domains/conversation/api/openai_routes.py`（OpenAI-compat 归 conversation），挂 `/v1/chat/completions` 保留 |
| `aperag/views/prompts.py` | 205 | **迁到** `aperag/domains/model_platform/api/prompt_routes.py`，挂 `/api/v1/user-prompts` 保留 |
| ~~`aperag/views/quota.py`~~ | ~~234~~ | ✅ Phase 8 #66 (G5b) **DONE** — carved to `aperag/domains/governance/api/quota_routes.py`, restored the previously unmounted quota/system-default contract at `/api/v2/quotas*` + `/api/v2/system/default-quotas`, and deleted the legacy view. |
| ~~`aperag/views/settings.py`~~ | ~~67~~ | ✅ Phase 8 #48 (G2) **DONE** — carved to `aperag/domains/knowledge_base/api/settings_routes.py`, hard-cut prefix `/api/v1/settings` → `/api/v2/settings` per D7-2 (msg=94f663f2 §3.2.2). Legacy file deleted. |
| `aperag/views/test.py` | 60 | **dev-only**；**直接 rm** 或迁 `tests/fixtures/`（PM 决定） |
| `aperag/views/utils.py` | 134 | `auth.py` + `config.py` 依赖；**随它们迁到 identity 域 `_utils.py`**，然后 rm |
| `aperag/views/chat_documents.py` | 21 | 空 stub，app.py 未 mount；**直接 rm** |

### 3.3 ACTIVE_LEGACY decision 原则

1. **保留 `/api/v1/` URL contract**：hard-cut 只动内部 canonical 归属，**不破坏 v1 客户端契约**。外部 curl/SDK 仍然走 `/api/v1/...`。
2. **若 v2 已完全覆盖 v1 功能**，v1 URL 可以一并删（上面 `collections.py` / `main.py` 两条），前提是 FE 和外部 SDK 已经切完（PM coordinate with FE + docs 团队）。
3. **OpenAPI schema byte-check**：`scripts/export_openapi.py --check` 必须 pass — 如果 v1 URL 还在契约里，迁移后 handler 签名 + response model 必须 bit-identical。

### 3.4 BE PR gate

```bash
# 所有 hard-cut 在一个 PR 里，所以 gate 一次性跑
uv run pytest tests/unit_test/test_modularization_boundaries.py -x -q       # G1-G19 全过
uv run pytest tests/ -x -q                                                    # 完整 pytest
uv run python scripts/export_openapi.py --check                               # OpenAPI bit-stable
uv run ruff check aperag/ tests/                                              # lint
grep -R "from aperag\.service\.\(api_key_service\|audit_service\|bot_service\|chat_collection_service\|chat_document_service\|chat_service\|chat_title_service\|collection_service\|collection_summary_service\|default_model_service\|document_service\|graph_service\|llm_available_model_service\|llm_provider_service\|marketplace_collection_service\|marketplace_service\|turn_feedback_service\)" aperag tests   # → 0 hits
grep -R "from aperag\.views\.\(agent_runtime\|api_key\|audit\|bots_v2\|chat\|collections_v2\|documents_v2\|evaluation_v2\|llm\|marketplace\|marketplace_collections\|providers_v2\)" aperag tests   # → 0 hits
grep -R "from aperag\.agent_runtime" aperag tests   # → 0 hits outside tests/aperag.domains.agent_runtime
grep -R "from aperag\.evaluation_v2" aperag tests   # → 0 hits
grep -R "from aperag\.schema\.view_models import" aperag tests | grep -v "FailResponse\|Settings\|ParserHealth\|Prompt\|UpdateUserPrompts\|UserPromptsResponse"   # → 0 hits（只剩 local schema 的 import）
```

---

## 4. 组织口径（per PM msg=ec47711e / msg=8fff00a8）

| Item | 决策 |
|------|------|
| PR 数量 | 目标 **2 PR**：`PR-BE destructive backend cleanup` + `PR-FE destructive frontend cleanup` |
| PR 规模 | 每个 PR 可以 **非常大**（backend 可能 ~5 000 LOC diff；frontend ~26 000 LOC diff，主要是删除）— 接受 |
| 允许压成 1 PR | 若 PM 判断 BE/FE 同 PR 更顺（如有跨 layer 的 `/api/v1/` contract FE 同步需要），接受合并为 1 PR |
| CR 口径 | Weston blocker-level minimal CR；Opus lane 做 canonical drift（每个主写 owner 自检 §1 白名单 + §3.4 / §2.5 gate） |
| Cadence | 10 min PM checkpoint；但 PR 本身不分 batch — 一次性做完一次性 review |
| 架构边界 | @符炫炜 inventory SSoT + canonical lock；PM 不做架构判断 (per earayu2 msg=92ddf593) |
| Coding owner | 优先 Opus（Bryce Phase 4 + chenyexuan Phase 5/6 + cuiwenbo Phase 3 + huangheng FE L1），GPT (dongdong / weihong) 补并行位 |

### 4.1 Owner 建议（PM 最终决定）

- **PR-BE 主 author**：Bryce（Phase 4 SME，跨域经验最广，能抓 view_models dual-hook 和 db.models 两个高危点）
- **PR-BE co-author**：chenyexuan（Phase 5/6 agent_runtime + evaluation + conversation 域） + cuiwenbo（Phase 3 KB + indexing + retrieval + KG 域） + Bryce 协调
- **PR-FE 主 author**：huangheng（FE L1 SME）
- **Single-author alternative**：若 PM 决定 BE 一人主写，Bryce 最合适；chenyexuan / cuiwenbo 做 parallel spot-check（不改同文件）

### 4.2 执行纪律

1. 主 author 开 branch 前 fetch latest main；PR body `Ghost-check:` line 引用 `docs/modularization/cleanup-inventory.md §1`（禁删白名单 self-check 确认）
2. 大 PR 在 branch 上 commit-by-commit 组织（便于 rollback 到任意 intermediate commit），最终 squash-merge
3. 每个高危 block（§3.1 E/F db.models re-export + view_models dual-hook）独立 commit，便于 bisect
4. PR 内 gate 全绿才请 Weston review；不做多轮 back-and-forth，一次性提交 final state

---

## 5. 历史：v0 → v1 diff

v0 (PR #1662 `97aa4bf8` / 345 LOC) → v1（本文件）主要变化：

1. §1 禁删白名单重写成 5 层 distinction（Layer A G18 alt 2 / Layer B standalone-infra 1 含 `search_pipeline_service` / Layer C Invitation / Layer D G17 / Layer E misc infra dirs）— 修正 Bryce msg=647b2b17 指出的 v0 Section 5 漏列 `search_pipeline_service` 的 canonical drift。
2. §6 原推荐"B1-B7 多 batch 小 PR" → v1 §4 重写为"1-2 大 PR hard-cut"（per earayu2 msg=be4c5c19 / msg=ac0a9243）。
3. §3 ACTIVE_LEGACY 从"audit 后单独决策" → v1 §3.2 直接给出每文件 canonical migration destination + 是否保留 v1 URL。
4. 加 §2.3 FE 白名单（`design-tokens.ts` reserved exports 不删）— per huangheng msg=8e203455 提的 "reserved export" 类比 BE standalone-infra。
5. §4 明确架构 / PM 角色边界（per earayu2 msg=92ddf593 / msg=90c75cce）。

这份 v1 替换 v0 作为 Phase 7 cleanup canonical SSoT。v0 PR 已 merge，历史记录保留在 git history。
