# Hurl coverage matrix

Maps every Hurl suite under `tests/e2e_http/hurl/**` to the v2 domain
that owns it and to the GitHub workflow job that runs it. Destructive
phases must update both the Hurl file and this matrix so nothing drifts.

## Suites → domains

| Hurl file | Domain (v2) | Notes |
| --- | --- | --- |
| `smoke/00_health.hurl` | `platform` | liveness / readiness; always on every workflow. |
| `smoke/01_auth.hurl` | `identity` | sign-in / session establishment. |
| `smoke/02_collection.hurl` | `knowledge_base` | collection CRUD minimal flow. |
| `smoke/03_document_basic.hurl` | `knowledge_base` + `indexing` | document create + minimal index status check. |
| `smoke/04_api_key.hurl` | `identity` | user API-key lifecycle. |
| `full/10_provider_llm.hurl` | `model_platform` | provider / model catalog full flow. |
| `full/11_document_full.hurl` | `knowledge_base` + `indexing` | document upload, rebuild, preview, object, download. |
| `full/12_bot.hurl` | `conversation` | bot CRUD + collection binding. |
| `full/13_chat_http.hurl` | `conversation` | chat turn shell via HTTP. |
| `full/14_graph_http.hurl` | `knowledge_graph` | graph query / curation / export. |
| `full/15_agent_runtime_v3.hurl` | `agent_runtime` | SSE turn / timeline / artifact contract — hard boundary. |
| `full/16_evaluation_v2.hurl` | `evaluation` | evaluation v2 dataset + run. |
| `full/17_chat_collection_flow.hurl` | `conversation` + `knowledge_base` | cross-domain chat over collection. |
| `full/18_web_access_http.hurl` | `web_access` | deterministic validation-only contract (missing body / bad body / malformed JSON). Provider-independent — never exercises live JINA / DuckDuckGo fetches. |
| `full/19_retrieval_http.hurl` | `retrieval` | Phase 2 domain hard-cut: validation + typed-empty-shape contract for `/api/v2/collections/{id}/searches*` (create / list / delete + missing query + unknown collection). Provider-independent: no real recall backend is touched because the test collection disables every strategy. |
| `full/20_knowledge_graph_http.hurl` | `knowledge_graph` | Phase 2 domain hard-cut: validation + typed-empty-shape contract for `/api/v2/collections/{id}/graphs*` (labels / subgraph / nodes-merge / merge-suggestions GET + action 404). Provider-independent: curation-run POST coverage is deferred to provider/full. |

## Domains → hurl coverage target

| Domain | Smoke Hurl | Full Hurl | Phase 2+ Hurl work |
| --- | --- | --- | --- |
| `identity` | `01_auth`, `04_api_key` | — | add Phase 4 identity hardening if v1 → v2 path rename. |
| `knowledge_base` | `02_collection`, `03_document_basic` | `11_document_full`, `17_chat_collection_flow` | Phase 3 rewires if KB / indexing boundaries change. |
| `indexing` | `03_document_basic` (status) | `11_document_full` | Phase 3 pilot DB model split — keep Alembic + status paths covered. |
| `retrieval` | *(none yet)* | `19_retrieval_http` | Phase 2 landed: `19_retrieval_http.hurl` covers validation + typed-empty-shape contract for `/api/v2/collections/{id}/searches*`. Provider-dependent live recall coverage must live in provider/full, never in smoke. |
| `knowledge_graph` | *(none yet)* | `14_graph_http` (legacy v1 — flagged for retirement), `20_knowledge_graph_http` | Phase 2 landed: `20_knowledge_graph_http.hurl` covers labels / subgraph / nodes-merge / merge-suggestions GET + action 404 on `/api/v2/`. Follow-up PR should retire or retarget the legacy `14_graph_http.hurl` once the v1 graph paths are fully decommissioned. |
| `web_access` | *(none yet)* | `18_web_access_http.hurl` | Phase 2a landed: `18_web_access_http.hurl` covers validation-only contract for `/api/v2/web/search` + `/api/v2/web/read` (missing body / bad body / malformed JSON). Provider-dependent live-fetch Hurl, if any, must live in the provider/full job so it cannot block smoke. |
| `conversation` | *(none yet)* | `12_bot`, `13_chat_http`, `17_chat_collection_flow` | Phase 5 ownership cleanup. |
| `agent_runtime` | *(none yet)* | `15_agent_runtime_v3` | Phase 5 — SSE shape unchanged unless explicitly redesigned. |
| `evaluation` | *(none yet)* | `16_evaluation_v2` | Phase 5 — keep v3 simplification guards (`V1_PATHS_REMOVED_IN_FINAL_SWEEP`, `test_e2e_http_uses_simplified_evaluation_v2_contract`). |
| `marketplace` | *(none yet)* | *(none yet)* | Phase 4 — add marketplace contract Hurl when public routes land. |
| `model_platform` | *(none yet)* | `10_provider_llm` | Phase 4 — confirm provider-secret Hurl stays in provider/full job. |
| `governance` | *(none yet)* | *(none yet)* | Phase 4 — add contract Hurl for quota / audit / settings once v2 paths land. |
| `platform` | `00_health` | — | — |

## GitHub workflow mapping

| Workflow | File | Jobs of interest | What it runs |
| --- | --- | --- | --- |
| CI (`cicd-push.yml` / `cicd-pull-request.yml`) | `.github/workflows/cicd-push.yml` | `Unit-Test` | `lint`, `make openapi-check`, unit tests with coverage. |
| E2E HTTP (local compose) | `.github/workflows/e2e-http-smoke.yml` | `e2e-http-smoke` (smoke only), `e2e-http-provider` (full, secrets-gated) | `tests/e2e_http/scripts/run_compose_smoke.sh` (smoke) and `run_compose_full.sh` (full) when provider secrets are present. |
| E2E ApeRAG on EKS (full) | `.github/workflows/e2e-aperag.yml` → `e2e-aperag-test.yml` | `e2e-test` | Both smoke and full Hurl suites over an EKS deployment, driven by `test_scope` input. |

## Per-phase PR-body gate (what each Phase PR must list)

- **Phase 1** (FE legacy SDK hard-delete, no route/schema change):
  unit tests + typed contract tests locally green; GitHub CI `Unit-Test`
  green; GitHub `e2e-http-smoke` must stay green (no Hurl changes
  expected, but the guardrail should not regress).
- **Phase 2** (retrieval / knowledge_graph / web_access hard-cut): Hurl
  files in the affected domain updated; `e2e-http-smoke` green;
  `e2e-http-provider` must either be green (preferred) or explicitly
  scoped out by PR body for deterministic-only behaviour. For
  `web_access` the PR body must separate **deterministic contract Hurl**
  (safe to run in smoke) from **provider-dependent Hurl** (must live in
  provider/full, not smoke) so flakes cannot mask real misses.
- **DB split phase** (separate): `e2e-aperag-test.yml` full run with
  Alembic migration; downgrade notes or a "destructive, no downgrade
  accepted" statement in the PR body.
- **Phase 5** (conversation / agent_runtime / evaluation cleanup): full
  Hurl suites for the touched domain must go green on GitHub; no SSE /
  event shape drift unless the PR body explicitly lists the breaking
  contract change.
