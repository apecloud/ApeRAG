# ApeRAG HTTP E2E

This suite is the new black-box HTTP E2E entrypoint for ApeRAG.

Design constraints:
- Targets a freshly started, empty ApeRAG service.
- Exercises user-visible behavior only through HTTP APIs.
- Keeps test content independent from the environment launcher.
- Treats `hurl/` as the current execution implementation, not the suite identity.

Top-level layout:
- `hurl/smoke/`: minimal provider-independent smoke coverage
- `hurl/full/`: broader HTTP coverage that may depend on providers or longer flows
- `bootstrap/`: prepares the minimum testable state on an empty service
- `runners/`: environment launchers such as Compose or K8s
- `scripts/`: wrappers that connect runner, bootstrap, and Hurl execution
- `testdata/`: stable input files used by HTTP tests

Current v1 scope:
- Readiness endpoint: `GET /health/ready` (legacy `GET /health` stays available for backward compatibility)
- Auth smoke: login, current user, logout
- Collection smoke: create, get, list, update, delete
- Document smoke: upload, get detail, delete
- API key smoke: create, use, update, delete
- Provider-aware full coverage:
  - inspect available models and provider configuration surfaces
  - configure public providers with user API keys
  - manage provider models, including slash-containing model names
  - set default embedding/completion/rerank models
  - query `/api/v1/available_models`
  - call `/api/v1/embeddings` and `/api/v1/rerank` through real external providers
  - cover document staged/confirm/download/rebuild paths
  - cover document status visibility, list search by name, and collection search/history HTTP contracts
  - cover bot CRUD + agent config get/update
  - cover chat create/list/get/update/delete
  - assert the stable OpenAI-shaped `/v1/chat/completions` contract backed by Agent Runtime V3
  - cover a provider-aware business flow: collection + document + bot-bound chat turn
  - cover graph labels + graph overview + parameter validation endpoints
  - cover a scripted chat business flow that waits for vector/fulltext indexing, then asserts a non-empty answer artifact and a non-empty reference bundle
  - cover a scripted graph business flow that waits for knowledge-graph indexing to become `ACTIVE`, then asserts non-empty labels, nodes, and edges

Non-goals for v1:
- Replacing every existing pytest-based E2E immediately
- Covering WebSocket or streaming chat flows
- Depending on external model providers in the smoke path

Important behavior notes from the current implementation:
- Smoke tests create and clean up their own business resources.
- `document` smoke intentionally validates upload and basic lifecycle only; it does not require provider-backed indexing to complete.
- `document` full validates stable document/search HTTP contracts without hard-coding async index completion or search relevance guarantees.
- `run_full.sh` now includes two supplemental scripted flows after the Hurl contracts:
  - `run_chat_collection_flow.sh` proves provider-backed indexing completes and bot chat returns both an answer artifact and references.
  - `run_graph_index_flow.sh` proves knowledge-graph indexing completes and graph APIs return real graph content.
- API key smoke explicitly logs out before bearer-only checks so cookie auth does not mask key behavior.
- Full Hurl tests are intentionally separated from smoke so provider/API-key failures do not dilute the minimal deployment contract.

Typical local flow against an existing ApeRAG instance:

```bash
export E2E_BASE_URL=http://127.0.0.1:8000
./tests/e2e_http/bootstrap/bootstrap.sh
./tests/e2e_http/scripts/run_smoke.sh
```

Typical local flow with the first Compose runner:

```bash
./tests/e2e_http/scripts/run_compose_smoke.sh
```

Provider-aware local flow:

```bash
export E2E_ALIBABACLOUD_API_KEY=...
export E2E_OPENROUTER_API_KEY=...
./tests/e2e_http/scripts/run_compose_full.sh
```

Typical local flow against a Kubernetes-backed environment:

```bash
export E2E_K8S_NAMESPACE=default
export E2E_K8S_SERVICE=aperag-api
./tests/e2e_http/scripts/run_k8s_smoke.sh
```
