# Target domain map (v2 destructive-first baseline)

Canonical backend path is `aperag/domains/<domain>/` unless stated
otherwise. FE adapter path is `web/src/features/<adapter>/` unless the
row says `none/currently internal-only`.

`search` and `graph` are independent domains even though their URLs are
collection-scoped: the URL is UX, the domain boundary is ownership
(query planning, search history, graph storage, curation, etc. are
reused outside the collection lifecycle).

| Domain | Backend canonical path | FE feature path(s) | Owned DB models (canonical target) | Owned API schemas | Owned routes (shape will follow Phase 2+ breaking tables) | Notes / contracts |
| --- | --- | --- | --- | --- | --- | --- |
| `identity` | `aperag/domains/identity` | `features/identity`, `features/api-key` (1:N mirror allowed) | `User`, `OAuthAccount`, `Invitation`, `ApiKey` | auth / user / api-key / invitation | auth, user, api-key routes | Replaces the current `views.auth` central-dependency pattern with an explicit contract. |
| `governance` | `aperag/domains/governance` | `features/quota`, `features/audit`, `features/settings` | `UserQuota`, `AuditLog`, `Setting`, `ConfigModel` | quota / audit / settings / config | `/quotas*`, `/audit-logs*`, `/settings*`, `/config` | Policy / recording; does not execute business side effects. |
| `model_platform` | `aperag/domains/model_platform` | `features/providers`, `features/prompt` | `ModelServiceProvider`, `LLMProvider`, `LLMProviderModel`, `PromptTemplate` | provider / default-model / prompt | `/providers*`, `/default-models`, `/prompts*` | Backend stays 1 domain; FE splits providers / prompt. |
| `knowledge_base` | `aperag/domains/knowledge_base` | `features/collection`, `features/document` | `Collection`, `Document` | collection / document lifecycle | `/collections*`, `/documents*` | Reads index status through an `indexing` contract; does not own `DocumentIndex`. |
| `retrieval` | `aperag/domains/retrieval` | `features/search` | `SearchHistory` | `SearchRequest`, `SearchResult*` | `/collections/{collection_id}/searches*` (URL stays collection-scoped) | Independent domain: query planning, retrieval orchestration, search history, multi-index fusion. |
| `indexing` | `aperag/domains/indexing` | surfaced through `features/document` (no direct FE adapter) | `DocumentIndex` | `RebuildIndexes*`, index status enums | document rebuild / status endpoints | Logical owner of the `DocumentIndex` state machine and scheduling. Physical model stays in the legacy aggregate until the DB split phase. |
| `knowledge_graph` | `aperag/domains/knowledge_graph` | `features/graph` | `GraphCurationRun`, `GraphCurationSuggestion` | `Graph*`, merge suggestion | `/collections/{collection_id}/graphs*` | Independent domain: graph storage, curation, merge suggestions, executor contract. |
| `conversation` | `aperag/domains/conversation` | `features/bot`, `features/chat` | `Bot`, `Chat`, `TurnFeedback` | bot / chat / feedback | `/bots*`, `/chats*` | Conversation is not the owner of agent runtime. `components/chat/` stays ownership-unassigned until the split phase; no wholesale component move is allowed. |
| `agent_runtime` | `aperag/agent_runtime` (kept top-level, not migrated in Phase 1/2) | `future features/agent-runtime` or existing chat runtime usage (not created in Phase 0/1) | `AgentTurn`, `AgentTimelineEvent`, `AgentArtifact` | Turn / Timeline / Artifact schemas | `/api/v2/agent/*` | Already-canonical product domain; SSE / event / artifact shape is a hard boundary. First round only adds boundary tests; physical move lives in its own later phase. |
| `evaluation` | `aperag/domains/evaluation` | `features/evaluation` | evaluation v2 tables (`EvaluationDataset`, `EvaluationRun`, `EvaluationRunItem`, `EvaluationRunItemAttempt`) | evaluation v2 schemas | `/api/v2/evaluation-*` | Legacy evaluation / question-set objects are a cleanup candidate; dropped only when references / tests / docs are updated. |
| `marketplace` | `aperag/domains/marketplace` | `features/marketplace` | `CollectionMarketplace`, `UserCollectionSubscription` | `SharedCollection*` | `/marketplace/collections*` | Reads collection / document only through the public `knowledge_base` contract. |
| `web_access` | `aperag/domains/web_access` | `none / currently internal-only` | — | `WebSearch*`, `WebRead*` | `/web/search`, `/web/read` | External web search / read capability. Not collection RAG. No FE adapter until a real FE caller appears. |
| `platform` | existing neutral infra / future `aperag/platform` | — | base / session / repository-base / adapters only | neutral only | none | Db session, object / vector / llm / docparser adapters, trace, concurrent control, Celery wiring. Never hosts business rules. |

## Cross-cutting invariants

- **URL scope ≠ domain owner.** Routes can stay nested under
  `/collections/{id}/*` while the owning backend/FE domain is
  independent.
- **Shared / lib minimalism.** `aperag/platform` and `web/src/lib`
  host only neutral infrastructure. Business rules stay in the
  owning domain.
- **Contracts over imports.** Cross-domain consumption goes through an
  explicit contract module under the owning domain; never through
  `aperag/service/*`, `aperag/schema/view_models`, or `aperag/db/models`.
- **No empty mirror modules.** When `FE = none / currently
  internal-only`, no empty `features/<d>/` is created.
