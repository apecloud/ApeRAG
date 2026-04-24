# Phase 4 — `identity` + `governance` + `model_platform` + `marketplace` backend hard-cut breaking-change table

## 1. Summary

- Owner: `@Bryce`
- Reviewer(s): `@huangheng` (incremental CR) / `@符炫炜` (architect) / `@架构师` (PM)
- Linked task: `#25 / PR #1633`.
- Canonical basis: `msg=d47fa490` + `msg=896584ee` + `msg=6ab7d211` + `msg=6d2ae86a`.
- Relies on: Phase 3 `#1629` MERGED (knowledge_base domain + indexing / retrieval / knowledge_graph DB split) + the Phase 3 G1 / G11 / G13 / G14 boundary gates.
- Rollback strategy: plain `git revert`. The DB layer is unchanged on disk (pure ORM relocation; `Base.metadata` picks up the same canonical tables through the per-domain modules). The revert restores the 13 Phase-4-moved DB classes, the four service modules (marketplace / marketplace_collection / audit / api_key / llm_provider / llm_available_model / default_model), the `UserManager` body in `aperag/views/auth.py`, the adapter wiring in `aperag/app.py`, and the 48 Pydantic schemas now living in the four domain `schemas.py` files.

## 2. API changes

HTTP surface is **byte-stable** — `scripts/export_openapi.py --check` passes after every sub-commit of Phase 4. URLs / verbs / response_model / status codes unchanged. The moved view modules collapse to router-import shims so pre-migration callers (tests, FE adapters, hurl fixtures) keep resolving the same `APIRouter` instances.

No URL moved, no verb changed, no schema name changed.

## 3. Backend import surface changes

Canonical import paths for the four Phase 4 domains. Pre-migration callers keep working through re-export shims (removed in Phase 6).

| Legacy import | New canonical import | Notes |
| --- | --- | --- |
| `from aperag.db.models import User, Role, OAuthAccount` | `from aperag.domains.identity.db.models import User, Role, OAuthAccount` | Step 4-S2a. `aperag.db.models` re-exports via a module-top import (`Role` is consumed by `Invitation` column type at class-body time, so the shim cannot live at the end-of-file like Phase 3). |
| `from aperag.db.models import ApiKey, ApiKeyStatus, AuditLog, AuditResource` | `from aperag.domains.governance.db.models import …` | Step 4-S2b. `aperag.db.models` end-of-file shim. |
| `from aperag.db.models import CollectionMarketplace, UserCollectionSubscription, CollectionMarketplaceStatusEnum` | `from aperag.domains.marketplace.db.models import …` | Step 4-S2c. |
| `from aperag.db.models import LLMProvider, LLMProviderModel, APIType` | `from aperag.domains.model_platform.db.models import …` | Step 4-S2d. `APIType` rides along because `LLMProviderModel.api` binds it as an `EnumColumn` at class-body time (minimum-necessary addendum to canonical scope; PM `msg=97baeb93` accepted as "minimum-necessary 附带搬运"). |
| `from aperag.schema.view_models import Auth, Auth0, Authing, Config, Logto, Register, User, Login, UserList, ChangePassword, Invitation, InvitationCreate, InvitationList` | `from aperag.domains.identity.schemas import …` | Step 4-S3a. Dual-hook re-export (cuiwenbo Step 4b Scenario A pattern) keeps `view_models` working. |
| `from aperag.schema.view_models import ApiKey, ApiKeyList, ApiKeyCreate, ApiKeyUpdate, AuditLog, AuditLogList` | `from aperag.domains.governance.schemas import …` | Step 4-S3b. |
| `from aperag.schema.view_models import SharedCollection, SharedCollectionConfig, SharedCollectionList` | `from aperag.domains.marketplace.schemas import …` | Step 4-S3c. |
| `from aperag.schema.view_models import TagFilterCondition, TagFilterRequest, ModelConfig, ModelConfigList, DefaultModelConfig, DefaultModelsResponse, DefaultModelsUpdateRequest, LlmProvider, LlmProviderModel, LlmConfigurationResponse, LlmProviderCreateWithApiKey, LlmProviderUpdateWithApiKey, LlmProviderModelList, LlmProviderModelCreate, LlmProviderModelCreateRequest, LlmProviderModelUpdate, EmbeddingRequest, EmbeddingData, EmbeddingUsage, EmbeddingResponse, Document1, RerankRequest, Document2, RerankDocument, RerankUsage, RerankResponse` | `from aperag.domains.model_platform.schemas import …` | Step 4-S3d. 26 schemas — the largest single-commit sub-step of Phase 4. |
| `from aperag.service.marketplace_service import marketplace_service` | `from aperag.domains.marketplace.service.marketplace_service import marketplace_service` | Step 4-S4. Legacy shim. |
| `from aperag.service.marketplace_collection_service import marketplace_collection_service` | `from aperag.domains.marketplace.service.marketplace_collection_service import marketplace_collection_service` | Step 4-S4. **Q2 public rename**: `_check_marketplace_access` → `check_marketplace_access` (drop leading underscore, per `msg=6ab7d211` Q2). The transitional `_MarketplaceCollectionOpsAdapter` in `aperag/app.py` dropped at the same time — the service now structurally satisfies the KB `MarketplaceCollectionOps` Protocol directly. |
| `from aperag.service.audit_service import audit_service` | `from aperag.domains.governance.service.audit_service import audit_service` | Step 4-S5. |
| `from aperag.service.api_key_service import api_key_service` | `from aperag.domains.governance.service.api_key_service import api_key_service` | Step 4-S5. |
| `from aperag.service.llm_provider_service import …` | `from aperag.domains.model_platform.service.llm_provider_service import …` | Step 4-S6. |
| `from aperag.service.llm_available_model_service import llm_available_model_service` | `from aperag.domains.model_platform.service.llm_available_model_service import llm_available_model_service` | Step 4-S6. |
| `from aperag.service.default_model_service import default_model_service` | `from aperag.domains.model_platform.service.default_model_service import default_model_service` | Step 4-S6 (brought along with the two sibling llm services — same domain). |
| `from aperag.views.marketplace import router` / `from aperag.views.marketplace_collections import router` | `from aperag.domains.marketplace.api.routes import router` | Step 4-S4. Two legacy routers collapse to one canonical router. |
| `from aperag.views.audit import router` / `from aperag.views.api_key import router` | `from aperag.domains.governance.api.routes import router` | Step 4-S5. |
| `from aperag.views.llm import router` | `from aperag.domains.model_platform.api.llm_routes import router` | Step 4-S6. Kept distinct from `providers_v2_routes` because the two routes mount under different prefixes (`/api/v1` vs `/api/v2`). The legacy `aperag/views/llm.py` shim explicitly re-exports the private `_build_rerank_response_items` helper because one existing unit test imports it. |
| `from aperag.views.providers_v2 import router` | `from aperag.domains.model_platform.api.providers_v2_routes import router` | Step 4-S6. |

### Non-moves (explicit)

- **`aperag.llm.*` stays** as shared infrastructure (`msg=d47fa490` Section 7). It is **not** part of the model_platform domain — it is a thin HTTP-client wrapper layer consumed by multiple domains (KB indexing, retrieval, agent_runtime) and therefore lives alongside `aperag.config` / `aperag.db.base`.
- **`aperag.service.quota_service` stays** legacy per `msg=d47fa490` Q1 / Phase 4 canonical; its permanent home is deferred to Phase 5 / Phase 6.
- **`aperag.service.bot_service` / `aperag.service.chat_collection_service` stay** legacy; Phase 5 conversation domain implementation moves them. Phase 4 wires the legacy implementations behind the identity `BotInitOps` / `ChatInitOps` adapter Protocols.
- **OAuth routers (GitHub / Google) stay** in `aperag/views/auth.py` for this phase; moving `fastapi_users.get_oauth_router` requires re-hosting `auth_backend` + the `fastapi_users` instance, which tightly couples back to the same module. A follow-up can lift them into `aperag/domains/identity/service/oauth.py` after Phase 4 merges if desired.

## 4. Consumer-owned Protocol surface (lesson 9a-quad)

Four domains each declare their own Protocols. Concrete providers satisfy them structurally; no provider-side import of the consumer's `ports.py` module (new boundary gates `G15` / `G16` / `G17` enforce this at PR time).

| Protocol | Owner | Consumer call site | Wired-at-startup concrete |
| --- | --- | --- | --- |
| `AuthenticatedUser` × 4 (per-domain) | identity / governance / marketplace / model_platform | handler parameter types for `Depends(required_user)` / `Depends(optional_user)` | legacy `aperag.db.models.User` — ORM class structurally satisfies the Protocol |
| `UserView` | governance | audit-subject lookup + admin-only permission check shape | identity `User` ORM satisfies `id` + `role` |
| `BotInitOps.create_default_bot_for_user` | identity | `UserManager.on_after_register` | `_BotInitOpsAdapter` in `aperag/app.py` wrapping legacy `bot_service.create_bot(BotCreate(...), skip_quota_check=True)` |
| `ChatInitOps.create_default_chat_for_user` | identity | `UserManager.on_after_register` | `_ChatInitOpsAdapter` wrapping legacy `chat_collection_service.initialize_user_chat_collection` |
| `QuotaInitOps.initialize_user_quota` | identity | `UserManager.on_after_register` | `_QuotaInitOpsAdapter` wrapping legacy `quota_service.initialize_user_quotas` |

The identity DI adapters collapse to direct structural satisfaction when Phase 5 moves `bot_service` / `chat_collection_service` into the conversation domain.

## 5. New boundary gates introduced in this PR (G15 / G16 / G17)

Added alongside the existing Phase 3 G1 / G11 / G13 / G14 / KB-scope boundary gates.

| Gate | Test | Rationale |
| --- | --- | --- |
| **G15** | `test_phase4_consumer_domains_never_import_role_enum` | Non-identity domains must compare `user.role == "admin"` by literal; the identity ``Role`` enum stays identity-internal. AST import-ban only — the literal value list is a soft convention enforced by reviewer CR (msg=6d2ae86a). |
| **G16** | `test_phase4_consumer_domains_never_import_user_orm_class` | Non-identity domains must use the per-domain ``AuthenticatedUser(Protocol)`` (or a narrower ``UserView(Protocol)``) instead of binding to the ``User`` ORM. |
| **G17** | `test_phase4_di_critical_wirings_at_app_startup` | Runtime smoke — after ``import aperag.app`` every entry in the ``CRITICAL_WIRINGS`` registry resolves to a non-``None`` Protocol instance. Catches a forgotten / re-ordered startup wire-up at CI time instead of at first request (msg=896584ee: use runtime state check, not AST setter-naming scan). |

Inherited boundary tests continue to pass unchanged.

## 6. Data / alembic / runtime

- **DB schema**: zero changes. The per-domain DB modules share `Base` via `aperag/db/base.py`; Alembic `autogenerate` continues to see every table / column / enum / index / constraint with identical shape.
- **Alembic migrations**: none.
- **Runtime**: `aperag/app.py` startup wires both the Phase 3 KB DI slots (marketplace / marketplace_collection / search_pipeline / quota — already merged in Phase 3) and the new Phase 4 identity DI slots (bot_init / chat_init / quota_init). Adapters dropped: `_MarketplaceCollectionOpsAdapter` (Phase 4 Step 4-S4 service rename makes it redundant).

## 7. Shim lifecycle

Every legacy path continues to work through re-export shims. The shim inventory for Phase 6 cleanup:

| Shim | File | Contents |
| --- | --- | --- |
| `aperag.db.models` | `aperag/db/models.py` | Re-exports the Phase 3 15 + Phase 4 13 = 28 DB symbols via four per-domain import blocks (three at end-of-file, the identity block at module top to satisfy `Invitation.role = Column(EnumColumn(Role), …)` binding). |
| `aperag.schema.view_models` | `aperag/schema/view_models.py` | End-of-file `try` blocks re-import the Phase 3 17 + Phase 4 48 = 65 Pydantic schemas from the per-domain `schemas.py` files. |
| `aperag.service.{marketplace,marketplace_collection,audit,api_key,llm_provider,llm_available_model,default_model}_service` | `aperag/service/*.py` | Pure re-export shims (7 files). `marketplace_collection_service` additionally documents the Q2 `_check_marketplace_access` → `check_marketplace_access` rename. |
| `aperag.views.{marketplace,marketplace_collections,audit,api_key,llm,providers_v2}` | `aperag/views/*.py` | Router-import shims (6 files). |
| Adapters in `aperag/app.py` | `_BotInitOpsAdapter` / `_ChatInitOpsAdapter` / `_QuotaInitOpsAdapter` | Three thin classes wrapping the legacy bot_service / chat_collection_service / quota_service method names onto the public identity Protocol surface. Collapse when Phase 5 moves bot/chat services into conversation domain. |

## 8. FastAPI / route lessons

- Route modules under `aperag/domains/**/api/` do not use `from __future__ import annotations` (lesson 9a-quatuordec from Phase 3 Step 5a carries forward).
- `aperag/views/auth.py` keeps its FastAPI integration plumbing (fastapi-users `FastAPIUsers` instance, OAuth router wiring, `required_user` / `optional_user` Depends factories, invitation / login / logout / user-admin handlers) and simply re-imports `UserManager` from the identity domain location. This minimizes blast radius in a file that is tightly coupled to fastapi-users' router-builder API.

## 9. Risks + rollback

- **Risk 1 — legacy service shim cascade**. Callers import `aperag.service.*` which re-exports from `aperag.domains.*`. If a Phase 5 follow-up renames a service method, both the domain module and the re-export shim must stay consistent. Mitigation: the legacy shim is intentionally `from ... import *` with an explicit `from ... import SpecificName` list, so missed symbols surface at import time.
- **Risk 2 — identity DI wire-up not firing**. `UserManager.on_after_register` raises `RuntimeError` with an actionable message when any of the three DI slots is `None`. Mitigation: G17 runtime smoke + the `CRITICAL_WIRINGS` registry; a forgotten wire-up fails at CI collection time.
- **Rollback**: plain `git revert`. Zero DB-layer state on disk changed.
