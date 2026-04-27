# Phase 3 — `knowledge_base` + `indexing` backend hard-cut breaking-change table

## 1. Summary

- Owner: `@Bryce`
- Reviewer(s): `@huangheng` (current-head CR) / `@符炫炜` (architect) / `@架构师` (PM)
- Linked task: `#16 / PR #1629` in `#模块化重构`; depends on merged
  Phase 2 (PR #1624) and the Phase 3 CI bootstrap (PR #1628).
- Relies on: Phase 0 boundary fixtures (G1 strict ban on
  ``aperag/domains/**`` importing ``aperag.service.*`` /
  ``aperag.schema.view_models`` / ``aperag.db.models``); Phase 2
  ``AuthenticatedUser(Protocol)`` lesson 9a-ter canonical; Phase 2b
  cuiwenbo Step 4b dual-hook re-export pattern (view_models ↔
  domain schemas symmetric load-order).
- Rollback strategy: revert the PR. Zero DB-layer migrations (the
  Phase 3 DB split is pure ORM relocation — tables / columns /
  enums / indexes / constraints unchanged on disk; ``Base.metadata``
  picks up the same canonical tables through the per-domain
  modules). The revert restores the legacy ``aperag.db.models``
  class definitions, the ``aperag/service/{collection,document,collection_summary}_service.py``
  bodies, the ``aperag/views/{collections,documents}_v2.py`` route
  decorators, and the ``aperag.schema.view_models.*`` class
  definitions for the 17 Pydantic schemas that were relocated to
  ``aperag.domains.knowledge_base.schemas``.

## 2. API changes

HTTP surface is **byte-stable** — ``scripts/export_openapi.py --check``
passes after Step 5a. Routes continue to live under the ``/api/v2``
prefix and keep their former paths / verbs / response_model / audit
decorator / status codes. The two former route modules
``aperag/views/collections_v2.py`` and ``aperag/views/documents_v2.py``
collapse to router-import shims re-exporting the single merged
``APIRouter`` at ``aperag.domains.knowledge_base.api.routes``.

No URL moved, no verb changed, no schema name changed — callers (FE
adapters, hurl fixtures, MCP, OpenAPI consumers) see zero diff.

## 3. Backend import surface changes

The canonical import paths change for everyone reaching into the KB
domain. Pre-migration callers keep working through re-export shims
(removed in Phase 6). Migration table for new call sites:

| Legacy import | New canonical import | Notes |
| --- | --- | --- |
| ``from aperag.db.models import Collection, CollectionSummary, Document, CollectionStatus, CollectionSummaryStatus, CollectionType, DocumentStatus`` | ``from aperag.domains.knowledge_base.db.models import …`` | 7 DB classes + 4 lifecycle enums moved in Step 4a. ``aperag.db.models`` re-exports them via explicit shim. |
| ``from aperag.db.models import DocumentIndex, DocumentIndexStatus, DocumentIndexType`` | ``from aperag.domains.indexing.db.models import …`` | DocumentIndex + 2 enums moved in Step 2 (indexing-domain shape) and re-exported from the legacy aggregate. |
| ``from aperag.db.models import SearchHistory`` | ``from aperag.domains.retrieval.db.models import SearchHistory`` | Step 3 sliver. |
| ``from aperag.db.models import GraphCurationRun, GraphCurationSuggestion, GraphCurationRunStatus, GraphCurationSuggestionStatus`` | ``from aperag.domains.knowledge_graph.db.models import …`` | Moved in Step 3 alongside graphindex rename. |
| ``from aperag.schema.view_models import Collection, CollectionCreate, CollectionUpdate, CollectionView, CollectionViewList, CollectionSummaryTriggerResponse, Document, DocumentList, DocumentPreview, RebuildIndexesRequest, RebuildIndexesResponse`` | ``from aperag.domains.knowledge_base.schemas import …`` | 11 schemas moved in Step 4b. ``view_models`` re-imports via the end-of-file ``try`` block + ``_bind_view_models_reexports`` hook. |
| ``from aperag.schema.view_models import UploadDocumentResponse, FailedDocument, ConfirmDocumentsResponse, FetchUrlResultItem, FetchUrlResponse, StagedDocumentsResponse`` | ``from aperag.domains.knowledge_base.schemas import …`` | 6 document-envelope schemas moved in Step 5b3. Same dual-hook. |
| ``from aperag.schema.view_models import ConfirmDocumentsRequest, FetchUrlRequest, DeleteDocumentsRequest, DeleteDocumentsResponse, SharingStatusResponse, MineruTokenTestRequest, MineruTokenTestResponse`` | ``from aperag.domains.knowledge_base.schemas import …`` | 7 route-level schemas moved in Step 5a. Same dual-hook. |
| ``from aperag.service.collection_service import CollectionService, collection_service`` | ``from aperag.domains.knowledge_base.service.collection_service import …`` | Moved in Step 5b2b; legacy path is a pure re-export shim (no runtime side-effects after Step 5b2c lifted wire-up into app.py). |
| ``from aperag.service.document_service import DocumentService, document_service`` | ``from aperag.domains.knowledge_base.service.document_service import …`` | Moved in Step 5b3. |
| ``from aperag.service.collection_summary_service import CollectionSummaryService, collection_summary_service`` | ``from aperag.domains.knowledge_base.service.collection_summary_service import …`` | Moved in Step 5b1. |
| ``from aperag.index.<mod>`` | ``from aperag.domains.indexing.<mod>`` | Step 2 renamed the whole subtree (``document_index_manager``, ``summary_index``, ``graph_index``, ``vision_index`` and friends). |
| ``from aperag.graphindex.<mod>`` | ``from aperag.domains.knowledge_graph.graphindex.<mod>`` | Step 3 renamed the subtree. |
| ``from aperag.views.collections_v2 import router`` / ``from aperag.views.documents_v2 import router`` | ``from aperag.domains.knowledge_base.api.routes import router`` | Merged into one router in Step 5a; legacy paths are router-import shims that re-export the same ``APIRouter`` instance. |

``aperag/db/base.py`` is a new G1-neutral infrastructure module
extracted in Step 2 to host ``Base = declarative_base()`` so the
per-domain DB modules and Alembic ``env.py`` can share the same
declarative base without importing the legacy aggregate. It is
deliberately **not** listed on the G1 ban list.

## 4. Consumer-owned Protocol surface (lesson 9a-quad)

The KB domain declares five Protocols in
``aperag/domains/knowledge_base/ports.py`` so it never static-imports
the legacy services it consumes. Concrete providers structurally
satisfy the Protocols — no provider-side import of the consumer's
ports module is allowed (new boundary gate
``test_knowledge_base_protocol_boundary_is_consumer_owned`` enforces
this).

| Protocol | Consumer call site | Wired-at-startup concrete |
| --- | --- | --- |
| ``AuthenticatedUser`` | ``api/routes.py`` typed ``Depends(required_user)`` parameters | legacy ``aperag.db.models.User`` ORM (structurally satisfies ``id: Any``) |
| ``MarketplaceOps`` (validate / get_sharing_status / publish / unpublish) | ``collection_service`` + ``api/routes.py`` sharing group + ``document_service`` marketplace check | legacy ``aperag.service.marketplace_service.marketplace_service`` |
| ``MarketplaceCollectionOps`` (``check_marketplace_access`` — public) | ``collection_service.create_search`` marketplace-subscriber fallback | ``_MarketplaceCollectionOpsAdapter`` in ``aperag/app.py`` bridging public name onto legacy ``_check_marketplace_access`` (adapter collapses when Phase 4 marketplace_collection_service rename lands; tracked via msg=6ab7d211 Q2) |
| ``SearchPipelineOps`` (``execute_search``) | ``collection_service.execute_search_flow`` | legacy ``aperag.service.search_pipeline_service.search_pipeline_service`` |
| ``QuotaOps`` (check_and_consume / release / get_user_quotas) | ``collection_service`` create/delete + ``document_service`` per-document / per-collection cap | legacy ``aperag.service.quota_service.quota_service`` |

``aperag/app.py`` module-scope block (Step 5b2c) wires all four
setters before FastAPI ``app`` is constructed; the sibling-import
pattern means ``document_service`` reads the same DI slots via
accessors imported from ``collection_service``.

## 5. New boundary gates introduced in this PR

Added in Step 7 alongside the existing G1 / G11 / G13 / G14 tests
the previous phases landed:

| Gate | Test | Rationale |
| --- | --- | --- |
| KB Protocol direction one-way | ``tests/unit_test/test_modularization_boundaries.py::test_knowledge_base_protocol_boundary_is_consumer_owned`` | Lesson 9a-quad: provider services (``marketplace_service``, ``marketplace_collection_service``, ``search_pipeline_service``, ``quota_service``) must never import ``aperag.domains.knowledge_base.ports``. |
| DI wire-up smoke | ``tests/unit_test/test_modularization_boundaries.py::test_knowledge_base_di_wire_up_populated_after_app_import`` | Runtime assertion: after ``import aperag.app`` all four KB consumer-owned Protocol globals are non-``None``. Guards against a re-ordered / forgotten startup wire-up landing silently. |

Inherited (from earlier phases, unchanged):

- ``G1`` (``test_aperag_domains_never_import_legacy_aggregate_modules``) — strict ban on the three god aggregates.
- ``G11`` (``test_aperag_db_models_reexports_full_phase3_set``) — 7 DB + 8 enum + 7 table re-export attest.
- ``G13`` (``test_phase3_classes_have_single_definition_site``) — each Phase 3 class lives in exactly one ``class Foo(Base):`` site.
- ``G14`` (``test_retrieval_kg_protocol_boundary_is_one_way``) — sibling canonical for retrieval ↔ knowledge_graph.
- ``test_aperag_domains_auth_dependency_is_not_any`` — now satisfied by KB routes via ``AuthenticatedUser(Protocol)``.

## 6. Data / alembic / runtime

- **DB schema**: zero changes. ``Base.metadata`` remains identical;
  the per-domain modules share the same ``Base`` via
  ``aperag/db/base.py`` so Alembic ``autogenerate`` continues to
  see every table / column / enum / index / constraint with the
  same definition.
- **Alembic migrations**: none needed. ``aperag/migration/env.py``
  explicitly imports ``aperag.db.models`` (which re-exports) plus
  ``aperag.domains.knowledge_graph.graphindex.models`` so the
  graphindex tables keep registering during ``alembic`` runs; the
  CI ``make db-check`` gate (introduced in PR #1628) would have
  caught any accidental drift.
- **Runtime**: ``aperag.app`` boot path unchanged for clients —
  same URL prefix, same response shapes, same status codes, same
  audit decorators, same authentication surface.

## 7. Shim lifecycle

Every legacy import path continues to work through re-export shims.
The shim count is tracked for Phase 6 cleanup:

| Shim | File | Contents | Phase 6 action |
| --- | --- | --- | --- |
| ``aperag.db.models`` | ``aperag/db/models.py`` | Re-exports 15 symbols + ``Base`` pass-through | Remove after all callers switch to per-domain DB modules. |
| ``aperag.schema.view_models`` | ``aperag/schema/view_models.py`` | End-of-file ``try`` block re-imports the 17 KB Pydantic schemas; ``_bind_view_models_reexports`` on the KB side handles the opposite load order. | Remove after all callers switch to per-domain schema modules. |
| ``aperag.service.collection_service`` | ``aperag/service/collection_service.py`` | Pure re-export of the KB domain service + the 4 DI setters. | Remove alongside the corresponding caller migration. |
| ``aperag.service.document_service`` | ``aperag/service/document_service.py`` | Pure re-export (no DI bootstrap — sibling-share via collection_service + app.py wire-up). | Remove alongside the corresponding caller migration. |
| ``aperag.service.collection_summary_service`` | ``aperag/service/collection_summary_service.py`` | Pure re-export. | Remove alongside the corresponding caller migration. |
| ``aperag.views.collections_v2`` | ``aperag/views/collections_v2.py`` | Router-import shim returning the merged KB router. | Remove once every caller imports the domain router directly. |
| ``aperag.views.documents_v2`` | ``aperag/views/documents_v2.py`` | Router-import shim (same router as above). | Same as above. |
| ``_MarketplaceCollectionOpsAdapter`` | ``aperag/app.py`` | Tiny adapter mapping the public Protocol method name onto the legacy underscore-prefixed method. | Collapse when Phase 4 ``marketplace_collection_service`` rename lands. |

## 8. FastAPI / route lessons

- Route modules under ``aperag/domains/**/api/routes.py`` must **not**
  use ``from __future__ import annotations``. FastAPI inspects
  return-type annotations at route registration time to decide
  whether a status-204 handler has a body; under PEP 563 the
  ``-> Response`` return annotation becomes the string ``"Response"``
  and the ``is_body_allowed_for_status_code(204)`` assertion fires
  at import. The moved KB router drops the future import for this
  reason; Phase 4 / 5 route moves should follow the same rule
  (candidate lesson ``9a-quatuordec``, flagged by @huangheng in the
  Step 5a CR).

## 9. Risks + rollback

- **Risk 1 — shim drift**: future contributors import the legacy
  path without noticing the canonical. Mitigation: G1 banlist plus
  the per-phase docstring trail. No revert needed — shims resolve.
- **Risk 2 — DI wire-up forgotten on a new deployment path** (e.g. a
  celery worker that imports the domain service but not
  ``aperag.app``). Mitigation: the accessor functions raise
  ``RuntimeError`` with an actionable message instead of letting
  ``None.method(...)`` blow up opaquely; the new
  ``test_knowledge_base_di_wire_up_populated_after_app_import``
  gate catches the happy path.
- **Risk 3 — alembic drift** between the legacy aggregate and the
  per-domain DB modules. Mitigation: the Phase 3 CI gate
  (``make db-check`` in PR #1628) runs ``alembic autogenerate``
  against every new commit; any divergence produces a red CI.
- **Rollback**: plain ``git revert``. The DB layer is unchanged on
  disk so no data-plane action is required.
