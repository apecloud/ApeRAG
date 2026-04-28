import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_server_typed_client_uses_api_root_not_legacy_v1_base_path():
    server_client = REPO_ROOT / "web/src/lib/api/typed/server.ts"
    env_template = REPO_ROOT / "web/deploy/env.local.template"
    configmap = REPO_ROOT / "web/deploy/yaml/configmap.yaml"

    server_source = server_client.read_text()

    assert "API_SERVER_BASE_PATH" in env_template.read_text()
    assert "API_SERVER_BASE_PATH" in configmap.read_text()
    assert "API_SERVER_BASE_PATH" not in server_source
    assert "process.env.API_SERVER_ENDPOINT || 'http://localhost:8000'" in server_source
    assert "if (!response.ok)" in server_source


def test_api_key_feature_uses_v2_typed_api_boundary():
    """#3 Phase 1a FE typed adapter skeleton + api-key sample migration.

    The user-scoped API key pages (`app/workspace/api-keys/**`) and the new
    `features/api-key/*` adapter must consume the typed OpenAPI client only.
    Any regression to `@/api` legacy SDK or raw `apiClient.defaultApi.apikeys*`
    calls inside this scope means the Phase 1a baseline (api-key moved off
    the FE legacy SDK allowlist) has been unwound.

    Negative assertions are scoped to `app/workspace/api-keys/**` and
    `features/api-key/**` so legitimate uses of the string "api key" in
    unrelated places (e.g. `features/providers` where `api_key` is a
    provider credential field, not a user API key) stay unaffected.
    """
    checked_paths = [
        REPO_ROOT / "web/src/app/workspace/api-keys",
        REPO_ROOT / "web/src/features/api-key",
    ]

    sources = {
        path: path.read_text()
        for root in checked_paths
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".ts", ".tsx"}
    }
    joined = "\n".join(sources.values())
    feature_sources = "\n".join(source for path, source in sources.items() if "/web/src/features/api-key/" in str(path))

    # Negative: the domain scope must not reach the old generated SDK or
    # the v1 ApiKey types imported via `@/api`.
    assert "from '@/api'" not in joined
    assert "apiClient.defaultApi.apikeys" not in joined
    assert "serverApi.defaultApi.apikeys" not in joined
    assert "defaultApi.apikeysGet" not in joined
    assert "defaultApi.apikeysPost" not in joined
    assert "defaultApi.apikeysApikeyIdPut" not in joined
    assert "defaultApi.apikeysApikeyIdDelete" not in joined

    # Positive: the adapter must only reach the typed v1 api-key paths
    # through the typed openapi client. Phase 1a does not rename the API
    # path — that is deferred to a later domain/API phase.
    assert "from '@/api'" not in feature_sources
    assert "fetch(" not in feature_sources
    assert "'/api/v2/apikeys'" in feature_sources
    assert "'/api/v2/apikeys/{apikey_id}'" in feature_sources

    # Positive: the business caller wiring goes through the feature adapter,
    # not the old generated SDK.
    actions_tsx = (REPO_ROOT / "web/src/app/workspace/api-keys/api-key-actions.tsx").read_text()
    assert "from '@/features/api-key/client-api'" in actions_tsx
    assert "from '@/features/api-key/types'" in actions_tsx

    table_tsx = (REPO_ROOT / "web/src/app/workspace/api-keys/api-key-table.tsx").read_text()
    assert "from '@/features/api-key/types'" in table_tsx

    page_tsx = (REPO_ROOT / "web/src/app/workspace/api-keys/page.tsx").read_text()
    assert "from '@/features/api-key/server-api'" in page_tsx
    assert "serverApi.defaultApi.apikeysGet" not in page_tsx


def test_quota_feature_uses_v2_typed_api_boundary():
    """#4 Phase 1b batch 2 — quota domain (workspace scope only).

    The workspace Quotas page and its server caller must reach the typed
    `/api/v2/quotas` endpoint through `features/quota/server-api`, not via
    the legacy `@/api` generated SDK or its indirect `getServerApi` path.
    The `GET /api/v2/quotas` typed response is a union
    `UserQuotaInfo | UserQuotaList` because the endpoint is shared with
    admin list views; the adapter narrows at its boundary so the page
    never has to `as UserQuotaInfo`.

    Scope: `app/workspace/quotas/**` + `features/quota/**`.
    """
    checked_paths = [
        REPO_ROOT / "web/src/app/workspace/quotas",
        REPO_ROOT / "web/src/features/quota",
    ]

    sources = {
        path: path.read_text()
        for root in checked_paths
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".ts", ".tsx"}
    }
    joined = "\n".join(sources.values())
    feature_sources = "\n".join(source for path, source in sources.items() if "/web/src/features/quota/" in str(path))

    # Negative: workspace quota scope must not reach the old generated SDK
    # (direct `@/api` import or indirect `getServerApi`), and must not
    # call `serverApi.quotasApi.quotasGet` / `apiClient.quotasApi.*`.
    assert "from '@/api'" not in joined
    assert "serverApi.quotasApi.quotasGet" not in joined
    assert "apiClient.quotasApi" not in joined
    assert "getServerApi" not in joined

    # Positive: the adapter must only reach the typed v2 quotas path
    # through the typed openapi client.
    assert "from '@/api'" not in feature_sources
    assert "fetch(" not in feature_sources
    assert "'/api/v2/quotas'" in feature_sources

    # Positive: the adapter narrows the union (UserQuotaInfo | UserQuotaList)
    # at the boundary so workspace callers never see an admin list.
    server_api = (REPO_ROOT / "web/src/features/quota/server-api.ts").read_text()
    assert "'items' in data" in server_api

    # Positive: business caller wiring goes through the feature adapter
    # and the page no longer casts `as UserQuotaInfo`.
    page_tsx = (REPO_ROOT / "web/src/app/workspace/quotas/page.tsx").read_text()
    assert "from '@/features/quota/server-api'" in page_tsx
    assert "as UserQuotaInfo" not in page_tsx

    chart_tsx = (REPO_ROOT / "web/src/app/workspace/quotas/quota-radial-chart.tsx").read_text()
    assert "from '@/features/quota/types'" in chart_tsx


def test_marketplace_feature_uses_v2_typed_api_boundary():
    """#4 Phase 1b batch 3 — marketplace domain.

    Migrates only marketplace-owned API callers under `app/marketplace/**`
    to `features/marketplace/*`. Per PM msg=7b4a3b06 / Weston msg=86bfb0cf
    / dongdong msg=9b7cbaad / 架构师 msg=ca06ddd8, two categories of legacy
    residue are **deliberately deferred** and must stay out of this test's
    scope:

    1. `app/marketplace/collections/[collectionId]/documents/{documents-table,
       document-index-status,[documentId]/document-detail}.tsx` are document
       subcomponents that currently hold document-domain legacy types and —
       for `documents-table.tsx` — a mixed `SharedCollection` legacy type.
       These deferred to the document batch or the marketplace/document
       boundary cleanup. They still match `@/api` but are *not* in this
       PR's scope, so the negative assertions below explicitly do not scan
       them.
    2. `app/workspace/collections/[collectionId]/graph/collection-graph.tsx`
       uses `apiClient.defaultApi.marketplaceCollectionsCollectionIdGraphGet`
       but belongs to the workspace graph tree; it is a cross-domain
       graph/marketplace boundary item deferred to #5 or a later graph
       cleanup. The negative assertions below do not scan workspace graph.

    In other words, this test intentionally *does not* try to prove that
    `marketplaceCollections*` is gone from the whole repo — only that the
    six migrated callers and `features/marketplace/**` are clean.
    """
    migrated_callers = [
        REPO_ROOT / "web/src/app/marketplace/page.tsx",
        REPO_ROOT / "web/src/app/marketplace/collection-list.tsx",
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/collection-header.tsx",
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/documents/page.tsx",
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/documents/[documentId]/page.tsx",
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/graph/page.tsx",
    ]

    feature_paths = [REPO_ROOT / "web/src/features/marketplace"]
    feature_sources_by_path: dict[Path, str] = {}
    for root in feature_paths:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".ts", ".tsx"}:
                feature_sources_by_path[path] = path.read_text()

    migrated_sources_by_path = {path: path.read_text() for path in migrated_callers}
    scoped_joined = "\n".join([*migrated_sources_by_path.values(), *feature_sources_by_path.values()])
    feature_sources = "\n".join(feature_sources_by_path.values())

    # Negative: the six migrated callers + features/marketplace/** must not
    # touch the legacy `@/api` SDK directly. Do not scan the deferred
    # document subcomponents or workspace graph here.
    assert "from '@/api'" not in scoped_joined
    assert "apiClient.defaultApi.marketplaceCollections" not in scoped_joined
    assert "serverApi.defaultApi.marketplaceCollections" not in scoped_joined
    assert "defaultApi.marketplaceCollectionsGet" not in scoped_joined
    assert "getServerApi" not in scoped_joined

    # Positive: adapter only reaches the typed v1 marketplace paths.
    assert "fetch(" not in feature_sources
    assert "'/api/v2/marketplace/collections'" in feature_sources
    assert "'/api/v2/marketplace/collections/{collection_id}'" in feature_sources
    assert "'/api/v2/marketplace/collections/{collection_id}/documents'" in feature_sources
    assert "'/api/v2/marketplace/collections/{collection_id}/documents/{document_id}/preview'" in feature_sources
    assert "'/api/v2/marketplace/collections/{collection_id}/subscribe'" in feature_sources

    # Positive: callers wire through the feature adapter.
    page_tsx = migrated_sources_by_path[REPO_ROOT / "web/src/app/marketplace/page.tsx"]
    assert "from '@/features/marketplace/server-api'" in page_tsx

    list_tsx = migrated_sources_by_path[REPO_ROOT / "web/src/app/marketplace/collection-list.tsx"]
    assert "from '@/features/marketplace/types'" in list_tsx

    header_tsx = migrated_sources_by_path[
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/collection-header.tsx"
    ]
    assert "from '@/features/marketplace/client-api'" in header_tsx
    assert "from '@/features/marketplace/types'" in header_tsx

    docs_page_tsx = migrated_sources_by_path[
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/documents/page.tsx"
    ]
    assert "from '@/features/marketplace/server-api'" in docs_page_tsx

    doc_detail_page_tsx = migrated_sources_by_path[
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/documents/[documentId]/page.tsx"
    ]
    assert "from '@/features/marketplace/server-api'" in doc_detail_page_tsx

    graph_page_tsx = migrated_sources_by_path[
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/graph/page.tsx"
    ]
    assert "from '@/features/marketplace/server-api'" in graph_page_tsx


def test_provider_feature_uses_v2_typed_api_boundary():
    """#4 Phase 1b batch 4 — provider domain.

    Migrates `components/providers/bot-provider.tsx` off the legacy
    `apiClient.defaultApi.availableModelsPost` + `ModelSpec from '@/api'` and
    onto the existing `features/providers/*` adapter (`getAvailableModels` +
    `ModelSpec` from `@/features/providers/types`). The `features/providers`
    adapter was migrated in an earlier phase to the typed
    `/api/v2/providers/*` surface, so this batch only swaps the caller and
    does **not** introduce new feature surface, new v2 routes, or change
    fallback semantics.

    Scope: `components/providers/bot-provider.tsx` + `features/providers/**`
    only. Two cross-domain residues are deliberately **deferred** and must
    stay out of scope:

    1. `app/workspace/collections/collection-form.tsx` also calls
       `availableModelsPost` + imports `ModelSpec` via `@/api`, but the same
       file imports `TitleGenerateRequestLanguageEnum` (collection/legacy
       form concern). A partial file migration would still leave the file
       on the legacy-api allowlist, masking the "allowlist row = file is
       legacy-free" invariant. Deferred to the future collection batch.
    2. `components/providers/app-provider.tsx` is auth domain
       (`loginPost/registerPost/logoutPost` + `/api/v2/auth/*`), not
       provider. Deferred to the future identity/auth batch; no
       `features/auth/*` exists yet.
    """
    checked_paths = [
        REPO_ROOT / "web/src/components/providers/bot-provider.tsx",
        REPO_ROOT / "web/src/features/providers",
    ]

    sources: dict[Path, str] = {}
    for entry in checked_paths:
        if entry.is_file():
            sources[entry] = entry.read_text()
            continue
        for path in entry.rglob("*"):
            if path.is_file() and path.suffix in {".ts", ".tsx"}:
                sources[path] = path.read_text()
    joined = "\n".join(sources.values())
    feature_sources = "\n".join(
        source for path, source in sources.items() if "/web/src/features/providers/" in str(path)
    )

    # Negative: the two scoped files must not reach the legacy SDK, the
    # `availableModelsPost` method, or the indirect `getServerApi` route-data
    # path.
    assert "from '@/api'" not in joined
    assert "apiClient.defaultApi.availableModelsPost" not in joined
    assert "serverApi.defaultApi.availableModelsPost" not in joined
    assert "defaultApi.availableModelsPost" not in joined
    assert "getServerApi" not in joined

    # Positive: the feature adapter must only reach v2 provider typed paths
    # and not fall back to the old `@/api` generated SDK or raw fetch.
    assert "from '@/api'" not in feature_sources
    assert "fetch(" not in feature_sources
    assert "'/api/v2/models'" in feature_sources
    assert "'/api/v2/model-providers'" in feature_sources

    # Positive: the migrated caller wires through the feature adapter and
    # imports the domain-scoped `ModelSpec` alias from `features/providers`.
    bot_provider = sources[REPO_ROOT / "web/src/components/providers/bot-provider.tsx"]
    assert "from '@/features/providers/client-api'" in bot_provider
    assert "from '@/features/providers/types'" in bot_provider
    assert "getAvailableModels(" in bot_provider


def test_prompt_feature_uses_v2_typed_api_boundary():
    """#4 Phase 1b batch 1 — prompt domain (FE `features/prompt`, backend
    canonical owner stays `model_platform` per v2 map).

    Same pattern as `test_api_key_feature_uses_v2_typed_api_boundary`:
    the Prompt Settings page and its client/server callers must only reach
    `/api/v2/prompts/user` + `/api/v2/prompts/user/{prompt_type}` through the
    typed openapi-fetch client behind `features/prompt/*`. Phase 8 #49 G3
    carved the route into the `model_platform` domain and hard-cut
    `/api/v1/prompts*` → `/api/v2/prompts*` per D7 canonical; the
    `prompt_type` enum and richer response component remain known gaps
    tracked under the `model_platform` breaking-changes table.

    Negative assertions are scoped to `app/workspace/prompts/**` and
    `features/prompt/**` so legitimate uses of "prompt" vocabulary in
    adjacent places (e.g. `page_prompts` i18n, provider/model prompt fields)
    stay unaffected. The scope also catches the indirect `getServerApi`
    route-data regression.
    """
    checked_paths = [
        REPO_ROOT / "web/src/app/workspace/prompts",
        REPO_ROOT / "web/src/features/prompt",
    ]

    sources = {
        path: path.read_text()
        for root in checked_paths
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".ts", ".tsx"}
    }
    joined = "\n".join(sources.values())
    feature_sources = "\n".join(source for path, source in sources.items() if "/web/src/features/prompt/" in str(path))

    # Negative: the domain scope must not reach the old generated SDK,
    # its indirect `getServerApi` path, or the v1 prompt types imported
    # via `@/api`.
    assert "from '@/api'" not in joined
    assert "apiClient.defaultApi.promptsUser" not in joined
    assert "serverApi.defaultApi.promptsUser" not in joined
    assert "defaultApi.promptsUserGet" not in joined
    assert "defaultApi.promptsUserPut" not in joined
    assert "defaultApi.promptsUserPromptTypeDelete" not in joined
    assert "getServerApi" not in joined

    # Positive: the adapter must only reach the typed v1 prompt paths
    # through the typed openapi client. Phase 1b does not rename the API
    # path — the v1→v2 hard-cut is deferred to the `model_platform` phase.
    assert "from '@/api'" not in feature_sources
    assert "fetch(" not in feature_sources
    assert "'/api/v2/prompts/user'" in feature_sources
    assert "'/api/v2/prompts/user/{prompt_type}'" in feature_sources

    # Positive: the business caller wiring goes through the feature adapter.
    settings_tsx = (REPO_ROOT / "web/src/app/workspace/prompts/prompt-settings.tsx").read_text()
    assert "from '@/features/prompt/client-api'" in settings_tsx
    assert "from '@/features/prompt/types'" in settings_tsx

    page_tsx = (REPO_ROOT / "web/src/app/workspace/prompts/page.tsx").read_text()
    assert "from '@/features/prompt/server-api'" in page_tsx


def test_evaluation_feature_uses_v2_typed_api_boundary():
    """Evaluation v3 simplification: FE must only touch the new
    `/api/v2/evaluation-datasets*` and `/api/v2/evaluation-runs*` surface.

    The old `Benchmark* / Dataset Version / dataset_version_id` concepts were
    dropped in the `#20` v3 simplification. Re-introducing any of them in the
    FE — even in a helper or search predicate — either resurrects a
    user-visible concept we deliberately removed or points the typed client at
    a path that no longer exists.
    """
    checked_paths = [
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]",
        REPO_ROOT / "web/src/app/workspace/bots/[botId]/evaluation",
        REPO_ROOT / "web/src/components/evaluation",
        REPO_ROOT / "web/src/features/evaluation",
    ]

    sources = {
        path: path.read_text()
        for root in checked_paths
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".ts", ".tsx"}
    }
    joined = "\n".join(sources.values())
    feature_sources = "\n".join(
        source for path, source in sources.items() if "/web/src/features/evaluation/" in str(path)
    )

    # Negative: v1 evaluation + old generated SDK must stay out of scope.
    assert "/api/v1/evaluations" not in joined
    assert "/api/v1/question-sets" not in joined
    assert "apiClient.evaluationApi" not in joined
    assert "from '@/api'" not in feature_sources
    assert "fetch(" not in feature_sources

    # Negative: no benchmark / dataset version residue anywhere under
    # evaluation scope. `benchmark-datasets` path, `dataset_version_id`
    # field and the old generated types must all be gone.
    assert "/api/v2/benchmark-datasets" not in joined
    assert "dataset_version_id" not in joined
    assert "BenchmarkDataset" not in joined
    assert "createBenchmarkDatasetVersion" not in joined
    assert "listBenchmarkDatasets" not in joined
    assert "DatasetVersionStatusBadge" not in joined
    assert "page_benchmarks" not in joined
    assert "benchmarksApi" not in joined

    # Positive: features/evaluation must reach both the new dataset and run
    # paths and nothing else.
    assert "/api/v2/evaluation-datasets" in feature_sources
    assert "/api/v2/evaluation-runs" in feature_sources

    # Positive: the new datasets panel search predicate keeps the null-safe
    # cast so `undefined` description or source_type values do not throw.
    datasets_panel = (REPO_ROOT / "web/src/components/evaluation/evaluation-datasets-panel.tsx").read_text()
    assert "String(value ?? '').toLowerCase()" in datasets_panel

    # Positive: Start Evaluation is gated on `dataset.item_count > 0` so a
    # freshly-created empty dataset cannot be run, matching msg=38d7e74d
    # UX patch F.
    collection_runs_panel = (REPO_ROOT / "web/src/components/evaluation/collection-runs-panel.tsx").read_text()
    assert "(dataset.item_count ?? 0) > 0" in collection_runs_panel
    assert "start_run_empty_dataset" in collection_runs_panel
    # Product flow must not expose a bot override in the Collection
    # Evaluation page; backend resolves the default evaluation bot.
    assert "bot_override_" not in collection_runs_panel
    assert "botId" not in collection_runs_panel
    assert "bot_id: startRunForm" not in collection_runs_panel
    # When the default-bot resolver fails the toast text is user-readable
    # (msg=38d7e74d UX patch G), not a raw ValidationException echo with
    # `bot_id` in it.
    assert "start_run_no_bot_error" in collection_runs_panel
    assert "isBotMissingError" in collection_runs_panel


def test_bot_feature_uses_v2_typed_api_boundary():
    """Bot feature adapter boundary — also covers the #4 Phase 1b batch 6
    ("chat type shell") 8-file migration.

    Batch 6 migrates eight `components/chat/*` type-only consumers off the
    legacy `@/api` SDK onto `features/bot/types`. `chat-input.tsx` is
    deferred (it has `apiClient.chatDocumentsApi.*` call-sites for chat-
    document attachment, which crosses the chat↔document domain boundary
    and is out of scope for a pure type-shell migration). `chat-messages.tsx`
    uses a raw `fetch(` against the agent-runtime streaming/artifact
    endpoints — that is an intentional non-SDK path, not a Phase 1b
    regression, so this test does not forbid `fetch(` in the migrated
    business code (only inside `features/bot/**`).
    """
    checked_paths = [
        REPO_ROOT / "web/src/app/workspace/bots",
        REPO_ROOT / "web/src/app/workspace/layout.tsx",
        REPO_ROOT / "web/src/components/providers/bot-provider.tsx",
        REPO_ROOT / "web/src/components/evaluation/evaluation-runs-panel.tsx",
        REPO_ROOT / "web/src/features/bot",
    ]

    sources = {}
    for entry in checked_paths:
        if entry.is_file():
            sources[entry] = entry.read_text()
            continue
        for path in entry.rglob("*"):
            if path.is_file() and path.suffix in {".ts", ".tsx"}:
                sources[path] = path.read_text()
    joined = "\n".join(sources.values())
    feature_sources = "\n".join(source for path, source in sources.items() if "/web/src/features/bot/" in str(path))

    # Business code under these paths must not touch the v1 bot surface or the old
    # generated bots SDK directly.
    assert "/api/v1/bots" not in joined
    assert "defaultApi.botsGet" not in joined
    assert "defaultApi.botsPost" not in joined
    assert "defaultApi.botsBotId" not in joined

    # features/bot adapter must go through the typed v2 paths and not fall back to
    # the old `@/api` SDK or raw fetch.
    assert "from '@/api'" not in feature_sources
    assert "fetch(" not in feature_sources
    assert "/api/v2/bots" in feature_sources

    # Title generate must build a fully-typed TitleGenerateRequest body with
    # runtime-safe concrete defaults. An under-specified `{ language }`-only
    # body would crash backend `chat_title_service.generate_title()` at
    # `max(1, turns)` / `min(max_length, 50)`, so the normaliser must never
    # emit `null` for any of the three required keys.
    bot_client_api = (REPO_ROOT / "web/src/features/bot/client-api.ts").read_text()
    assert "buildTitleGenerateRequest" in bot_client_api
    assert "toTitleLanguage" in bot_client_api
    assert "DEFAULT_TITLE_MAX_LENGTH = 20" in bot_client_api
    assert "DEFAULT_TITLE_TURNS = 1" in bot_client_api
    assert "DEFAULT_TITLE_LANGUAGE: TitleLanguage = 'zh-CN'" in bot_client_api
    assert "max_length: input.max_length ?? DEFAULT_TITLE_MAX_LENGTH" in bot_client_api
    assert "turns: input.turns ?? DEFAULT_TITLE_TURNS" in bot_client_api
    # Guard the negative case: no `?? null` fallback for any of the three
    # required keys should reappear.
    assert "max_length: input.max_length ?? null" not in bot_client_api
    assert "turns: input.turns ?? null" not in bot_client_api

    # #4 Phase 1b batch 6 — chat type shell migration. The eight type-only
    # consumers below must only reach `features/bot/types` for chat-domain
    # types. The legacy `@/api` SDK and `FeedbackTag{Enum,TypeEnum}` enum
    # classes are banned across this scope. The `chat-input.tsx` caller is
    # **deliberately out of scope** (deferred to the chat/document boundary
    # batch).
    # Phase 8 D8.4b (#77) replaced `agent-turn-card.tsx` with
    # `agent-turn-renderer.tsx`; the contract guard moves with it.
    batch6_chat_paths = [
        REPO_ROOT / "web/src/components/chat/agent-turn-renderer.tsx",
        REPO_ROOT / "web/src/components/chat/chat-messages.tsx",
        REPO_ROOT / "web/src/components/chat/message-feedback.tsx",
        REPO_ROOT / "web/src/components/chat/message-part-ai.tsx",
        REPO_ROOT / "web/src/components/chat/message-parts-ai.tsx",
        REPO_ROOT / "web/src/components/chat/message-parts-user.tsx",
        REPO_ROOT / "web/src/components/chat/message-reference.tsx",
        REPO_ROOT / "web/src/components/chat/message-timestamp.tsx",
    ]
    batch6_sources = {path: path.read_text() for path in batch6_chat_paths}
    batch6_joined = "\n".join(batch6_sources.values())

    # Negative: the 8 migrated chat shell files must not reach the legacy
    # SDK or the legacy enum classes. Note that `fetch(` is allowed here
    # (chat-messages.tsx uses it for agent-runtime streaming, not for the
    # legacy SDK).
    assert "from '@/api'" not in batch6_joined
    assert "FeedbackTagEnum" not in batch6_joined
    assert "FeedbackTypeEnum" not in batch6_joined
    assert "apiClient.defaultApi" not in batch6_joined
    assert "serverApi.defaultApi" not in batch6_joined
    assert "apiClient.chatDocumentsApi" not in batch6_joined
    assert "getServerApi" not in batch6_joined

    # Positive: every migrated chat-shell file now imports from
    # `features/bot/types`; the feedback form uses the derived
    # `FEEDBACK_TAGS` const + the `FeedbackType` alias instead of the old
    # enum classes.
    for source in batch6_sources.values():
        assert "from '@/features/bot/types'" in source
    message_feedback = batch6_sources[REPO_ROOT / "web/src/components/chat/message-feedback.tsx"]
    assert "FEEDBACK_TAGS" in message_feedback
    assert "FeedbackType" in message_feedback

    # Positive: features/bot/types exposes the schema-derived chat shell
    # aliases + the runtime `FEEDBACK_TAGS` const (constrained by
    # `satisfies readonly FeedbackTag[]`), matching the locked gate.
    #
    # ``ChatMessage`` and ``Reference`` were originally schema-derived
    # but the underlying OpenAPI components were removed/refactored in
    # Wave 8 D8.5+ (history → ``AgentTurnSnapshot``; reference →
    # ``DataCitationPart`` / ``SourceDocumentPart`` / ``SourceUrlPart``).
    # PR #1774 swept the broken ``components['schemas'][...]`` lookups
    # and replaced them with FE-local types pending a proper renderer
    # migration (TODO W9-3). Pin the FE-local shape so a future
    # contributor doesn't silently re-introduce the broken schema
    # lookup.
    bot_types = (REPO_ROOT / "web/src/features/bot/types.ts").read_text()
    assert "components['schemas']['Feedback']" in bot_types
    assert "NonNullable<Feedback['tag']>" in bot_types
    assert "NonNullable<Feedback['type']>" in bot_types
    assert "satisfies readonly FeedbackTag[]" in bot_types
    assert "export type ChatMessage = {" in bot_types
    assert "export type Reference = {" in bot_types


def test_collection_feature_uses_v2_typed_api_boundary():
    """Collection + Sharing FE typed adapter boundary.

    Extended in #4 Phase 1b batch 5 ("Option B — core + export") to also
    cover the collection-form / collection-list / collection-header /
    export-dialog callers and the export-task adapter. Two residues are
    deliberately **deferred** and must stay out of scope:

    - `app/workspace/collections/feature-visibility.ts` imports
      `RebuildIndexesRequestIndexTypesEnum` / `SearchResultItem*` from
      `@/api` — indexing/retrieval domain, not collection — deferred to
      the document/indexing batch.
    - `app/workspace/collections/tools.ts` imports `DocumentStatusEnum`
      from `@/api` — document domain — deferred to the document batch.
    - `[collectionId]/documents/**`, `graph/**`, `graph-showcase/**`,
      `search/**` subtrees stay in their own domain batches.

    Business code under these paths must not call the old generated
    `defaultApi.collections*` / `defaultApi.collectionsCollectionId*` /
    `defaultApi.collectionsCollectionIdSharing*` /
    `defaultApi.availableModelsPost` / `defaultApi.createExportTask` /
    `defaultApi.getExportTask` SDK directly, and the
    `features/collection/*` adapter must only reach
    `/api/v2/collections*` typed paths (plus the still-v1 export-task
    paths) without a `@/api` fallback or raw `fetch(`.
    """
    checked_paths = [
        REPO_ROOT / "web/src/app/workspace/collections/page.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/collection-form.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/collection-list.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/layout.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/collection-delete.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/collection-header.tsx",
        REPO_ROOT / "web/src/components/collections/export-dialog.tsx",
        REPO_ROOT / "web/src/components/providers/collection-provider.tsx",
        REPO_ROOT / "web/src/components/providers/bot-provider.tsx",
        REPO_ROOT / "web/src/features/collection",
    ]

    sources = {}
    for entry in checked_paths:
        if entry.is_file():
            sources[entry] = entry.read_text()
            continue
        for path in entry.rglob("*"):
            if path.is_file() and path.suffix in {".ts", ".tsx"}:
                sources[path] = path.read_text()
    joined = "\n".join(sources.values())
    feature_sources = "\n".join(
        source for path, source in sources.items() if "/web/src/features/collection/" in str(path)
    )

    # Business code under scope must not reach the old generated
    # collection SDK or the v1 collections routes directly.
    assert "defaultApi.collectionsGet" not in joined
    assert "defaultApi.collectionsPost" not in joined
    assert "defaultApi.collectionsCollectionIdGet" not in joined
    assert "defaultApi.collectionsCollectionIdPut" not in joined
    assert "defaultApi.collectionsCollectionIdDelete" not in joined
    assert "defaultApi.collectionsCollectionIdSharingGet" not in joined
    assert "defaultApi.collectionsCollectionIdSharingPost" not in joined
    assert "defaultApi.collectionsCollectionIdSharingDelete" not in joined

    # #4 batch 5 additions: the collection-form provider call and the
    # export-dialog task calls must be gone from the four migrated
    # callers. `@/api` is banned across the migrated call-sites; the
    # `features/providers::getAvailableModels` cross-domain adapter call
    # is allowed and is not what these negative assertions target.
    form_tsx = sources[REPO_ROOT / "web/src/app/workspace/collections/collection-form.tsx"]
    list_tsx = sources[REPO_ROOT / "web/src/app/workspace/collections/collection-list.tsx"]
    header_tsx = sources[REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/collection-header.tsx"]
    export_dialog_tsx = sources[REPO_ROOT / "web/src/components/collections/export-dialog.tsx"]
    migrated_joined = "\n".join([form_tsx, list_tsx, header_tsx, export_dialog_tsx])
    assert "from '@/api'" not in migrated_joined
    assert "apiClient.defaultApi.availableModelsPost" not in migrated_joined
    assert "apiClient.defaultApi.createExportTask" not in migrated_joined
    assert "apiClient.defaultApi.getExportTask" not in migrated_joined
    assert "defaultApi.createExportTask" not in migrated_joined
    assert "defaultApi.getExportTask" not in migrated_joined

    # Positive: migrated callers wire through the feature adapter(s).
    assert "from '@/features/providers/client-api'" in form_tsx
    assert "from '@/features/collection/types'" in form_tsx
    assert "from '@/features/collection/types'" in list_tsx
    assert "from '@/features/collection/types'" in header_tsx
    assert "from '@/features/collection/client-api'" in export_dialog_tsx
    assert "from '@/features/collection/types'" in export_dialog_tsx

    # features/collection adapter must only reach v2 typed paths and not
    # fall back to the old `@/api` generated SDK or raw fetch. Export
    # endpoints were migrated v1→v2 by Phase 8 #47 G1 (D7 hard-cut).
    assert "from '@/api'" not in feature_sources
    assert "fetch(" not in feature_sources
    assert "/api/v2/collections" in feature_sources
    assert "'/api/v2/collections/{collection_id}/export'" in feature_sources
    assert "'/api/v2/export-tasks/{task_id}'" in feature_sources

    # Positive: schema-derived types + fail-fast export-task adapter.
    types_ts = (REPO_ROOT / "web/src/features/collection/types.ts").read_text()
    assert "NonNullable<CollectionView['status']>" in types_ts
    assert "NonNullable<\n  components['schemas']['TitleGenerateRequest']['language']\n>" in types_ts
    assert "satisfies readonly TitleLanguage[]" in types_ts
    assert "components['schemas']['ExportTaskResponse']" in types_ts

    client_api_ts = (REPO_ROOT / "web/src/features/collection/client-api.ts").read_text()
    # `createExportTask` / `getExportTask` return typed schema components
    # with `export_task_id` + `status` as required fields; an empty body
    # means the backend broke its contract, so the adapter must throw
    # instead of returning a typed-shape-missing `{}` (new server/client
    # adapter gate, #1612 / #1611 follow-up).
    assert "createExportTask: empty response body" in client_api_ts
    assert "getExportTask: empty response body" in client_api_ts


def test_document_feature_uses_v2_typed_api_boundary():
    """#24b / #28 Document FE typed adapter boundary.

    Document list/detail/delete/rebuild/download/object/preview call sites
    must not reach the old generated ``defaultApi.collectionsCollectionIdDocuments*``
    SDK or the v1 document path, and the ``features/document/*`` adapter must
    only reach ``/api/v2/collections/{collection_id}/documents*`` typed paths
    (no ``@/api`` fallback, no raw ``fetch(``). Upload/confirm/fetch-url stay
    out of scope — they remain with the upload-flow hotfix and will be
    migrated in a later slice.
    """
    workspace_documents = REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents"

    document_scope_paths = [
        workspace_documents / "page.tsx",
        workspace_documents / "documents-table.tsx",
        workspace_documents / "document-delete.tsx",
        workspace_documents / "document-rebuild-index.tsx",
        workspace_documents / "document-rebuild-failed-index.tsx",
        workspace_documents / "[documentId]",
        REPO_ROOT / "web/src/features/document",
    ]

    sources = {}
    for entry in document_scope_paths:
        if entry.is_file():
            sources[entry] = entry.read_text()
            continue
        for path in entry.rglob("*"):
            if path.is_file() and path.suffix in {".ts", ".tsx"}:
                sources[path] = path.read_text()

    # Upload-flow files under documents/upload/** are intentionally excluded
    # from the #28 scope; drop any that slipped in via rglob.
    sources = {path: text for path, text in sources.items() if "/documents/upload/" not in str(path)}

    joined = "\n".join(sources.values())
    feature_sources = "\n".join(text for path, text in sources.items() if "/web/src/features/document/" in str(path))
    business_sources = "\n".join(
        text for path, text in sources.items() if "/web/src/features/document/" not in str(path)
    )

    # Business code under #28 scope must not reach the old generated document
    # SDK calls or the v1 document routes directly. The `features/document`
    # adapter wraps the upload-flow paths (Phase 1b batch 8 originally
    # typed-wrapped them at v1; Phase 8 #63 G5a flipped them to
    # ``/api/v2/collections/.../documents/{staged,upload,confirm,fetch-url}``).
    # The v1-path ban here targets business code only, not the adapter.
    assert "defaultApi.collectionsCollectionIdDocumentsGet" not in joined
    assert "defaultApi.collectionsCollectionIdDocumentsDocumentIdGet" not in joined
    assert "defaultApi.collectionsCollectionIdDocumentsDocumentIdDelete" not in joined
    assert "defaultApi.collectionsCollectionIdDocumentsDocumentIdRebuildIndexesPost" not in joined
    assert "defaultApi.collectionsCollectionIdRebuildFailedIndexesPost" not in joined
    assert "serverApi.defaultApi.getDocumentPreview" not in joined
    assert "/api/v1/collections/" not in business_sources

    # features/document adapter must only reach v2 typed paths (or the
    # still-v1 upload-flow paths wrapped by batch 8) and not fall back to
    # the old `@/api` generated SDK or raw fetch.
    assert "from '@/api'" not in feature_sources
    assert "fetch(" not in feature_sources
    assert "/api/v2/collections/{collection_id}/documents" in feature_sources

    # The PDF preview viewer must use the v2 object URL helper rather than
    # hand-building a v1 path.
    document_detail = (workspace_documents / "[documentId]/document-detail.tsx").read_text()
    assert "buildDocumentObjectUrl" in document_detail

    # #4 Phase 1b batch 7 — document type shell migration. Seven type-only
    # callers must drop `@/api` in favour of `features/document/types` (plus
    # a `SharedCollection` type import from `features/marketplace/types` for
    # the marketplace documents-table). The feature-visibility.ts carryover
    # is **deliberately excluded** because it mixes indexing and retrieval
    # enums and is deferred to the retrieval/search batch; the upload-flow
    # trio (`document-upload.tsx`, `url-import.tsx`, `text-import.tsx`)
    # remains with the future upload batch.
    batch7_document_paths = [
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/documents/[documentId]/document-detail.tsx",
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/documents/document-index-status.tsx",
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/documents/documents-table.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/document-index-status.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/document-rebuild-index.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/documents-table.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/tools.ts",
    ]
    batch7_sources = {path: path.read_text() for path in batch7_document_paths}
    batch7_joined = "\n".join(batch7_sources.values())

    # Negative: the 7 migrated type-only files must not reach the legacy
    # `@/api` SDK or the legacy enum classes that this batch replaces with
    # schema-derived aliases. `apiClient`/`defaultApi`/`serverApi`/`fetch(`
    # are already banned transitively via the #28 assertions above plus the
    # file-level lint; this block is the scoped `@/api` + enum check.
    assert "from '@/api'" not in batch7_joined
    assert "DocumentStatusEnum" not in batch7_joined
    assert "DocumentVectorIndexStatusEnum" not in batch7_joined
    assert "RebuildIndexesRequestIndexTypesEnum" not in batch7_joined

    # Positive: every migrated file now imports from `features/document/types`;
    # the marketplace documents-table additionally pulls its SharedCollection
    # type from `features/marketplace/types` instead of the legacy SDK.
    for source in batch7_sources.values():
        assert "from '@/features/document/types'" in source
    marketplace_documents_table = batch7_sources[
        REPO_ROOT / "web/src/app/marketplace/collections/[collectionId]/documents/documents-table.tsx"
    ]
    assert "from '@/features/marketplace/types'" in marketplace_documents_table

    # Positive: `features/document/types.ts` exposes the schema-derived
    # aliases + the runtime `DOCUMENT_INDEX_TYPES` const (constrained by
    # `satisfies readonly DocumentIndexType[]`). This is the locked gate
    # for batch 7; a regression here would mean a handwritten enum or
    # union slipped back in.
    document_types = (REPO_ROOT / "web/src/features/document/types.ts").read_text()
    assert "NonNullable<Document['status']>" in document_types
    assert "NonNullable<Document['vector_index_status']>" in document_types
    assert "RebuildIndexesRequest['index_types'][number]" in document_types
    assert "satisfies readonly DocumentIndexType[]" in document_types

    # #4 Phase 1b batch 8 — document upload trio. Three upload/import
    # callers (`document-upload.tsx`, `url-import.tsx`, `text-import.tsx`)
    # migrate off `apiClient.defaultApi.collectionsCollectionIdDocuments{
    # StagedGet,UploadPost,ConfirmPost,FetchUrlPost}` onto new
    # `features/document/client-api` methods (`listStagedDocuments`,
    # `uploadDocument`, `confirmDocuments`, `fetchUrlDocuments`).
    # `text-import.tsx` was only on the route-data allowlist (no `@/api`
    # type imports) but still hits the upload endpoint directly; this
    # batch removes its raw SDK call so it drops from route-data too.
    batch8_upload_paths = [
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/upload/import/url-import.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/upload/import/text-import.tsx",
    ]
    batch8_sources = {path: path.read_text() for path in batch8_upload_paths}
    batch8_joined = "\n".join(batch8_sources.values())

    # Negative: the three migrated callers must not reach the legacy SDK
    # or the legacy enum classes that this batch replaces with
    # schema-derived aliases.
    assert "from '@/api'" not in batch8_joined
    assert "apiClient.defaultApi" not in batch8_joined
    assert "serverApi.defaultApi" not in batch8_joined
    assert "UploadDocumentResponseStatusEnum" not in batch8_joined
    assert "FetchUrlResultItemFetchStatusEnum" not in batch8_joined
    assert "collectionsCollectionIdDocumentsStagedGet" not in batch8_joined
    assert "collectionsCollectionIdDocumentsUploadPost" not in batch8_joined
    assert "collectionsCollectionIdDocumentsConfirmPost" not in batch8_joined
    assert "collectionsCollectionIdDocumentsFetchUrlPost" not in batch8_joined

    # Positive: every migrated caller now imports from
    # `features/document/client-api`; the `document-upload.tsx` caller
    # additionally uses the `UploadDocumentStatus` type alias instead of
    # the legacy enum class.
    for source in batch8_sources.values():
        assert "from '@/features/document/client-api'" in source
    document_upload_tsx = batch8_sources[
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx"
    ]
    assert "UploadDocumentStatus" in document_upload_tsx

    # Positive: `features/document/client-api.ts` exposes the four upload
    # adapters with empty-body throw guards (lesson 9a: typed components
    # with required fields must not accept `data ?? {}`). The multipart
    # upload explicitly uses a `FormData` `bodySerializer` to get the
    # right `Content-Type: multipart/form-data; boundary=...` header from
    # `fetch`.
    document_client_api = (REPO_ROOT / "web/src/features/document/client-api.ts").read_text()
    assert "listStagedDocuments" in document_client_api
    assert "uploadDocument" in document_client_api
    assert "confirmDocuments" in document_client_api
    assert "fetchUrlDocuments" in document_client_api
    assert "listStagedDocuments: empty response body" in document_client_api
    assert "uploadDocument: empty response body" in document_client_api
    assert "confirmDocuments: empty response body" in document_client_api
    assert "fetchUrlDocuments: empty response body" in document_client_api
    assert "bodySerializer" in document_client_api
    assert "new FormData()" in document_client_api
    assert "'/api/v2/collections/{collection_id}/documents/staged'" in document_client_api
    assert "'/api/v2/collections/{collection_id}/documents/upload'" in document_client_api
    assert "'/api/v2/collections/{collection_id}/documents/confirm'" in document_client_api
    assert "'/api/v2/collections/{collection_id}/documents/fetch-url'" in document_client_api

    # Positive: `features/document/types.ts` exposes the upload schema
    # aliases (required-shape components) + the schema-derived nullable
    # status unions.
    assert "components['schemas']['UploadDocumentResponse']" in document_types
    assert "components['schemas']['StagedDocumentsResponse']" in document_types
    assert "components['schemas']['ConfirmDocumentsRequest']" in document_types
    assert "components['schemas']['FetchUrlResponse']" in document_types
    assert "UploadDocumentResponse['status']" in document_types
    assert "FetchUrlResultItem['fetch_status']" in document_types


def test_documents_upload_regression_guards():
    """Regression guards for #前端 #16 document upload UX fixes.

    1. `document-upload.tsx` must not auto-abort the bulk upload on unmount.
       The old `useEffect(() => () => stopUpload(), [stopUpload])` killed
       in-flight uploads when the user navigated away in the same tab,
       making the uploader appear to silently stop.
    2. `documents-table.tsx` must handle TanStack Table's
       `onPaginationChange` updater as both a value and a function. The
       old `@ts-expect-error` variant threw at runtime whenever TanStack
       dispatched a plain value, reverting the visible page to page 1
       after a click.
    """
    upload_tsx = (
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx"
    ).read_text()
    assert "() => stopUpload()" not in upload_tsx
    assert "() => () => stopUpload" not in upload_tsx

    table_tsx = (
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/documents-table.tsx"
    ).read_text()
    assert "@ts-expect-error onPaginationChange" not in table_tsx
    assert "typeof updater === 'function'" in table_tsx


V1_PATHS_REMOVED_IN_FINAL_SWEEP = frozenset(
    {
        # All bot v1 routes (bot_router deleted; #25 FE migrated; #1579)
        "/api/v1/bots",
        "/api/v1/bots/{bot_id}",
        # v1 bot-scoped chat shell (chat_router trimmed; replaced by bots_v2
        # bot-scoped chat routes in #1577 + #25 FE adapter)
        "/api/v1/bots/{bot_id}/chats",
        "/api/v1/bots/{bot_id}/chats/{chat_id}",
        "/api/v1/bots/{bot_id}/chats/{chat_id}/title",
        # v1 collection CRUD / sharing / summary / mineru (collections_router
        # trimmed; replaced by collections_v2 in #1573 + #24a FE in #1581)
        "/api/v1/collections",
        "/api/v1/collections/test-mineru-token",
        "/api/v1/collections/{collection_id}",
        "/api/v1/collections/{collection_id}/sharing",
        "/api/v1/collections/{collection_id}/summary/generate",
        # v1 documents read-side (replaced by documents_v2 in #1575 + #24b FE
        # in #1585; upload/confirm/fetch-url/staged deliberately kept for
        # the upload flow slice)
        "/api/v1/collections/{collection_id}/documents",
        "/api/v1/collections/{collection_id}/documents/{document_id}",
        "/api/v1/collections/{collection_id}/documents/{document_id}/download",
        "/api/v1/collections/{collection_id}/documents/{document_id}/object",
        "/api/v1/collections/{collection_id}/documents/{document_id}/preview",
        "/api/v1/collections/{collection_id}/documents/{document_id}/rebuild_indexes",
        "/api/v1/collections/{collection_id}/rebuild_failed_indexes",
    }
)


def test_final_sweep_v1_ghost_paths_are_gone_from_exported_spec():
    """The `#26` final sweep removes every v1 route that has a v2 replacement
    and is no longer consumed by FE business code. Re-introducing any of
    them would either resurrect a dead handler or break the hard-cut
    promise; fail the test instead of silently rolling back.

    The preserved v1 routes (upload / confirm / fetch-url / staged /
    searches / graphs / feedback / chat documents / marketplace / admin
    / settings / auth / api-keys / prompts / export / llm / config / web)
    are deliberately not in this set — they still have FE consumers or
    no v2 replacement yet.
    """
    from aperag.app import app
    from aperag.openapi_spec import build_full_openapi_spec, filter_public_openapi

    full_spec = build_full_openapi_spec(app)
    public_spec = filter_public_openapi(full_spec)

    for spec_name, spec in (("full", full_spec), ("public", public_spec)):
        present = set(spec["paths"]) & V1_PATHS_REMOVED_IN_FINAL_SWEEP
        assert not present, (
            f"#26 final sweep regression: {spec_name} OpenAPI spec still "
            f"contains removed v1 path(s) {sorted(present)}. If this was "
            "intentional, update V1_PATHS_REMOVED_IN_FINAL_SWEEP here and "
            "explain in the PR."
        )


def test_e2e_http_does_not_call_removed_v1_paths():
    """HTTP e2e assets must not call routes removed by the final v1 sweep.

    Search, graph, and upload-flow v1 routes are intentionally still allowed
    because they were explicitly preserved in #26. This guard only blocks the
    deleted bot shell, collection CRUD, and document read-side paths that broke
    post-merge e2e smoke when Hurl still called `/api/v1/collections`.
    """
    e2e_root = REPO_ROOT / "tests/e2e_http"
    sources = {
        path: path.read_text() for path in e2e_root.rglob("*") if path.is_file() and path.suffix in {".hurl", ".sh"}
    }

    removed_route_patterns = {
        r"/api/v1/bots(?:[/?\"\s]|\$\{)": "bot v1 routes were replaced by /api/v2/bots*",
        r"/api/v1/collections(?:[?\"\s]|\Z)": "collection list/create must use /api/v2/collections",
        r"/api/v1/collections/(?:\{\{collection_id\}\}|\$\{collection_id\})(?:[?\"\s]|\Z)": (
            "collection get/update/delete must use /api/v2/collections/{collection_id}"
        ),
        r"/api/v1/collections/(?:\{\{collection_id\}\}|\$\{collection_id\})/documents(?:[?\"\s]|\Z)": (
            "document list/create must use /api/v2/collections/{collection_id}/documents"
        ),
        r"/api/v1/collections/(?:\{\{collection_id\}\}|\$\{collection_id\})/documents/"
        r"(?:\{\{document_id\}\}|\$\{document_id\})(?:[/?\"\s]|\Z)": (
            "document detail/delete/download/preview/object/rebuild must use v2 document paths"
        ),
        r"/api/v1/collections/(?:\{\{collection_id\}\}|\$\{collection_id\})/rebuild_failed_indexes": (
            "rebuild_failed_indexes must use /api/v2/collections/{collection_id}/rebuild_failed_indexes"
        ),
    }

    offenders: list[str] = []
    for path, source in sources.items():
        for pattern, reason in removed_route_patterns.items():
            if re.search(pattern, source):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {reason}")

    assert not offenders, "e2e HTTP assets still call removed v1 route(s):\n" + "\n".join(sorted(offenders))


def test_e2e_http_uses_simplified_evaluation_v2_contract():
    """The full evaluation Hurl suite must follow the post-#1590 API.

    ``/api/v2/benchmark-datasets*``, dataset versions, and
    ``dataset_version_id`` were removed from the public evaluation contract.
    Keep this static guard near the e2e assets so a backend contract switch
    cannot leave provider/full Hurl coverage on dead routes again.
    """
    e2e_root = REPO_ROOT / "tests/e2e_http"
    sources = {
        path: path.read_text() for path in e2e_root.rglob("*") if path.is_file() and path.suffix in {".hurl", ".sh"}
    }

    forbidden = {
        "/api/v2/benchmark-datasets": "use /api/v2/evaluation-datasets instead",
        "/versions": "dataset version routes were removed from evaluation v2",
        "dataset_version_id": "runs now use dataset_id and snapshot dataset items on creation",
    }

    offenders: list[str] = []
    for path, source in sources.items():
        for token, reason in forbidden.items():
            if token in source:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token} ({reason})")

    assert not offenders, "e2e HTTP assets still use retired evaluation v2 contract token(s):\n" + "\n".join(
        sorted(offenders)
    )


def test_documents_upload_regression_hardening():
    """Stronger #30 regression guards layered on top of the minimal
    `test_documents_upload_regression_guards`. The hotfix in #1580 left
    two runtime-sensitive choices that a future unrelated edit could
    quietly flip back:

    1. `document-upload.tsx` must still expose a **manual** stop path for
       the user (the "Stop" button). The hotfix only removed the *unmount
       cleanup* auto-abort; it did not remove user-initiated abort. A
       refactor that collapses the callback would make the uploader
       unstoppable.
    2. `documents-table.tsx` must keep the fully normalised
       `onPaginationChange` handler: read `current`, call
       `updater(current)` only when `updater` is a function, otherwise
       use the value directly, and dispatch `next.pageIndex + 1` /
       `next.pageSize` through the URL search-params helper. Without
       the normalisation the handler either crashes on value-form
       updaters (snapping back to page 1) or silently discards pageSize
       changes.

    Any future edit that drops these guarantees fails these positive
    assertions, so the reviewer has to consciously update the test
    baseline — that acknowledgement is the point.
    """

    upload_tsx = (
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx"
    ).read_text()

    # Manual-stop affordance must remain: a `stopUpload` callback and a
    # user-facing `onClick={stopUpload}` wiring in the JSX. Combined with
    # the negative `() => stopUpload()` / `() => () => stopUpload` asserts
    # above, this pins "user can still stop; component unmount cannot".
    assert "const stopUpload = useCallback(" in upload_tsx, (
        "stopUpload callback must remain for the manual Stop button path"
    )
    assert "onClick={stopUpload}" in upload_tsx, (
        "The Stop button must continue to wire up stopUpload so users can still abort a bulk upload on intent"
    )
    # Defensive: no `useEffect` anywhere in the file may dispatch
    # `stopUpload` from its cleanup. This catches both the original
    # one-liner and any refactor that splits the cleanup body across
    # several lines before calling `stopUpload()`.
    cleanup_call_pattern = re.compile(
        r"useEffect\s*\([\s\S]*?=>\s*\(\s*\)\s*=>\s*[\s\S]*?stopUpload\s*\(\s*\)",
        re.MULTILINE,
    )
    assert cleanup_call_pattern.search(upload_tsx) is None, (
        "document-upload.tsx must not call stopUpload() from any "
        "useEffect cleanup — same-tab navigation would abort uploads"
    )

    table_tsx = (
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/documents/documents-table.tsx"
    ).read_text()

    # Positive: the normalised updater handler body is intact.
    assert "onPaginationChange: (updater)" in table_tsx, (
        "onPaginationChange must accept `updater` explicitly so the "
        "function-vs-value branch is visible at the call site"
    )
    assert "typeof updater === 'function' ? updater(current) : updater" in (table_tsx), (
        "onPaginationChange must normalise both Updater shapes before reading pageIndex/pageSize"
    )
    assert "next.pageIndex" in table_tsx and "next.pageSize" in table_tsx, (
        "pagination click must dispatch the normalised next.pageIndex / next.pageSize, not the stale `current` snapshot"
    )

    # Negative: the old `fn({...}).pageIndex` / `fn({...}).pageSize`
    # shape must not reappear — that was the broken form that assumed
    # `updater` was always a function.
    assert "fn({" not in table_tsx, (
        "documents-table.tsx must not reuse the old `fn({...})` updater "
        "pattern; use the normalised `updater(current)` branch instead"
    )
    assert "// @ts-expect-error" not in table_tsx, (
        "documents-table.tsx must not silence TanStack's Updater typing with `@ts-expect-error`"
    )


def test_phase1_fe_complete_identity_auth_admin_audit_adapter_boundary():
    """#13 Phase 1 FE complete — identity / auth / admin / audit adapter
    boundary. All remaining ``@/api`` callers (auth + identity + admin +
    audit + chat-input + feature-visibility residual) migrate onto
    ``features/<d>/{client,server}-api``; the legacy ``web/src/api/`` tree
    and the low-level ``lib/api/{client,server}.ts`` wrappers are
    deleted.

    Counting canonical (post-#12 merge → post-#13): ``legacy 14 → 0 /
    raw_schema 13 → 16 / route_data 9 → 0``. The raw_schema +3 adds
    identity / auth / admin typed adapters; audit is a hand-written
    mirror (lesson 9a-ter boundary exception for hidden-path endpoints)
    and does *not* import ``@/api-v2/schema``. Phase 4 governance
    (msg=659a98da decisions R + S) unhides audit + config and promotes
    both to typed wrappers.
    """
    # identity — User canonical source is `features/identity/types`.
    identity_types = (REPO_ROOT / "web/src/features/identity/types.ts").read_text()
    assert "components['schemas']['User']" in identity_types
    assert "from '@/api-v2/schema'" in identity_types

    # auth — typed `login` / `register` / `logout` + raw-fetch OAuth
    # authorize (redirect, no typed body) + raw-fetch `/api/v2/config`
    # (hidden from public OpenAPI; lesson 9a-ter boundary exception).
    auth_client_api = (REPO_ROOT / "web/src/features/auth/client-api.ts").read_text()
    auth_server_api = (REPO_ROOT / "web/src/features/auth/server-api.ts").read_text()
    assert "from '@/lib/api/typed/browser'" in auth_client_api
    assert "from '@/lib/api/typed/server'" in auth_server_api
    assert "oauthAuthorize" in auth_client_api
    assert "getAuthConfig" in auth_server_api
    assert "/api/v2/config" in auth_server_api

    # admin — umbrella adapter for settings / system default quotas /
    # per-user quota / admin user list (msg=5f0a370b decision L).
    admin_client_api = (REPO_ROOT / "web/src/features/admin/client-api.ts").read_text()
    admin_server_api = (REPO_ROOT / "web/src/features/admin/server-api.ts").read_text()
    for method in (
        "testMineruToken",
        "updateSettings",
        "getSystemDefaultQuotas",
        "updateSystemDefaultQuotas",
        "getUserQuota",
        "updateUserQuota",
        "recalculateUserQuota",
    ):
        assert method in admin_client_api, f"features/admin/client-api.ts missing `{method}`"
    for method in ("getSettings", "getSystemDefaultQuotas", "listUsers"):
        assert method in admin_server_api, f"features/admin/server-api.ts missing `{method}`"
    assert "/api/v2/system/default-quotas" in admin_client_api
    assert "/api/v2/system/default-quotas" in admin_server_api
    assert "/api/v2/quotas" in admin_client_api
    assert "/api/v2/quotas/{user_id}" in admin_client_api
    assert "/api/v2/quotas/{user_id}/recalculate" in admin_client_api
    assert "/api/v1/system/default-quotas" not in admin_client_api
    assert "/api/v1/system/default-quotas" not in admin_server_api
    assert "/api/v1/quotas" not in admin_client_api

    # audit — hand-written mirror + raw fetch (hidden path
    # `/api/v2/audit-logs`). `features/audit/types.ts` must NOT import
    # raw schema; its header must self-document the Phase 4 +1
    # raw_schema impact so future readers don't hunt for history.
    audit_types = (REPO_ROOT / "web/src/features/audit/types.ts").read_text()
    audit_client_api = (REPO_ROOT / "web/src/features/audit/client-api.ts").read_text()
    audit_server_api = (REPO_ROOT / "web/src/features/audit/server-api.ts").read_text()
    assert "from '@/api-v2/schema'" not in audit_types
    assert "raw_schema 13 → 16" in audit_types
    assert "/api/v2/audit-logs" in audit_client_api
    assert "/api/v2/audit-logs" in audit_server_api
    assert "/api/v1/audit-logs" not in audit_client_api
    assert "/api/v1/audit-logs" not in audit_server_api

    # chat-input — #13 Option A thin caller migrates off
    # `apiClient.chatDocumentsApi.*` onto `features/bot/client-api`.
    chat_input = (REPO_ROOT / "web/src/components/chat/chat-input.tsx").read_text()
    assert "from '@/api'" not in chat_input
    assert "apiClient.chatDocumentsApi" not in chat_input
    assert "from '@/features/bot/client-api'" in chat_input
    bot_client_api = (REPO_ROOT / "web/src/features/bot/client-api.ts").read_text()
    assert "uploadChatDocument" in bot_client_api
    assert "getChatDocument" in bot_client_api
    assert "'/api/v2/chats/{chat_id}/documents'" in bot_client_api

    # feature-visibility.ts — document residual swaps
    # `RebuildIndexesRequestIndexTypesEnum` → `DOCUMENT_INDEX_TYPES`.
    feature_visibility = (REPO_ROOT / "web/src/app/workspace/collections/feature-visibility.ts").read_text()
    assert "from '@/api'" not in feature_visibility
    assert "RebuildIndexesRequestIndexTypesEnum" not in feature_visibility
    assert "from '@/features/document/types'" in feature_visibility
    assert "DocumentIndexType" in feature_visibility

    # Layout identity wiring — root / workspace / admin resolve the
    # authenticated user through `features/auth/server-api::getCurrentUser`.
    for layout_path in (
        REPO_ROOT / "web/src/app/layout.tsx",
        REPO_ROOT / "web/src/app/workspace/layout.tsx",
        REPO_ROOT / "web/src/app/admin/layout.tsx",
    ):
        layout_source = layout_path.read_text()
        assert "from '@/api'" not in layout_source, f"{layout_path} still imports legacy @/api"
        assert "getCurrentUser" in layout_source, (
            f"{layout_path} must resolve the user via features/auth/server-api::getCurrentUser"
        )

    # Phase 1c deletion — legacy SDK tree + low-level wrappers are gone.
    assert not (REPO_ROOT / "web/src/api").exists()
    assert not (REPO_ROOT / "web/src/lib/api/client.ts").exists()
    assert not (REPO_ROOT / "web/src/lib/api/server.ts").exists()
