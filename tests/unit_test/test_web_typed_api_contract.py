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


def test_evaluation_feature_uses_v2_typed_api_boundary():
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
        source
        for path, source in sources.items()
        if "/web/src/features/evaluation/" in str(path)
    )

    assert "/api/v1/evaluations" not in joined
    assert "/api/v1/question-sets" not in joined
    assert "apiClient.evaluationApi" not in joined
    assert "from '@/api'" not in feature_sources
    assert "fetch(" not in feature_sources
    assert "/api/v2/benchmark-datasets" in feature_sources
    assert "/api/v2/evaluation-runs" in feature_sources

    benchmarks_panel = (
        REPO_ROOT / "web/src/components/evaluation/benchmarks-panel.tsx"
    ).read_text()
    assert "String(value ?? '').toLowerCase()" in benchmarks_panel


def test_bot_feature_uses_v2_typed_api_boundary():
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
    feature_sources = "\n".join(
        source
        for path, source in sources.items()
        if "/web/src/features/bot/" in str(path)
    )

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
    bot_client_api = (
        REPO_ROOT / "web/src/features/bot/client-api.ts"
    ).read_text()
    assert "buildTitleGenerateRequest" in bot_client_api
    assert "toTitleLanguage" in bot_client_api
    assert "DEFAULT_TITLE_MAX_LENGTH = 20" in bot_client_api
    assert "DEFAULT_TITLE_TURNS = 1" in bot_client_api
    assert "DEFAULT_TITLE_LANGUAGE: TitleLanguage = 'zh-CN'" in bot_client_api
    assert (
        "max_length: input.max_length ?? DEFAULT_TITLE_MAX_LENGTH"
        in bot_client_api
    )
    assert "turns: input.turns ?? DEFAULT_TITLE_TURNS" in bot_client_api
    # Guard the negative case: no `?? null` fallback for any of the three
    # required keys should reappear.
    assert "max_length: input.max_length ?? null" not in bot_client_api
    assert "turns: input.turns ?? null" not in bot_client_api


def test_collection_feature_uses_v2_typed_api_boundary():
    """#24a Collection + Sharing FE typed adapter boundary.

    Business code under these paths must not call the old generated
    `defaultApi.collections*` / `defaultApi.collectionsCollectionId*` /
    `defaultApi.collectionsCollectionIdSharing*` SDK directly, and the
    `features/collection/*` adapter must only reach `/api/v2/collections*`
    typed paths (no `@/api` fallback, no raw `fetch(`). Document-specific
    routes (`/documents*`) are deliberately out of scope — they stay with
    the documents slice and the upload-flow hotfix.
    """
    checked_paths = [
        REPO_ROOT / "web/src/app/workspace/collections/page.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/collection-form.tsx",
        REPO_ROOT / "web/src/app/workspace/collections/[collectionId]/layout.tsx",
        REPO_ROOT
        / "web/src/app/workspace/collections/[collectionId]/collection-delete.tsx",
        REPO_ROOT
        / "web/src/app/workspace/collections/[collectionId]/collection-header.tsx",
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
        source
        for path, source in sources.items()
        if "/web/src/features/collection/" in str(path)
    )

    # Business code under #24a scope must not reach the old generated
    # collection SDK or the v1 collections routes directly.
    assert "defaultApi.collectionsGet" not in joined
    assert "defaultApi.collectionsPost" not in joined
    assert "defaultApi.collectionsCollectionIdGet" not in joined
    assert "defaultApi.collectionsCollectionIdPut" not in joined
    assert "defaultApi.collectionsCollectionIdDelete" not in joined
    assert "defaultApi.collectionsCollectionIdSharingGet" not in joined
    assert "defaultApi.collectionsCollectionIdSharingPost" not in joined
    assert "defaultApi.collectionsCollectionIdSharingDelete" not in joined

    # features/collection adapter must only reach v2 typed paths and not
    # fall back to the old `@/api` generated SDK or raw fetch.
    assert "from '@/api'" not in feature_sources
    assert "fetch(" not in feature_sources
    assert "/api/v2/collections" in feature_sources


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
        REPO_ROOT
        / "web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx"
    ).read_text()
    assert "() => stopUpload()" not in upload_tsx
    assert "() => () => stopUpload" not in upload_tsx

    table_tsx = (
        REPO_ROOT
        / "web/src/app/workspace/collections/[collectionId]/documents/documents-table.tsx"
    ).read_text()
    assert "@ts-expect-error onPaginationChange" not in table_tsx
    assert "typeof updater === 'function'" in table_tsx
