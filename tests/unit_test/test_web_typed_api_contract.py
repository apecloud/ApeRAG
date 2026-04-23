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

    # Title generate must build a fully-typed TitleGenerateRequest body. If a
    # future edit drops either field or skips the locale normaliser, the
    # generated TS schema would quietly allow an `{ language }`-only body and
    # re-introduce the type risk @ApeRAG专家 caught on ff89876.
    bot_client_api = (
        REPO_ROOT / "web/src/features/bot/client-api.ts"
    ).read_text()
    assert "buildTitleGenerateRequest" in bot_client_api
    assert "toTitleLanguage" in bot_client_api
    assert "max_length: input.max_length ?? null" in bot_client_api
    assert "turns: input.turns ?? null" in bot_client_api
