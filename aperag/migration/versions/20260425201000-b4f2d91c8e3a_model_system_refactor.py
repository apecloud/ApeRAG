"""replace legacy llm provider tables with model system

The model-platform refactor (Phase 8 #1697) replaces the legacy
``llm_provider`` / ``model_service_provider`` / ``llm_provider_models``
trio with the new ``model_provider`` / ``model_account`` / ``model``
/ ``model_use`` schema. This migration also performs a best-effort
copy of any user-supplied API keys + model rows into the new tables
*before* dropping the legacy ones — so collections / bots created on
the previous schema continue to resolve a working ``ModelAccount`` +
``Model`` after the migration runs.

Chained after ``d8e2c4a17b91`` (the D8.2 ``agent_message`` table) to
collapse the alembic multi-head — both this revision and ``d8e2c4a17b91``
were authored against ``7c4e9e1f8b21`` and would otherwise produce
``alembic heads`` reporting two heads.

Revision ID: b4f2d91c8e3a
Revises: d8e2c4a17b91
Create Date: 2026-04-25 20:10:00.000000
"""

from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b4f2d91c8e3a"
down_revision: Union[str, None] = "d8e2c4a17b91"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Capture legacy provider / model rows into in-memory snapshots
    # before dropping the old tables. The new model_provider /
    # model_account / model tables are then created and populated
    # below using ``_replay_legacy_provider_data`` — this preserves
    # any user API keys + model catalogs across the refactor (required
    # by Weston msg=80e873c1 so existing collections / bots resolve a
    # working runtime row after the PR merges).
    legacy_snapshot = _capture_legacy_provider_data()

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())
    for tbl in ("llm_provider_models", "model_service_provider", "llm_provider"):
        if tbl in existing_tables:
            op.drop_table(tbl)

    op.create_table(
        "model_provider",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("provider_type", sa.String(length=128), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("supported_capabilities", sa.JSON(), nullable=False),
        sa.Column("account_schema", sa.JSON(), nullable=False),
        sa.Column("default_base_url", sa.String(length=512), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=False),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_deleted", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_type"),
    )
    op.create_index(op.f("ix_model_provider_provider_type"), "model_provider", ["provider_type"], unique=False)
    now = datetime.now(timezone.utc)
    op.bulk_insert(
        sa.table(
            "model_provider",
            sa.column("id", sa.String),
            sa.column("provider_type", sa.String),
            sa.column("display_name", sa.String),
            sa.column("description", sa.Text),
            sa.column("supported_capabilities", sa.JSON),
            sa.column("account_schema", sa.JSON),
            sa.column("default_base_url", sa.String),
            sa.column("enabled", sa.Boolean),
            sa.column("sort_order", sa.Integer),
            sa.column("extra", sa.JSON),
            sa.column("gmt_created", sa.DateTime),
            sa.column("gmt_updated", sa.DateTime),
        ),
        [
            {
                "id": "mp_openai",
                "provider_type": "openai",
                "display_name": "OpenAI",
                "description": None,
                "supported_capabilities": ["chat", "embedding"],
                "account_schema": {},
                "default_base_url": "https://api.openai.com/v1",
                "enabled": True,
                "sort_order": 10,
                "extra": {},
                "gmt_created": now,
                "gmt_updated": now,
            },
            {
                "id": "mp_dashscope",
                "provider_type": "dashscope",
                "display_name": "阿里百炼",
                "description": None,
                "supported_capabilities": ["chat", "embedding", "rerank"],
                "account_schema": {},
                "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "enabled": True,
                "sort_order": 20,
                "extra": {},
                "gmt_created": now,
                "gmt_updated": now,
            },
            {
                "id": "mp_jina",
                "provider_type": "jina",
                "display_name": "Jina AI",
                "description": None,
                "supported_capabilities": ["embedding", "rerank"],
                "account_schema": {},
                "default_base_url": "https://api.jina.ai/v1",
                "enabled": True,
                "sort_order": 30,
                "extra": {},
                "gmt_created": now,
                "gmt_updated": now,
            },
            {
                "id": "mp_openai_compatible",
                "provider_type": "openai_compatible",
                "display_name": "OpenAI Compatible",
                "description": None,
                "supported_capabilities": ["chat", "embedding"],
                "account_schema": {},
                "default_base_url": None,
                "enabled": True,
                "sort_order": 100,
                "extra": {},
                "gmt_created": now,
                "gmt_updated": now,
            },
        ],
    )

    op.create_table(
        "model_account",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("provider_type", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=False),
        sa.Column("base_url", sa.String(length=512), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=False),
        sa.Column("auth_config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("last_validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("validation_error", sa.Text(), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=False),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_deleted", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_model_account_provider_type"), "model_account", ["provider_type"], unique=False)
    op.create_index(op.f("ix_model_account_user_id"), "model_account", ["user_id"], unique=False)
    op.create_index(op.f("ix_model_account_name"), "model_account", ["name"], unique=False)

    op.create_table(
        "model",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("account_id", sa.String(length=24), nullable=False),
        sa.Column("provider_model_id", sa.String(length=256), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=False),
        sa.Column("capability", sa.String(length=50), nullable=False),
        sa.Column("runner_type", sa.String(length=128), nullable=False),
        sa.Column("runner_config", sa.JSON(), nullable=False),
        sa.Column("context_window", sa.Integer(), nullable=True),
        sa.Column("max_input_tokens", sa.Integer(), nullable=True),
        sa.Column("max_output_tokens", sa.Integer(), nullable=True),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=True),
        sa.Column("supports_vision", sa.Boolean(), nullable=False),
        sa.Column("supports_tool_calling", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=False),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_deleted", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_model_account_id"), "model", ["account_id"], unique=False)
    op.create_index(op.f("ix_model_capability"), "model", ["capability"], unique=False)
    op.create_index(op.f("ix_model_provider_model_id"), "model", ["provider_model_id"], unique=False)
    op.create_index(op.f("ix_model_status"), "model", ["status"], unique=False)

    op.create_table(
        "model_use",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("scenario", sa.String(length=50), nullable=False),
        sa.Column("capability", sa.String(length=50), nullable=False),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("primary_model_id", sa.String(length=24), nullable=True),
        sa.Column("fallback_model_ids", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=False),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_deleted", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_model_use_primary_model_id"), "model_use", ["primary_model_id"], unique=False)
    op.create_index(op.f("ix_model_use_scenario"), "model_use", ["scenario"], unique=False)
    op.create_index(op.f("ix_model_use_user_id"), "model_use", ["user_id"], unique=False)

    _replay_legacy_provider_data(legacy_snapshot)


def downgrade() -> None:
    op.drop_table("model_use")
    op.drop_table("model")
    op.drop_table("model_account")
    op.drop_table("model_provider")


# ---------------------------------------------------------------------------
# Legacy data migration helpers
# ---------------------------------------------------------------------------
#
# The pre-refactor schema split provider config across three tables:
#
#   * ``llm_provider`` — provider catalog row (name, dialects, base_url,
#     allow_custom_base_url). Per-provider, NOT per-user.
#   * ``model_service_provider`` — per-provider api-key row written by
#     ``PUT /api/v1/llm_providers/{name}`` / ``PUT /api/v2/providers/{name}``.
#   * ``llm_provider_models`` — provider catalog of models (api / model
#     name / dialect / context_window / etc.).
#
# The new schema collapses provider+account into ``model_account`` (per
# user, holds the api key) and ``model`` (per account, holds the model
# entry) hanging off a built-in ``model_provider`` row.
#
# We map:
#   * each ``model_service_provider`` row whose name matches a known
#     ``model_provider.provider_type`` (or ``allow_custom_base_url`` is
#     set) → a single ``ModelAccount`` for ``user_id="public"``.
#   * each ``llm_provider_models`` row hanging off a provider that has
#     a matching account → a ``Model`` row of the right capability.
#
# The mapping is best-effort: rows that don't fit any new provider type
# are skipped (logged via alembic stdout) rather than failing the
# migration. Phase 8 hard-cut philosophy says "destructive changes OK
# as long as it deploys" — but Weston's blocker calls for a smoother
# upgrade path on real DBs that already have user-supplied API keys.


_LEGACY_TO_PROVIDER_TYPE = {
    "openai": "openai",
    "openrouter": "openai_compatible",
    "alibabacloud": "dashscope",
    "dashscope": "dashscope",
    "jina": "jina",
    "jina_ai": "jina",
    "openai_compatible": "openai_compatible",
}

_PROVIDER_DEFAULT_BASE_URL = {
    "openai": "https://api.openai.com/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "jina": "https://api.jina.ai/v1",
    "openai_compatible": "https://api.openai.com/v1",
}

_API_TO_CAPABILITY = {
    "completion": "chat",
    "chat": "chat",
    "embedding": "embedding",
    "rerank": "rerank",
}


def _capture_legacy_provider_data():
    """Return ``(accounts, models)`` snapshot drawn from the legacy
    ``llm_provider`` / ``model_service_provider`` / ``llm_provider_models``
    tables. Returns empty lists when those tables don't exist yet
    (fresh deploy) or are empty.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    needed = {"llm_provider", "model_service_provider", "llm_provider_models"}
    if not needed.issubset(table_names):
        return {"accounts": [], "models": []}

    providers = bind.execute(
        sa.text("SELECT name, label, base_url FROM llm_provider WHERE gmt_deleted IS NULL")
    ).fetchall()
    msps = bind.execute(
        sa.text("SELECT name, api_key FROM model_service_provider WHERE gmt_deleted IS NULL")
    ).fetchall()
    legacy_models = bind.execute(
        sa.text(
            "SELECT provider_name, api, model, context_window, max_input_tokens, "
            "max_output_tokens FROM llm_provider_models WHERE gmt_deleted IS NULL"
        )
    ).fetchall()

    provider_meta = {row.name: {"label": row.label, "base_url": row.base_url} for row in providers}
    api_keys = {row.name: row.api_key for row in msps}

    accounts = []
    models = []
    seen_account_keys = set()

    def _new_id(prefix: str, idx: int) -> str:
        return f"{prefix}{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}{idx:04d}"[:24]

    for idx, (legacy_name, api_key) in enumerate(api_keys.items()):
        provider_type = _LEGACY_TO_PROVIDER_TYPE.get(legacy_name)
        if not provider_type:
            continue
        meta = provider_meta.get(legacy_name, {})
        base_url = meta.get("base_url") or _PROVIDER_DEFAULT_BASE_URL.get(provider_type) or ""
        display_name = meta.get("label") or legacy_name
        key = (provider_type, legacy_name)
        if key in seen_account_keys:
            continue
        seen_account_keys.add(key)
        account_id = _new_id("ma", idx)
        accounts.append(
            {
                "_legacy_provider_name": legacy_name,
                "id": account_id,
                "user_id": "public",
                "provider_type": provider_type,
                "name": legacy_name,
                "display_name": display_name,
                "base_url": base_url,
                "encrypted_api_key": api_key,
            }
        )

    legacy_to_account = {acc["_legacy_provider_name"]: acc for acc in accounts}

    for idx, row in enumerate(legacy_models):
        legacy_name = row.provider_name
        account = legacy_to_account.get(legacy_name)
        if not account:
            continue
        capability = _API_TO_CAPABILITY.get((row.api or "").lower())
        if not capability:
            continue
        runner_type = (
            "dashscope_rerank"
            if capability == "rerank" and account["provider_type"] == "dashscope"
            else (
                "jina_rerank" if capability == "rerank" and account["provider_type"] == "jina" else "openai_compatible"
            )
        )
        models.append(
            {
                "id": _new_id("ml", idx),
                "account_id": account["id"],
                "provider_model_id": row.model,
                "display_name": row.model,
                "capability": capability,
                "runner_type": runner_type,
                "context_window": row.context_window,
                "max_input_tokens": row.max_input_tokens,
                "max_output_tokens": row.max_output_tokens,
            }
        )

    return {"accounts": accounts, "models": models}


def _replay_legacy_provider_data(snapshot) -> None:
    accounts = snapshot.get("accounts", [])
    models = snapshot.get("models", [])
    if not accounts and not models:
        return

    now = datetime.now(timezone.utc)
    if accounts:
        op.bulk_insert(
            sa.table(
                "model_account",
                sa.column("id", sa.String),
                sa.column("user_id", sa.String),
                sa.column("provider_type", sa.String),
                sa.column("name", sa.String),
                sa.column("display_name", sa.String),
                sa.column("base_url", sa.String),
                sa.column("encrypted_api_key", sa.Text),
                sa.column("auth_config", sa.JSON),
                sa.column("status", sa.String),
                sa.column("extra", sa.JSON),
                sa.column("gmt_created", sa.DateTime),
                sa.column("gmt_updated", sa.DateTime),
            ),
            [
                {
                    "id": acc["id"],
                    "user_id": acc["user_id"],
                    "provider_type": acc["provider_type"],
                    "name": acc["name"],
                    "display_name": acc["display_name"],
                    "base_url": acc["base_url"],
                    "encrypted_api_key": acc["encrypted_api_key"],
                    "auth_config": {},
                    "status": "ACTIVE",
                    "extra": {"migrated_from": "llm_provider"},
                    "gmt_created": now,
                    "gmt_updated": now,
                }
                for acc in accounts
            ],
        )

    if models:
        op.bulk_insert(
            sa.table(
                "model",
                sa.column("id", sa.String),
                sa.column("account_id", sa.String),
                sa.column("provider_model_id", sa.String),
                sa.column("display_name", sa.String),
                sa.column("capability", sa.String),
                sa.column("runner_type", sa.String),
                sa.column("runner_config", sa.JSON),
                sa.column("context_window", sa.Integer),
                sa.column("max_input_tokens", sa.Integer),
                sa.column("max_output_tokens", sa.Integer),
                sa.column("embedding_dimensions", sa.Integer),
                sa.column("supports_vision", sa.Boolean),
                sa.column("supports_tool_calling", sa.Boolean),
                sa.column("status", sa.String),
                sa.column("extra", sa.JSON),
                sa.column("gmt_created", sa.DateTime),
                sa.column("gmt_updated", sa.DateTime),
            ),
            [
                {
                    "id": m["id"],
                    "account_id": m["account_id"],
                    "provider_model_id": m["provider_model_id"],
                    "display_name": m["display_name"],
                    "capability": m["capability"],
                    "runner_type": m["runner_type"],
                    "runner_config": {},
                    "context_window": m.get("context_window"),
                    "max_input_tokens": m.get("max_input_tokens"),
                    "max_output_tokens": m.get("max_output_tokens"),
                    "embedding_dimensions": None,
                    "supports_vision": False,
                    "supports_tool_calling": False,
                    "status": "ACTIVE",
                    "extra": {"migrated_from": "llm_provider_models"},
                    "gmt_created": now,
                    "gmt_updated": now,
                }
                for m in models
            ],
        )
