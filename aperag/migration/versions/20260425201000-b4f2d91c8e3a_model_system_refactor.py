"""replace legacy llm provider tables with model system

Revision ID: b4f2d91c8e3a
Revises: 7c4e9e1f8b21
Create Date: 2026-04-25 20:10:00.000000
"""

from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b4f2d91c8e3a"
down_revision: Union[str, None] = "7c4e9e1f8b21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("llm_provider_models")
    op.drop_table("model_service_provider")
    op.drop_table("llm_provider")

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


def downgrade() -> None:
    op.drop_table("model_use")
    op.drop_table("model")
    op.drop_table("model_account")
    op.drop_table("model_provider")
