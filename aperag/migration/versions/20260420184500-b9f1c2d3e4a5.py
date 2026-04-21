"""add agent runtime v3 tables

Revision ID: b9f1c2d3e4a5
Revises: a1b2c3d4e5f6
Create Date: 2026-04-20 18:45:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b9f1c2d3e4a5"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_turn",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("chat_id", sa.String(length=24), nullable=False),
        sa.Column("user", sa.String(length=256), nullable=False),
        sa.Column("bot_id", sa.String(length=24), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("client_idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("model_profile", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("answer_artifact_id", sa.String(length=24), nullable=True),
        sa.Column("reference_bundle_artifact_id", sa.String(length=24), nullable=True),
        sa.Column("timeline_cursor", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("gmt_started", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gmt_finished", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_turn")),
        sa.UniqueConstraint("chat_id", "client_idempotency_key", name="uq_agent_turn_chat_idempotency"),
    )
    op.create_index(op.f("ix_agent_turn_chat_id"), "agent_turn", ["chat_id"], unique=False)
    op.create_index(op.f("ix_agent_turn_user"), "agent_turn", ["user"], unique=False)
    op.create_index(op.f("ix_agent_turn_bot_id"), "agent_turn", ["bot_id"], unique=False)
    op.create_index(op.f("ix_agent_turn_request_id"), "agent_turn", ["request_id"], unique=True)
    op.create_index(op.f("ix_agent_turn_status"), "agent_turn", ["status"], unique=False)
    op.create_index(op.f("ix_agent_turn_answer_artifact_id"), "agent_turn", ["answer_artifact_id"], unique=False)
    op.create_index(
        op.f("ix_agent_turn_reference_bundle_artifact_id"),
        "agent_turn",
        ["reference_bundle_artifact_id"],
        unique=False,
    )
    op.create_index("idx_agent_turn_chat_created", "agent_turn", ["chat_id", "gmt_created"], unique=False)
    op.create_index("idx_agent_turn_user_status", "agent_turn", ["user", "status"], unique=False)

    op.create_table(
        "agent_timeline_event",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("turn_id", sa.String(length=24), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("type", sa.String(length=128), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=True),
        sa.Column("actor", sa.String(length=50), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_timeline_event")),
        sa.UniqueConstraint("turn_id", "sequence", name="uq_agent_timeline_event_turn_sequence"),
    )
    op.create_index(op.f("ix_agent_timeline_event_turn_id"), "agent_timeline_event", ["turn_id"], unique=False)
    op.create_index(op.f("ix_agent_timeline_event_timestamp"), "agent_timeline_event", ["timestamp"], unique=False)
    op.create_index(op.f("ix_agent_timeline_event_type"), "agent_timeline_event", ["type"], unique=False)
    op.create_index(
        "idx_agent_timeline_event_turn_timestamp", "agent_timeline_event", ["turn_id", "timestamp"], unique=False
    )

    op.create_table(
        "agent_artifact",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("turn_id", sa.String(length=24), nullable=False),
        sa.Column("artifact_type", sa.String(length=50), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("storage_ref", sa.Text(), nullable=True),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_artifact")),
    )
    op.create_index(op.f("ix_agent_artifact_turn_id"), "agent_artifact", ["turn_id"], unique=False)
    op.create_index(op.f("ix_agent_artifact_artifact_type"), "agent_artifact", ["artifact_type"], unique=False)
    op.create_index("idx_agent_artifact_turn_type", "agent_artifact", ["turn_id", "artifact_type"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_agent_artifact_turn_type", table_name="agent_artifact")
    op.drop_index(op.f("ix_agent_artifact_artifact_type"), table_name="agent_artifact")
    op.drop_index(op.f("ix_agent_artifact_turn_id"), table_name="agent_artifact")
    op.drop_table("agent_artifact")

    op.drop_index("idx_agent_timeline_event_turn_timestamp", table_name="agent_timeline_event")
    op.drop_index(op.f("ix_agent_timeline_event_type"), table_name="agent_timeline_event")
    op.drop_index(op.f("ix_agent_timeline_event_timestamp"), table_name="agent_timeline_event")
    op.drop_index(op.f("ix_agent_timeline_event_turn_id"), table_name="agent_timeline_event")
    op.drop_table("agent_timeline_event")

    op.drop_index("idx_agent_turn_user_status", table_name="agent_turn")
    op.drop_index("idx_agent_turn_chat_created", table_name="agent_turn")
    op.drop_index(op.f("ix_agent_turn_reference_bundle_artifact_id"), table_name="agent_turn")
    op.drop_index(op.f("ix_agent_turn_answer_artifact_id"), table_name="agent_turn")
    op.drop_index(op.f("ix_agent_turn_status"), table_name="agent_turn")
    op.drop_index(op.f("ix_agent_turn_request_id"), table_name="agent_turn")
    op.drop_index(op.f("ix_agent_turn_bot_id"), table_name="agent_turn")
    op.drop_index(op.f("ix_agent_turn_user"), table_name="agent_turn")
    op.drop_index(op.f("ix_agent_turn_chat_id"), table_name="agent_turn")
    op.drop_table("agent_turn")
