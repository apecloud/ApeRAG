"""hard cut message feedback into turn feedback

Revision ID: f17a6b9c2d4e
Revises: a1e2f3d4c5b6
Create Date: 2026-04-21 23:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f17a6b9c2d4e"
down_revision: Union[str, None] = "a1e2f3d4c5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.rename_table("message_feedback", "turn_feedback")
    op.execute("ALTER INDEX IF EXISTS message_feedback_pkey RENAME TO turn_feedback_pkey")

    op.drop_constraint("uq_feedback_chat_message_deleted", "turn_feedback", type_="unique")
    op.drop_index(op.f("ix_message_feedback_user"), table_name="turn_feedback")
    op.drop_index(op.f("ix_message_feedback_status"), table_name="turn_feedback")
    op.drop_index(op.f("ix_message_feedback_gmt_deleted"), table_name="turn_feedback")

    op.alter_column("turn_feedback", "message_id", new_column_name="turn_id")
    op.alter_column("turn_feedback", "type", existing_type=sa.String(length=50), nullable=False)

    op.drop_column("turn_feedback", "question")
    op.drop_column("turn_feedback", "status")
    op.drop_column("turn_feedback", "original_answer")
    op.drop_column("turn_feedback", "revised_answer")
    op.drop_column("turn_feedback", "gmt_deleted")

    op.create_index(op.f("ix_turn_feedback_user"), "turn_feedback", ["user"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_turn_feedback_user"), table_name="turn_feedback")

    op.add_column("turn_feedback", sa.Column("gmt_deleted", sa.DateTime(timezone=True), nullable=True))
    op.add_column("turn_feedback", sa.Column("revised_answer", sa.Text(), nullable=True))
    op.add_column("turn_feedback", sa.Column("original_answer", sa.Text(), nullable=True))
    op.add_column("turn_feedback", sa.Column("status", sa.String(length=50), nullable=True))
    op.add_column("turn_feedback", sa.Column("question", sa.Text(), nullable=True))

    op.alter_column("turn_feedback", "type", existing_type=sa.String(length=50), nullable=True)
    op.alter_column("turn_feedback", "turn_id", new_column_name="message_id")

    op.create_index(op.f("ix_message_feedback_gmt_deleted"), "turn_feedback", ["gmt_deleted"], unique=False)
    op.create_index(op.f("ix_message_feedback_status"), "turn_feedback", ["status"], unique=False)
    op.create_index(op.f("ix_message_feedback_user"), "turn_feedback", ["user"], unique=False)
    op.create_unique_constraint(
        "uq_feedback_chat_message_deleted", "turn_feedback", ["chat_id", "message_id", "gmt_deleted"]
    )

    op.execute("ALTER INDEX IF EXISTS turn_feedback_pkey RENAME TO message_feedback_pkey")
    op.rename_table("turn_feedback", "message_feedback")
