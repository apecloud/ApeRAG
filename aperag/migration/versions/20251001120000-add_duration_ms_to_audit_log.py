"""add duration_ms to audit_log

Revision ID: a1b2c3d4e5f6
Revises: ef8cf2222205
Create Date: 2025-10-01 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "ef8cf2222205"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add duration_ms column and its index to audit_log.

    The column stores the pre-computed request duration (end_time - start_time)
    in milliseconds so it can be sorted and filtered efficiently without
    requiring a computed expression index.

    Existing rows will have duration_ms = NULL.  The application back-fills the
    value on-the-fly when it lists audit logs, so no data migration is needed.
    """
    op.add_column(
        "audit_log",
        sa.Column(
            "duration_ms",
            sa.BigInteger(),
            nullable=True,
            comment="Request duration in milliseconds (end_time - start_time)",
        ),
    )
    op.create_index("idx_audit_duration_ms", "audit_log", ["duration_ms"])


def downgrade() -> None:
    op.drop_index("idx_audit_duration_ms", table_name="audit_log")
    op.drop_column("audit_log", "duration_ms")
