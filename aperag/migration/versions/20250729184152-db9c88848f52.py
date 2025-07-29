"""Create PostgreSQL extensions (pgvector)

Revision ID: db9c88848f52
Revises: 
Create Date: 2025-07-29 18:41:52.296210

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from aperag.migration.utils import execute_sql_file


# revision identifiers, used by Alembic.
revision: str = 'db9c88848f52'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create PostgreSQL extensions."""
    # Execute extensions initialization SQL
    execute_sql_file("extensions_init.sql")


def downgrade() -> None:
    """Drop PostgreSQL extensions."""
    # Note: Dropping extensions should be done carefully in production
    # as it may affect existing data and other databases
    op.execute(sa.text("DROP EXTENSION IF EXISTS vector CASCADE"))
