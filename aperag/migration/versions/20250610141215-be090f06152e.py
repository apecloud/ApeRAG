"""empty message

Revision ID: be090f06152e
Revises: 6b261afb99c8
Create Date: 2025-06-10 14:12:15.758254

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from aperag.migration.utils import execute_sql_file


# revision identifiers, used by Alembic.
revision: str = 'be090f06152e'
down_revision: Union[str, None] = '6b261afb99c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    execute_sql_file("model_configs_init.sql")


def downgrade() -> None:
    """Downgrade schema."""
    pass
