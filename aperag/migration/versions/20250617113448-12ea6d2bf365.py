"""Initialize model configurations data

Revision ID: 12ea6d2bf365
Revises: 6b5cab1cd8d8
Create Date: 2025-06-11 22:35:16.362747

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from aperag.migration.utils import execute_sql_file


# revision identifiers, used by Alembic.
revision: str = '12ea6d2bf365'
down_revision: Union[str, None] = 'dcc0b6c56552'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Initialize model configurations data."""
    # Execute model configurations initialization SQL
    execute_sql_file("model_configs_init.sql")


def downgrade() -> None:
    """Remove model configurations data."""
    # Clean up model configurations data
    op.execute(sa.text("DELETE FROM llm_provider_models"))
    op.execute(sa.text("DELETE FROM llm_provider"))
