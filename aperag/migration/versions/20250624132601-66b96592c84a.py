"""empty message

Revision ID: 66b96592c84a
Revises: 850b2c5dc08f
Create Date: 2025-06-24 13:26:01.031627

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from aperag.migration.utils import execute_sql_file


# revision identifiers, used by Alembic.
revision: str = '66b96592c84a'
down_revision: Union[str, None] = '850b2c5dc08f'
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
