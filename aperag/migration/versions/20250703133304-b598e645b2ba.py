"""Initialize LLM model configurations

Revision ID: b598e645b2ba
Revises: 0b274fcc91e2
Create Date: 2025-07-03 13:33:04.315167

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from aperag.migration.utils import execute_sql_file


# revision identifiers, used by Alembic.
revision: str = 'b598e645b2ba'
down_revision: Union[str, None] = '0b274fcc91e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Initialize LLM model configurations data."""
    # Execute model configurations initialization SQL
    execute_sql_file("model_configs_init.sql")


def downgrade() -> None:
    """Remove LLM model configurations data."""
    # Clean up model configurations data
    op.execute(sa.text("DELETE FROM llm_provider_models"))
    op.execute(sa.text("DELETE FROM llm_provider"))
