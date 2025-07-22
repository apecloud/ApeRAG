"""Merge branches

Revision ID: 4873db3367ce
Revises: 56a24ace49af, b3a2c218442f
Create Date: 2025-07-22 21:18:36.375134

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4873db3367ce'
down_revision: Union[str, None] = ('56a24ace49af', 'b3a2c218442f')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
