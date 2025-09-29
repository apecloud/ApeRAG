"""add use_markitdown as default setting

Revision ID: ef8cf2222205
Revises: d112e0332219
Create Date: 2025-09-29 18:37:57.365896

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ef8cf2222205'
down_revision: Union[str, None] = 'd112e0332219'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("INSERT INTO setting (key, value, gmt_created, gmt_updated) VALUES ('use_markitdown', 'true', NOW(), NOW())")


def downgrade() -> None:
    op.execute("DELETE FROM setting WHERE key = 'use_markitdown'")
